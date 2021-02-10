import cv2
import numpy as np
import os
import glob
from optical_flow import *
from scipy.ndimage.filters import gaussian_filter
from scipy import ndimage
from scipy.ndimage import convolve
# todo: make independent of scipy, if possible
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment


# from dvs import DVS
import scipy.io as sio
import matplotlib.pyplot as plt
# todo: make independent of the dvs module


EVENT_THRESH = 0.2
# Standard deviation of the threshold
# According to Lichtsteiner et al '08, this should be approx. 2.1% for thresh < 0.35, and up to 3.5% for thresh = 0.6
SIGMA_THRESH = (2.1E-2) * EVENT_THRESH

# Constant to convert from brightness to photocurrent
PHOTO_CURRENT_CONST = 1.5E-2


def get_events_from_frames(cur_gray, cur_t, prev_gray, prev_t, prev_sensor_val=None):
    """
    Summary line.

    This is the simulation of the event camera from grayscale images

    Parameters:
    ------------
    cur_gray: current grayscale image at time cur_t
    cur_t: more recent time, cur_t > prev_t
    prev_gray: previous grayscale image at time prev_t
    prev_t: previous time, prev_t < cur_t
    prev_sensor_val: sensor val at the previous frame

    Returns:
    ------------
    4xN array of events, number of events as an array the size of the image, and the final sensor value
    """

    if prev_sensor_val is None:
        prev_sensor_val = np.zeros_like(cur_gray)

    # Probabilistically generate the threshold values for the individual pixels
    thresholds = np.random.normal(EVENT_THRESH, SIGMA_THRESH, cur_gray.shape)

    dt = cur_t - prev_t

    # Compute photocurrent for both frames (I_1 and I_2)
    # threshold each of them to a minimum of 1, to avoid things blowing up if we take the log of 0
    photocurrent_cur = np.maximum(PHOTO_CURRENT_CONST * cur_gray, 1)
    photocurrent_prev = np.maximum(PHOTO_CURRENT_CONST * prev_gray, 1)

    # Check change in photocurrent to see how many events would be triggered assuming first order hold
    # in between previous frame and current frame

    # Change in current from frame to frame
    # Note: num_events has negative entries for negative changes and positive entries otherwise
    # num_events = np.floor_divide(np.log(I_2) - self.prev_sensor_val, self.thresholds).astype('int')
    g = np.log(photocurrent_cur) - np.log(photocurrent_prev) + prev_sensor_val
    num_events = (np.floor_divide(np.abs(g), thresholds) * np.sign(g)).astype('int')

    # The value for the last event in this frame (if any)
    final_sensor_val = g - thresholds * num_events

    event_idx = (abs(num_events) > 0)

    # get list of event positions and counts
    event_positions = np.indices(cur_gray.shape)[:, event_idx]
    event_count = num_events[event_idx]
    abs_count = np.abs(event_count)

    if len(event_count) > 0:
        all_positions = np.repeat(event_positions, abs_count, axis=1)

        # get all event times. assume multiple events at a single position are linearly spaced over time
        cum_count = np.cumsum(abs_count)
        cum_count_rolled = np.roll(cum_count, 1)
        cum_count_rolled[0] = 0
        count_rep = np.repeat(abs_count, abs_count)
        time_index = np.arange(all_positions.shape[1]) + 1 - np.repeat(cum_count_rolled, abs_count)
        all_times = cur_t - (time_index - 1) / count_rep * dt

        # get all event polarities
        all_event_types = np.repeat(np.sign(event_count), abs_count)

        # stack event data into a single data structure for this frame and for all history
        return np.vstack((all_positions, all_times, all_event_types)), num_events, final_sensor_val
    #
    return [], num_events, final_sensor_val


def get_edge_index(event_density, grad_ed, sigma_1=0.2, sigma_2=0.3):
    edge_index = abs(event_density) > sigma_2
    return edge_index


def get_circle_kernel(diameter, array_len=None):
    if array_len is None:
        array_len = diameter
    c = round((array_len - 1) / 2)
    a = np.zeros((array_len, array_len, 3))
    if diameter > 0:
        a[c, c, 1] = 1
    for i in range(round((diameter-1)/2)):
        a = ndimage.binary_dilation(a, structure=ndimage.generate_binary_structure(3, 1)).astype(a.dtype)
    return a


def get_smoothing_kernel(inner_size=1, outer_size=5):
    a = get_circle_kernel(outer_size)
    b = get_circle_kernel(inner_size, array_len=outer_size)
    c = a - b
    return c


def visualize_events(event_count_array, thresh=2):
    positive_events = np.copy(event_count_array)
    positive_events[positive_events < thresh] = 0
    negative_events = - np.copy(event_count_array)
    negative_events[negative_events < thresh] = 0
    rgb_event_img = np.zeros((event_count_array.shape[0], event_count_array.shape[1], 3))
    rgb_event_img[:, :, 0] = negative_events * 255 / np.max((np.max(negative_events), 1))
    rgb_event_img[:, :, 2] = positive_events * 255 / np.max((np.max(positive_events), 1))
    return rgb_event_img


def get_event_optical_flow(all_events, flow_shape, bins, t, dt=1./300, window_size=10):

    on_events = all_events[:, all_events[3, :] > 0]
    off_events = all_events[:, all_events[3, :] < 0]

    H_on, edges_on = np.histogramdd(on_events[0:3, :].T, bins=bins)
    H_off, edges_off = np.histogramdd(off_events[0:3, :].T, bins=bins)

    hist1 = H_on - H_off
    # hist = num_events

    # Scale up the firing rate for units in ms
    firing_rate = 5*hist1 * (1. / dt)
    avg_firing_rate = gaussian_filter(firing_rate, sigma=(0.8, 0.8, 2))

    grad_fr = np.gradient(avg_firing_rate, (3.5E-3) / 160, (3.5E-3) / 160, 1)
    # grad_fr = np.gradient(avg_firing_rate, (3.5E-3) / 160, (3.5E-3) / 160, t / window_size)
    edge_idx = get_edge_index(avg_firing_rate, grad_fr, 0.3, 0.3)

    u = -grad_fr[2] * grad_fr[1]
    v = -grad_fr[2] * grad_fr[0]

    u[np.isnan(u)] = 0
    v[np.isnan(v)] = 0
    # print('event_flow_size', u.shape)

    # todo: kernel size should be in terms of fraction of the sensor size
    smoothing_kernel = get_smoothing_kernel(inner_size=3, outer_size=7)
    smoothing_kernel_normalized = smoothing_kernel / np.sum(smoothing_kernel)
    u_avg = convolve(u, smoothing_kernel_normalized)
    v_avg = convolve(v, smoothing_kernel_normalized)

    u_edges = np.zeros(np.shape(u))
    v_edges = np.zeros(np.shape(v))

    u_edges[edge_idx] = u_avg[edge_idx]
    v_edges[edge_idx] = v_avg[edge_idx]

    # this is the optical flow
    v_x = u_edges
    v_y = v_edges

    # take the most recent optical flow estimate
    vx = v_x[:, :, -1]
    vy = v_y[:, :, -1]

    # add 0-padding to flow to match image size
    ### todo: why stack zeros into vx and vy???
    # print('flow_shape_inloop', flow_shape[0])
    n_rows = flow_shape[0] - vx.shape[0]
    n_cols = flow_shape[1] - vx.shape[1]
    vx = np.vstack((vx, np.zeros((n_rows, vx.shape[1]))))
    vy = np.vstack((vy, np.zeros((n_rows, vy.shape[1]))))
    vx = np.hstack((vx, np.zeros((vx.shape[0], n_cols))))
    vy = np.hstack((vy, np.zeros((vy.shape[0], n_cols))))

    flow_out = np.zeros((vx.shape[0], vx.shape[1], 2))
    flow_out[:, :, 0] = -vx
    flow_out[:, :, 1] = vy
    # return flow_out * .000000005
    return flow_out * .00000005
    # return flow_out * 5


