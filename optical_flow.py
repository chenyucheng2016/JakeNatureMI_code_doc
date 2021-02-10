import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import convolve
from event_camera import *


def get_flow_from_firing_rate(firing_rate):
    """
    Summary line.

    Compute the optical flow (vx, vy) from the firing rate of an event-camera.

    Parameters:
    -----------
    firing_rate: A 3-dimensional np.ndarray (dtype=double) of the firing rate


    Returns:
    -----------
    Tuple of np.ndarrays (vx, vy) of the optical flow, computed on the same grid as the firing rate
    """
    ft, fy, fx = np.gradient(firing_rate)

    grad_mag = np.sqrt(ft ** 2 + fy ** 2 + fx ** 2)
    ft_norm = ft/grad_mag
    fx_norm = fx/grad_mag
    fy_norm = fy/grad_mag

    firing_rate_grad_angle = np.arctan2(fy_norm, fx_norm)
    a1, a2, a3 = np.gradient(firing_rate_grad_angle)

    mag_a = np.sqrt(a1 ** 2 + a2 ** 2 + a3 ** 2)
    a1 /= mag_a
    a2 /= mag_a
    a3 /= mag_a

    b1 = ft_norm
    b2 = fy_norm
    b3 = fx_norm

    # The optical flow is the cross product of a (the gradient of firing_rate_grad_angle) with the firing rate gradient
    flow_1 = a2*b3 - a3*b2
    flow_2 = a3*b1 - a1*b3
    flow_3 = a1*b2 - a2*b1

    # Flip any vectors that are pointing the wrong way (backwards in time)
    flip_mask = flow_1 < 0
    flow_2[flip_mask] *= -1
    flow_3[flip_mask] *= -1
    flow_1[flip_mask] *= -1

    v_x = flow_3 / np.abs(flow_1)
    v_y = flow_2 / np.abs(flow_1)

    window = np.ones((5, 5, 5))
    weights = np.exp(-10 * (1 - np.abs(flow_1)))
    v_x_num = convolve(v_x*weights, window)
    v_y_num = convolve(v_y*weights, window)
    den = convolve(weights, window)

    v_x_smooth = np.zeros(flow_1.shape)
    v_y_smooth = np.zeros(flow_1.shape)
    mask_to_estimate = (firing_rate > 0.015)
    v_x_smooth[mask_to_estimate] = (v_x_num[mask_to_estimate] / den[mask_to_estimate])
    v_y_smooth[mask_to_estimate] = (v_y_num[mask_to_estimate] / den[mask_to_estimate])

    return v_x_smooth, v_y_smooth


def compute_optical_flow(frames_or_events, dt=1/30, is_frames=True, sensor_size=None):
    """
    Summary line.

    Compute the optical flow for rgb frames using opencv routine

    Parameters:
    -----------
    frames_or_events: 
    dt: the time difference between two frames
    is_frames: a boolan variable specifying whether the optical flow is computed for rgb frames or events 
    sensor_size:

    Returns:
    ----------
    flow_out: the optical flow of the given consecutive frames
    """
    if is_frames:
        frames = frames_or_events
        flow_out = cv2.calcOpticalFlowFarneback(prev=frames[0],
                                                next=frames[1],
                                                flow=None,
                                                pyr_scale=0.5,
                                                levels=3,
                                                winsize=15,
                                                iterations=3,
                                                poly_n=5,
                                                poly_sigma=1.2,
                                                flags=0)
    else:
        events = frames_or_events

        flow_out = None
    return flow_out

def visualize_flow(img_in, flow_in, decimation=15, scale=10, method=0):
    """
    Summary line.

    A function to visualize optical flow over the original image

    Parameters:
    -----------
    img_in: rgb image - if none is provided, the flow is shown alone
    flow_in: this is the dense optical flow
    decimation: how dense the arrows are shown in the vector field
    scale: scale of the magnitude in the vector field
    method: vector field or hsv plot

    Returns:
    ----------
    None
    """
    img_out = np.copy(img_in)
    if method == 0:
        # quiver plot
        y = list(range(int(img_out.shape[0])))[0::decimation]
        x = list(range(int(img_out.shape[1])))[0::decimation]
        xv, yv = np.meshgrid(x, y)
        u = scale * flow_in[yv, xv, 0]
        v = scale * flow_in[yv, xv, 1]
        start_points = np.array([xv.flatten(), yv.flatten()]).T.astype(int).tolist()
        end_points = np.array([xv.flatten() - u.flatten(), yv.flatten() - v.flatten()]).T.astype(int).tolist()
        for i in range(len(start_points)):
            cv2.arrowedLine(img_out, tuple(start_points[i]), tuple(end_points[i]), [255, 100, 30], thickness=1)
        return img_out
    else:
        hsv = np.zeros_like(img_out)
        hsv[..., 1] = 255

        mag, ang = cv2.cartToPolar(flow_in[..., 0], flow_in[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
