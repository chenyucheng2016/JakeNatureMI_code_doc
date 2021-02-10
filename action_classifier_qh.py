import numpy as np
import glob
import cv2
from numpy.linalg import inv
import random
import matplotlib.pyplot as plt
# from sklearn import svm
# from sklearn.externals import joblib
import os
import airsim
from collections import deque
# from dvs import DVS
from dvs_QH import DVS


def calmeimhi(video_clip, taumin, taumax, slide_step, tol, sensor_size, is_frames=True):
    """
    Summary line.
    
    Calculate the MEI(Enegery Image) and MHI (Motion History Image) of a stream of images
    
    Parameters:
    ------------
    video_clip: a stream of orginal rgb frames
    taumin:
    taumax:
    slide_step:
    tol:
    sensor_size:
    is_frame:a boolean variable specifying if the MEI and MHI are computed for rgb frames or events
    
    Returns:
    ------------
    mei: the energy image of the given stream of frames
    mhi: the motion history image of the given stream frame
    
    """
    if is_frames:
        # assume input frames with size (tauMax-tauMin) x width x height
        frames = video_clip[taumin:taumax]
        mhi = np.zeros((frames.shape[1], frames.shape[2]))
        tau = taumax - taumin
        for f in range(tau-1):
            # convert frame to np array
            # load mask & compute
            # box = rectangle_from_img_mask(mask[f])
            # box_next = rectangle_from_img_mask(mask[f + 1])
            # warp_mat = np.array([[1.0, 0.0, box_next[0][0] - box[0][0]], [0.0, 1.0, box_next[0][1] - box[0][1]]])
            image = frames[f]
            image_next = frames[f + 1]
            # image_next = cv2.warpAffine(image_next, warp_mat, (image_next.shape[1], image_next.shape[0]))
            # calculate frame difference
            # if box is None:
            image_diff = image_next - image
            # else:
            #     # print('123',flightmode.value)
            #     if flightmode.value == 1:
            #         image_diff_box = image_next[int(box[0, 0]): int(box[0, 2]), int(box[0, 1]): int(box[0, 3])] - \
            #                      image[int(box[0, 0]): int(box[0, 2]), int(box[0, 1]): int(box[0, 3])]
            #         image_diff = np.zeros((frames.shape[1], frames.shape[2]))
            #         image_diff[int(box[0, 0]): int(box[0, 2]), int(box[0, 1]): int(box[0, 3])] = image_diff_box
            #     else:
            #         image_diff = image_next - image
            # calculate mhi
            # set the pixel value to tau for the region where the image difference is above tolerance
            mhi[abs(image_diff) > tol] = tau
            # reduce the pixel value of remaining regions by slide steps
            mhi[~(abs(image_diff) > tol)] = np.floor(mhi[~(abs(image_diff) > tol)] - slide_step)
            # set the value of non-moving region back to 0
            mhi[mhi < 0] = 0
        # obtain mei by thresholding mhi
        mei = (mhi > 0).astype(int)

        # cv2.imshow('mhi', mhi)

        # cv2.imshow(mhi, cmap='gray', vmin=0, vmax=tau)
        # plt.show()
        # plt.imshow(mei, cmap='gray', vmin=0, vmax=1)
        # plt.show()
        ################################################################################################################
    else:
        # calculate mhi, mei
        t_total = np.unique(video_clip)
        # t_total, indices, counts = np.unique(video_clip[:, 2], return_inverse=True, return_counts=True)
        mhi = np.zeros((sensor_size[0], sensor_size[1]))
        # tau = len(t_total)
        count = 1
        # cur_clip = video_clip[indices == count]
        # print('before mhi calc')
        # print('number of firing times', len(t_total))
        for tt in t_total:
            mhi[video_clip == tt] = count
            # mhi[np.int_(cur_clip[:, 0]), np.int_(cur_clip[:, 1])] = count
            # mhi[np.int_(event_clip[event_clip[:, 2] == tt, 0]), np.int_(
            #     event_clip[event_clip[:, 2] == tt, 1])] = count
            count += 1
        # threshold
        # print('after mhi calc')
        # print('mhi_max', np.max(mhi))
        mei = (mhi > count/5).astype(int)                        #### to do: tune
        mhi[mhi < count/5] = 0                                   #### to do: tune
    return mei, mhi


def hu_moments(m):
    # central crop to eliminate background noise
    h_ratio = 1
    w_ratio = 1
    m = m[0:int(m.shape[0]*h_ratio), int(m.shape[1] * (1 - w_ratio) / 2):int(m.shape[1] * (1 + w_ratio) / 2)]
    # opencv to calculate hu moments
    if m.dtype == 'int32':
        moments = cv2.moments(m, binaryImage=True)
    else:
        # normalize mhi to eliminate the influence of motion speed
        m = m/np.amax(m)
        moments = cv2.moments(m)
    hu_vec = cv2.HuMoments(moments)
    # return hu_vec
    # log to match the magnitude
    return -1*np.multiply(np.sign(hu_vec), np.log10(abs(hu_vec)))

def train_model(data_dir, is_frames):
    labels = list(['walking', 'jogging', 'waving', 'hovering',  'movewalking', 'pointing_outdoor'])
    store_mei = np.empty((7, 1))
    store_mhi = np.empty((7, 1))
    mean_mei_hu = np.empty((7, 1))
    mean_mhi_hu = np.empty((7, 1))
    for f in labels:
        # data loader, training data shape: num_trials x video shape (num_frames x width x height)
        mode = 'train'
        image_path = data_dir + "/%s/" % f
        image_folder = os.listdir(image_path)
        mei_hu = np.zeros((7, 1))
        mhi_hu = np.zeros((7, 1))
        for tri in image_folder:
            train_data = data_loader(image_path+tri, mode)
            start_clip = 0
            end_clip = train_data.shape[0]
            slide_step = 1
            tol = 45
            # calc mei & mhi
            if is_frames:
                mei, mhi = calmeimhi(train_data, start_clip, end_clip, slide_step, tol, sensor_size=[], is_frames=True)
            else:
                sensor_size = [480, 640]  # resize to 360p
                dvs = DVS(sensor_size=sensor_size, thresh=0.1)  # load dvs class
                all_events_arr, time_event, tf = dvs.get_event_sequence(np.asarray(train_data), 1.0 / 10)  #
                mei, mhi = calmeimhi(time_event, None, None, None, None,
                                     sensor_size=(train_data.shape[1], train_data.shape[2]), is_frames=False)
            # mei, mhi = calmeimhi(train_data, start_clip, end_clip, slide_step, tol, sensor_size=[], is_frames=True)
            # calc & accumulate & store hu moments of mei & mhi
            mei_hu = mei_hu + hu_moments(mei)
            mhi_hu = mhi_hu + hu_moments(mhi)
            store_mei = np.concatenate((store_mei, hu_moments(mei)), axis=1)
            store_mhi = np.concatenate((store_mhi, hu_moments(mhi)), axis=1)
        mean_mei_hu = np.concatenate((mean_mei_hu, mei_hu/len(image_folder)), axis=1)
        mean_mhi_hu = np.concatenate((mean_mhi_hu, mhi_hu/len(image_folder)), axis=1)
        # print('mean_mei_hu', mean_mei_hu)
        # print('mean_mhi_hu', mean_mhi_hu)

    print('mean_mei_hu_all',mean_mei_hu)
    print('mean_mhi_hu_all', mean_mhi_hu)
    # calc pooled independent cov
    cov_mei = np.diag(np.diag(np.cov(store_mei[:, 1:])))
    cov_mhi = np.diag(np.diag(np.cov(store_mhi[:, 1:])))

    # save trained data
    if is_frames:
        fname_mei_cov = data_dir + '/train/moments/cov_mei.npy'
        fname_mhi_cov = data_dir + '/train/moments/cov_mhi.npy'
        fname_mei_hu = data_dir + '/train/moments/mean_mei_hu.npy'
        fname_mhi_hu = data_dir + '/train/moments/mean_mhi_hu.npy'
    else:
        fname_mei_cov = data_dir + '/train/moments/cov_mei_event.npy'
        fname_mhi_cov = data_dir + '/train/moments/cov_mhi_event.npy'
        fname_mei_hu = data_dir + '/train/moments/mean_mei_hu_event.npy'
        fname_mhi_hu = data_dir + '/train/moments/mean_mhi_hu_event.npy'
    np.save(fname_mei_cov, cov_mei)
    np.save(fname_mhi_cov, cov_mhi)
    np.save(fname_mei_hu, mean_mei_hu[:, 1:])
    np.save(fname_mhi_hu, mean_mhi_hu[:, 1:])
    return fname_mei_cov, fname_mhi_cov, fname_mei_hu, fname_mhi_hu


def classify_action(data_dir, cov_mei, cov_mhi, mean_mei_hu, mean_mhi_hu, is_frames=True):
    mode = 'test'
    test_labels = list(['walking'])  # action labels
    # test_labels = list(['walking', 'jogging', 'waving', 'pointing'])  # action labels
    train_labels = list(['walking', 'jogging', 'waving', 'hovering',  'movewalking', 'pointing_outdoor'])
    classified_labels = []
    for k in test_labels:
        image_path = data_dir + "/%s/" % k
        image_folder = os.listdir(image_path)
        test_clip = data_loader(image_path+image_folder[random.randint(0, len(image_folder)-1)], mode)
        # assume test data with shape: num_frames x width x height
        start_clip = 0
        end_clip = test_clip.shape[0]
        slide_step = 1
        tol = 25
        # calc mei & mhi
        if is_frames:
            mei, mhi = calmeimhi(test_clip, start_clip, end_clip, slide_step, tol, [], is_frames=True)
        else:
            sensor_size = [480, 640]  # resize to 360p
            dvs = DVS(sensor_size=sensor_size, thresh=0.025)  # load dvs class
            all_events_arr, time_event, tf = dvs.get_event_sequence(np.asarray(test_clip), 1.0/90)  # current frame rate
            mei, mhi = calmeimhi(time_event, None, None, None, None, is_frames=False)
        mei_hu = hu_moments(mei)
        mhi_hu = hu_moments(mhi)
        label_record = []
        mahad_record = []
        # template matching
        for f in range(mean_mei_hu.shape[1]):
            mahad_mei = np.sqrt(np.dot(np.matmul((mei_hu - np.expand_dims(mean_mei_hu[:, f], axis=1)).T, inv(cov_mei)),
                                       (mei_hu - np.expand_dims(mean_mei_hu[:, f], axis=1))))
            # if mahad_mei[0] < 28:
            mahad_mhi = np.sqrt(np.dot(np.matmul((mhi_hu - np.expand_dims(mean_mhi_hu[:, f], axis=1)).T, inv(cov_mhi)),
                                       (mhi_hu - np.expand_dims(mean_mhi_hu[:, f], axis=1))))
                # if mahad_mhi[0] < 28:
            label_record.append(train_labels[f])
            mahad_record.append(mahad_mei+mahad_mhi)
        # obtain label of the testing data
        classified_labels.append(label_record[mahad_record.index(min(mahad_record))])
        # abc = sorted(range(len(mahad_record)), key=lambda i: mahad_record[i])[0]
    accuracy = 100*sum((classified_labels[i] == test_labels[i] for i in range(len(test_labels))))/len(test_labels)
    # todo: add top 5 accuracy
    return classified_labels, accuracy


def data_loader(folder_name, mode):
    if mode == "train":
        print("loading training data {fname}".format(fname=folder_name))
        width = 640
        height = 480
        image_data = np.empty((height, width))
        #mask_data = np.empty((height, width))
        #for i in glob.glob(folder_name + "/scene/*.jpg"):
        count = 0
        for i in glob.glob(folder_name + "/*.jpg"):
            if count % 10 == 0:
                image = cv2.imread(i)
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image_data = np.dstack((image_data, np.array(image)))
            count = count + 1
        #print("loading mask data {fname}".format(fname=folder_name))
        #for i in glob.glob(folder_name + "/segmentation/*.png"):
            #image = cv2.imread(i)
            #image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            #mask_data = np.dstack((mask_data, np.array(image)))
        return np.transpose(image_data[:, :, 1:], (2, 0, 1))
    else:
        print("loading test data {fname}".format(fname=folder_name))
        width = 640
        height = 480
        image_data = np.empty((height, width))
        #for i in glob.glob(folder_name + "/scene/*.jpg"):
        for i in glob.glob(folder_name + "/*.jpg"):
            image = cv2.imread(i)
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_data = np.dstack((image_data, np.array(image)))
        return np.transpose(image_data[:, :, 1:], (2, 0, 1))


def rectangle_from_img_mask(mask_in, pad=10):
    """
    Summary line.
    
    Compute a bounding box based on the mask processoed from frame(s)
    
    Parameters:
    ------------
    mask: binary image with action/object detected
    pad: size of pad added at the edge of the mask
    
    Returns:
    ------------
    rectangular bounding box
    
    """

    pad = 10
    idxs = np.nonzero(mask_in.any(axis=0))[0]
    if idxs.shape[0] > 0:
        xmin = np.max((idxs[0] - pad, 0))
        xmax = np.min((idxs[-1] + pad, mask_in.shape[1]))
    else:
        return None
    # indices of non empty rows
    idxs = np.nonzero(mask_in.any(axis=1))[0]
    if idxs.shape[0] > 0:
        ymin = np.max((idxs[0] - pad, 0))
        ymax = np.min((idxs[-1] + pad, mask_in.shape[0]))
    else:
        return None
    return [(xmin, ymin), (xmax, ymax)]


def real_time_classification(cov_mei, cov_mhi, mean_mei_hu, mean_mhi_hu, is_frames=True):
    # parameters
    sim_speed = 1  # you can use this to slow down the simulation if your machine can't keep up
    rgb_images = deque()
    seg_images = deque()
    scores_cache = deque()
    slide_step = 1
    average_size = 3
    train_labels = list(['walking', 'jogging', 'waving', 'pointing','hovering'])
    scores_sum = np.zeros(len(train_labels))
    # connect to the AirSim simulator
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    # set object ID
    found = client.simSetSegmentationObjectID("[\w]*", 2, True)
    print("Set background object id: %r" % found)
    found = client.simSetSegmentationObjectID("BP_Personnel_2", 0, True)
    print("Set target 1 object id: %r" % found)
    target_colors = []
    palette = cv2.imread("seg_palette.png")
    target_colors.append(palette[0, 0, :])
    target_colors.append(palette[0, 1, :])
    # real-time classification with sliding window method. results is the average of last 3 windows.
    if is_frames:
        t_max = 50.0 / 30  # simulation length
        dt = 1.0 / 30  # time between frames
        tol = 35
        window_size = 30
    else:
        t_max = 100.0 / 300  # simulation length
        dt = 1.0 / 30  # time between frames
        sensor_size = [360, 480]  # resize
        dvs = DVS(sensor_size=sensor_size, thresh=0.25)  # load dvs class
        window_size = 50
    client.simContinueForTime(dt / sim_speed)
    client.simPause(True)
    for i in range(int(t_max / dt)):
        print("collecting frame {} / {}".format(i, int(t_max / dt)))
        # get raw, depth, and segmentation images
        responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)  # get numpy array
        img = img1d.reshape(response.height, response.width, 3)
        img = cv2.resize(img, (480, 360), interpolation=cv2.INTER_AREA)
        rgb_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        client.simContinueForTime(dt / sim_speed)
        if len(rgb_images) == window_size:
            if is_frames:
                mei, mhi = calmeimhi(np.asarray(rgb_images).astype(np.float64), 0, window_size-1, slide_step, tol, is_frames=True)
            else:
                all_events_arr, time_event, tf = dvs.get_event_sequence(np.asarray(rgb_images).astype(np.float64), dt)
                mei, mhi = calmeimhi(time_event, None, None, None, None, is_frames=False)
            mei_hu = hu_moments(mei)
            mhi_hu = hu_moments(mhi)
            label_record = []
            mahad_record = []
            for f in range(mean_mei_hu.shape[1]):
                mahad_mei = np.sqrt(
                    np.dot(np.matmul((mei_hu - np.expand_dims(mean_mei_hu[:, f], axis=1)).T, inv(cov_mei)),
                           (mei_hu - np.expand_dims(mean_mei_hu[:, f], axis=1))))
                # if mahad_mei[0] < 28:
                mahad_mhi = np.sqrt(
                    np.dot(np.matmul((mhi_hu - np.expand_dims(mean_mhi_hu[:, f], axis=1)).T, inv(cov_mhi)),
                           (mhi_hu - np.expand_dims(mean_mhi_hu[:, f], axis=1))))
                    # if mahad_mhi[0] < 28:
                label_record.append(train_labels[f])
                mahad_record.append(mahad_mei + mahad_mhi)
            scores_cache.append(mahad_record)
            scores_sum += np.asarray(mahad_record)[:, 0, 0]
            if len(scores_cache) == average_size:
                classified_labels = label_record[int(np.argmin(scores_sum))]
                print('classified labels: {labels}'.format(labels=classified_labels))
                scores_sum -= np.asarray(scores_cache.popleft())[:, 0, 0]
            rgb_images.popleft()
    client.simPause(False)
    client.armDisarm(False)
    client.reset()
    client.enableApiControl(False)


