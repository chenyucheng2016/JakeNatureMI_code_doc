import numpy as np
import cv2
from scipy.ndimage import uniform_filter
from skimage import draw
import glob
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score
from optical_flow import compute_optical_flow, visualize_flow
from event_camera import get_events_from_frames, get_event_optical_flow, visualize_events
from sklearn.cluster import DBSCAN
# size of target downsampling for generating feature vectors
DIM = (64, 64)


def hof(flow_in, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=False,
        normalise=False, motion_threshold=1.):
    """
    Extract Histogram of Optical Flow (HOF) for a given optical flow field
        1. (optional) global image normalisation
        2. computing the dense optical flow
        3. computing flow histograms
        4. normalising across blocks
        5. flattening into a feature vector
    Parameters
    ----------
    flow_in : (M, N) ndarray
        Input optical flow field (x and y flow images).
    orientations : int
        Number of orientation bins.
    pixels_per_cell : 2 tuple (int, int)
        Size (in pixels) of a cell.
    cells_per_block  : 2 tuple (int,int)
        Number of cells in each block.
    visualise : bool, optional
        Also return an image of the hof.
    normalise : bool, optional
        Apply power law compression to normalise the image before
        processing.
    motion_threshold : threshold for no motion
    Returns
    -------
    newarr : ndarray
        hof for the image as a 1D (flattened) array.
    hof_image : ndarray (if visualise=True)
        A visualisation of the hof image.
    References
    ----------
    * https://github.com/colincsl/pyKinectTools/blob/master/pyKinectTools/algs/HistogramOfOpticalFlow.py
    * http://en.wikipedia.org/wiki/Histogram_of_oriented_gradients
    * Dalal, N and Triggs, B, Histograms of Oriented Gradients for
      Human Detection, IEEE Computer Society Conference on Computer
      Vision and Pattern Recognition 2005 San Diego, CA, USA
    """
    flow_in = np.atleast_2d(flow_in)

    # optional global image normalisation
    if normalise:
        flow_in = np.sqrt(flow_in)

    # first order image gradients.
    # todo: [Jake] I think this is only looks at the x-derivative in the x-component of the flow, etc.
    if flow_in.dtype.kind == 'u':
        # convert uint image to float
        # to avoid problems with subtracting unsigned numbers in np.diff()
        flow_in = flow_in.astype('float')

    gx = np.zeros(flow_in.shape[:2])
    gy = np.zeros(flow_in.shape[:2])
    # gx[:, :-1] = np.diff(flow_in[:, :, 1], n=1, axis=1)
    # gy[:-1, :] = np.diff(flow_in[:, :, 0], n=1, axis=0)

    gx = flow_in[:, :, 1]
    gy = flow_in[:, :, 0]

    # compute flow histograms
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * (180 / np.pi) % 180

    sy, sx = flow_in.shape[:2]
    cx, cy = pixels_per_cell
    bx, by = cells_per_block

    n_cellsx = int(np.floor(sx // cx))  # number of cells in x
    n_cellsy = int(np.floor(sy // cy))  # number of cells in y

    # compute orientations integral images
    orientation_histogram = np.zeros((n_cellsy, n_cellsx, orientations))
    subsample = np.index_exp[int(cy / 2):int(cy * n_cellsy):cy, int(cx / 2):int(cx * n_cellsx):cx]
    for i in range(orientations-1):
        # create new integral image for this orientation
        # isolate orientations in this range

        temp_ori = np.where(orientation < 180 / orientations * (i + 1),
                            orientation, -1)
        temp_ori = np.where(orientation >= 180 / orientations * i,
                            temp_ori, -1)
        # select magnitudes for those orientations
        cond2 = (temp_ori > -1) * (magnitude > motion_threshold)
        temp_mag = np.where(cond2, magnitude, 0)

        temp_filt = uniform_filter(temp_mag, size=(cy, cx))
        orientation_histogram[:, :, i] = temp_filt[subsample]

    # calculate the 'no-motion' bin
    temp_mag = np.where(magnitude <= motion_threshold, magnitude, 0)

    temp_filt = uniform_filter(temp_mag, size=(cy, cx))
    orientation_histogram[:, :, -1] = temp_filt[subsample]

    # compute the histogram for each cell
    hof_image = None

    if visualise:
        radius = min(cx, cy) // 2 - 1
        hof_image = np.zeros((sy, sx), dtype=float)
        for x in range(n_cellsx):
            for y in range(n_cellsy):
                for o in range(orientations-1):
                    centre = tuple([y * cy + cy // 2, x * cx + cx // 2])
                    dx = int(radius * np.cos(float(o) / orientations * np.pi))
                    dy = int(radius * np.sin(float(o) / orientations * np.pi))
                    rr, cc = draw.line(centre[0] - dy, centre[1] - dx,
                                                   centre[0] + dy, centre[1] + dx)
                    hof_image[rr, cc] += orientation_histogram[y, x, o]

    # normalization
    n_blocksx = (n_cellsx - bx) + 1
    n_blocksy = (n_cellsy - by) + 1
    normalised_blocks = np.zeros((n_blocksy, n_blocksx, by, bx, orientations))

    for x in range(n_blocksx):
        for y in range(n_blocksy):
            block = orientation_histogram[y:y+by, x:x+bx, :]
            eps = 1e-5
            normalised_blocks[y, x, :] = block / np.sqrt(block.sum()**2 + eps)

    # return as a feature vector
    if visualise:
        return normalised_blocks.ravel(), hof_image
    else:
        return normalised_blocks.ravel()


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
    rectangular bounding box: a 2x2 array containig two vertics of the bounding box
    
    """
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

def detect_object_qh(flow_in, gamma=0.5, return_mask=False, is_frames=False):
    """
    Summary line.
    
    Detect the existence of an object from optical flow
    
    Parameters:
    ------------
    flow_in: The optical flow computed by either rgb frames or a series of events 
    gamma: threshold of optical flow vector magnitude 
    return_mask=False: a boolean variable specifying whether a mask is returned
    is_frames=False:a boolean variable specifying whether the input optical flow is computed by rgb frames or events
    
    Returns:
    ------------
    mask: binary image with object detected
    rectangular bounding box: a 2x2 array containig two vertics of the bounding box
    
    """
    mag, ang = cv2.cartToPolar(flow_in[..., 0], flow_in[..., 1])
    # ignore 2% pad around the image border which is usually noisy for OF
    pad = np.max([2, int(flow_in.shape[1] * .02)])
    mag[:, :pad] = 0
    mag[:, -pad:] = 0
    mag[:pad, :] = 0
    mag[-pad:, :] = 0

    # generate a mask that ignores flow below gamma % of the max magnitude
    gamma = np.clip(gamma, 0.01, 0.99)
    mask = mag > gamma * np.max(mag)

    # generate binary image from the mask
    img = np.zeros_like(mask, dtype=np.uint8)
    img[mask] = 255

    # plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    # plt.show()

    # small erosion to eliminate noise
    if is_frames:
        kernel = np.ones((5, 5), dtype=np.uint8)
        img = cv2.erode(img, kernel, iterations=2)

        # apply Gaussian blur and threshold to reduce noise
        img = cv2.GaussianBlur(img, (25, 25), 0)
        img = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)[1]

        # dilate+erode (close) the features to capture more of the object
        kernel = np.ones((9, 9), dtype=np.uint8)
        img = cv2.dilate(img, kernel, iterations=3)
        kernel = np.ones((3, 3), dtype=np.uint8)
        img = cv2.erode(img, kernel, iterations=3)
    else:


        # apply Gaussian blur and threshold to reduce noise
        img = cv2.GaussianBlur(img, (11, 11), 0)
        img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)[1]
        kernel = np.ones((9, 9), dtype=np.uint8)
        img = cv2.dilate(img, kernel, iterations=3)
        kernel = np.ones((3, 3), dtype=np.uint8)
        img = cv2.erode(img, kernel, iterations=3)

        # remove noise from backgrounds
        img[0:160, :] = 0

    mask = img > 0
    if return_mask:
        return mask
    else:
        return rectangle_from_img_mask(mask)


def detect_object(flow_in, box2draw, gamma=0.5,  return_mask=False):
    mag, ang = cv2.cartToPolar(flow_in[..., 0], flow_in[..., 1])
    # ignore 2% pad around the image border which is usually noisy for OF
    pad = np.max([2, int(flow_in.shape[1] * .05)])
    mag[:, :pad] = 0
    mag[:, -pad:] = 0
    mag[:pad, :] = 0
    mag[-pad:, :] = 0
    # generate a mask that ignores flow below gamma % of the max magnitude
    gamma = np.clip(gamma, 0.01, 0.99)
    mask = mag > gamma * np.max(mag)

    # generate binary image from the mask
    img = np.zeros_like(mask, dtype=np.uint8)
    img2 = np.zeros_like(mask, dtype=np.uint8)
    img[mask] = 255

    # small erosion to eliminate noise
    kernel = np.ones((4, 4), dtype=np.uint8)
    kernel = np.ones((4, 4), dtype=np.uint8)
    img = cv2.erode(img, kernel, iterations=2)

    # apply Gaussian blur and threshold to reduce noise
    #img = cv2.GaussianBlur(img, (3, 3), 0)
    #img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)[1]

    # dilate+erode (close) the features to capture more of the object
    kernel = np.ones((5, 5), dtype=np.uint8)
    img = cv2.dilate(img, kernel, iterations=3)
    kernel = np.ones((5, 5), dtype=np.uint8)
    img = cv2.erode(img, kernel, iterations=3)
    xys = np.where(img == 255)
    xpix = np.array([xys[0]])
    ypix = np.array([xys[1]])
    positive_pixels = np.concatenate((xpix, ypix), axis=0).T
    clustering = DBSCAN(eps=10, min_samples=150).fit(positive_pixels)
    labels = clustering.labels_
    sample_clusters = dict()
    for lb in labels:
        if lb != -1:
            sample_clusters[str(lb)] = []
    for i in range(positive_pixels.shape[0]):
        fpt = positive_pixels[i]
        lb = labels[i]
        if lb > -1:
            sample_clusters[str(lb)].append(fpt)

    if box2draw is None:
        maxVolKey = ''
        maxVolNum = -1
        for key in sample_clusters.keys():
            print(np.mean(sample_clusters[key], axis=0))
            print('\n')
            if len(sample_clusters[key]) > maxVolNum:
                maxVolNum = len(sample_clusters[key])
                maxVolKey = key
        if len(maxVolKey) > 0:
            maxCluster = np.array(sample_clusters[maxVolKey])
            for j in range(maxCluster.shape[0]):
                fpt = maxCluster[j]
                img2[fpt[0], fpt[1]] = 255
    else:
        b2d_center_x = (box2draw[0, 0] + box2draw[0, 2]) / 2
        b2d_center_y = (box2draw[0, 1] + box2draw[0, 3]) / 2
        maxVolKey = ''
        maxScore = -1
        beta = 0.003
        for key in sample_clusters.keys():
            cl_center = np.mean(sample_clusters[key], axis=0)
            dist = 1.0/(np.sqrt((cl_center[0] - b2d_center_x)**2 + (cl_center[1] - b2d_center_y)**2)) * 5000
            volume = len(sample_clusters[key])
            #print('dist',dist)
            #print('vol',volume)
            score = beta*volume + (1-beta)*dist
            if  score > maxScore:
                maxScore = score
                maxVolKey = key
        if len(maxVolKey) > 0:
            maxCluster = np.array(sample_clusters[maxVolKey])
            for j in range(maxCluster.shape[0]):
                fpt = maxCluster[j]
                img2[fpt[0], fpt[1]] = 255

    mask = img2> 0



    if return_mask:
        return mask
    else:
        return rectangle_from_img_mask(mask)


def compute_hof_features(flow_in, scene_imgs_in, mask_imgs_in, is_frames=True):
    """returns an array of HOF feature vectors for a directory of sequential images making up a video"""

    # how many frames to use to compute temporal gradient at each time step
    window_size = 10
    img_scale = 1
    width = int(scene_imgs_in.shape[1] * img_scale)
    height = int(scene_imgs_in.shape[0] * img_scale)
    sensor_size = (height, width)
    sensor_indices = np.indices(sensor_size)

    # Set up bins for histograms
    bin_row = range(1, sensor_size[0] - 1)
    bin_col = range(1, sensor_size[1] - 1)

    # prev_gray = None
    # sensor_val = None
    # all_events = None
    features_out = []
    # for idx, path in enumerate(scene_imgs_in):  # image folder
    #     # todo: add ability to 'set' the frame rate and resolution
    #     if not idx % 1 == 0:
    #         continue
    #     t = idx * dt
    #     print("Computing HOF features {} / {}".format(idx, len(scene_imgs_in)))
    #
    #     # get frames and convert to grayscale
    #     frame = cv2.imread(path)  # image folder
    #     mask = cv2.imread(mask_imgs_in[idx])
    #
    #     # resize image for optical flow calculation
    #     width = int(frame.shape[1] * img_scale)
    #     height = int(frame.shape[0] * img_scale)
    curr_gray = scene_imgs_in
    # curr_gray = cv2.resize(scene_imgs_in, (width, height), interpolation=cv2.INTER_AREA)
    mask = mask_imgs_in

    # curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # if prev_gray is None:
    #     prev_gray = curr_gray

    # if is_frames:
    #     flow = compute_optical_flow([curr_gray, prev_gray])
    # else:
    #         # flow = compute_optical_flow([curr_gray, prev_gray])
    #         # convert frames to events
    #     frame_events, num_events, sensor_val = get_events_from_frames(curr_gray, t, prev_gray, t - dt, sensor_val)
    #     prev_gray = curr_gray
    #
    #     # event_img = visualize_events(num_events, thresh=2)
    #     # cv2.imshow('event image', event_img)
    #     # cv2.waitKey(1)
    #
    #         # lump frames
    #     if all_events is None:
    #         all_events = frame_events
    #     else:
    #         all_events = np.hstack((all_events, frame_events))
    #         # only use events from the previous n *dt seconds
    #     tau = 5 * dt
    #     event_window = all_events[:, all_events[2, :] > t - tau]
    #     # get optical flow from event data
    #     flow = get_event_optical_flow(event_window, flow_shape=curr_gray.shape, bins=(bin_row, bin_col, window_size), t=t, dt=dt)

        # visualize_hof_algorithm(frame, flow, mask)

        # get target bounding box
        # box = rectangle_from_img_mask(mask)
    box = mask

    if box is not None:
        # get optical flow in the detected region for classification
        # print('box_test', box.shape)
        # print('flowin', flow_in.shape)
        flow_obj = flow_in[int(box[0,1]):int(box[0,3]), int(box[0,0]):int(box[0,2]), :]

        # "resize" flow to fit the object and extract HOF features
        flow_resized = cv2.resize(flow_obj, DIM, interpolation=cv2.INTER_AREA)

        # get HOF features
        features_out.append(hof(flow_resized, visualise=False))
        # update for next frame
        # prev_gray = curr_gray
    return features_out


def visualize_hof_algorithm(img_in, flow_in, mask_in):
    """
    Summary line.

    superimpose the optical flow and histogram of flow with the original rgb image captured by the quadcopter front camera and display the superimposed image

    Parameters:
    ------------
    img_in: the orginal rgb frame
    flow_in: the optical flow computed either by rgb frames or by events
    mask_in: a binary image with action/object detected
    
    Returns:
    ------------
    None

    """

    # 1) original image
    cv2.imshow('raw image', img_in)

    # 2) optical flow over original image
    of_img = visualize_flow(img_in, flow_in, decimation=30, scale=10)
    cv2.imshow('optical flow', of_img)

    # 3) bounding box over original image
    box = rectangle_from_img_mask(mask_in)
    if box is not None:
        cv2.rectangle(img_in, box[0], box[1], [100, 255, 100], thickness=3)
        cv2.imshow('object detection', img_in)

        # 4) cropped target region with optical flow - scaled up
        flow_obj = flow_in[box[0][1]:box[1][1], box[0][0]:box[1][0], :]
        img_obj = img_in[box[0][1]:box[1][1], box[0][0]:box[1][0], :]
        of_img_obj = visualize_flow(img_obj, flow_obj)

        scale = 4.00
        width = int(of_img_obj.shape[1] * scale)
        height = int(of_img_obj.shape[0] * scale)
        of_img_obj = cv2.resize(of_img_obj, (width, height), interpolation=cv2.INTER_AREA)
        cv2.imshow('detected object flow', of_img_obj)

        # 5) HOF features of resized optical flow image
        img_obj_resized = cv2.resize(img_obj, DIM, interpolation=cv2.INTER_AREA)
        flow_obj_resized = cv2.resize(flow_obj, DIM, interpolation=cv2.INTER_AREA)

        resized_of_img_obj = visualize_flow(img_obj_resized, flow_obj_resized)
        cv2.imshow('resized image for HOF calculation', resized_of_img_obj)
        cv2.waitKey(1)

        # get HOF image
        _, hof_img = hof(flow_obj_resized, visualise=True)

        # scale up HOF image for visualization
        scale = 4.00
        width = int(hof_img.shape[1] * scale)
        height = int(hof_img.shape[0] * scale)
        hof_img_resized = cv2.resize(hof_img, (width, height), interpolation=cv2.INTER_AREA)
        cv2.imshow('hof image', hof_img_resized)


########################################################################################################################


