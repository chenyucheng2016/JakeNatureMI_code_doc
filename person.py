import numpy as np
import cv2
from glasses_detector import get_glasses_features
from enum import Enum
from collections import deque
from dvs import DVS
from action_classifier_qh import calmeimhi, hu_moments
from numpy.linalg import inv
import time


class Joints(Enum):
    #encode the different parts of a human body
    nose = 0
    neck = 1
    r_shoulder = 2
    r_elbow = 3
    r_wrist = 4
    l_shoulder = 5
    l_elbow = 6
    l_wrist = 7
    mid_hip = 8
    r_hip = 9
    r_knee = 10
    r_ankle = 11
    l_hip = 12
    l_knee = 13
    l_ankle = 14
    r_eye = 15
    l_eye = 16
    r_ear = 17
    l_ear = 18
    l_big_toe = 19
    l_small_toe = 20
    l_heel = 21
    r_big_toe = 22
    r_small_toe = 23
    r_heel = 24
    r_background = 25


class ActionLabel(Enum):
    #The set of all different action labels
    walking = 0
    jogging = 1
    waving = 2
    # pointing = 3
    hovering = 3
    movewalking = 4
    pointing_outdoors = 5
    other = 6

class ObjectLabel(Enum):
    #The set of all different object labels
    person = 0
    bicycle = 1
    car = 2

action_label_values = [e.value for e in ActionLabel]

class Person:
    _person_ID = 0
    _eye_region_pad_x = 10
    _eye_region_pad_y = 40
    _gamma = 1.25              # gamma = L_t / L, where L_t is torso length and L is total arm length

    def __init__(self, classifier_buffer_len=18, action_confidence_thresh=0.8, glasses_confidence_thresh=0.7):
        # self.pose_keypoints = np.array([25, 3])
        self.keypoints_buffer = np.zeros([3, 25, 3])
        self.t_stamps = np.zeros(3)
        self.action_buffer = None
        self.buffer_len = classifier_buffer_len
        self.rh_action_label_buffer = np.zeros(classifier_buffer_len, dtype=int)
        self.is_wearing_glasses_buffer = np.zeros(int(classifier_buffer_len))
        self.is_wearing_glasses_filtered = 0
        self.action_confidence_threshold = action_confidence_thresh
        self.glasses_confidence_threshold = glasses_confidence_thresh
        self.neck_position = np.array([0, 0])
        self.mid_hip_position = np.array([0, 0])
        self.ID = Person._person_ID
        self.torso_length = 0
        # todo: change to width and height
        self.time_event = np.zeros((480, 640))
        self.time_event_prev = np.zeros((480, 640))
        self.num_event_count = deque()
        self.tt = 0
        self.count = 0
        # self.event_count = deque()
        # self.all_events = []
        Person._person_ID += 1

    @staticmethod
    def get_neck_pos(data, frame_width, frame_height):
        if np.size(data) > 1:
            neck_pos = data[:, 1, 0:2]
            neck_pos_normalized = (neck_pos - np.array([[frame_width / 2, frame_height / 2]])) / (frame_height/2)
        # each row contains the x and y positions, measured from the lower left corner, normalized to be between -1 and 1
        else:
            neck_pos_normalized = np.zeros([0,2])
        return neck_pos_normalized

    @staticmethod
    def get_mid_hip_pos(data, frame_width, frame_height):
        if np.size(data) > 1:
            mid_hip_pos = data[:, 8, 0:2]
            mid_hip_pos_normalized = (mid_hip_pos - np.array([[frame_width / 2, frame_height / 2]])) / (frame_height/2)
        else:
            mid_hip_pos_normalized = np.zeros([0,2])
        # each row contains the x and y positions, measured from the lower left corner, normalized to be between -1 and 1
        return mid_hip_pos_normalized

    @staticmethod
    def denormalize_pos(data, frame_width, frame_height):
        return data * (frame_height/2) + np.array([[frame_width / 2, frame_height / 2]])

    @staticmethod
    def get_r_arm_angles(data):
        r_shoulder = data[:, 2, 0:2]
        r_elbow = data[:, 3, 0:2]
        r_wrist = data[:, 4, 0:2]
        r_e_s = r_elbow - r_shoulder
        r_w_e = r_wrist - r_elbow
        r_forearm_angle = np.arctan2(-r_w_e[:, 1], r_w_e[:, 0])
        r_bicep_angle = np.arctan2(-r_e_s[:, 1], r_e_s[:, 0])
        # Make the forearm angle relative
        r_forearm_angle -= r_bicep_angle

        return r_forearm_angle, r_bicep_angle

    @staticmethod
    def get_r_arm_lengths(data):
        r_neck = data[:, 1, 0:2]
        r_shoulder = data[:, 2, 0:2]
        r_elbow = data[:, 3, 0:2]
        r_wrist = data[:, 4, 0:2]
        r_midhip = data[:, 8, 0:2]
        r_e_s = r_elbow - r_shoulder
        r_w_e = r_wrist - r_elbow
        r_n_h = r_neck - r_midhip
        len_forearm = np.sqrt(r_w_e[:,0]**2 + r_w_e[:,1]**2)
        len_bicep = np.sqrt(r_e_s[:,0]**2 + r_e_s[:,1]**2)
        len_torso = np.sqrt(r_n_h[:,0]**2 + r_n_h[:,1]**2)

        # Check for invalid entries
        mask_bad_rows = np.any(np.hstack([r_neck, r_elbow, r_shoulder, r_wrist, r_midhip]) == 0, axis=1)
        # Return forearm length as a fraction of torso length
        forearm_length_normalized = len_forearm
        forearm_length_normalized[~mask_bad_rows] /= len_torso[~mask_bad_rows]
        forearm_length_normalized[mask_bad_rows] = np.nan

        # Check for invalid entries
        mask_bad_rows = np.any(np.hstack([r_elbow, r_shoulder, r_wrist, r_midhip]) == 0, axis=1)
        # Return bicep length as a fraction of torso length
        bicep_length_normalized = len_bicep
        bicep_length_normalized[~mask_bad_rows] /= len_torso[~mask_bad_rows]
        bicep_length_normalized[mask_bad_rows] = np.nan

        return forearm_length_normalized, bicep_length_normalized

    @staticmethod
    def get_torso_length(neck_pos, mid_hip_pos):
        if np.any(neck_pos == 0) or np.any(mid_hip_pos == 0):
            return None
        else:
            return np.sqrt((neck_pos[0] - mid_hip_pos[0]) ** 2 + (neck_pos[1] - mid_hip_pos[1]) ** 2)

    @staticmethod
    def get_rh_action_features(keypoints_data, t_log):
        # Compute the absolute angle the forearm makes with the horizontal for both sets of data
        r_forearm_angle, r_bicep_angle = Person.get_r_arm_angles(keypoints_data)
        r_forearm_angle_dot = np.gradient(r_forearm_angle, t_log)
        r_forearm_angle_ddot = np.gradient(r_forearm_angle_dot, t_log)
        r_bicep_angle_dot = np.gradient(r_bicep_angle, t_log)
        r_bicep_angle_ddot = np.gradient(r_bicep_angle_dot, t_log)

        r_forearm_length, r_bicep_length = Person.get_r_arm_lengths(keypoints_data)
        r_forearm_length_dot = np.gradient(r_forearm_length, t_log)
        r_bicep_length_dot = np.gradient(r_bicep_length, t_log)

        features = np.vstack([r_forearm_angle,
                              r_forearm_angle_dot,
                              r_forearm_angle_ddot,
                              r_bicep_angle,
                              r_bicep_angle_dot,
                              r_bicep_angle_ddot,
                              r_forearm_length,
                              r_forearm_length_dot,
                              r_bicep_length,
                              r_bicep_length_dot]).T

        return features

    @staticmethod
    def get_valid_features_mask(features):
        valid_row_mask = ~np.any(np.isnan(features), -1)
        return valid_row_mask

    @staticmethod
    def get_rh_angle_with_image_plane(keypoints_data, focal_length):
        #  Compute the angle the arm makes with the image plane
        p2 = keypoints_data[Joints.r_shoulder.value, 0:2]
        p3 = keypoints_data[Joints.r_wrist.value, 0:2]

        neck_pos = keypoints_data[Joints.neck.value, 0:2]
        mid_hip_pos = keypoints_data[Joints.mid_hip.value, 0:2]
        if np.any(np.concatenate((neck_pos, mid_hip_pos)) == 0):
            return None
        else:
            l_t = Person.get_torso_length(neck_pos, mid_hip_pos)

            if l_t is None or l_t == 0:
                return 0
            else:
                theta_1 = np.arctan(p2[0]/focal_length)
                theta_2 = np.arctan(p3[0]/focal_length)

                tmp_1 = np.arcsin(np.clip(Person._gamma * focal_length / l_t * np.sqrt(theta_1**2 + 1) * np.sin(theta_2 - theta_1), -1, 1))
                tmp_2 = np.pi - tmp_1

                alpha_1 = np.pi/2 - theta_2 - tmp_1
                alpha_2 = np.pi/2 - theta_2 - tmp_2

                # Assume for now that the min solution is the correct one (this yields the arm orientation pointing towards the camera)
                # Alpha is the angle the arm makes with the image plane
                alpha = min(alpha_1, alpha_2)
                return alpha

    @staticmethod
    def is_looking_at_camera(keypoints_data):
        if np.any(keypoints_data[[0, 15, 16], 0:2] == 0):
            return False
        else:
            # check if facing the camera
            if keypoints_data[16, 0] <= keypoints_data[15, 0]:
                return False

            # check if looking directly at the camera
            if np.any(keypoints_data[17, 0:2] == 0):
                distance_right_eye_right_ear = 0
            else:
                distance_right_eye_right_ear = \
                    np.linalg.norm(keypoints_data[15, 0:2] - keypoints_data[17, 0:2])
            if np.any(keypoints_data[18, 0:2] == 0):
                distance_left_eye_left_ear = 0
            else:
                distance_left_eye_left_ear = \
                    np.linalg.norm(keypoints_data[16, 0:2] - keypoints_data[18, 0:2])
            distance_between_eyes = \
                np.linalg.norm(keypoints_data[15, 0:2] - keypoints_data[16, 0:2])

            max_eye_ear_distance = max(distance_left_eye_left_ear, distance_right_eye_right_ear)

            # adjust the value in the if statement to change sensitivity to view angle
            #   smaller values make it less sensitive - lower false positive rate
            #   larger values make it more sensitive - lower false negative rate
            if max_eye_ear_distance > 0:
                # print(distance_between_eyes / max_eye_ear_distance)
                if distance_between_eyes / max_eye_ear_distance < 0.5:
                    return False
                else:
                    return True
            else:
                return False

    @staticmethod
    def distance_between_points(p1, p2):
        if len(np.shape(p1)) > 1:
            return np.sqrt(np.sum((p1 - p2) ** 2, axis=1))
        else:
            return np.sqrt(np.sum((p1 - p2) ** 2))

    def init_action_buffer(self, buffer_length, num_classes):
        self.action_buffer = np.zeros((buffer_length, num_classes))

    def update_pose_keypoints(self, keypoints, t_stamp):
        self.keypoints_buffer[...] = np.roll(self.keypoints_buffer, 1, axis=0)
        self.keypoints_buffer[0, :, :] = keypoints

        self.t_stamps[1:len(self.t_stamps)] = self.t_stamps[0:(len(self.t_stamps)-1)]
        self.t_stamps[0] = t_stamp
        return self

    def set_neck_pos(self, neck_pos, filter_bad_values=False, frame_width=1, frame_height=1):
        if filter_bad_values:
            if not np.all(Person.denormalize_pos(neck_pos, frame_width, frame_height) == 0):
                self.neck_position = neck_pos
            else:
                pass
        else:
            self.neck_position = neck_pos
        return self

    def set_mid_hip_pos(self, mid_hip_pos):
        self.mid_hip_position = mid_hip_pos
        return self

    def set_torso_length(self, neck_pos, mid_hip_pos):
        self.torso_length = Person.get_torso_length(neck_pos, mid_hip_pos)
        return self

    # def update_is_waving(self, classifier, use_probability=False):
    #     if np.sum(self.t_stamps == 0) > 1:
    #         is_waving_now = False
    #     else:
    #         features = Person.get_rh_action_features(self.keypoints_buffer, self.t_stamps)[1:2, :]
    #
    #         # Check here for nans in the features - set is_waving_now = False if there are any
    #         if np.any(np.isnan(features)):
    #             is_waving_now = False
    #         else:
    #             if use_probability:
    #                 is_waving_now = classifier.predict_proba(features)[0][0]
    #             else:
    #                 is_waving_now = classifier.predict(features)[0]
    #
    #     self.is_waving_buffer = np.roll(self.is_waving_buffer, 1)
    #     self.is_waving_buffer[0] = is_waving_now
    #     self.is_waving_filtered = np.mean(self.is_waving_buffer) > self.waving_confidence_thresh
    #     return is_waving_now

    def update_action_classification(self, classifier, use_probability=False):
        label = ActionLabel.other
        # print('{label_p}'.format(label_p = label))
        label_probabilities = np.zeros(len(classifier.classes_))
        if np.sum(self.t_stamps == 0) <= 1:
            features = Person.get_rh_action_features(self.keypoints_buffer, self.t_stamps)[1:2, :]

            # Check here for nans in the features - must assume action is Other if there are any
            if (not np.any(np.isnan(features))) and Person.is_looking_at_camera(self.keypoints_buffer[0,:,:]):
                if use_probability:
                    label_probabilities = classifier.predict_proba(features)[0]
                    label_value = np.argmax(label_probabilities)
                else:
                    label_value = classifier.predict(features)[0]
                label = ActionLabel(label_value)

        # if use_probability:
        #     self._update_action_buffers(label, use_probabilities=True, probabilities=label_probabilities)
        # else:
        #     self._update_action_buffers(label)

        return label

        ### add by QH
    def update_action_classification_qh(self, frame, cov_mei, cov_mhi, mean_mei_hu, mean_mhi_hu, rgb_images, scores_cache,
                                        is_frames, use_probability):
        """
        Summary line.
    
        recognize the action label from a stream of frames. This method is used in both frame and event based approaches.
    
        Parameters:
        ------------
        frame:
        cov_mei:
        cov_mhi:
        mean_mei_hu: 
        mean_mhi_hu: 
        rgb_images:
        scores_cache:
        is_frames:
        use_probability:
    
        Returns:
        ------------
        label: action classification label
        frame: processed frame 
    
        """
        label = ActionLabel.other
        label_probabilities = np.zeros(5)

        slide_step = 1
        tol = 45                                              # tune tolerance
        window_size = 15                                      # tune
        dt = 1.0 / 3  # frame rate
        train_labels = list(['walking', 'jogging', 'waving', 'hovering', 'movewalking', 'pointing_outdoors'])
        scores_sum = np.zeros(len(train_labels))

        sensor_size = [frame.shape[0], frame.shape[1]]  # sensor size
        average_size = 1                                       #### to do: tune
        dvs = DVS(sensor_size=sensor_size, thresh=0.15)  # tune
        rgb_images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        if len(rgb_images) > 1:
            time_event, tt, diff = dvs.get_event_sequence(np.asarray([rgb_images[-2], rgb_images[-1]]).astype(np.float64),
                                                    self.time_event_prev, self.tt, dt)
            # append event id & number between two frames to a deque()
            # self.event_count.append(all_events_arr.shape[0])\
            print(np.sum(time_event))
            self.num_event_count.append(diff)
            # print('current_time_event', np.sum(diff))
            # print('event_count', len(self.num_event_count))
            # print('event_count_1', (self.num_event_count[0]))

            # assign value to time event
            self.time_event_prev = time_event
            self.tt = tt
        else:
            print("initiate events")

        # print('image sequence length: {len}'.format(len=len(rgb_images)))
        if len(rgb_images) == window_size:
            if is_frames:
                mei, mhi = calmeimhi(np.asarray(rgb_images).astype(np.float64), 0, window_size-1, slide_step, tol,
                                     sensor_size, is_frames)
            else:
                # print('t_frame', time.time())
                # all_events_arr, time_event, tf = dvs.get_event_sequence(np.asarray(rgb_images).astype(np.float64), dt)
                # print('t_events', time.time())
                print('et_before_sub', np.sum(self.time_event_prev))
                mei, mhi = calmeimhi(self.time_event_prev, None, None, None, None, sensor_size, is_frames)
                self.time_event_prev -= self.num_event_count.popleft()
                print('et_after_sub', np.sum(self.time_event_prev))

                self.tt = 0
                # self.count += 1

            mei_hu = hu_moments(mei)
            mhi_hu = hu_moments(mhi)
            label_record = []
            mahad_record = []

            # print('classified labels: {labels}'.format(labels='other'))
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
                # mahad_record.append(mahad_mei)
                mahad_record.append(mahad_mei + mahad_mhi)
            scores_cache.append(mahad_record)
            scores_sum += np.asarray(mahad_record)[:, 0, 0]
            # print('classified labels: {labels}'.format(labels='second'))
            # print('scores_cache_len', len(scores_cache))
            if len(scores_cache) == average_size:
                label_text = label_record[int(np.argmin(scores_sum))]
                # print out label prob
                label_probabilities = np.exp(-scores_sum[1:])/np.max(np.exp(-scores_sum[1:]))

                print("action score", scores_sum)
                # if label_text == 'waving' or label_text == 'pointing':
                #     if np.abs(scores_sum[2]-scores_sum[3])/scores_sum[3] < 0.4:
                #         label_text = 'waving'
                #         label = ActionLabel(2)
                #elif label_text == 'walking' or label_text == 'jogging':
                    #label_text = 'other'
                    #label = ActionLabel(0)
                if label_text == 'waving':
                    label = ActionLabel(2)
                elif label_text == 'walking':
                    label = ActionLabel(0)
                elif label_text == 'jogging':
                    label = ActionLabel(1)
                elif label_text == 'hovering':
                    label = ActionLabel(3)
                elif label_text == 'movewalking':
                    label = ActionLabel(0)
                elif label_text == 'pointing_outdoors':
                    label = ActionLabel(5)

                # print('classified labels: {labels}'.format(labels='third'))
                # if label_text == 'waving':
                #     label = ActionLabel(1)
                # if label_text == 'pointing':
                #     label = ActionLabel(2)
                print('classified labels: {labels}'.format(labels=label_text))
                scores_sum -= np.asarray(scores_cache.popleft())[:, 0, 0]
            rgb_images.popleft()

        if use_probability:
            self._update_action_buffers(label, use_probabilities=True, probabilities=label_probabilities)
        else:
            self._update_action_buffers(label)
        print(label)

        if len(rgb_images) < 2:
            return label, frame
        else:
            return label, rgb_images[-1]

    def update_action_classification_qh_new(self, frame, cov_mei, cov_mhi, mean_mei_hu, mean_mhi_hu, rgb_images,
                                            scores_cache, is_frames, use_probability):
        label = ActionLabel.other
        label_probabilities = np.zeros(3)

        slide_step = 1
        tol = 70                                             #### to do: tune tolerance
        window_size = 15                                       #### to do: tune
        dt = 1.0 / 30  # frame rate
        train_labels = list(['walking', 'jogging', 'waving', 'pointing', 'hovering'])
        scores_sum = np.zeros(len(train_labels))

        sensor_size = [frame.shape[0], frame.shape[1]]  # sensor size
        average_size = 3                                       #### to do: tune
        dvs = DVS(sensor_size=sensor_size, thresh=0.01)  # to do: tune
        rgb_images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        # calc events for every two frames

        if len(rgb_images) > 1:
            time_event, tt = dvs.get_event_sequence(np.asarray([rgb_images[-2], rgb_images[-1]]).astype(np.float64),
                                                                    self.time_event, self.tt, dt)
            # append event id & number between two frames to a deque()
            # self.event_count.append(all_events_arr.shape[0])
            self.num_event_count.append(time_event)
            print('event_count', len(self.num_event_count))
            print('event_count_1',(self.num_event_count[0]))

            # assign value to time event
            self.time_event = time_event
            self.tt = tt
        else:
            print("initiate events")

        # print('image sequence length: {len}'.format(len=len(rgb_images)))
        if len(rgb_images) == window_size:
            if is_frames:
                mei, mhi = calmeimhi(np.asarray(rgb_images).astype(np.float64), 0, window_size-1, slide_step, tol,
                                     sensor_size, is_frames)
            else:
                mei, mhi = calmeimhi(self.time_event, None, None, None, None, sensor_size, is_frames)

                # self.all_events[0:self.event_count[self.count], 0]
                self.time_event -= self.num_event_count.popleft()
                # self.num_event_count[self.count]
                self.tt = 0
                self.count += 1

            mei_hu = hu_moments(mei)
            mhi_hu = hu_moments(mhi)
            label_record = []
            mahad_record = []

            # print('classified labels: {labels}'.format(labels='other'))
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
            # print('classified labels: {labels}'.format(labels='second'))
            # print('scores_cache_len', len(scores_cache))
            if len(scores_cache) == average_size:
                label_text = label_record[int(np.argmin(scores_sum))]
                # print out label prob
                label_probabilities = np.exp(-scores_sum[1:])/np.max(np.exp(-scores_sum[1:]))

                print("action score", scores_sum)
                if np.abs(scores_sum[2]-scores_sum[3])/scores_sum[3] < 0.4:
                    label_text = 'waving'
                    label = ActionLabel(1)

                # print('classified labels: {labels}'.format(labels='third'))
                # if label_text == 'waving':
                #     label = ActionLabel(1)
                # if label_text == 'pointing':
                #     label = ActionLabel(2)
                print('classified labels: {labels}'.format(labels=label_text))
                scores_sum -= np.asarray(scores_cache.popleft())[:, 0, 0]
            rgb_images.popleft()

        if use_probability:
            self._update_action_buffers(label, use_probabilities=True, probabilities=label_probabilities)
        else:
            self._update_action_buffers(label)
        print(label)

        if len(rgb_images) < 2:
            return label, frame
        else:
            return label, rgb_images[-2]



    def get_filtered_label_probabilities(self):
        if self.action_buffer is None:
            return 0
        else:
            return np.mean(self.action_buffer, axis=0)

    def is_current_action(self, label):
        probabilities = self.get_filtered_label_probabilities()
        current_action = np.argmax(probabilities)
        return current_action == label.value

    def _update_action_buffers(self, label, use_probabilities = False, probabilities = None):
        if self.action_buffer is None:
            self.init_action_buffer(self.buffer_len, len(probabilities))
        self.rh_action_label_buffer = np.roll(self.rh_action_label_buffer, 1)
        self.action_buffer = np.roll(self.action_buffer, 1, axis=0)

        self.rh_action_label_buffer[0] = label.value
        if not use_probabilities:
            self.action_buffer[0, ActionLabel.waving.value] = 1 if label is ActionLabel.waving else 0
            self.action_buffer[0, ActionLabel.pointing.value] = 1 if label is ActionLabel.pointing else 0
        else:
            self.action_buffer[0,:] = probabilities

    