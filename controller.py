# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time
import json
import numpy as np
import glob
from person import Person, ActionLabel
import scipy.optimize
from enum import Enum
from action_classifier import get_action_classifier
from glasses_detector import get_glasses_classifier
from red_object_detector import get_robot_position
from object_classifier import get_object_classifier
import colorsys
from itertools import compress
from object_recognition import detect_object, compute_hof_features, detect_object_qh
from optical_flow import compute_optical_flow, visualize_flow
from event_camera import get_events_from_frames, get_event_optical_flow, visualize_events
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
# import joblib

# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(os.path.join(dir_path,'..', 'dependencies', 'Python'))
        # sys.path.append(dir_path + '/../../python/openpose/Release')
        # os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
        os.environ['PATH'] += ';' + os.path.join(dir_path, '..', 'dependencies', 'x64', 'Release') + ';' + os.path.join(dir_path, '..', 'dependencies', 'bin') + ';'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append('../../python')
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", default="../../../examples/media/", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
# parser.add_argument("--net_resolution", default="1x176")
parser.add_argument("--disable_multi_thread", default=True)
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "../models/"
# params['render_threshold'] = 0.25

# Add others in path?
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1])-1: next_item = args[1][i+1]
    else: next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace('-','')
        if key not in params:  params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace('-','')
        if key not in params: params[key] = next_item


USE_GPC = False
REACQUIRE_RADIUS_GROWTH_RATE = 1.5
REACQUIRE_RADIUS_GROWTH_OFFSET = 6
# MIN_COMMAND_CHANGE_TIME = 2
MIN_COMMAND_CHANGE_TIME = 10
# MIN_COMMAND_CHANGE_TIME_POINT = 5
STARTUP_DELAY = 2


class FlightMode(Enum):
    land = 0
    hover = 1
    follow = 2
    pointing_follow = 3
    robot_follow = 4


class Controller:

    def __init__(self, save_video=False, save_raw_video=False):
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()
        self.t_last_tracked = 0
        self.t_start = time.time()
        self.t_last_update_position = time.time()
        self.launch_time = time.time()

        self.box_window = np.empty([1, 4])
        self.box_window_point = np.empty([1, 4])

        # self.classifier = joblib.load('../models/oc_saved/Saved_SVM_model_frame20_1')

        self.r_forearm_angle_buffer = np.zeros(3)
        self.t_stamps = np.zeros(3)

        self.command_change_time = STARTUP_DELAY - MIN_COMMAND_CHANGE_TIME

        self.save_video = save_video
        self.save_raw_video = save_raw_video

        # self.tracked_persons = np.empty(0, Person)
        self.tracked_persons = []
        self.active_person = None

        # added by JG -- store current position estimate for the robot being tracked
        self.robot_position = None

        self.hover_count = 0

        self.counter = 0

        self.t_yaw = None

        self.all_events = None

        print('Initializing classifier...')
        self.action_classifier = get_action_classifier(use_gpc=USE_GPC, enable_pointing=True)
        self.glasses_classifier = get_glasses_classifier()
        self.object_classifier_frame = joblib.load('../models/oc_saved/Saved_SVM_model_lat')
        self.object_classifier_firing = joblib.load('../models/oc_saved/Saved_SVM_model_lat')
        # self.object_classifier = get_object_classifier()

        # Velocity proportional and integral gains
        # self.k_p_v = 3.2 # outdoors
        self.box_mean = None
        self.k_p_v = 0.1   #0.08
        #k_p_v 0.08
        self.k_d_v = 0
        # self.k_d_v = 0.01
        # self.k_d_v = 1.6
        self.k_i_v = 0.01
        # Yaw rate proportional and integral gains
        # self.k_p_yaw = 40
        # self.k_d_yaw = 14
        # self.k_i_yaw = 6 * 0
        self.k_p_yaw = 0.03
        self.k_d_yaw = 0
        self.k_i_yaw = 6 * 0
        # Speed to travel at (in m/s) when following a pointing command
        self.point_speed = 0.25
        # self.point_speed = 0.1
        # 7 seconds @ 2 m/s for going across the creek at wee stinky glen
        self.point_duration = 10.0  # How long the drone will fly in a direction after a point command is given, in seconds
        self.robot_look_delay = 0.5
        # self.point_duration = 0.1

        self.yaw_estimation = None
        self.yaw_rate_prev = None
        self.yaw_prev = None

        self.max_move_dist_per_frame = 0.1 # Person hip and neck can only move by ___ normalized image units per frame
        self.max_torso_diff_per_frame = 0.35 # Torso can only change 20% in length between frames
        self.torso_length_target = 0.5
        self.neck_error_sum = 0
        self.neck_error_rate = 0
        self.neck_error_prev = 0
        self.body_size_error_sum = 0
        self.body_size_error_rate = 0
        self.body_size_error_prev = 0

        ### bounding box error parameters
        self.x_error_sum = 0
        self.y_error_sum = 0
        self.bb_size_error_sum = 0
        self.x_error_rate = 0
        self.y_error_rate = 0
        self.bb_size_error_rate = 0
        self.x_error_prev = 0
        self.y_error_prev = 0
        self.bb_size_error_prev = 0

        self.v_x_weights = self._log_normal(np.arange(15) + 1, 1.4, 0.5)
        self.v_y_weights = self._log_normal(np.arange(15) + 1, 1.4, 0.5)
        self.yaw_weights = np.ones(3)
        self.v_x_buffer = np.zeros(len(self.v_x_weights))
        self.v_y_buffer = np.zeros(len(self.v_y_weights))
        self.yaw_buffer = np.zeros(len(self.yaw_weights))
        self.flight_mode_command = FlightMode.land

        self.frame_width = None
        self.frame_height = None
        self.x_desired = None
        self.y_desired = None
        self.w_desired = None
        self.h_desired = None

        # self.file_yaw_rate = open("yaw_rate/yaw_rate_degree.txt", "a")

        self.file_vx_rate = open('drone_state/vx_{0}.txt'.format(time.strftime('%Y-%m-%d-%H%M%S', time.localtime())),"a")
        self.file_vy_rate = open('drone_state/vy_{0}.txt'.format(time.strftime('%Y-%m-%d-%H%M%S', time.localtime())),"a")
        self.file_vz_rate = open('drone_state/vz_{0}.txt'.format(time.strftime('%Y-%m-%d-%H%M%S', time.localtime())),"a")
        self.file_yaw = open('drone_state/yaw_{0}.txt'.format(time.strftime('%Y-%m-%d-%H%M%S', time.localtime())),"a")
        # self.file_t = open("yaw_rate/yaw_rate_t.txt", "a")

        self.datum = op.Datum()

        # Focal length should be 0.8150585 * image_width
        # self.focal_length = 484
        # self.focal_length = None

        self.ActionCount = -15

        N = len(ActionLabel)
        hsv_tuples = [(x/N*1.0, 0.5, 0.8) for x in range(N)]
        self._color_list = []
        for hsv in hsv_tuples:
            rgb = colorsys.hsv_to_rgb(*hsv)
            self._color_list.append([val*255 for val in rgb])

        self.hof_list = []
        self.bb_p_error = []
        self.bb_size_error = []

        self.is_frames = False
        self.ego_flow = np.zeros((480, 640, 2))

        self.object_label_value = None
        self.flow = None

    def set_frame_size(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.x_desired = self.frame_width/2
        self.y_desired = self.frame_height/2
        self.w_desired = 300
        self.h_desired = 180

        # Focal length should be 0.8150585 * image_width
        self.focal_length_pointing = 0.8150585 * self.frame_width
        self.focal_length = 132.28 # 35mm


        if self.save_video:
            date_time_now = time.strftime('%Y-%m-%d-%H%M%S', time.localtime())
            video_out_file = 'saved_footage/video_{0}.avi'.format(date_time_now)
            self.out_video_writer = cv2.VideoWriter(video_out_file,
                                                    cv2.VideoWriter_fourcc(*'MJPG'),
                                                    4,
                                                    (self.frame_width, self.frame_height))

        if self.save_raw_video:
            date_time_now = time.strftime('%Y-%m-%d-%H%M%S', time.localtime())
            video_out_file = 'saved_footage/raw_video_{0}.avi'.format(date_time_now)
            video_out_file_op = 'saved_footage_comp1/OFR_video_{0}.avi'.format(date_time_now)
            video_out_file_ops = 'saved_footage_comp1/OFS_video_{0}.avi'.format(date_time_now)
            video_out_file_ego = 'saved_footage_comp1/EGO_video_{0}.avi'.format(date_time_now)


            self.out_raw_video_writer = cv2.VideoWriter(video_out_file,
                                                    cv2.VideoWriter_fourcc(*'MJPG'),
                                                    5,
                                                    (self.frame_width, self.frame_height))

            self.out_raw_video_writer_op = cv2.VideoWriter(video_out_file_op,
                                                    cv2.VideoWriter_fourcc(*'MJPG'),
                                                    5,
                                                    (self.frame_width, self.frame_height))

            self.out_raw_video_writer_ops = cv2.VideoWriter(video_out_file_ops,
                                                    cv2.VideoWriter_fourcc(*'MJPG'),
                                                    5,
                                                    (self.frame_width, self.frame_height))

            self.out_raw_video_writer_ego = cv2.VideoWriter(video_out_file_ego,
                                                    cv2.VideoWriter_fourcc(*'MJPG'),
                                                    5,
                                                    (self.frame_width, self.frame_height))


    # def add_frame_overlay(self, img, probabilities, glasses_probability, box, alpha=None):
    def add_frame_overlay(self, img, glasses_probability, box, action_label, object_label, alpha=None):
        cv2.line(img,
                 (int(box[0,0]), int(box[0,1])),
                 (int(box[0,0]), int(box[0,3])),
                 (255, 255, 255),
                 3)
        cv2.line(img,
                 (int(box[0,0]),int( box[0,3])),
                 (int(box[0,2]),int(box[0,3])),
                 (255, 255, 255),
                 3)
        cv2.line(img,
                 (int(box[0,2]), int(box[0,3])),
                 (int(box[0,2]), int(box[0,1])),
                 (255, 255, 255),
                 3)
        cv2.line(img,
                 (int(box[0,2]), int(box[0,1])),
                 (int(box[0,0]), int(box[0,1])),
                 (255, 255, 255),
                 3)

        cv2.putText(img, object_label,
                    (500, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2)

    def add_action_labels(self, img, action_label):
        cv2.putText(img, action_label,
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 128, 0),
                    2)

    def draw_boundingbox(self, img, box):
        # draw desired bounding box
        cv2.line(img,
                 (int(240-self.h_desired/2), int(320-self.w_desired/2)),
                 (int(240-self.h_desired/2), int(320+self.w_desired/2)),
                 (255, 255, 255),
                 3)

        cv2.line(img,
                 (int(240-self.h_desired/2), int(320+self.w_desired/2)),
                 (int(240+self.h_desired/2), int(320+self.w_desired/2)),
                 (255, 255, 255),
                 3)

        cv2.line(img,
                 (int(240+self.h_desired/2), int(320+self.w_desired/2)),
                 (int(240+self.h_desired/2), int(320-self.w_desired/2)),
                 (255, 255, 255),
                 3)

        cv2.line(img,
                 (int(240+self.h_desired/2), int(320-self.w_desired/2)),
                 (int(240-self.h_desired/2), int(320-self.w_desired/2)),
                 (255, 255, 255),
                 3)
        # draw person box
        cv2.line(img,
                 (box[0][0], box[0][1]),
                 (box[0][0], box[1][1]),
                 (255, 255, 255),
                 3)
        cv2.line(img,
                 (box[0][0], box[1][1]),
                 (box[1][0], box[1][1]),
                 (255, 255, 255),
                 3)
        cv2.line(img,
                 (box[1][0], box[1][1]),
                 (box[1][0], box[0][1]),
                 (255, 255, 255),
                 3)
        cv2.line(img,
                 (box[1][0], box[0][1]),
                 (box[0][0], box[0][1]),
                 (255, 255, 255),
                 3)

    def process_frame_with_openpose(self, frame):
        self.datum.cvInputData = frame
        self.opWrapper.emplaceAndPop([self.datum])

        return self.datum.cvOutputData

    def process_new_frame(self, frame, rgb_images, scores_cache, frame_prev, sensor_val, drone_yaw, drone_vx, drone_vy, drone_vz, is_frames):
        """
        Summary line.
        The method admits the sensor data from the DJI quadcopter, and prcesses the data to complete tasks: action recognition, object recognition and fight control.

        Parameters:
        ------------
        frame: the orginal rgb frame captured from the DJI quadcopter front camera
        rgb_images:
        scores_cache:
        frame_prev: 
        sensor_val:
        drone_yaw: the yaw angle of the quadcopter frm onboard sensor
        drone_vx: the velocity along the longitudinal axis of the quadcopter
        drone_vy: the velocity along the lateral axis of the quadcopter
        drone_vz: the velocity along the vertical axis of the quadcopter pointing toward the earth
        is_frames: 

        Returns:
        ------------
        frame: the processed image frame

        """


        tt = time.time()

        self.file_vx_rate.write("%f \r\n" % drone_vx)
        self.file_vy_rate.write("%f \r\n" % drone_vy)
        self.file_vz_rate.write("%f \r\n" % drone_vz)
        self.file_yaw.write("%f \r\n" % drone_yaw)

        # print('t0', time.time())
        if self.frame_height is None:
            self.set_frame_size(np.shape(frame)[1], np.shape(frame)[0])

        # comment openpose part
        # t0 = time.time()
        self.process_frame_with_openpose(frame)
        # print('t00', time.time())
        # t00 = time.time()

        # Process the new keypoints, updating the list of tracked people accordingly
        self._process_keypoints(self.datum.poseKeypoints, time.time() - self.t_start)
        # trained action recognition moments
        data_dir = '../models/trained'
        if self.is_frames:
            fname_mei_cov = data_dir + '/moments/cov_mei.npy'
            fname_mhi_cov = data_dir + '/moments/cov_mhi.npy'
            fname_mei_hu = data_dir + '/moments/mean_mei_hu.npy'
            fname_mhi_hu = data_dir + '/moments/mean_mhi_hu.npy'
        else:
            fname_mei_cov = data_dir + '/moments/cov_mei_event.npy'
            fname_mhi_cov = data_dir + '/moments/cov_mhi_event.npy'
            fname_mei_hu = data_dir + '/moments/mean_mei_hu_event.npy'
            fname_mhi_hu = data_dir + '/moments/mean_mhi_hu_event.npy'
        label = ActionLabel(0)
        openpose_label = ActionLabel(6)
        box = None
        box_mean = None

        #
        # print("yaw", drone_yaw)
        ego_vx, ego_vy = self.rotational_ego_motion(drone_yaw, frame)
        for person in self.tracked_persons:
            openpose_label = person.update_action_classification(self.action_classifier, use_probability=False)
            # if self.flight_mode_command == FlightMode.land:
            self.ActionCount +=1
            print('Action Classification Count', self.ActionCount)
            label, frame_disp = person.update_action_classification_qh(frame, np.load(fname_mei_cov), np.load(fname_mhi_cov),
                                                   np.load(fname_mei_hu), np.load(fname_mhi_hu), rgb_images, scores_cache,
                                                           is_frames=self.is_frames, use_probability=True)        
            person.update_is_wearing_glasses(frame, self.glasses_classifier)


        people_waving = (label.value == 2 or label.value == 3)
        # people_waving = True

        people_wearing_glasses = [p.is_wearing_glasses_filtered for p in self.tracked_persons]

        # Change flight modes:
        # When someone puts glasses on, take off.
        # If it is flying and someone waves at it, follow them
        # If someone waves at it while it is following, land
        if time.time() - self.command_change_time > MIN_COMMAND_CHANGE_TIME:
            print("people_waving", people_waving)
            if np.any(people_waving) and \
                    (self.flight_mode_command == FlightMode.land):
                # ((self.flight_mode_command == FlightMode.hover) or
                    #  (self.flight_mode_command == FlightMode.land) or
                    #  (self.flight_mode_command == FlightMode.pointing_follow)):
                print(self.flight_mode_command)
                self.command_change_time = time.time()
                self.t_last_tracked = time.time()
                self.launch_time = time.time()
                self.flight_mode_command = FlightMode.follow
                self.active_person = self.tracked_persons[np.argmax(people_waving)]
                # Reset the errors (including the integral terms in the controller) after taking off, before
                # starting to follow someone
                self._update_position_error()
                print('Switched to follow')

            elif self.flight_mode_command == FlightMode.follow:
                print('tracking function test')
                # elif (self.flight_mode_command == FlightMode.follow) and self._is_actively_tracking():
                if openpose_label.value == 2:
                    print('label1', openpose_label.value)
                    self.command_change_time = time.time()
                    self.flight_mode_command = FlightMode.pointing_follow
                    self.point_start_time = time.time()
                    self._send_quad_point_command(self.active_person.keypoints_buffer[0, :, :], None)
                    print("Switched to pointing follow")
                elif label.value == 2 or label.value == 3:
                    # print('Action Label 1', label.value)
                    # elif self.active_person.is_current_action(ActionLabel.waving) or self.active_person.is_current_action(ActionLabel.hovering):
                    if time.time() - self.t_start > 4.5*MIN_COMMAND_CHANGE_TIME:
                        self.command_change_time = time.time()
                        print('com_change_time',self.command_change_time)
                        # self.command_change_time = time.time()
                        self._send_quad_land_command()
                        # print('Action Label', label.value)
                        print('Switched to land (follow)')
                        # cv2.putText(frame_prev[-1], 'waving',
                        #             (10, 70),
                        #             cv2.FONT_HERSHEY_SIMPLEX,
                        #             1,
                        #             (255, 255, 255),
                        #             2)

        if self.flight_mode_command == FlightMode.follow:
            box, object_label, self.flow = self.update_object_classification_qh(frame, frame_prev, self.all_events,
                                                                    time.time(), sensor_val, ego_vx, ego_vy,
                                                                           is_frames=self.is_frames, is_classify=True)

            # print('t00000', time.time())

            # print(box)
            if box is None:
                print("no person tracked")
            else:
                print("track person")
                self.box_window = np.append(self.box_window, np.array(box).reshape(1, 4), axis=0)
                # if self.counter == 0:
                #     self.box_window = np.delete(self.box_window, 0, 0)
                # else:
                # print('box_window', self.box_window)
                # print('box_window', self.box_window.shape[0])

                if self.box_window.shape[0] >= 4:
                    # smooth the bbox
                    box_mean = np.mean(self.box_window[1:], axis=0)
                    # print('box_mean', box_mean)
                    # print('box_mean_shape', box_mean.shape)
                    self.box_mean = np.array(box_mean).reshape(1,4)
                    # print('box_mean_shape', self.box_mean[0, 0])
                    self._update_boundingbox_error((self.box_mean[0, 0] + self.box_mean[0, 2]) / 2, (self.box_mean[0,1] + self.box_mean[0, 3]) / 2,
                                                    self.box_mean[0, 2] - self.box_mean[0, 0], self.box_mean[0, 3] - self.box_mean[0, 1])
                    # define object classifier
                    # self.object_recognition_test(flow, self.classifier, frame, self.box_mean)
                    # print('t01', time.time())

                    # np.array([0, 480 / 3, 640, 480]).reshape(1, 4)
                    hof_feature, self.object_label_value = self.object_classifier(self.flow, frame, self.box_mean, is_frames=self.is_frames)

                    if (time.time()-self.launch_time) > 2:
                        self._send_quad_follow_command_qh(self.box_mean, self.active_person.neck_position)
                    self.box_window = np.delete(self.box_window, 0, 0)
                self.counter += 1
            # if people_waving:
            #     self.flight_mode_command = FlightMode.land

            # else:
            #     self._update_position_error()
            #     self._send_quad_follow_command(None, None)
        elif self.flight_mode_command == FlightMode.pointing_follow:
            if time.time() < self.point_start_time + self.point_duration: # and self.active_person.is_current_action(ActionLabel.pointing):
                print('start pointing follow')
                self._send_quad_point_command(self.active_person.keypoints_buffer[0, :, :], None)
            else:
                self.command_change_time = time.time()
                self.flight_mode_command = FlightMode.hover
        elif self.flight_mode_command == FlightMode.hover:
            if label.value == 2 or label.value == 3:
                if time.time() - self.t_start > 4.2 * MIN_COMMAND_CHANGE_TIME:
                    self.command_change_time = time.time()
                    self._send_quad_land_command()
                    print('Switched to land')
        elif self.flight_mode_command == FlightMode.robot_follow:
            if self.robot_position is not None and len(self.robot_position) > 1:
                self.robot_position = get_robot_position(frame,
                                                         prev_position=self.robot_position,
                                                         target_area=0.15,
                                                         gamma=0.5,
                                                         track_tol=0.8,
                                                         min_sv=[240, 100])
            else:
                self.robot_position = get_robot_position(frame,
                                                         target_area=0.15,
                                                         gamma=0.5,
                                                         track_tol=0.8,
                                                         min_sv=[240, 100])

            self._send_robot_follow_command(self.robot_position)

        img = self.datum.cvOutputData

        if self.save_raw_video:
            self.out_raw_video_writer.write(frame)

        if self._is_actively_tracking():
            probabilities = self.active_person.get_filtered_label_probabilities()
            glasses_probability = np.mean(self.active_person.is_wearing_glasses_buffer)
        else:
            probabilities = np.zeros(len(self.action_classifier.classes_))
            if len(self.tracked_persons) > 0:
                glasses_probability = np.mean(self.tracked_persons[0].is_wearing_glasses_buffer)
            else:
                glasses_probability = 0.0
        if self._is_actively_tracking():
            alpha = Person.get_rh_angle_with_image_plane(self.active_person.keypoints_buffer[0, :, :], self.focal_length_pointing)
        else:
            alpha = None
        self.add_action_labels(frame, action_label=label.name)

        if self.box_mean is None:
            print("no bbox shown")
        else:
            # self.draw_boundingbox(img, box)
            if self.flight_mode_command == FlightMode.land or self.flight_mode_command == FlightMode.pointing_follow or \
                    self.flight_mode_command == FlightMode.hover:
                print("landing or pointing, no flow")
            else:
                if self.object_label_value is None:
                    print('no object')
                else:
                    self.add_frame_overlay(frame, box=self.box_mean, action_label=label.name, object_label=self.object_label_value[0],
                                           alpha=alpha, glasses_probability=glasses_probability)
                    frame = visualize_flow(frame, self.flow, decimation=5, scale=1)
        if self.save_video:
            self.out_video_writer.write(frame)

        self.t_yaw = time.time() - tt
        print('flight mode',self.flight_mode_command)
        return frame

    def get_full_command(self):
        command = {
            'mode': self.flight_mode_command.value,
            'v_x': self.get_v_x_command(),
            'v_y': self.get_v_y_command(),
            'yaw_rate': self.get_yaw_rate_command()
        }
        return command

    def send_override_flight_mode_command(self, override_command: int):
        print('Override Received: {0}'.format(override_command))
        if override_command == 0:
            self._send_quad_land_command()
        elif override_command == 1:
            self._send_quad_hover_command()

    def finalize(self):
        # self.file_yaw_rate.close()
        # self.file_t.close()
        self.file_vx_rate.close()
        self.file_vy_rate.close()
        self.file_vz_rate.close()
        self.file_yaw.close()

        date_time_now = time.strftime('%Y-%m-%d-%H%M%S', time.localtime())
        p_name = 'error_data/position_error_{0}.npy'.format(date_time_now)
        np.save(p_name, np.array(self.bb_p_error))
        s_name = 'error_data/size_error_{0}.npy'.format(date_time_now)
        np.save(s_name, np.array(self.bb_size_error))

        if self.save_video:
            self.out_video_writer.release()
        if self.save_raw_video:
            self.out_raw_video_writer.release()
            self.out_raw_video_writer_op.release()
            self.out_raw_video_writer_ops.release()
            self.out_raw_video_writer_ego.release()

    def _read_training_data(self, file_names):
        training_data = np.zeros([0, 25, 3])
        t_log = np.zeros([0])
        for file_name in file_names:
            with open(file_name) as json_file:
                data = json.load(json_file)
                pos_data_arr = np.array(data['body_keypoints'])
                t_data_arr = np.array(data['time_stamps'])
                training_data = np.concatenate((training_data, pos_data_arr), axis=0)
                t_log = np.concatenate((t_log, t_data_arr), axis=0)

        return training_data, t_log

    def _get_tracking_radius(self):
        x = (time.time() - self.t_last_tracked) * REACQUIRE_RADIUS_GROWTH_RATE - REACQUIRE_RADIUS_GROWTH_OFFSET
        return 2 * (np.exp(x) / (np.exp(x) + 1))

    def _process_keypoints(self, pose_keypoints, t_stamp):
        """
        Summary line.

        Process the new keypoint measurements and match them with the existing persons list, adding or removing tracked people as necessary

        Parameters:
        -----------
        pose_keypoints: new peoplpe pose keypoints measurements
        t_stamp: the time stamp of the particular pose keypoints

        Returns:
        -----------
        None
        """


        if np.size(pose_keypoints) > 1:
            neck_pos_new = Person.get_neck_pos(pose_keypoints, self.frame_width, self.frame_height)
            mid_hip_pos_new = Person.get_mid_hip_pos(pose_keypoints, self.frame_width, self.frame_height)
        else:
            neck_pos_new = np.zeros([0, 2])
            mid_hip_pos_new = np.zeros([0, 2])

        # Throw out incomplete detections, TODO: unless a partial detection is insanely close to the currently active person - will require tracking torso height as a class member and modifying the complete_detection_row_mask below
        complete_detection_row_mask = ~np.any(np.hstack((neck_pos_new, mid_hip_pos_new)) == -1, axis=1)

        neck_pos_new = neck_pos_new[complete_detection_row_mask, :]
        mid_hip_pos_new = mid_hip_pos_new[complete_detection_row_mask, :]
        active_person_idx = None

        neck_pos_old = np.array([p.neck_position for p in self.tracked_persons])
        mid_hip_pos_old = np.array([p.mid_hip_position for p in self.tracked_persons])
        # remove_active_person = False

        if self._is_actively_tracking() and (np.size(neck_pos_new) > 0):
            dist_mat = np.array(
                [np.sqrt((pos[0] - neck_pos_new[:, 0]) ** 2 + (pos[1] - neck_pos_new[:, 1]) ** 2) for pos in [self.active_person.neck_position]])
            dist_mat += np.array(
                [np.sqrt((pos[0] - mid_hip_pos_new[:, 0]) ** 2 + (pos[1] - mid_hip_pos_new[:, 1]) ** 2) for pos in [self.active_person.mid_hip_position]])
            rows, cols = scipy.optimize.linear_sum_assignment(dist_mat)
            active_person_idx = np.argmin([p.ID - self.active_person.ID for p in self.tracked_persons])
            self.tracked_persons[active_person_idx].set_neck_pos(neck_pos_new[cols[0], :],
                                                       filter_bad_values=True,
                                                       frame_width=self.frame_width,
                                                       frame_height=self.frame_height)
            self.tracked_persons[active_person_idx].set_mid_hip_pos(mid_hip_pos_new[cols[0], :])
            self.tracked_persons[active_person_idx].update_pose_keypoints(pose_keypoints[cols[0], :, :], t_stamp)
            self.tracked_persons[active_person_idx].torso_length = Person.get_torso_length(neck_pos_new[cols[0], :],
                                                                                 mid_hip_pos_new[cols[0], :])

            neck_pos_old = np.array([p.neck_position for p in self.tracked_persons if p.ID != self.active_person.ID])
            mid_hip_pos_old = np.array([p.mid_hip_position for p in self.tracked_persons if p.ID != self.active_person.ID])

            unused_rows = np.full(neck_pos_new.shape[0], True, dtype=bool)
            unused_rows[cols[0]] = False
            neck_pos_new = neck_pos_new[unused_rows, :]
            mid_hip_pos_new = mid_hip_pos_new[unused_rows, :]

        if np.any(np.hstack((neck_pos_old, mid_hip_pos_old)) == -1):
            print('Found null value for hips or neck!')

        if (np.size(neck_pos_new) > 0) & (np.size(neck_pos_old) > 0):
            dist_mat = np.array(
                [np.sqrt((pos[0] - neck_pos_new[:, 0]) ** 2 + (pos[1] - neck_pos_new[:, 1]) ** 2) for pos in neck_pos_old])
            dist_mat += np.array(
                [np.sqrt((pos[0] - mid_hip_pos_new[:, 0]) ** 2 + (pos[1] - mid_hip_pos_new[:, 1]) ** 2) for pos in mid_hip_pos_old])
            rows, cols = scipy.optimize.linear_sum_assignment(dist_mat)

            # if self._is_actively_tracking():
            #     weights = np.ones(neck_pos_new.shape[0])
            #     dist_mat *= np.array([weights * 5 if p.ID == self.active_person.ID else weights for p in self.tracked_persons])
        else:
            rows = []
            cols = []

        # Update the positions for all existing persons using the new detections
        for i in range(len(rows)):
            idx_old = rows[i]
            idx_new = cols[i]
            if active_person_idx is not None and rows[i] >= active_person_idx:
                idx_old += 1

            self.tracked_persons[idx_old].set_neck_pos(neck_pos_new[idx_new, :],
                                                       filter_bad_values=True,
                                                       frame_width=self.frame_width,
                                                       frame_height=self.frame_height)
            self.tracked_persons[idx_old].set_mid_hip_pos(mid_hip_pos_new[idx_new, :])
            self.tracked_persons[idx_old].update_pose_keypoints(pose_keypoints[idx_new, :, :], t_stamp)
            self.tracked_persons[idx_old].torso_length = Person.get_torso_length(neck_pos_new[idx_new, :],
                                                                                 mid_hip_pos_new[idx_new, :])

        # Remove any people who we don't have detections for now
        index_to_remove = np.setdiff1d(np.arange(neck_pos_old.shape[0]), rows)
        if active_person_idx is not None:
            index_to_remove[index_to_remove >= active_person_idx] += 1

        for i_remove in sorted(index_to_remove, reverse=True):
            # idx = i_remove
            # if active_person_idx is not None and i_remove >= active_person_idx:
            #     idx += 1
            del self.tracked_persons[i_remove]

        # if remove_active_person:
        #     del self.tracked_persons[np.argmin([p.ID - self.active_person.ID for p in self.tracked_persons])]

        # Add any new people that we now have detections for
        index_to_add = np.setdiff1d(np.arange(neck_pos_new.shape[0]), cols)
        for i in index_to_add:
            new_person = Person().set_neck_pos(neck_pos_new[i, :]) \
                .set_mid_hip_pos(mid_hip_pos_new[i, :]) \
                .update_pose_keypoints(pose_keypoints[i, :, :], t_stamp)\
                .set_torso_length(neck_pos_new[i, :], mid_hip_pos_new[i, :])
            # self.tracked_persons = np.append(self.tracked_persons, new_person)
            self.tracked_persons.append(new_person)

        if not self._is_actively_tracking():
            if (self.active_person is not None) & (len(self.tracked_persons) > 0):
                print('Attempting to re-acquire person {0}'.format(self.active_person.ID))
                #         Attempt to re-acquire
                reacquire_radius = self._get_tracking_radius()

                neck_positions = np.array([p.neck_position for p in self.tracked_persons])
                torso_lengths = np.array([Person.get_torso_length(p.neck_position, p.mid_hip_position) for p in self.tracked_persons])
                dist = np.sqrt((self.active_person.neck_position[0] - neck_positions[:, 0]) ** 2 + (
                        self.active_person.neck_position[1] - neck_positions[:, 1]) ** 2)
                if (np.min(dist) < reacquire_radius) and \
                        (np.all(neck_positions[np.argmin(dist), :]) > 0): # and \
                        # (torso_lengths[np.argmin(dist)] - self.active_person.torso_length)/self.active_person.torso_length > -self.torso_reacquire_max_diff:
                    self.active_person = self.tracked_persons[np.argmin(dist)]
        else:
            self.t_last_tracked = time.time()

    def _is_actively_tracking(self):
        # return True
        if self.active_person is None:
            return False
        else:
            return self.active_person.ID in [p.ID for p in self.tracked_persons]

    def _send_robot_follow_command(self, robot_pos):
        print('robot_pos',robot_pos)
        if robot_pos is None:
            u_yaw = 0.0
            v_x = 0.0
        else:
            u_yaw = self.k_p_yaw * (robot_pos[0] - 0.5)
            robot_pos_error = (robot_pos[1] - 0.5)
            # v_x = np.clip(- self.k_p_v * 0.3 * robot_pos_error, -.4, .4)
            v_x = np.clip(- self.k_p_v * 0.4 * robot_pos_error, -.4, .4)
            print('robot_pos = {}; u_yaw = {}; v_x = {};'.format(robot_pos, u_yaw, v_x))
        self.flight_mode_command = FlightMode.robot_follow
        # print(f'yaw={u_yaw}; v_x={v_x}')
        self._set_v_x_command(v_x)
        self._set_v_y_command(0.0)
        self._set_yaw_rate_command(u_yaw)

    def _send_quad_follow_command(self, neck_pos, mid_hip_pos):
        if neck_pos is None or mid_hip_pos is None:
            u_yaw = 0.0
            v_x = 0.0
        else:
            u_yaw = self._get_yaw_control(neck_pos)
            if np.any(neck_pos == -1) or np.any(mid_hip_pos == -1):
                v_x = 0
            else:
                torso_length = Person.get_torso_length(neck_pos, mid_hip_pos)
                if torso_length is None or torso_length == 0:
                    v_x = 0
                else:
                    body_size_error = self.torso_length_target - torso_length
                    v_x = np.clip(
                        self.k_p_v * body_size_error + self.k_i_v * self.body_size_error_sum + self.k_d_v * self.body_size_error_rate,
                        -0.01, 1)

        self.flight_mode_command = FlightMode.follow
        self._set_v_x_command(v_x)
        self._set_v_y_command(0.0)
        self._set_yaw_rate_command(u_yaw)

    def _send_quad_point_command(self, keypoints=None, neck_pos=None):
        if keypoints is None:
            self._set_v_x_command(0.0)
            self._set_v_y_command(0.0)
        else:
            alpha = Person.get_rh_angle_with_image_plane(keypoints, self.focal_length_pointing)
            # alpha = 3.14159/6
            # print('angle alpha', alpha)
            # print('v_x_point_follow', self.point_speed * np.sin(alpha))
            # print('v_y_point_follow', self.point_speed * np.cos(alpha))

            if alpha is not None:
                for i in range(len(self.v_x_buffer)):
                    self._set_v_x_command(self.point_speed * np.sin(alpha))
                    self._set_v_y_command(self.point_speed * np.cos(alpha))

        self.flight_mode_command = FlightMode.pointing_follow
        if neck_pos is None:
            u_yaw = 0.0
        else:
            u_yaw = self._get_yaw_control(neck_pos)
        self._set_yaw_rate_command(u_yaw)

    # Qingze: added PID controllers for vx & yaw control
    def _send_quad_follow_command_qh(self, box, neck_pos):
        if box is None:
            u_yaw = 0.0
            v_x = 0.0
        else:
            # TODO: add yaw controller
            # u_yaw = self._get_yaw_control(neck_pos)
            # u_yaw = 0

            bb_position_error = (box[0, 2]+box[0, 0])/2 - self.x_desired
            self.bb_p_error.append(bb_position_error)

            u_yaw = np.clip(self.k_p_yaw*bb_position_error +
                            self.k_i_yaw*self.y_error_sum + self.k_d_yaw*self.y_error_rate, -5, 5)

            h = np.abs(box[0,3] - box[0,1])/2
            w = np.abs(box[0,2] - box[0,0])/2

            # scale the error due to the unit different: pixel and m.
            bb_size_error = -(np.sqrt(w*w + h*h) - np.sqrt(self.w_desired*self.w_desired + self.h_desired*self.h_desired))/100
            self.bb_size_error.append(bb_size_error)
            # print('h', h)
            # print('w', w)
            # print('de_h', self.h_desired)
            # print('de_w', self.w_desired)

            # TODO: adjust PID parameters
            v_x = np.clip(
                self.k_p_v * bb_size_error + self.k_i_v * self.bb_size_error_sum + self.k_d_v * self.bb_size_error_rate,
                -0.1, 0.7)
            #v_x = 0.0
            # print('error', bb_size_error)
            # print('error_rate', self.bb_size_error_rate)
            # print('error_sum', self.bb_size_error_sum)
            print('v_x', v_x)
            print('u_yaw', u_yaw)
        # change to hover if test action recognition
        self.flight_mode_command = FlightMode.follow
        self._set_v_x_command(v_x)
        # print('error', bb_size_error)
        # print('error_rate', self.bb_size_error_rate)
        # print('error_sum', self.bb_size_error_sum)

        self._set_v_y_command(0.0)
        self._set_yaw_rate_command(0.0)

    def _get_yaw_control(self, neck_pos):
        return self.k_p_yaw * neck_pos[0] + self.k_i_yaw * self.neck_error_sum + self.k_d_yaw * self.neck_error_rate

    def _send_quad_hover_command(self):
        self.flight_mode_command = FlightMode.hover
        self._set_v_x_command(0)
        self._set_v_y_command(0)
        self._set_yaw_rate_command(0)

    def _send_quad_land_command(self):
        self.flight_mode_command = FlightMode.land
        self._set_v_x_command(0)
        self._set_v_y_command(0)
        self._set_yaw_rate_command(0)

    @staticmethod
    def _log_normal(x, mu, sigma):
        expon = (np.log(x) - mu)**2 / (2*sigma**2)
        coef = 1./(sigma * np.sqrt(2*np.pi))
        return 1./x * coef * np.exp(-expon)

    def _set_v_x_command(self, v_x):
        self.v_x_buffer[:] = np.roll(self.v_x_buffer, 1)
        self.v_x_buffer[0] = v_x

    def _set_v_y_command(self, v_y):
        self.v_y_buffer[:] = np.roll(self.v_y_buffer, 1)
        self.v_y_buffer[0] = v_y

    def _set_yaw_rate_command(self, yaw_rate):
        self.yaw_buffer[:] = np.roll(self.yaw_buffer, 1)
        self.yaw_buffer[0] = yaw_rate

    def get_v_x_command(self):
        return np.sum(self.v_x_weights * self.v_x_buffer) / np.sum(self.v_x_weights)

    def get_v_y_command(self):
        return np.sum(self.v_y_weights * self.v_y_buffer) / np.sum(self.v_y_weights)

    def get_yaw_rate_command(self):
        return np.sum(self.yaw_weights * self.yaw_buffer) / np.sum(self.yaw_weights)

    def _update_position_error(self, neck_position = None, mid_hip_position = None):
        if (neck_position is None) or (mid_hip_position is None):
            self.neck_error_sum = 0
            self.body_size_error_sum = 0
            self.neck_error_prev = 0
        else:
            alpha = 0.5
            dt = (time.time() - self.t_last_update_position)
            self.neck_error_sum = np.clip(self.neck_error_sum + neck_position[0], -10, 10)
            self.neck_error_rate = (neck_position[0] - self.neck_error_prev)/dt

            torso_length = Person.get_torso_length(neck_position, mid_hip_position)
            if torso_length is None or torso_length == 0:
                body_size_error = 0
            else:
                body_size_error = self.torso_length_target - torso_length
            self.body_size_error_sum = np.clip(self.body_size_error_sum + body_size_error * dt, -2, 2)
            body_size_error_rate = (body_size_error - self.body_size_error_prev)/dt
            self.body_size_error_rate = alpha * body_size_error_rate + (1 - alpha) * self.body_size_error_rate
            self.neck_error_prev = neck_position[0]
            self.body_size_error_prev = body_size_error
            self.t_last_update_position = time.time()

    def _update_boundingbox_error(self, x, y, w, h):

        dt = (time.time() - self.t_last_update_position)
        x_error = np.abs(self.x_desired - x)
        y_error = np.abs(y - self.y_desired)
        #print("bounding box error", (np.sqrt(w*w+h*h)- np.sqrt(self.w_desired*self.w_desired - self.h_desired*self.h_desired))/100)
        bb_size_error = -(np.sqrt(w*w+h*h) - np.sqrt(self.w_desired*self.w_desired + self.h_desired*self.h_desired))/100

        self.x_error_sum = np.clip(self.x_error_sum + x_error, -10, 10)
        self.y_error_sum = np.clip(self.y_error_sum + y_error, -10, 10)
        self.bb_size_error_sum = np.clip(self.bb_size_error_sum + bb_size_error, -10, 10)

        self.x_error_rate = (x_error - self.x_error_prev)/dt
        self.y_error_rate = (y_error - self.y_error_prev)/dt
        self.bb_size_error_rate = (bb_size_error - self.bb_size_error_prev)/dt

        self.x_error_prev = x_error
        self.y_error_prev = y_error
        self.bb_size_error_prev = bb_size_error


    # @staticmethod
    def update_object_classification_qh(self, frame, frame_prev, all_events, t, sensor_val, ego_vx, ego_vy, is_frames, is_classify):
        """
        Summary line.

        This method uses the current and previous rgb frames to detect and (or) recognize the designated object

        Parameters:
        -----------
        frame: the current rgb frame capture by the quadcopter
        frame_prev: all previous rgb frames
        all_events: all events up till this time step
        t: current time step
        sensor_val:
        ego_vx: the quadcopter's self longitudinal velocity
        ego_vy: the quadcopter's self lateral velocity
        is_frames: a boolean variable specifying whether computing over rgb frames or events
        is_classify: a boolean variable specifying whether classifying targets after detecting the target's existence

        Returns:
        -----------
        mask: a binary image with detected target
        label: object target label (if classified)
        flow: the optical flow of the current the previous frames our events
        """

        dt = 1 / 30.0 ### change to real time rate
        img_scale = 1.0
        width = int(frame.shape[0] * img_scale)
        height = int(frame.shape[1] * img_scale)

        sensor_size = (width, height)
        bin_row = range(1, sensor_size[0] - 1)
        bin_col = range(1, sensor_size[1] - 1)
        #10
        window_size = 10

        # object labels
        label = 'None'
        # car or person or bicycle
        label_probabilities = np.zeros(3)

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if len(frame_prev) == 0:
            frame_prev.append(curr_gray)
        # box
        if is_frames:

            flow = compute_optical_flow([curr_gray, frame_prev[-1]])

            # visualize optical flow
            img1 = visualize_flow(frame, flow, decimation=20, scale=3)
            cv2.imshow('Optical Flow', img1)
            cv2.namedWindow('Optical Flow', cv2.WINDOW_NORMAL)
            if self.save_raw_video:
                self.out_raw_video_writer_op.write(img1)

            if ego_vx is None:
                print("on ground")
            else:
                # flow_trans_x = ego_vx
                # flow_trans_y = ego_vy
                mask = detect_object_qh(flow, gamma=0.3, return_mask=True, is_frames=True)

                flow_trans_x = flow[:, :, 0] + ego_vx*mask
                flow_trans_y = flow[:, :, 1] + ego_vy*mask
                # print('sub_flow', flow)
                flow[:, :, 0] = flow_trans_x
                flow[:, :, 1] = flow_trans_y

            # print('ego_vx_max', np.max(ego_vx))
            # print('ego_vy_max', np.max(ego_vy))

            # print('flow_shape', flow.shape)
            # print('flow', flow)
            mask = detect_object_qh(flow, gamma=0.3, return_mask=False, is_frames=True)
            # print('{mask_frame}'.format(mask_frame=mask))

            # visualize subtracted optical flow
            img2 = visualize_flow(frame, flow, decimation=20, scale=3)
            cv2.imshow('Optical Flow Subtracted', img2)
            cv2.namedWindow('Optical Flow Subtracted', cv2.WINDOW_NORMAL)
            if self.save_raw_video:
                self.out_raw_video_writer_ops.write(img2)

            self.ego_flow[:, :, 0] = ego_vx
            self.ego_flow[:, :, 1] = ego_vy
            img3 = visualize_flow(frame, self.ego_flow, decimation=20, scale=3)
            cv2.imshow('Ego', img3)
            cv2.namedWindow('Ego', cv2.WINDOW_NORMAL)
            if self.save_raw_video:
                self.out_raw_video_writer_ego.write(img3)

        else:
            t1 = time.time()
            frame_events, num_events, sensor_val_out = get_events_from_frames(curr_gray, t, frame_prev[-1],
                                                                              t - dt, sensor_val)
            if self.all_events is None:
                self.all_events = frame_events
                # print('first event', frame_events)
                mask = None
                img = None
                flow = np.zeros((height, width, 2))
            else:
                # print('all_e_else', np.shape(self.all_events))
                # print('frame_e_else', np.shape(frame_events))
                if len(self.all_events) < 1:
                    # print("empty event")
                    self.all_events = frame_events
                    mask = None
                    img = None
                    flow = np.zeros((height, width, 2))
                else:
                    self.all_events = np.hstack((self.all_events, frame_events))

                    # tau = 5 * dt
                    tau = 1 * dt

                    event_window = self.all_events[:, self.all_events[2, :] > t-tau]
                    # print('event_window', event_window)
                    # get optical flow from event data
                    t2 = time.time()

                    flow = get_event_optical_flow(frame_events, flow_shape=frame.shape, bins=(bin_row, bin_col, window_size),
                                                  t=t, dt=dt/10)
                    # print('events to flow time', time.time() - t2)

                    if ego_vx is None:
                        print("on ground")
                    else:
                        # flow_trans_x = np.transpose(flow[:, :, 0])
                        # flow_trans_y = np.transpose(flow[:, :, 1])
                        print('no subtraction', flow)
                    mask = detect_object_qh(flow, gamma=0.3, return_mask=False, is_frames=False)
        frame_prev.append(curr_gray)

        return mask, label, flow

    def rotational_ego_motion(self, drone_yaw, frame):
        u = None
        v = None

        self.yaw_estimation = drone_yaw
        if self.yaw_prev is None:
            print("skip first frame for yaw rate calc")
            self.yaw_prev = self.yaw_estimation
        else:
            if self.yaw_rate_prev is None:
                print("skip second frame for previous yaw rate calc")
                self.yaw_rate_prev = self.yaw_estimation - self.yaw_prev
            else:
                yaw_rate_value = self.yaw_rate_estimate(self.yaw_rate_prev, self.yaw_prev, drone_yaw)
                print('yaw_rate_value', yaw_rate_value)

                f_c = self.focal_length

                # add rotational(yaw) ego-motion subtraction
                # calc matrix of H
                # x = 0 1 2 ... frame.shape[0]
                # y = 0 1 2 ... frame.shape[1]
                # axis = 0, down
                # axis = 1, right
                width = frame.shape[1]
                height = frame.shape[0]
                print('width', width)

                h_array = np.arange(-int(height / 2), int(height / 2)+1)
                w_array = np.arange(-int(width / 2), int(width / 2)+1)

                h1 = (np.matmul(np.delete(h_array, int(height / 2)).reshape(height, 1),
                               np.delete(w_array, int(width / 2)).reshape(1, width)))/f_c
                #
                # # check h2 later
                h2 = - f_c * np.ones((height, width)) - \
                     (np.repeat(np.square(np.delete(w_array, int(width / 2))).reshape(1, width) / f_c, height,
                               axis=0))

                h3 = (np.repeat(np.arange(-int(height/2), int(height/2)).reshape(height, 1), width, axis=1))
                # print('h3',h3)

                # check h4 later
                h4 = f_c * np.ones((width, height)) + \
                     np.transpose(np.repeat(np.square(np.delete(h_array, int(height / 2))).reshape(height, 1) / f_c, width,
                               axis=1))

                h5 = -h1

                h6 = -np.repeat(np.arange(-int(width/2), int(width/2)).reshape(1, int(width)), height, axis=0)

                # rotational ego-motion
                u = h2 * yaw_rate_value

                v = h5 * yaw_rate_value

                # print("ego_u", u)
                # print("ego_v", v)

                self.yaw_prev = self.yaw_estimation
                self.yaw_rate_prev = yaw_rate_value

        return u, v

    def yaw_rate_estimate(self, yaw_rate_prev, yaw_angle_prev, yaw_angle):
        # print('yaw_angle', yaw_angle)
        # print('yaw_angle_prev', yaw_angle_prev)
        # print('yaw_angle', yaw_angle)
        if self.t_yaw is None:
            dt = 1/2
        else:
            dt = self.t_yaw
        print(dt)
        # counterclockwise is postive for yaw_rate
        alpha = 0.5
        if yaw_angle < 0:
            # print('yaw1')
            yaw_angle = yaw_angle + 360
        if yaw_angle_prev < 0:
            yaw_angle_prev = yaw_angle_prev + 360
            # print('yaw2')

        yaw_error = yaw_angle - yaw_angle_prev
        if abs(yaw_error) < 180:
            # print('yaw3')
            yaw_value_est = alpha * yaw_rate_prev + (1 - alpha) * yaw_error / dt
        else:
            if yaw_error > 0:
                # print('yaw4')
                yaw_value_est = alpha * yaw_rate_prev + (1 - alpha) * (yaw_error - 360) / dt
            else:
                # print('yaw5')
                yaw_value_est = alpha * yaw_rate_prev + (1 - alpha) * (360 + yaw_error) / dt

        return yaw_value_est*0.0174533

    def object_classifier(self, flow, curr_gray, mask, is_frames):

        feature_list = compute_hof_features(flow, curr_gray, mask, is_frames=True)
        feature_arr = np.array(feature_list)
        #print("feature_arr", feature_arr)
        if mask is None:
            print("object label is none")
        else:
            if is_frames:
                label_value = self.object_classifier_frame.predict(feature_arr.reshape(1, -1))
            else:
                label_value = self.object_classifier_firing.predict(feature_arr.reshape(1, -1))

            print("object label", label_value)
        return feature_list, label_value

    