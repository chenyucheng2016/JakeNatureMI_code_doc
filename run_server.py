from skimage import feature
from flask import jsonify, Flask, request
import sys
import cv2
import os
import pybase64
from PIL import Image
import io
from sys import platform
import argparse
from datetime import datetime
import threading
import numpy as np
from multiprocessing import Process, Queue, Value
import multiprocessing as mp
import ctypes
from controller import Controller
import time
from collections import deque
import pyzed.sl

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
parser.add_argument("--net_resolution", default="1x176")
parser.add_argument("--disable_multi_thread", default=True)  # what is multi-thread
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "../models/"
# params['render_threshold'] = 0.15

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

is_running = True
save_video = True
save_raw_video = True
url = '192.168.43.224'
# url = '128.253.227.253'
port = '8000'

def convert_images(server_to_converter, shared_img_arr, img_arr_shape):
    """
    Summary line.

    This function converts the encoded b4bit array from DJI quadcopter to a RGB image for processing
    in the PC

    Parameters:
    -----------
    server_to_converter: a queue that contains the sensor data (image, veclocities) from DJI quadcopter
    shared_img_arr: a multiprocessing array that can be accessed by multiple processes
    img_arr_shape: dimentions (width and height) of the image

    Returns:
    ---------
    None

    """
    img_arr = np.frombuffer(shared_img_arr.get_obj(), dtype=np.uint8).reshape(img_arr_shape)
    while True:
        item = server_to_converter.get()
        if item is None:
            break
        img_data = pybase64.b64decode(item['encoded_data'])
        bytes_data = io.BytesIO(img_data)
        image = Image.open(bytes_data)
        img_arr[...] = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)



def run_controller(shared_img_arr, img_arr_shape, shared_cmd_arr, override_flight_mode_command, rgb_images,
                   scores_cache, frame_prev, sensor_val, drone_vx, drone_vy, drone_vz, drone_yaw):
    

    frame = np.frombuffer(shared_img_arr.get_obj(), dtype=np.uint8).reshape(img_arr_shape)
    cmd_arr = np.frombuffer(shared_cmd_arr.get_obj(), dtype=np.double)
    quad_controller = Controller(save_video=save_video, save_raw_video=save_raw_video)
    t_prev = time.time()

    cv2.namedWindow('Processed Frame', cv2.WINDOW_NORMAL)
    new_cmd = np.zeros(4)
    cmd_alpha = 1.0

    while True:

        processed_frame = quad_controller.process_new_frame(frame, rgb_images, scores_cache, frame_prev, sensor_val, drone_yaw.value, drone_vx.value, drone_vy.value, drone_vz.value, is_frames=False)
        fps = 1./(time.time() - t_prev)

        new_cmd[...] = [quad_controller.flight_mode_command.value,
                        quad_controller.get_v_x_command(),
                        quad_controller.get_v_y_command(),
                        quad_controller.get_yaw_rate_command()]


        # low-pass filter the velocity commands to reduce jitter
        cmd_arr[[0, 3]] = new_cmd[[0, 3]]
        cmd_arr[[1, 2]] = cmd_alpha * new_cmd[[1, 2]] + (1 - cmd_alpha) * cmd_arr[[1, 2]]
        # print('cmd_array', cmd_arr)

        cv2.putText(processed_frame, '{0:3.1f}'.format(fps), (processed_frame.shape[1] - 80, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Processed Frame', processed_frame)
        t_prev = time.time()
        # print('OVERRIDE',override_flight_mode_command.value)
        if override_flight_mode_command.value > -1:
            # Send the override command and then reset the override value to -1 so that it doesn't send continuously
            quad_controller.send_override_flight_mode_command(int(override_flight_mode_command.value))
            override_flight_mode_command.value = -1

        key = cv2.waitKey(1)
        if key == 27:
            quad_controller.finalize()
            break

if __name__ == '__main__':
    app = Flask(__name__)
    next_frame = None
    data_buffer = [None, None]
    server_to_converter = Queue()

    override_flight_mode_command = Value(ctypes.c_int, -1)
    drone_vx = Value(ctypes.c_float,-1.0)
    drone_vy = Value(ctypes.c_float, -1.0)
    drone_vz = Value(ctypes.c_float, -1.0)
    drone_yaw = Value(ctypes.c_float, -1.0)
    test = 2

    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    @app.route('/', methods=['GET', 'POST'])
    def handle_request():
        """
        Summary line.

        Communicate with the Android app. The function recieves the rgb image from the DJI quadcopter front camera,
        the vecloices along longitudinal, lateral and vertical axis and sends control commands to the Android app.

        """
        cmd_arr = np.frombuffer(shared_cmd_arr.get_obj(), dtype=np.double)
        date_object = datetime.now()
        current_time = date_object.strftime('%H:%M:%S')
        if request.method == 'POST':
            server_to_converter.put({'encoded_data': request.json['encodedData'],
                                     'time': request.json['time']})
            override_command = request.json['overrideFlightMode']
            drone_vx.value = request.json['velocity_x']
            # print('drone_vx', drone_vx.value)
            drone_vy.value = request.json['velocity_y']
            drone_vz.value = request.json['velocity_z']
            drone_yaw.value = request.json['yaw_angle']
            if override_command > -1:
                override_flight_mode_command.value = override_command
            queue_size = server_to_converter.qsize()
            if queue_size > 1:
                print('queue: {0}'.format(queue_size))
            return jsonify(time=current_time)
        else:
            command = {
                'mode': cmd_arr[0],
                'v_x': cmd_arr[1],
                'v_y': cmd_arr[2],
                'yaw_rate': cmd_arr[3]
            }
            # print(command)
            return jsonify(command)


    processes = []
    num_conversion_processes = 6

    img_arr = np.zeros((480, 640, 3), dtype=np.uint8)
    shape_img_arr = img_arr.shape
    img_arr.shape = img_arr.size
    shared_img_arr = mp.Array('B', img_arr)

    cmd_array = np.zeros(4)
    shared_cmd_arr = mp.Array(ctypes.c_double, cmd_array)

    for i in range(num_conversion_processes):
        image_conversion_process = Process(target=convert_images, args=(server_to_converter, shared_img_arr, shape_img_arr))
        image_conversion_process.start()
        processes.append(image_conversion_process)

    ### New Paras
    rgb_images=deque()
    scores_cache=deque()
    sensor_val = None
    frame_prev = deque()

    controller_process = Process(target=run_controller, args=(shared_img_arr, shape_img_arr, shared_cmd_arr,
                                                              override_flight_mode_command, rgb_images, scores_cache,
                                                              frame_prev, sensor_val, drone_vx, drone_vy, drone_vz, drone_yaw))
    controller_process.start()

    app.run(host=url, port=port)
    controller_process.join()
    for i in range(len(processes)):
        processes[i].join()




