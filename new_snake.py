### find ports:
# conda install pyserial
# python -m serial.tools.list_ports

import numpy as np
import argparse
from datetime import datetime
import serial
import os

from utils.dobot import *
from reskin_sensor import ReSkinProcess
from utils.force_sensor import ForceSensor, _ForceSensorSetting

def save_data_npz(res_bl, res_contact, fs_bl, fs_contact, xs, ys, mag_num, file_name):
    col_names = ['T', 'Bx', 'By', 'Bz']
    fields = []
    for i in range(mag_num):
        for colname in col_names:
            fields.append(colname + str(i))
    fields += ['X_force', 'Y_force', 'Z_force', 'indent_ID', 'X_location', 'Y_location']

    bl_dict, contact_dict = [], []

    # i is the indent_id, and j is the sample_id
    # change to .npz file
    for i in range(len(res_bl)):
        for j in range(len(res_bl[i])):
            reskin_data = res_bl[i][j][2:22]
            force_data = fs_bl[i][0][j]
            location = [xs[i], ys[i]]
            indent = i
            dict = {'ReSkin_Data': reskin_data, "Force_Data": force_data, "Location": location, "Indent": indent}
            bl_dict.append(dict)

    for i in range(len(res_contact)):
        for j in range(len(res_contact[i])):
            reskin_data = res_contact[i][j][2:22]
            force_data = fs_contact[i][0][j]
            location = [xs[i], ys[i]]
            indent = i
            dict = {'ReSkin_Data': reskin_data, "Force_Data": force_data, "Location": location, "Indent": indent}
            contact_dict.append(dict)

    np.savez(file_name, bl_arr = bl_dict, cont_arr = contact_dict)

def getSingleIterationData(robot, reskin_sensor, fs, r, depth, num_samples, filename):
    robot.setOrigin(r)

    # Initialize movement variables
    x, y, z = 0, 0, 0
    xmove, ymove = 4, 4
    x_indents, y_indents = int(16/xmove) + 1, int(16/ymove) + 1

    # Initialize data collection variables
    fs_bl, fs_contact = [], []
    res_bl, res_contact = [], []
    x_loc, y_loc = [], []
    num_samples = num_samples

    # Start snake path
    for j in range(y_indents):
        for i in range(int(x_indents)):
            if not ((x < 4 or x > 12) and (y < 4 or y > 12)):
                # Move robot to correct position and record location
                robot.move([x, y, 0])
                x_loc.append(x)
                y_loc.append(y)

                # Collect baseline data from ReSkin and Force sensors
                res_bl.append(reskin_sensor.get_data(num_samples))
                fs_bl.append(fs.get_data(num_samples))

                # Make indentation with robot and collect contact data
                robot.move([x, y, -depth])
                # time.sleep(0.1)
                res_contact.append(reskin_sensor.get_data(num_samples*10))
                fs_contact.append(fs.get_data(num_samples*10))

                # Finish indentation
                robot.move([x, y, 0])

            x += xmove
        xmove *= -1
        x += xmove
        y += ymove

    save_data_npz(res_bl, res_contact, fs_bl, fs_contact, x_loc, y_loc, reskin_sensor.num_mags, filename)
    print("Iteration saved.")

def getSingleSkinData(port, pid, origin, depths, db, fs, num_samples, directory_path):
    # Initialize ReSkin sensor
    print("Using port: " + port)
    sensor_stream = ReSkinProcess(
        num_mags=5,
        port=port,
        baudrate=115200,
        burst_mode=True,
        device_id=1,
        temp_filtered=True,
        reskin_data_struct=False,
    )
    sensor_stream.start()
    sensor_stream.start_streaming()

    # Start data collection at various depths for a particular ReSkin sensor

    for i, d in enumerate(depths):
        getSingleIterationData(db, sensor_stream, fs, origin, d, num_samples, directory_path +
                               'port_' + str(pid + 1) + '_depth_' + str(i + 1) + '.npz')

    # Stop sensor stream
    sensor_stream.pause_streaming()
    sensor_stream.join()


# Initialize Dobot
db = Dobot(port='/dev/ttyUSB0')

# Initialize force sensor
fs_settings = _ForceSensorSetting(device_name_prefix="Dev",
                                  device_ids=[1],
                                  sensor_names=["FT34108"],
                                  calibration_folder="./utils",
                                  reverse_scaling={1: ["Fz"], 2: ["Fz"]},
                                  # key: device_id, parameter. E.g.:if x & z dimension of sensor 1 and z dimension of sensor 2 has to be flipped use {1: ["Fx", "Fz"], 2: ["Fz"]}
                                  remote_control=False, ask_filename=False, write_Fx=True,
                                  write_Fy=True, write_Fz=True, write_Tx=False, write_Ty=False,
                                  write_Tz=False, write_trigger1=True, write_trigger2=False,
                                  zip_data=True, convert_to_forces=True,
                                  priority='normal')
force_sensor = ForceSensor(fs_settings)
force_sensor.start_recording()

##########################################################################################################
##########################################################################################################
# Set ReSkin ports, origins, and depths for each iteration
port_names = ['/dev/ttyACM0','/dev/ttyACM1','/dev/ttyACM2']
force_sensor_height = 13
# [east-west, south-north, down-up]
origins = [[178.00, -196.75, -87.25 + force_sensor_height],
           [178.00, -95.25, -87.25 + force_sensor_height],
           [178.50, 78.50, -87.25 + force_sensor_height]]
depths = [7]
##########################################################################################################
##########################################################################################################


# Create data directories
time_string = datetime.now().strftime("%m_%d_%Y_%H:%M")
dataset_path = '/home/rbhirang/code/kaushik_reskin/reskin/'
os.mkdir(dataset_path + 'datasets/' + time_string)
directory_path = dataset_path + 'datasets/' + time_string + '/raw/'
os.mkdir(directory_path)
os.mkdir(dataset_path + 'datasets/' + time_string + '/processed/')
os.mkdir(dataset_path + 'datasets/' + time_string + '/normalized/')

# Iterate over each port/origin
for pid,port in enumerate(port_names):
    getSingleSkinData(port, pid, origins[pid], depths, db, force_sensor, 10, directory_path)

force_sensor.pause_recording()