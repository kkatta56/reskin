### find ports:
# conda install pyserial
# python -m serial.tools.list_ports

import numpy as np
import argparse
import time
import serial
import csv

from utils.dobot import *
from reskin_sensor import ReSkinProcess
from utils.force_sensor import ForceSensor, _ForceSensorSetting


def save_data_csv(buff_dat, xs, ys, depth, mag_num, file_name):
    col_names = ['T', 'Bx', 'By', 'Bz']
    fields = []
    for i in range(mag_num):
        for colname in col_names:
            fields.append(colname + str(i))
    fields += ['indent_ID', 'X_location', 'Y_location', 'Z_location']

    rows = []
    for indent_id, indent_data in enumerate(buff_dat):
        #for i in range(len(indent_data)-550, len(indent_data)):
        for i in range(len(indent_data)):
            indent_data[i].data.append(indent_id)
            indent_data[i].data.append(xs[indent_id])
            indent_data[i].data.append(ys[indent_id])
            indent_data[i].data.append(depth)
            rows.append(indent_data[i].data)

    with open(file_name, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)

def getSingleIterationData(robot, reskin_sensor, fs, r, depth, filename):
    # Start buffering
    if reskin_sensor.is_alive() == False:
        return "Error: stream has not started"

    # Start snake path
    buffer_start = time.time()
    robot.setOrigin(r)
    x = 0;
    y = 0;
    z = 0;
    xmove = 2;
    ymove = 2;
    buffered_data = []
    x_loc = []
    y_loc = []

    for j in range(9):
        for i in range(9):
            if not ((x < 4 or x > 12) and (y < 4 or y > 12)):
                x_loc.append(x)
                y_loc.append(y)
                reskin_sensor.start_buffering(overwrite=True)
                robot.move([x, y, 0])
                robot.move([x, y, -depth])
                time.sleep(1.5)
                robot.move([x, y, 0])
                reskin_sensor.pause_buffering()
                buffered_data.append(reskin_sensor.get_buffer())
            x += xmove
        xmove *= -1
        x += xmove
        y += ymove

    # Get force sensor data
    ##############buffered_fsdata = fs.get_buffer()

    # FIGURE OUT SAVING FORCE SENSOR DATA
    save_data_csv(buffered_data, x_loc, y_loc, depth, reskin_sensor.num_mags, filename + ".csv")
    print("Iteration saved.")

def getSingleSkinData(port, pid, origin, depths, db, fs):
    # Initialize reskin sensor
    sensor_stream = ReSkinProcess(
        num_mags=5,
        port=port,
        baudrate=115200,
        burst_mode=True,
        device_id=1,
        temp_filtered=False,
    )
    print("Using port: " + port)
    sensor_stream.start()

    # Start data collection at various depths for  a particular reskin sensor
    for i, d in enumerate(depths):
        getSingleIterationData(db, sensor_stream, fs, origin, d,
                          "datasets/raw/port_" + str(pid + 1) + "_depth_" + str(i + 1))

    # Stop sensor stream
    sensor_stream.pause_streaming()
    #################force_sensor.pause_recording()
    sensor_stream.join()


if __name__ == "__main__":

    # Initialize Dobot
    db = Dobot(port='/dev/ttyUSB0')

    # Initializing force sensor
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
    #################force_sensor.start_recording()

    # Set ports, origins, and depths for experiment
    port_names = ['/dev/ttyACM0','/dev/ttyACM1','/dev/ttyACM2']
    origins = [[177.29611206054688, -197.7755126953125, -87.25672912597656],
               [177.29611206054688, -95.56216430664062, -87.25672912597656],
               [177.29611206054688, 77.446216430664062, -87.25672912597656]]
    depths = [8,10]

    # Iterate over each port/origin
    for pid,port in enumerate(port_names):
        getSingleSkinData(port, pid, origins[pid], depths, db, force_sensor)
