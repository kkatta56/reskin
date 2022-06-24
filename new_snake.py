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


def save_data(buff_dat, mag_num, file_name):
    col_names = ['T', 'Bx', 'By', 'Bz']
    fields = []
    for i in range(mag_num):
        for colname in col_names:
            fields.append(colname + str(i))

    rows = []
    for sample in buff_dat:
        rows.append(sample.data)

    with open(file_name, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)

def getSingleSkinData(robot, reskin_sensor, fs, r, depth, filename):
    # Start buffering
    if reskin_sensor.is_alive():
        reskin_sensor.start_buffering()
        buffer_start = time.time()
    else:
        return "Error: stream has not started"

    # Start snake path
    robot.startSnakePath(r,depth)

    # Stop buffer
    reskin_sensor.pause_buffering()
    buffer_stop = time.time()

    # Get buffered data
    buffered_data = reskin_sensor.get_buffer()
    ##############buffered_fsdata = fs.get_buffer()

    # Print buffered data summary
    if buffered_data is not None:
        print(
            "Time elapsed: {}, Number of datapoints: {}".format(
                buffer_stop - buffer_start, len(buffered_data)
            )
        )

    # FIGURE OUT SAVING FORCE SENSOR DATA
    save_data(buffered_data, reskin_sensor.num_mags, filename)
    print("Iteration saved.")

def startExperiment(port, pid, origin, depths, db, fs):
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
        getSingleSkinData(db, sensor_stream, fs, origin, d,
                          "port_" + str(pid + 1) + "_depth_" + str(i + 1) + ".csv")

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
    depths = [5,5.5]

    # Iterate over each port/origin
    for pid,port in enumerate(port_names):
        startExperiment(port, pid, origins[pid], depths, db, force_sensor)
