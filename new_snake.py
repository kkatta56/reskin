#python test.py -p /dev/ttyACM0

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

def startSnakeProcess(robot, reskin_sensor, fs, r, depth, filename):
    # Start buffering
    if reskin_sensor.is_alive():
        reskin_sensor.start_buffering()
        buffer_start = time.time()
    else:
        return "Error: stream has not started"

    # Start snake path
    print("Snake pattern started for iteration.")
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

if __name__ == "__main__":
    # Parse terminal command for arguments
    parser = argparse.ArgumentParser(
        description="Test code to run a ReSkin streaming process in the background. Allows data to be collected without code blocking"
    )
    # fmt: off
    parser.add_argument("-p", "--port", type=str, help="port to which the microcontroller is connected", required=True,)
    parser.add_argument("-b", "--baudrate", type=str, help="baudrate at which the microcontroller is streaming data", default=115200,)
    parser.add_argument("-n", "--num_mags", type=int, help="number of magnetometers on the sensor board", default=5,)
    parser.add_argument("-tf", "--temp_filtered", action="store_true", help="flag to filter temperature from sensor output",)
    # fmt: on
    args = parser.parse_args()

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

    # Create and start sensor stream
    sensor_stream = ReSkinProcess(
        num_mags=args.num_mags,
        port=args.port,
        baudrate=args.baudrate,
        burst_mode=True,
        device_id=1,
        temp_filtered=args.temp_filtered,
    )
    sensor_stream.start()
    #################force_sensor.start_recording()

    # Initialize Dobot
    db = Dobot(port='/dev/ttyUSB0')
    origin = [177.29611206054688, -95.56216430664062, -87.25672912597656]

    # Start data collection
    depths = np.arange(5,6,0.2)
    for i,depth in enumerate(depths):
        startSnakeProcess(db, sensor_stream, force_sensor, origin, depth, "res_test_"+str(i)+".csv")
    print("Finished data collection")

    # Stop sensor stream
    sensor_stream.pause_streaming()
    #################force_sensor.pause_recording()
    sensor_stream.join()
