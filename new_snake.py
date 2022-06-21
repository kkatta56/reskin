#python test.py -p /dev/ttyACM0

import numpy as np
import argparse
import time
import serial
import csv

from utils.dobot import *
from reskin_sensor import ReSkinProcess

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

def startDataProcess(robot, reskin_sensor, r, filename, path="snake"):
    # Start sensor stream and buffering data
    reskin_sensor.start()
    if reskin_sensor.is_alive():
        reskin_sensor.start_buffering()
        buffer_start = time.time()
    else:
        return "Error: stream has not started"

    # Action during buffer
    if path == "snake":
        robot.startSnakePath(r)
    elif path == "updown":
        robot.setOrigin(r)
        for i in range(3):
            robot.move(8, 8, 0)
            robot.move(8, 8, -8)
        robot.move(8, 8, 0)
    else:
        return "Error: no path specified"

    # Stop buffer
    reskin_sensor.pause_buffering()
    buffer_stop = time.time()

    # Get buffered data
    buffered_data = reskin_sensor.get_buffer()

    # Pause sensor stream
    reskin_sensor.pause_streaming()
    reskin_sensor.join()

    # Print buffered data summary
    if buffered_data is not None:
        print(
            "Time elapsed: {}, Number of datapoints: {}".format(
                buffer_stop - buffer_start, len(buffered_data)
            )
        )

    save_data(buffered_data, reskin_sensor.num_mags, filename)



if __name__ == "__main__":
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

    # Create sensor stream
    sensor_stream = ReSkinProcess(
        num_mags=args.num_mags,
        port=args.port,
        baudrate=args.baudrate,
        burst_mode=True,
        device_id=1,
        temp_filtered=args.temp_filtered,
    )

    # Initialize Dobot
    db = Dobot(port='/dev/ttyUSB0')
    origin = [177.79611206054688, -197.7755126953125, -87.25672912597656]
    startDataProcess(db, sensor_stream, origin, "res_test.csv", path="snake")
