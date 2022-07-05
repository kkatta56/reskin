#python test.py -p /dev/ttyACM0

import numpy as np
import argparse
import time
import serial
import csv

from utils.dobot import *
from reskin_sensor import ReSkinProcess

if __name__ == "__main__":

    # Initialize Dobot
    db = Dobot(port='/dev/ttyUSB0')




    # Set ports, origins, and depths for experiment
    port = '/dev/ttyACM0'
    origin = [177.29611206054688, -197.7755126953125, -87.25672912597656]
    d = 8

    # Iterate over each port/origin
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

    # Start buffering
    if sensor_stream.is_alive():
        buffer_start = time.time()
    else:
        print("Error: stream has not started")

    baseline_data = []
    touch_data = []
    # Start robot path
    db.setOrigin(origin)
    x = 0;
    y = 0;
    z = 0;
    xmove = 8;
    ymove = 8;

    for j in range(3):
        for i in range(3):
            if not ((x < 4 or x > 12) and (y < 4 or y > 12)):
                sensor_stream.start_buffering(overwrite=True)
                db.move([x, y, 0])
                db.move([x, y, -d])
                db.move([x, y, 0])
                sensor_stream.pause_buffering()
                touch_data.append(sensor_stream.get_buffer())
            x += xmove
        xmove *= -1
        x += xmove
        y += ymove

    # Stop buffer
    buffer_stop = time.time()

    # Stop sensor stream
    sensor_stream.pause_streaming()
    sensor_stream.join()


def save_data(buff_dat, mag_num, file_name):
    col_names = ['T', 'Bx', 'By', 'Bz']
    fields = []
    for i in range(mag_num):
        for colname in col_names:
            fields.append(colname + str(i))
    fields.append('indent_ID')

    rows = []
    for indent_id, sample in enumerate(buff_dat):
        for dat in sample:
            dat.data.append(indent_id)
            rows.append(dat.data)

    with open(file_name, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)

save_data(touch_data, 5, 'touch_1.csv')
