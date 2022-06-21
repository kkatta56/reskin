#python test.py -p /dev/ttyACM0

import numpy as np
import argparse
import time
import serial
import numpy as np
import csv

from utils.dobot import *
from reskin_sensor import ReSkinProcess

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

    # Start sensor stream
    sensor_stream.start()
    time.sleep(0.1)

    # Buffer data for two seconds and return buffer
    if sensor_stream.is_alive():
        sensor_stream.start_buffering()
        buffer_start = time.time()

        # Action during buffer
        db.startSnakePath([177.79611206054688, -96.56216430664062, -83.77088928222656])

        # Stop buffer
        sensor_stream.pause_buffering()
        buffer_stop = time.time()

        # Get buffered data
        buffered_data = sensor_stream.get_buffer()

        # Pause sensor stream
        sensor_stream.pause_streaming()
        sensor_stream.join()

        if buffered_data is not None:
            print(
                "Time elapsed: {}, Number of datapoints: {}".format(
                    buffer_stop - buffer_start, len(buffered_data)
                )
            )

        # Save buffered data as csv file
        col_names = ['T','Bx','By','Bz']
        fields = []
        for i in range(sensor_stream.num_mags):
            for colname in col_names:
                fields.append(colname + str(i))

        rows = []
        for sample in buffered_data:
            rows.append(sample.data)

        filename = "reskin_results.csv"
        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)
            csvwriter.writerows(rows)

        # Print buffered data
        #print(
        #    "Columns: ",
        #    ", \t".join(
        #        [
        #            "T{0}, \tBx{0}, \tBy{0}, \tBz{0}".format(ind)
        #            for ind in range(args.num_mags)
        #        ]
        #    ),
        #)
        #for sid, sample in enumerate(buffered_data):
        #    print(
        #        "Sample {}: ".format(sid + 1)
        #        + str(["{:.2f}".format(d) for d in sample.data])
        #    )
