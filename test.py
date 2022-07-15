#python test.py -p /dev/ttyACM0

import numpy as np
import argparse
import time
import serial
import csv

from utils.dobot import *
from reskin_sensor import ReSkinProcess
from utils.force_sensor import ForceSensor, _ForceSensorSetting
from utils.libraries.forceDAQ.force import *


# Initialize Dobot
db = Dobot(port='/dev/ttyUSB0')

# Set ports, origins, and depths for experiment
force_sensor_height = 13
origin = [177.29611206054688, -197.7755126953125, -87.25672912597656+force_sensor_height]
d = 8
port = '/dev/ttyACM0'

rs = ReSkinProcess(
    num_mags=5,
    port=port,
    baudrate=115200,
    burst_mode=True,
    device_id=1,
    temp_filtered=False,
)

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

fs = ForceSensor(fs_settings)
rs.start()
rs.start_streaming()
fs.start_recording()


fs_bl, fs_contact = [], []
res_bl, res_contact = [], []

db.setOrigin(origin)

db.move([0, 0, 0])
res_bl.append(rs.get_data(3))
fs_bl.append(fs.get_data(3))
db.move([8, 0, 0])
db.move([8, 0,-8])
res_contact.append(rs.get_data(3))
fs_contact.append(fs.get_data(3))
db.move([8, 0, 0])
db.move([8, 16, 0])
db.move([8, 16, -8])
db.move([8, 16, 0])
db.move([16, 8, 0])
db.move([16, 8, -8])
db.move([16, 8, 0])
db.move([0, 8, 0])
db.move([0, 8, -8])
db.move([0, 8, 0])



