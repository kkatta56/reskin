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


a = np.load("datasets/raw/port_1_depth_1.npz", allow_pickle=True)

res_bls = []
force_bls = []
#for i in range(len(a['bl_arr'])):


print(a['bl_arr'][0])