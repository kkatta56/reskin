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
j = 0

for i in range(a['bl_arr'][-1]['Indent']+1):
    indent_res_data = []
    indent_force_data = []
    while i == a['bl_arr'][j]['Indent']:
        indent_res_data.append(a['bl_arr'][j]['ReSkin Data'])
        indent_force_data.append(a['bl_arr'][j]['Force Data'])
        j += 1
        if j == len(a['bl_arr']):
            break
    indent_res_data, indent_force_data = np.array(indent_res_data), np.array(indent_force_data)
    res_bls.append(np.mean(indent_res_data, axis=0))
    force_bls.append(np.mean(indent_force_data, axis=0))

raw_contact_data = a['cont_arr']
temp_indices = [0,4,8,12,16]
for entry in raw_contact_data:
    entry['ReSkin Data'] -= res_bls[entry['Indent']]
    entry['ReSkin Data'] = np.delete(entry['ReSkin Data'], temp_indices)
    entry['Force Data'] -= force_bls[entry['Indent']]

processed_data = raw_contact_data

res_data = []
force_data = []
for entry in processed_data:
    res_data.append(entry['ReSkin Data'])
    force_data.append(entry['Force Data'])
res_data = np.array(res_data)
force_data = np.array(force_data)

res_mins, res_maxs = res_data.min(axis=0), res_data.max(axis=0)
force_mins, force_maxs = force_data.min(axis=0), force_data.max(axis=0)

for entry in processed_data:
    entry['ReSkin Data'] = ( 2 * (entry['ReSkin Data'] - res_mins) / (res_maxs - res_mins) ) - 1
    entry['Force Data'] = ( 2 * (entry['Force Data'] - force_mins) / (force_maxs - force_mins) ) - 1

print(processed_data)
