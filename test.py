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

db = Dobot(port='/dev/ttyUSB0')
db.stopQueueExec()
print(db.checkConnection())
