import os
import time 
import struct 

import serial
import numpy as np
from datetime import datetime

# Function defined here just to maintain consistency. Currently going
# to be written in two forms. One as simple easy-to-use numpy arrays. 
# Another as pandas dataframe that I might not write abhi even.
def save_sensor_data(raw_data, path='./data'):
    # if directory == '':
    #     directory = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        
    # save_path = os.path.join(path,directory)
    # if not os.path.isdir(save_path):
    #     os.mkdir(save_path)
    # np.save(os.path.join(save_path,'sensor_data.npy'), raw_data)
    # return save_path
    np_data = np.stack(raw_data)
    print(np_data.shape)
    if not os.path.isdir(path):
        os.mkdir(path)
    np.save(os.path.join(path,'sensor_data.npy'), raw_data)
    return path

class Sensor:
    def __init__(self, port, baud_rate=115200, burst_mode=False, name=''):
        self.port = port
        self.baud_rate = baud_rate
        self.burst_mode = burst_mode
        self.save_dir = './'
        self.timeout = 5.
        self._initialize(port, baud_rate, burst_mode)
        self.data = []
        
        if name == '':
            self.name = 'sensor_'+port
    
    def _initialize(self, port, baud_rate, burst_mode):
        self.sensor = serial.Serial(port, baud_rate, timeout=1)
        self.sensor.flush()
        print("Press reset on the arduino")
        init_start = time.time()
        while True:
            if not burst_mode:
                if(self.sensor.in_waiting):
                    zero_bytes = self.sensor.readline()
                    decoded_zero_bytes = zero_bytes.decode('utf-8')
                    decoded_zero_bytes = decoded_zero_bytes.strip()
                    if decoded_zero_bytes == 'Ready!':
                        print(decoded_zero_bytes)
                        break
                    else:
                        print(decoded_zero_bytes) 
                        break
                else:
                    if time.time() - init_start > self.timeout:
                        return -1
            else:
                if self.sensor.in_waiting>=115:
                    zero_bytes = self.sensor.read(82)
                    decoded_zero_bytes = struct.unpack('@20fcc', zero_bytes)
                    print(' '.join('{:.2f}'.format(x) for x in decoded_zero_bytes[:20]))
                    # print('--------------------------------')
                    # time.sleep(.5)
                    break
                else:
                    if time.time() - init_start > self.timeout:
                        print('Warning! Initialization failed.')
                        return -1
        return 0
        

    # Add comment with the ordering of data here.
    def collect_data(self, num_samples=100):
        # self.sensor.reset_input_buffer()
        if self.sensor.in_waiting > 4000:
            self.sensor.reset_input_buffer()
            while True:
                # print('Had to reset')
                if self.sensor.in_waiting >=115:
                    if self.sensor.read(82)[-1] == 10:
                        break
                # print('sticks here')
                    self.sensor.reset_input_buffer()
        k = 0
        data = []
        collect_start = time.time()
        if not self.burst_mode:
            while k < num_samples:
                if(self.sensor.in_waiting):
                    zero_bytes = self.sensor.readline()
                    decoded_zero_bytes = zero_bytes.decode('utf-8')
                    decoded_zero_bytes = decoded_zero_bytes.strip()
                    # data += [decoded_zero_bytes]
                    new_data = [float(x) for x in decoded_zero_bytes.split()]
                    
                    if len(new_data) == 15:
                        data += [new_data]
                        # print(k, decoded_zero_bytes)
                        k+=1
                    if k%50 == 0:
                        print(k, decoded_zero_bytes)
                else:
                    if time.time() - collect_start > self.timeout:
                        break
        else:
            while k<num_samples:
                if self.sensor.in_waiting >=115:
                    zero_bytes = self.sensor.read(82)
                    # if not zero_bytes[-1] == 10:
                    #     self.sensor.reset_input_buffer()
                    #     print('stuck here')
                    #     continue
                    # print('never gets here')
                    decoded_zero_bytes = struct.unpack('@20fcc', zero_bytes)[:20]
                    
                    data += [decoded_zero_bytes]
                    k+=1

                    # print(self.sensor.in_waiting)
                    # print(k, ' '.join('{:.2f}'.format(x) for x in decoded_zero_bytes))
                    # print('--------------------------------')
                    # time.sleep(.001)
                    # self.sensor.reset_input_buffer()
                else:
                    if time.time() - collect_start > self.timeout:
                        break
        # self.sensor.reset_input_buffer()
        # self.data += data
        return data, self.data

    def flush_data(self):
        self.data = []