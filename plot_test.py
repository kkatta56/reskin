import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

def readData(filename):
    # Open NPZ files
    data = np.load(filename, allow_pickle=True)['arr_0']

    # Populate arrays with data
    res_data = []
    force_data = []
    for entry in data:
        res_data.append(entry['ReSkin Data'])
        force_data.append(entry['Force Data'])

    return np.array(res_data), np.array(force_data)



r1, f1 = readData('datasets/normalized/port_1_depth_1.npz')
r2, f2 = readData('datasets/normalized/port_2_depth_1.npz')
r3, f3 = readData('datasets/normalized/port_3_depth_1.npz')

column = 6

plt.plot(r1[:,column], label="Skin 1")
plt.plot(r2[:,column], label="Skin 2")
plt.plot(r3[:,column], label="Skin 3")
plt.legend()

plt.show()

