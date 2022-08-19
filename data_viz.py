import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def readData(filename):
    # Open NPZ files
    data = np.load(filename, allow_pickle=True)['arr_0']

    # Populate arrays with data
    res_data = []
    force_data = []
    for entry in data:
        res_data.append(entry['ReSkin_Data'])
        force_data.append(entry['Force_Data'])

    # Return ReSkin Data and Force Data arrays
    return np.array(res_data), np.array(force_data)

var = 'magnet'
port = 2
data_type = 'normalized'
r_b2s2, f_b2s2 = readData('datasets/'+var+'_base/'+data_type+'/port_'+str(port)+'_depth_1.npz')
r_b2s1, f_b2s1 = readData('datasets/'+var+'_switch_1_and_2/'+data_type+'/port_'+str(port)+'_depth_1.npz')
r_b2s3, f_b2s3 = readData('datasets/'+var+'_switch_2_and_3/'+data_type+'/port_'+str(port)+'_depth_1.npz')

start = 1100
end = 1200

mean_r_b2s2 = np.mean(r_b2s2[start:end], axis=0)
mean_r_b2s1 = np.mean(r_b2s1[start:end], axis=0)
mean_r_b2s3 = np.mean(r_b2s3[start:end], axis=0)

sd_r_b2s2 = np.std(r_b2s2[start:end], axis=0)
sd_r_b2s1 = np.std(r_b2s1[start:end], axis=0)
sd_r_b2s3 = np.std(r_b2s3[start:end], axis=0)

x = range(15)

plt.errorbar(x, mean_r_b2s2, sd_r_b2s2, linestyle='None', marker='o', label="Board 1, Skin 1")
plt.errorbar(x, mean_r_b2s1, sd_r_b2s1, linestyle='None', marker='o', label="Board 1, Skin 2")
plt.errorbar(x, mean_r_b2s3, sd_r_b2s3, linestyle='None', marker='o', label="Board 1, Skin 3")

plt.legend()
plt.show()

