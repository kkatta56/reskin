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
port = 1
data_type = 'normalized'
r1, f1 = readData('datasets/'+var+'_base/'+data_type+'/port_'+str(port)+'_depth_1.npz')
r2, f2 = readData('datasets/'+var+'_switch_1_and_2/'+data_type+'/port_'+str(port)+'_depth_1.npz')
r3, f3 = readData('datasets/'+var+'_switch_1_and_3/'+data_type+'/port_'+str(port)+'_depth_1.npz')
r4, f4 = readData('datasets/'+var+'_switch_2_and_3/'+data_type+'/port_'+str(port)+'_depth_1.npz')


columns = ['Bx0', 'By0', 'Bz0', 'Bx1', 'By1', 'Bz1', 'Bx2', 'By2', 'Bz2',
           'Bx3', 'By3', 'Bz3', 'Bx4', 'By4', 'Bz4']
# Plot data
for column in range(15):
    plt.plot(r2[:, column], label="switch 1 and 2")
    plt.plot(r3[:, column], label="switch 1 and 3")
    plt.plot(r4[:, column], label="switch 2 and 3")
    plt.plot(r1[:, column], label="original")
    plt.title(str(columns[column]))
    plt.legend()
    plt.show()

