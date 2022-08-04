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

# Read data
time_string = '08_04_2022_15:10:38'
r1, f1 = readData('datasets/'+time_string+'/normalized/port_1_depth_1.npz')
r2, f2 = readData('datasets/'+time_string+'/normalized/port_2_depth_1.npz')
r3, f3 = readData('datasets/'+time_string+'/normalized/port_3_depth_1.npz')

x1 = StandardScaler().fit_transform(r1)
x2 = StandardScaler().fit_transform(r2)
x3 = StandardScaler().fit_transform(r3)
x = np.concatenate((x1,x2,x3), axis=0)

y = np.array(["skin 1"] * len(x1) + ["skin 2"] * len(x2) + ["skin 3"] * len(x3))

pca = PCA(n_components=1)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1'])
principalDf['Skin Number'] = y

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Time', fontsize = 15)
ax.set_ylabel('Principal Component 1', fontsize = 15)
ax.set_title('1 component PCA', fontsize = 20)

skin1 = principalDf.loc[principalDf['Skin Number'] == 'skin 1'].reset_index()
skin2 = principalDf.loc[principalDf['Skin Number'] == 'skin 2'].reset_index()
skin3 = principalDf.loc[principalDf['Skin Number'] == 'skin 3'].reset_index()

plt.plot(skin1['principal component 1'], label="Skin 1")
plt.plot(skin2['principal component 1'], label="Skin 2")
plt.plot(skin3['principal component 1'], label="Skin 3")
plt.legend()


# Plot data
#column = 10
#plt.plot(r1[:,column], label="Skin 1")
#plt.plot(r2[:,column], label="Skin 2")
#plt.plot(r3[:,column], label="Skin 3")
#plt.title("original setup")
#plt.legend()
plt.show()

