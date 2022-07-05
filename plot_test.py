import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

def openFile(filename):
    data = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for rid, row in enumerate(reader):
            if rid == 0:
                column_names = row
            else:
                desired_array = [float(numeric_string) for numeric_string in row]
                data.append(desired_array)
    return data, column_names

def findBaselines(df):
    baselines = []
    for indent_ID in range(int(df['indent_ID'].max())+1):
        sample_vals = df.loc[df['indent_ID'] == indent_ID][1:101]
        bl = sample_vals.mean()
        baselines.append(bl)
    return baselines

def processData(df,bls):
    for i in range(len(df)):
        ind_id = int(df.loc[i].indent_ID)
        df.loc[i] -= bls[ind_id]
    return df

# Open CSV files and create dataframe
d, col_names = openFile('samples/port_1_depth_1.csv')
proc_df = pd.DataFrame(d, columns=col_names)

# Find baselines
#bls = findBaselines(df)

# Process data
#proc_df = processData(df,bls)

# Plot data
proc_df.plot(y="Bx0", kind="line")
plt.show()



def findBaselines_numpy(arr):
    baselines = []
    for indent in arr:
        sample_vals = pd.DataFrame(indent[1:11])
        bl = sample_vals.mean()
        # make the baseline values for indent_ID, x_loc, y_loc, z_loc = 0
        # in order to preserve those data points
        bl[20:24] = [0,0,0,0]
        baselines.append(bl)
    return baselines

def processData_numpy(arr,bls):
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            arr[i][j] -= bls[i]
    return arr

np_train = np.load('raw/port_1_depth_1.npy', allow_pickle=True)
bls = findBaselines_numpy(np_train)
proc_data = processData_numpy(np_train,bls)

np.save('test.npy', proc_data)

upload = np.load('processed/port_1_depth_1.npy', allow_pickle=True)


