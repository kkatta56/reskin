import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

# Open raw CSV file
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

# Find the baselines for each indentation
def findBaselines(df):
    baselines = []
    for indent_ID in range(int(df['indent_ID'].max())+1):
        sample_vals = df.loc[df['indent_ID'] == indent_ID][1:11]
        bl = sample_vals.mean()
        # make the baseline values for indent_ID, x_loc, y_loc, z_loc = 0
        # in order to preserve those data points
        bl[20:24] = [0,0,0,0]
        baselines.append(bl)
    return baselines

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

# Process the data for each of the baselines
def processData(df,bls):
    for i in range(len(df)):
        ind_id = int(df.loc[i].indent_ID)
        df.loc[i] -= bls[ind_id]
    return df

def sample_process(df,bls):
    sample_df = []
    for indent_ID in range(int(df['indent_ID'].max()) + 1):
        sample_df.append(df.loc[df['indent_ID'] == indent_ID][300:350])
    return pd.concat(sample_df)

def processData_numpy(arr,bls):
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            arr[i][j] -= bls[i]
    return arr

# Take first line as baseline
def oneBaseline(dat):
    bl = dat[0]
    proc_dat = []
    for row in dat:
        proc_arr = [num-bl[i] for i,num in enumerate(row)]
        proc_dat.append(proc_arr)
    return proc_dat

# Input number of depths/sensors
depths = 2
ports = 3

# Run process over all raw data files
for i in range(1,ports+1):
    for j in range(1,depths+1):

        # Open .csv/.npy files and create dataframe
        d, col_names = openFile('raw/port_'+str(i)+'_depth_'+str(j)+'.csv')
        df = pd.DataFrame(d, columns=col_names)
        np_data = np.load('raw/port_'+str(i)+'_depth_'+str(j)+'.npy', allow_pickle=True)

        # Find baselines
        bls = findBaselines(df)
        bls_np = findBaselines_numpy(np_data)

        # Process data
        proc_df = processData(df, bls)
        proc_np = processData_numpy(np_data, bls_np)
        samples_df = sample_process(df, bls)

        # Save data
        proc_df.to_csv('processed/port_' + str(i) + '_depth_' + str(j) + '.csv', encoding='utf-8', index = False)
        samples_df.to_csv('samples/port_' + str(i) + '_depth_' + str(j) + '.csv', encoding='utf-8', index = False)
        np.save('processed/port_' + str(i) + '_depth_' + str(j) + '.npy', proc_np)

        print('Processed raw/port_'+str(i)+'_depth_'+str(j))
