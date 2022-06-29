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
        baselines.append(bl)
    return baselines

# Process the data for each of the baselines
def processData(df,bls):
    for i in range(len(df)):
        ind_id = int(df.loc[i].indent_ID)
        df.loc[i] -= bls[ind_id]
    return df

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

        # Open CSV files and create dataframe
        d, col_names = openFile('raw/port_'+str(i)+'_depth_'+str(j)+'.csv')
        df = pd.DataFrame(d, columns=col_names)

        # Find baselines
        bls = findBaselines(df)

        # Process data
        proc_df = processData(df, bls)

        # Save data
        proc_df.to_csv('processed/port_' + str(i) + '_depth_' + str(j) + '.csv', encoding='utf-8')
