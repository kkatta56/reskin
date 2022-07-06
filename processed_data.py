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
        sample_vals = df.loc[df['indent_ID'] == indent_ID][0:50]
        bl = sample_vals.mean()
        # make the baseline values for indent_ID, x_loc, y_loc, z_loc = 0
        # in order to preserve those data points
        bl[20:24] = [0,0,0,0]
        baselines.append(bl)
    return baselines

# Process the data for each of the baselines
def processData(df,bls):
    #subtract from baseline
    for i in range(len(df)):
        ind_id = int(df.loc[i].indent_ID)
        df.loc[i] -= bls[ind_id]

    final_df = []
    for indent_ID in range(int(df['indent_ID'].max()) + 1):
        final_df.append(df.loc[df['indent_ID'] == indent_ID][-1100:])

    return pd.concat(final_df)

def sample_process(df):
    sample_df = []
    for indent_ID in range(int(df['indent_ID'].max()) + 1):
        sample_df.append(df.loc[df['indent_ID'] == indent_ID][400:800])
    return pd.concat(sample_df)

def normalize(df):
    df.iloc[:, 0:20] = (df.iloc[:, 0:20] - df.iloc[:, 0:20].min()) / (df.iloc[:, 0:20].max() - df.iloc[:, 0:20].min())
    df.drop('T0', axis=1, inplace=True)
    df.drop('T1', axis=1, inplace=True)
    df.drop('T2', axis=1, inplace=True)
    df.drop('T3', axis=1, inplace=True)
    df.drop('T4', axis=1, inplace=True)
    return df

# Input number of depths/sensors
depths = 2
ports = 3

# Run process over all raw data files
for i in range(1,ports+1):
    for j in range(1,depths+1):

        # Open .csv/.npy files and create dataframe
        d, col_names = openFile('datasets/raw/port_'+str(i)+'_depth_'+str(j)+'.csv')
        df = pd.DataFrame(d, columns=col_names)

        # Find baselines
        bls = findBaselines(df)

        # Process data
        proc_df = processData(df, bls)
        samples_df = sample_process(proc_df)
        norm_df = normalize(samples_df)

        # Save data
        proc_df.to_csv('datasets/processed/port_' + str(i) + '_depth_' + str(j) + '.csv', encoding='utf-8', index = False)
        samples_df.to_csv('datasets/samples/port_' + str(i) + '_depth_' + str(j) + '.csv', encoding='utf-8', index = False)
        norm_df.to_csv('datasets/normalized/port_' + str(i) + '_depth_' + str(j) + '.csv', encoding='utf-8',
                          index=False)

        print('Processed raw/port_'+str(i)+'_depth_'+str(j))
