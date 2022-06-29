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
        sample_vals = df.loc[df['indent_ID'] == indent_ID][1:11]
        bl = sample_vals.mean()
        baselines.append(bl)
    return baselines

def processData(df,bls):
    for i in range(len(df)):
        ind_id = int(df.loc[i].indent_ID)
        df.loc[i] -= bls[ind_id]
    return df

# Open CSV files and create dataframe
d, col_names = openFile('raw/port_1_depth_1.csv')
df = pd.DataFrame(d, columns=col_names)

# Find baselines
bls = findBaselines(df)

# Process data
proc_df = processData(df,bls)

# Plot data
proc_df.plot(y="Bz0", kind="line")
plt.show()
