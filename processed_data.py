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
        d, col_names = openFile('raw/port_'+str(i)+'_depth_'+str(j)+'.csv')
        proc_df = pd.DataFrame(oneBaseline(d), columns=col_names)
        proc_df.to_csv('processed/oneBL/port_' + str(i) + '_depth_' + str(j) + '.csv', encoding='utf-8')
