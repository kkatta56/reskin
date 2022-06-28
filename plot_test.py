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

def processBaselines(df):
    baselines = []
    for indent_ID in range(int(proc_df['indent_ID'].max())+1):
        sample_vals = df.loc[df['indent_ID'] == indent_ID][1:11]
        bl = sample_vals.mean()
        baselines.append(bl)

    #for row in df:


    return baselines


d, col_names = openFile('raw/port_1_depth_1.csv')
proc_df = pd.DataFrame(d, columns=col_names)
bls = processBaselines(proc_df)
#proc_df.plot(y="Bx0", kind="line")
#plt.show()
print(proc_df[1])
