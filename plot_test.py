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

# Open CSV files and create dataframe
d, col_names = openFile('datasets/normalized/port_1_depth_1.csv')
norm_df = pd.DataFrame(d, columns=col_names)
d, col_names = openFile('datasets/normalized/port_1_depth_2.csv')
norm_df2 = pd.DataFrame(d, columns=col_names)
d, col_names = openFile('datasets/normalized/port_1_depth_3.csv')
norm_df3 = pd.DataFrame(d, columns=col_names)


# Plot data
dir = "x"
mag_id = "2"

plot_val = "B"+dir+mag_id
#plot_val = "Z_force"

ax = norm_df[0:400].plot(y=plot_val, kind="line")
norm_df2[0:400].plot(y=plot_val, kind="line", ax=ax)
norm_df3[0:400].plot(y=plot_val, kind="line", ax=ax)



plt.show()
