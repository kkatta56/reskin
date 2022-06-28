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

d, col_names = openFile('touch_1.csv')
proc_df = pd.DataFrame(d, columns=col_names)
proc_df.plot(y="Bx0", kind="line")
plt.show()
print(proc_df['Bz0'])

