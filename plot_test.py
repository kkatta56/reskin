import numpy as np
import matplotlib.pyplot as plt
import csv

x = []
y = []
z = []

with open('res_test.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        x.append(row['Bx0'])
        y.append(row['By0'])
        z.append(row['Bz0'])

plt.plot(x)
plt.show()

