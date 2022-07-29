import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os


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
def findBaselines(raw_data):
    # Initialize variables
    res_bls, force_bls = [], []
    j = 0

    # Iterate through each indent
    for i in range(raw_data[-1]['Indent'] + 1):
        indent_res_data, indent_force_data = [], []

        # Group data of the same indent together
        while i == raw_data[j]['Indent']:
            indent_res_data.append(raw_data[j]['ReSkin_Data'])
            indent_force_data.append(raw_data[j]['Force_Data'])
            j += 1
            if j == len(raw_data):
                break

        # Find the mean data of each indent in order to get baseline
        indent_res_data, indent_force_data = np.array(indent_res_data), np.array(indent_force_data)
        res_bls.append(np.mean(indent_res_data, axis=0))
        force_bls.append(np.mean(indent_force_data, axis=0))

    return res_bls, force_bls

# Process the data for each of the baselines
def processData(raw_data, res_bls, force_bls):
    # Drop temp values
    temp_indices = [0,4,8,12,16]

    # Subtract from baseline
    for entry in raw_data:
        entry['ReSkin Data'] -= res_bls[entry['Indent']]
        entry['ReSkin Data'] = np.delete(entry['ReSkin Data'], temp_indices)
        entry['Force Data'] -= force_bls[entry['Indent']]

    return raw_contact_data

def normalize(data):
    # Initialize empty arrays
    res_data = []
    force_data = []

    # Populate arrays with data
    for entry in data:
        res_data.append(entry['ReSkin Data'])
        force_data.append(entry['Force Data'])
    res_data = np.array(res_data)
    force_data = np.array(force_data)

    # Find the minimums and maximums for each column
    res_mins, res_maxs = res_data.min(axis=0), res_data.max(axis=0)
    force_mins, force_maxs = force_data.min(axis=0), force_data.max(axis=0)

    # Normalize Data
    for entry in data:
        entry['ReSkin Data'] = (2 * (entry['ReSkin Data'] - res_mins) / (res_maxs - res_mins)) - 1
        entry['Force Data'] = (2 * (entry['Force Data'] - force_mins) / (force_maxs - force_mins)) - 1

    return data

# Input number of depths/sensors and date/time
time_string = '07_29_2022_12:39:26'
start_port = 1
end_port = 3
start_depth = 1
end_depth = 3

# Make new directories
dataset_path = '/home/rbhirang/code/kaushik_reskin/reskin/'
os.mkdir(dataset_path + 'datasets/' + time_string + '/processed/')
os.mkdir(dataset_path + 'datasets/' + time_string + '/normalized/')

# Run process over all raw data files
for i in range(start_port, end_port+1):
    for j in range(start_depth, end_depth+1):

        # Open .csv/.npy files and create dataframe
        a = np.load('datasets/'+time_string+'/raw/port_'+str(i)+'_depth_'+str(j)+'.npz', allow_pickle=True)
        raw_baseline_data = a['bl_arr']
        raw_contact_data = a['cont_arr']

        # Find baselines
        res_bls, force_bls = findBaselines(raw_baseline_data)

        # Process and save data
        processed_data = processData(raw_contact_data, res_bls, force_bls)
        np.savez('datasets/'+time_string+'/processed/port_'+str(i)+'_depth_'+str(j)+'.npz', processed_data)


        # Normalize and save data
        normalized_data = normalize(processed_data)
        np.savez('datasets/'+time_string+'/normalized/port_'+str(i)+'_depth_'+str(j)+'.npz', normalized_data)

        print('Processed raw/'+time_string+'/port_'+str(i)+'_depth_'+str(j))
