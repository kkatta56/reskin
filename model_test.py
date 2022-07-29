import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.model import *

def train_model(train_loader, plot = False):
    # Initialize the MLP
    mlp = MLP()

    # Define the loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    # Run the training loop
    losses = []
    for epoch in range(0, 5):  # 5 epochs at maximum

        # Print epoch
        print(f'Starting epoch {epoch + 1}')

        # Set current loss value
        current_loss = 0.0
        epoch_losses = []
        # Iterate over the DataLoader for training data
        for i, data in enumerate(train_loader, 0):
            # Get and prepare inputs
            inputs, xyF = data
            inputs, xyF = inputs.float(), xyF.float()
            xyF = xyF.reshape((xyF.shape[0], 3))

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            xyF_model = mlp(inputs)

            # Compute loss
            loss = loss_function(xyF_model, xyF)

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Print statistics
            current_loss += loss.item()
            epoch_losses.append(current_loss)
            if i % 10 == 0:
                print('Loss after batch %5d: %.3f' %
                      (i + 1, current_loss))
                current_loss = 0.0
        losses.append(np.array(epoch_losses).min())

    # Process is complete.
    print('Training process has finished.')

    # Plot Loss vs Epoch
    if plot:
        plt.scatter(range(5),losses)
        plt.show()

    return mlp

def test_model(test_loader, mlp, tolerances, f_tolerances):
    with torch.no_grad():
        results = []
        for tol in tolerances:
            for f_tol in f_tolerances:
                loc_correct, force_correct, tot_correct, total = 0, 0, 0, 0
                for inputs, xyF in test_loader:
                    x_loc, y_loc, force = xyF[:, 0], xyF[:, 1], xyF[:, 2]
                    inputs, x_loc, y_loc, force = inputs.float(), x_loc.float(), y_loc.float(), force.float()
                    pred = mlp(inputs)

                    x_pred, y_pred, force_pred = pred[:, 0], pred[:, 1], pred[:, 2]
                    x_pred, y_pred, force_pred = np.reshape(x_pred, -1), np.reshape(y_pred, -1), np.reshape(force_pred,
                                                                                                            -1)

                    total += inputs.size(0)
                    loc_correct += ((x_loc - tol <= x_pred) &
                                    (x_pred <= x_loc + tol) &
                                    (y_loc - tol <= y_pred) &
                                    (y_pred <= y_loc + tol)).sum().item()
                    force_correct += ((force - f_tol <= force_pred) &
                                      (force_pred <= force + f_tol)).sum().item()
                    tot_correct += ((x_loc - tol <= x_pred) &
                                    (x_pred <= x_loc + tol) &
                                    (y_loc - tol <= y_pred) &
                                    (y_pred <= y_loc + tol) &
                                    (force - f_tol <= force_pred) &
                                    (force_pred <= force + f_tol)).sum().item()
                print('For location tolerance {} and force tolerance {}'.format(tol,f_tol))
                print('Location value accuracy of the network on the test values: {}%'.format(100 * loc_correct / total))
                print('Force value accuracy of the network on the test values: {}%'.format(100 * force_correct / total))
                print('Accuracy of the network on the test values: {}%'.format(100 * tot_correct / total))
                results.append([tol, f_tol, 100 * loc_correct / total, 100 * force_correct / total, 100 * tot_correct / total])
        return results


def readData(filename):
    # Open NPZ files
    data = np.load(filename, allow_pickle=True)['arr_0']

    # Populate arrays with data
    res_data = []
    force_data = []
    for entry in data:
        res_data.append(entry['ReSkin Data'])
        force_data.append(entry['Force Data'])

    # Return ReSkin Data and Force Data arrays
    return np.array(res_data), np.array(force_data)

def split_dataset(train_proportion, full_dataset):
    train_size = int(train_proportion * len(full_dataset))
    test_size = len(full_dataset) - train_size
    return torch.utils.data.random_split(full_dataset, [train_size, test_size])

def combine_datasets(urls):
    rs = []
    fs = []
    for url in urls:
        r, f = readData(url)
        rs.append(r)
        fs.append(f)
    rs = np.concatenate(rs)
    fs = np.concatenate(fs)
    return ResDataSet(rs,fs)

def plotPredVsTrue(model, test_loader, category):
    true_value = []
    pred_value = []
    for inputs, xyF in test_loader:
        for i in range(len(xyF)):
            true_value.append(xyF[i][category].item())
            pred_value.append(model(inputs[i].float())[category].item())
    plt.scatter(true_value, pred_value)
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.show()

time_string = '07_29_2022_12:39:26'

############ Train with multiple datasets / Test with multiple datasets ##################
train_urls = ['datasets/'+time_string+'/normalized/port_2_depth_1.npz',
              'datasets/'+time_string+'/normalized/port_2_depth_2.npz',
              'datasets/'+time_string+'/normalized/port_2_depth_2.npz',
              'datasets/'+time_string+'/normalized/port_3_depth_1.npz',
              'datasets/'+time_string+'/normalized/port_3_depth_2.npz',
              'datasets/'+time_string+'/normalized/port_3_depth_3.npz',
              ]
test_urls = ['datasets/'+time_string+'/normalized/port_1_depth_1.npz',
             'datasets/'+time_string+'/normalized/port_1_depth_2.npz',
             'datasets/'+time_string+'/normalized/port_1_depth_3.npz'
             ]
train_dataset = combine_datasets(train_urls)
test_dataset = combine_datasets(test_urls)

######################### Train and test on same datasets #################################
#urls = ['datasets/'+time_string+'/normalized/port_1_depth_1.npz',
#        'datasets/'+time_string+'/normalized/port_1_depth_2.npz',
#        'datasets/'+time_string+'/normalized/port_1_depth_3.npz',
#        'datasets/'+time_string+'/normalized/port_2_depth_1.npz',
#        'datasets/'+time_string+'/normalized/port_2_depth_2.npz',
#        'datasets/'+time_string+'/normalized/port_2_depth_3.npz',
#        'datasets/'+time_string+'/normalized/port_3_depth_1.npz',
#        'datasets/'+time_string+'/normalized/port_3_depth_2.npz',
#        'datasets/'+time_string+'/normalized/port_3_depth_3.npz',
#        ]
#full_dataset = combine_datasets(urls)
#train_dataset, test_dataset = split_dataset(0.9, full_dataset)


batch_size = 10
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)
tolerances = [2, 1, 0.5]
f_tolerances = [0.5, 0.2]
model = train_model(train_loader)
#plotPredVsTrue(model, test_loader, 2)
print(np.array(test_model(test_loader, model, tolerances, f_tolerances)))