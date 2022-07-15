import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd

class ResDataSet(Dataset):
    def __init__(self, arr):
        self.b_vals = arr[:, 0:15]
        self.loc = arr[:, [19, 20, 17]]

    def __len__(self):
        return len(self.loc)

    def __getitem__(self, idx):
        b_vals = self.b_vals[idx]
        loc = self.loc[idx]
        return b_vals, loc

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(15, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.Linear(200, 40),
            nn.Linear(40, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 3)
        )

    def forward(self, x):
        return self.layers(x)

def train_model(train_loader):
    # Initialize the MLP
    mlp = MLP()

    # Define the loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    # Run the training loop
    for epoch in range(0, 5):  # 5 epochs at maximum

        # Print epoch
        print(f'Starting epoch {epoch + 1}')

        # Set current loss value
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(train_loader, 0):
            # Get and prepare inputs
            inputs, loc = data
            inputs, loc = inputs.float(), loc.float()
            loc = loc.reshape((loc.shape[0], 3))

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            loc_val = mlp(inputs)

            # Compute loss
            loss = loss_function(loc_val, loc)

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Print statistics
            current_loss += loss.item()
            if i % 10 == 0:
                print('Loss after mini-batch %5d: %.3f' %
                      (i + 1, current_loss / 500))
                current_loss = 0.0

    # Process is complete.
    print('Training process has finished.')
    return mlp

def test_model(test_loader, mlp, tolerance, f_tolerance):
    with torch.no_grad():
        x_correct, y_correct, tot_correct, force_correct, total = 0, 0, 0, 0, 0
        for inputs, loc in test_loader:
            x_loc, y_loc, force = loc[:,0], loc[:,1], loc[:,2]
            inputs, x_loc, y_loc, force = inputs.float(), x_loc.float(), y_loc.float(), force.float()
            pred = mlp(inputs)
            x_pred, y_pred, force_pred = pred[:,0], pred[:,1], pred[:,2]
            x_pred, y_pred, force_pred = np.reshape(x_pred, -1), np.reshape(y_pred, -1), np.reshape(force_pred, -1)
            total += inputs.size(0)
            x_correct += ((x_loc - tolerance <= x_pred) & (x_pred <= x_loc + tolerance)).sum().item()
            y_correct += ((y_loc - tolerance <= y_pred) & (y_pred <= y_loc + tolerance)).sum().item()
            force_correct += ((force - f_tolerance <= force_pred) & (force_pred <= force + f_tolerance)).sum().item()
            tot_correct += ((x_loc - tolerance <= x_pred) & (x_pred <= x_loc + tolerance) & (
                        y_loc - tolerance <= y_pred) & (y_pred <= y_loc + tolerance) &
                            (force - f_tolerance <= force_pred) & (force_pred <= force + f_tolerance)).sum().item()
        print('X value accuracy of the network on the test values: {}%'.format(100 * x_correct / total))
        print('Y value accuracy of the network on the test values: {}%'.format(100 * y_correct / total))
        print('Force value accuracy of the network on the test values: {}%'.format(100 * force_correct / total))
        print('Accuracy of the network on the test values: {}%'.format(100 * tot_correct / total))
        return [100 * x_correct / total, 100 * y_correct / total, 100 * force_correct / total, 100 * tot_correct / total]

def split_dataset(train_proportion, filename):
    full_dataset = ResDataSet(pd.read_csv(filename).to_numpy())
    train_size = int(train_proportion * len(full_dataset))
    test_size = len(full_dataset) - train_size
    return torch.utils.data.random_split(full_dataset, [train_size, test_size])

batch_size = 10

train_dataset = ResDataSet(pd.read_csv('datasets/processed/port_1_depth_1.csv').to_numpy())
test_dataset = ResDataSet(pd.read_csv('datasets/processed/port_2_depth_1.csv').to_numpy())
#train_dataset, test_dataset = split_dataset(0.8,'datasets/processed/port_1_depth_1.csv')

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)
tolerance = 0.5
f_tolerance = 0.1
data = []
model = train_model(train_loader)
results = test_model(test_loader, model, tolerance, f_tolerance)
print(results)
