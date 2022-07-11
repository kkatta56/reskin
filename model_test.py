import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd

class ResDataSet(Dataset):
    def __init__(self, arr):
        self.b_vals = arr[:, 0:15]
        self.x_loc = arr[:, 16]
        #self.y_loc = arr[:, 17]

    def __len__(self):
        return len(self.x_loc)

    def __getitem__(self, idx):
        b_vals = self.b_vals[idx]
        x_loc = self.x_loc[idx]
        #y_loc = self.y_loc[idx]
        return b_vals, x_loc

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(15, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 1)
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == '__main__':

    batch_size = 50

    full_dataset = ResDataSet(pd.read_csv('datasets/normalized/port_2_depth_1.csv').to_numpy())
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    #train_dataset = ResDataSet(pd.read_csv('datasets/normalized/port_2_depth_1.csv').to_numpy())
    #test_dataset = ResDataSet(pd.read_csv('datasets/normalized/port_3_depth_1.csv').to_numpy())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

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
            inputs, x_loc = data
            inputs, x_loc = inputs.float(), x_loc.float()
            x_loc = x_loc.reshape((x_loc.shape[0], 1))

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = mlp(inputs)

            # Compute loss
            loss = loss_function(outputs, x_loc)

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

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, x_loc in test_loader:
        inputs, x_loc = inputs.float(), x_loc.float()
        predicted = mlp(inputs)
        predicted = np.reshape(predicted, -1)
        print(predicted, x_loc)
        total += inputs.size(0)
        tolerance = 0.5
        correct += ((x_loc-tolerance <= predicted) & (predicted <= x_loc+tolerance)).sum().item()
    print('Accuracy of the network on the test values: {} %'.format(100 * correct / total))
