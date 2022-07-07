import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


class ResDataSet(Dataset):
    def __init__(self, arr):
        self.b_vals = arr[:, 0:15]
        self.indent_num = arr[:, 15]

    def __len__(self):
        return len(self.indent_num)

    def __getitem__(self, idx):
        b_vals = self.b_vals[idx]
        indent_num = self.indent_num[idx]
        return b_vals, indent_num

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 15
hidden_size = 500
num_classes = 65
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# dataset
train_dataset = ResDataSet(pd.read_csv('datasets/normalized/port_3_depth_1.csv').to_numpy())
test_dataset = ResDataSet(pd.read_csv('datasets/normalized/port_2_depth_1.csv').to_numpy())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (b_vals, indent_num) in enumerate(train_loader):
        # Move tensors to the configured device
        b_vals = b_vals.to(device)
        indent_num = indent_num.type(torch.LongTensor).to(device)

        # Forward pass
        outputs = model(b_vals.float())
        loss = criterion(outputs, indent_num)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for b_vals, indent_num in test_loader:
        b_vals = b_vals.to(device)
        indent_num = indent_num.type(torch.LongTensor).to(device)
        outputs = model(b_vals.float())
        _, predicted = torch.max(outputs.data, 1)
        total += b_vals.size(0)
        correct += (predicted == indent_num).sum().item()

    print('Accuracy of the network on the test values: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')

https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-create-a-neural-network-for-regression-with-pytorch.md
