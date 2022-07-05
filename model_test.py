# -*- coding: utf-8 -*-
import os
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import csv
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

#######################################################
#               Define Reskin Data Class
#######################################################

class ResDataSet(Dataset):
    def __init__(self, arr):
        self.b_vals = arr[:, 0:20]
        self.labels = arr[:, 20]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        b_vals = self.b_vals[idx]
        labels = self.labels[idx]
        return b_vals, labels

class ResDataSet_np(Dataset):
    def __init__(self, arr):
        self.b_vals = []
        self.labels = []
        for indent_id, indent in enumerate(arr):
            a = []
            for data_point in indent:
                a.append(data_point[0:20])
            self.b_vals.append(a)
            self.labels.append(indent_id)
        self.b_vals = np.array(self.b_vals)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        b_vals = self.b_vals[idx]
        labels = self.labels[idx]
        return b_vals, labels

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(20, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 65),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# Upload CSV data as ResDataSet object
# reading two csv files
data1 = pd.read_csv('processed/port_1_depth_1.csv')
data2 = pd.read_csv('processed/port_2_depth_1.csv')

# using merge function by setting how='inner'
output1 = pd.merge(data1, data2,
                   on='T0',
                   how='inner')

#res_dat_train = ResDataSet(output1.to_numpy())
res_dat_train = ResDataSet(pd.read_csv('samples/port_1_depth_1.csv').to_numpy())
res_dat_test = ResDataSet(pd.read_csv('samples/port_3_depth_1.csv').to_numpy())
upload_train = ResDataSet_np(np.load('processed/port_1_depth_1.npy', allow_pickle=True))
upload_test = ResDataSet_np(np.load('processed/port_3_depth_1.npy', allow_pickle=True))

# Initialize data loaders
batch_size = 50
train_dataloader = DataLoader(res_dat_train, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(res_dat_test, batch_size=batch_size, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

#for batch, (X, y) in enumerate(train_dataloader):
#    y = y.type(torch.LongTensor)
#    X, y = X.to(device), y.to(device)
#    pred = model(X.float())
#    loss = loss_fn(pred, y)
#    print(loss)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        y = y.type(torch.LongTensor)
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X.float())
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            y = y.type(torch.LongTensor)
            X, y = X.to(device), y.to(device)
            pred = model(X.float())
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

#torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
