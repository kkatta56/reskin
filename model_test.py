import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.model import *

############ Train with multiple datasets / Test with multiple datasets ##################
#train_urls = ['datasets/normalized/port_1_depth_1.npz',
#              'datasets/normalized/port_1_depth_2.npz',
#              'datasets/normalized/port_2_depth_1.npz',
#              'datasets/normalized/port_2_depth_2.npz',
#              ]
#test_urls = ['datasets/normalized/port_3_depth_1.npz',
#             'datasets/normalized/port_3_depth_2.npz'
#             ]
#train_dataset = combine_datasets(train_urls)
#test_dataset = combine_datasets(test_urls)

######################### Train and test on same datasets #################################
urls = ['datasets/normalized/port_1_depth_1.npz',
        'datasets/normalized/port_1_depth_2.npz',
        'datasets/normalized/port_1_depth_3.npz',
        'datasets/normalized/port_2_depth_1.npz',
        'datasets/normalized/port_2_depth_2.npz',
        'datasets/normalized/port_2_depth_3.npz',
        'datasets/normalized/port_3_depth_1.npz',
        'datasets/normalized/port_3_depth_2.npz',
        'datasets/normalized/port_3_depth_3.npz',
        ]
full_dataset = combine_datasets(urls)
train_dataset, test_dataset = split_dataset(0.9, full_dataset)


batch_size = 10
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)
tolerance = 1
f_tolerance = 0.2
model = train_model(train_loader, plot=True)
#plotPredVsTrue(model, test_loader, 2)
print(test_model(test_loader, model, tolerance, f_tolerance))