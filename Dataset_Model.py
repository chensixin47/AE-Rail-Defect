# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 10:50:17 2019

@author: Autumn
"""

#import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Dataset_class():
    def __init__(self, features, labels, feature_shape):
        self.features = features
        self.labels = labels
        self.feature_shape = feature_shape
    def __len__(self):
        return len(self.labels)
    def __getitem__(self,idx):
        feature = self.features[idx].reshape( self.feature_shape).astype('f')
        feature_item = torch.from_numpy(feature)
        label = self.labels[idx].squeeze().astype('int64')
        label_item = torch.from_numpy(label)
        return feature_item,label_item

class CNN_class(nn.Module):
    def __init__(self, input_size, p, hidden_size):
        super(CNN_class, self).__init__()
        self.input_size = input_size
        self.p = p
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p = self.p, inplace=False)
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, 4)
    def forward(self,x):
        x = x.view(-1,self.input_size)
        x = F.relu(self.dropout(self.fc1(x)))
        x = self.fc2(x)
        return x