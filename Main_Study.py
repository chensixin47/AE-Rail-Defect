# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 20:59:12 2019

@author: Autumn
"""

#%%
#import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
from Dataset_Model import Dataset_class, CNN_class
#import focalloss
import training_step
import math
#from imblearn.over_sampling import SMOTE
training_size = 2880
#USE_SMOTE = False

#%%
with np.load('dataset.npz') as dataset:
    features = dataset['features']
    labels = dataset['labels']

#%%
np.random.seed(42)
permutation = list(np.random.permutation(len(features)))

#%%
training_permutation = permutation[0:training_size]
testing_permutation = permutation[training_size:5536]
#seperate training and testing
training_features = features[training_permutation,:]
training_labels = labels[training_permutation,:]
print(training_features.shape)
print(training_labels.shape)

#%%
#smote = SMOTE('minority')
#training_features_sm, training_labels_sm = smote.fit_sample(training_features, training_labels)
#print(training_features_sm.shape)
#print(training_labels_sm.shape)
#training_size_sm = len(training_labels_sm)

#%%
#if USE_SMOTE == True:
#    training_features = training_features_sm
#    training_labels = training_labels_sm.reshape(-1,1)
#    training_size = training_size_sm

#%%
training_batch_size = 32
training_batch_num= math.ceil(training_size/training_batch_size)
print(training_batch_num)

#%%
training_dataset = Dataset_class(features = training_features, labels = training_labels, feature_shape = (1,1024))
training_dataset_loader = DataLoader(training_dataset, batch_size=training_batch_size, shuffle=False)
cnn = CNN_class(input_size = 1024, p = 0.5, hidden_size = 64)
print(cnn)

#%%
criterion0 = nn.CrossEntropyLoss()
#criterion1 = focalloss.FocalLoss(gamma=1)
optimizer0 = optim.Adam(cnn.parameters(),lr=0.001)
optimizer1 = optim.Adam(cnn.parameters(),lr=0.0001)
j = 0
j = training_step.main(range_=range(training_batch_num), optimizer=optimizer0, criterion=criterion0, training_batch_num=training_batch_num, training_dataset_loader=training_dataset_loader, cnn=cnn, training_batch_size=training_batch_size, j=j)
training_step.main(range_=range(training_batch_num, 2*training_batch_num), optimizer=optimizer1, criterion=criterion0, training_batch_num=training_batch_num, training_dataset_loader=training_dataset_loader, cnn=cnn, training_batch_size=training_batch_size, j=j)

#%%
testing_batch_size = 16
#testing_size = 5536 - training_size
#seperate training and testing
testing_features = features[testing_permutation,:]
testing_labels = labels[testing_permutation,:]
print(testing_features.shape)
print(testing_labels.shape)
testing_dataset = Dataset_class(features = testing_features, labels = testing_labels, feature_shape = (1,1024))
testing_dataset_loader = DataLoader(testing_dataset, batch_size=testing_batch_size, shuffle=False)

#%%
class_correct = list(0 for i in range(4))
class_total = list(0 for i in range(4))
nb_classes = 4
confusion_matrix = torch.zeros(nb_classes, nb_classes)

with torch.no_grad():
    for (feature_items, label_items) in testing_dataset_loader:
#        print(label_items.shape)
        cnn.eval()
        output_items = cnn(feature_items)
        _, prediction_items = torch.max(output_items, 1)#Keep the batch dimension, reduce the probability dimension
        c = (prediction_items == label_items)
        for i in range(testing_batch_size):
            label_item = label_items[i]
#            print(label_item.numpy())
            class_correct[label_item] += c[i].item()
            class_total[label_item] += 1
        for label, prediction in zip(label_items.view(-1), prediction_items.view(-1)):
            confusion_matrix[label.long(), prediction.long()] += 1

classes = ('S0','S1','S2','S3')
for i in range(4):
    print('Accuracy of %1s: %.3f %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
print(confusion_matrix.numpy().astype('int'))

#%%
#print("Model's state_dict:")
#for param_tensor in cnn.state_dict():
#    print(param_tensor, "\t", cnn.state_dict()[param_tensor].size())
#torch.save(cnn.state_dict(),'./Model_v3_70%.pth')