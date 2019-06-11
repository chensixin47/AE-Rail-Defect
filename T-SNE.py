# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 20:51:15 2019

@author: Autumn
"""

#%%
#import scipy.io as sio
import numpy as np
#import torch
#from torch.utils.data import DataLoader
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
#from Dataset_Model import Dataset_class, CNN_class

#%%
with np.load('dataset.npz') as dataset:
    features = dataset['features']
    labels = dataset['labels']

#%%
#import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets

#digits = datasets.load_digits(n_class=6)
X, y = features,labels
n_samples, n_features = X.shape

#'''显示原始数据'''
#n = 20  # 每行20个数字，每列20个数字
#img = np.zeros((10 * n, 10 * n))
#for i in range(n):
#    ix = 10 * i + 1
#    for j in range(n):
#        iy = 10 * j + 1
#        img[ix:ix + 8, iy:iy + 8] = X[i * n + j].reshape((8, 8))
#plt.figure(figsize=(8, 8))
#plt.imshow(img, cmap=plt.cm.binary)
#plt.xticks([])
#plt.yticks([])
#plt.show()
#%%
#'''t-SNE'''
tsne = manifold.TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
X_tsne = tsne.fit_transform(X)
#%%
#Scatter 2D
print("Org data dimension is {}. \n Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

#'''嵌入空间可视化'''
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
plt.figure(figsize=(16, 16))
for i in range(X_norm.shape[0]):
    plt.text(X_norm[i, 0], X_norm[i, 1], str(int(y[i])), color=plt.cm.Set1(int(y[i])), fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.savefig('Feature_Visualization_v4.jpg')
plt.show()
#%%
#Scatter 3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
fig = plt.figure(figsize=(16, 16))
ax = Axes3D(fig)
x_0 = X_norm[:3524,0]
y_0 = X_norm[:3524,1]
z_0 = X_norm[:3524,2]
ax.scatter(x_0, y_0, z_0, color = 'black')
x_1 = X_norm[3524:4470,0]
y_1 = X_norm[3524:4470,1]
z_1 = X_norm[3524:4470,2]
ax.scatter(x_1, y_1, z_1, color = 'green')
x_2 = X_norm[4470:4952,0]
y_2 = X_norm[4470:4952,1]
z_2 = X_norm[4470:4952,2]
ax.scatter(x_2, y_2, z_2, color = 'blue')
x_3 = X_norm[4952:,0]
y_3 = X_norm[4952:,1]
z_3 = X_norm[4952:,2]
ax.scatter(x_3, y_3, z_3, color = 'red')
plt.show()
#%%
#Different View of 3D
plt.figure(figsize=(16, 16))
for i in range(X_norm.shape[0]):
    plt.text(X_norm[i, 0], X_norm[i, 2], str(int(y[i])), color=plt.cm.Set1(int(y[i])), fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.show()