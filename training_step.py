# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 09:51:09 2019

@author: Autumn
"""

import torch

def main(range_, optimizer, criterion, training_batch_num, training_dataset_loader, cnn, training_batch_size, j):
    for epoch in range_:
        running_loss = 0.0
        for i, (feature_items, label_items) in enumerate(training_dataset_loader, 0):
    #        features, labels = data
            optimizer.zero_grad()
            output_items = cnn(feature_items)
            loss = criterion(output_items, label_items)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i == j % training_batch_num:
                print('[Epoch %d, Batch %d]' %(epoch, i))
                print('Loss: %.3f' %(running_loss/training_batch_num))
                _, prediction_items = torch.max(output_items,1)
                print('Accuracy: %.1f%%' %(torch.sum(prediction_items==label_items).numpy()/training_batch_size*100))
                running_loss = 0.0
        j = j + 1
    print('Finished Training')    
    return j