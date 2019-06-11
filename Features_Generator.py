# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 16:11:01 2019

@author: Autumn
"""

#%%
#import os
import scipy.io as sio
import numpy as np
import csv
import AE_Feature_Extractor
#import torch.nn.functional as Fx
def main(stage_interest, n_fft, hop_length, globalpoolfn):
    dir_interest = 'C:\\Users\\Autumn\\Documents\\PhD Main Turnout Multi-Class'
    exception_list = []
    with open(dir_interest+'\\STAGE_'+stage_interest+'_useful_name_list.csv', 'r') as f:
        reader = csv.reader(f)
        name_list = list(reader)
    features = np.zeros((len(name_list),1024))
    for ii, item in enumerate(name_list):
        print(ii)
        try:
            audio_data = sio.loadmat(dir_interest+'\\Multi STAGE '+stage_interest+' MAT\\'+item[0])['Data'].squeeze()
        except:
            exception_list.append(str(ii))
        feature = AE_Feature_Extractor.main(audio_data, n_fft, hop_length, globalpoolfn)
        features[ii,:] = feature
#        if ii >= 3:
#            break
    return features, len(name_list), exception_list