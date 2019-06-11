# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 20:39:49 2019

@author: Autumn
"""

import Features_Generator
import numpy as np
import torch.nn.functional as Fx
features_0, length0, exception_list0 = Features_Generator.main('0', n_fft = 2048, hop_length = 1024, globalpoolfn = Fx.avg_pool2d)
features_1, length1, exception_list1 = Features_Generator.main('1', n_fft = 2048, hop_length = 1024, globalpoolfn = Fx.avg_pool2d)
features_2, length2, exception_list2 = Features_Generator.main('2', n_fft = 2048, hop_length = 1024, globalpoolfn = Fx.avg_pool2d)
features_3, length3, exception_list3 = Features_Generator.main('3', n_fft = 2048, hop_length = 1024, globalpoolfn = Fx.avg_pool2d)
features = np.concatenate((features_0,features_1,features_2,features_3))

stage0_labels = np.zeros((length0,1))
stage1_labels = np.ones((length1,1))
stage2_labels = 2*np.ones((length2,1))
stage3_labels = 3*np.ones((length3,1))
labels = np.concatenate((stage0_labels,stage1_labels,stage2_labels,stage3_labels))

np.savez('dataset_2048',features = features, labels = labels)