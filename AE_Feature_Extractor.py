# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:38:40 2019

@author: Autumn
"""

#%%
import librosa as lib
import numpy as np
import network_architectures as netark
import torch.nn.functional as Fx
import torch
from torch.autograd import Variable
#import sys,os
from collections import OrderedDict
import extractor as exm
#import matplotlib.pyplot as plt
#import librosa.display as display

#%%
usegpu = False

#%%
#n_fft = 2048
#hop_length = 1024

#n_fft = 1024
#hop_length = 512

n_mels = 128
trainType = 'weak_mxh64_1024'
pre_model_path = 'mx-h64-1024_0d3-1.17.pkl'
featType = 'layer18' # or layer 19 -  layer19 might not work well
#globalpoolfn = Fx.avg_pool2d # can use max also
netwrkgpl = Fx.avg_pool2d # keep it fixed
srate=60000
#%%
def load_model(netx,modpath):
    #load through cpu -- safest
    state_dict = torch.load(modpath,map_location=lambda storage, loc: storage)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    netx.load_state_dict(new_state_dict)

#%%
def getFeat(extractor,inpt, globalpoolfn):
    # return pytorch tensor 
    extractor.eval()
    with torch.no_grad():
        indata = Variable(torch.Tensor(inpt))
        if usegpu:
            indata = indata.cuda()
    
        pred = extractor(indata)
        print(pred.size())
        if len(pred.size()) > 2:
            gpred = globalpoolfn(pred,kernel_size=pred.size()[2:])
            gpred = gpred.view(gpred.size(0),-1)    
        return gpred

#%%MAIN
#filename = '125523.wav'
def main(audio_data,n_fft, hop_length, globalpoolfn):

#try:
#    y, sr = lib.load(filename, sr=None)
#except:
#    raise IOError('Give me an audio file which I can read!')
#if len(y.shape) > 1:
#    print('Mono Conversion')
#    y = lib.to_mono(y)
#if sr != srate:
#    print('Resampling to {}'.format(srate))
#    y = lib.resample(y,sr,srate)

    mel_feat = lib.feature.melspectrogram(y=audio_data,sr=srate,n_fft=n_fft, hop_length=hop_length)
    inpt = lib.power_to_db(mel_feat)
#    fig1 = plt.figure()
    #display.specshow(inpt,y_axis='mel',fmax=8000,x_axis='s')
#    display.specshow(inpt,y_axis='mel',fmax=8000,x_axis='s')
    inpt = inpt.T
#    plt.colorbar(format='%+2.0f db')
#    plt.title('Mel spectrogram')
#    plt.tight_layout()

    if inpt.shape[0]<128:
        inpt = np.concatenate((inpt,np.zeros((128-inpt.shape[0],n_mels))),axis=0)

    # input needs to be 4D, batch_size X 1 X inpt_size[0] X inpt_size[1]
    inpt = np.reshape(inpt,(1,1,inpt.shape[0],inpt.shape[1]))
    print(inpt.shape)

    netType = getattr(netark,trainType)
    netx = netType(527,netwrkgpl)
    load_model(netx,pre_model_path)

    feat_extractor = exm.featExtractor(netx,featType)
#    feat_extractor2 = feat_extractor#.data.numpy()
    pred = getFeat(feat_extractor,inpt, globalpoolfn)
    feature = pred.data.numpy()
    print(feature.shape)

#    fig2 = plt.figure()
#    plt.plot(feature[0,:])
    return feature#, feat_extractor2