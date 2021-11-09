#!/usr/bin/python
# A simple data loader that imports the train and test mat files
# from the `filename` and converts them to torch.tesnors()
# to be loaded for training and testing DLoc network
# `features_wo_offset`: targets for the consistency decoder
# `features_w_offset` : inputs for the network/encoder
# `labels_gaussian_2d`: targets for the location decoder
import torch
import h5py
import scipy.io
import numpy as np

def load_data(filename):
    print('Loading '+filename)
    arrays = {}
    f = h5py.File(filename,'r')
    features_wo_offset = torch.tensor(np.transpose(np.array(f.get('features_wo_offset'), dtype=np.float32)), dtype=torch.float32)
    features_w_offset = torch.tensor(np.transpose(np.array(f.get('features_w_offset'), dtype=np.float32)), dtype=torch.float32)
    labels_gaussian_2d = torch.tensor(np.transpose(np.array(f.get('labels_gaussian_2d'), dtype=np.float32)), dtype=torch.float32)
        
    return features_wo_offset,features_w_offset, labels_gaussian_2d