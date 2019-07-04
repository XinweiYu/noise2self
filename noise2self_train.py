#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 07:20:08 2019
This code is used for testing the noise2self code on our medical images.
@author: yxw
"""

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from util import show, plot_images, plot_tensors
from mask import Masker
from models.babyunet import BabyUnet
from torch.nn import MSELoss
from torch.optim import Adam
from PIL import Image


class XrayDataset(Dataset):
    # this class is used to read the x-ray dataset after format transform
    def __init__(self, data_dir, mode='train', tsfm=None):
        #tsfm stands for transform.  
        
        self.data_dir = data_dir
        # load the image file names in the data folder.
        self.img_list = os.listdir(data_dir)
        
        
        #print(len(self.img_list))
        # train or test mode
        self.mode = mode
        # transform is process of the loaded image
        self.transform = tsfm
    
    def __len__(self):
        # for the number of image data.
        return len(self.img_list)
    
    def __getitem__(self, idx):
        # get idx th image.
        img_name = os.path.join(self.data_dir, self.img_list[idx])
        image = Image.open(img_name)
        #image = io.imread(img_name)        
        
        if self.transform:
            image = self.transform(image)
        sample = {'image': image}    
        
        return sample
    
    
if __name__ == "__main__":
    # prepare dataset.
    tsfm = transforms.Compose([transforms.RandomCrop(size=300, pad_if_needed=True),\
                               transforms.ToTensor()])
    
    noisy_mnist_train = XrayDataset('data/LowQ_digest_train', mode='train', tsfm=tsfm)
    
    # initialize mask for J-invariant fuc
    masker = Masker(width = 4, mode='interpolate')
    
    # initialize network
    model = BabyUnet()
    
    # set loss function
    loss_function = MSELoss()
    
    # set optimizer
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # train the model
    data_loader = DataLoader(noisy_mnist_train, batch_size=32, shuffle=True, num_workers=4)
    # set a count number to get different mask idx
    count_batch = 0
    
    # train the network
    num_epoch = 10
    for epoch_idx in range(num_epoch):
        for i, batch in enumerate(data_loader):
            model.train()
            noisy_images = batch['image']
            
            net_input, mask = masker.mask(noisy_images, count_batch)
            net_output = model(net_input)
            # only use the masked pixel to calculate loss.
            loss = loss_function(net_output*mask, noisy_images*mask)
            
            optimizer.zero_grad()
         
            loss.backward()
            
            optimizer.step()
            count_batch += 1
            print(count_batch)
            if i % 1 == 0:
                print("Loss (", i, "): \t", round(loss.item(), 4))
                
            if i == 100:
                break    
            
    torch.cuda.empty_cache()     
    
#    #%% test the model on test image.          
#    noisy_mnist_test = XrayDataset('data/LowQ_digest_test', mode='test', tsfm=tsfm)            
#    test_data_loader = DataLoader(noisy_mnist_train, batch_size=32, shuffle=False, num_workers=4)     
#    i, test_batch = next(enumerate(test_data_loader))
#    noisy = test_batch['image']
#    model.eval()
#    # calculate the denoise result on test set.
#    simple_output = model(noisy)
#    #model.eval()
#    #invariant_output = masker.infer_full_image(noisy, model)
#    idx = 8
#    plot_tensors([noisy[idx], simple_output[idx]],\
#            ["Noisy Image", "Single Pass Inference"])
    
    
    
    
    
    