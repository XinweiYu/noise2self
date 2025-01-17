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
from PIL import Image, ImageOps
from util_data import tensor_to_image
import time


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
        if self.mode == 'train':
            # hard coded 2^n padding. not a good way.
            image = pad_to_target(image, 256, 256, label=0)
        else:
            image = pad_to_target(image, 1024, 1024, label=0)
        #print(image.size)
        if self.transform:
            image = self.transform(image)
        sample = {'image': image}    
        
        return sample
    
def pad_to_target(img, target_height, target_width, label=0):
    # Pad image with zeros to the specified height and width if needed
    # This op does nothing if the image already has size bigger than target_height and target_width.
    w, h = img.size
    left = top = right = bottom = 0
    doit = False
    if target_width > w:
        delta = target_width - w
        left = delta // 2
        right = delta - left
        doit = True
    if target_height > h:
        delta = target_height - h
        top = delta // 2
        bottom = delta - top
        doit = True
    #print(img.size)
    if doit:
        img = ImageOps.expand(img, border=(left, top, right, bottom), fill=label)
    #print(img.size)
    assert img.size[0] >= target_width
    assert img.size[1] >= target_height
    return img 


if __name__ == "__main__":
    
    # set up parameter
    use_gpu = True
    
    
    
    # prepare dataset.
    tsfm = transforms.Compose([transforms.RandomCrop(size=256, pad_if_needed=True),\
                               transforms.ToTensor()])
#    tsfm = transforms.Compose([transforms.ToTensor()])
    noisy_mnist_train = XrayDataset('data/LowQ_digest_train', mode='train', tsfm=tsfm)
    
    # initialize mask for J-invariant fuc
    masker = Masker(width = 4, mode='interpolate')
    
    # initialize network
    model = BabyUnet()
    if use_gpu:
        model.cuda()
    
    # set loss function
    loss_function = MSELoss()
    
    # set optimizer
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # train the model
    data_loader = DataLoader(noisy_mnist_train, batch_size=32, shuffle=False, num_workers=4)
    # set a count number to get different mask idx
    count_batch = 0
    
    # train the network
    num_epoch = 51
    
                
    # load a single image for test
    tsfm = transforms.ToTensor()
    noisy_mnist_test_in_sample = XrayDataset('data/LowQ_digest_train', mode='test', tsfm=tsfm)
    noisy_mnist_test_out_sample = XrayDataset('data/LowQ_digest_test', mode='test', tsfm=tsfm)
    test_data_loader_in_sample = DataLoader(noisy_mnist_test_in_sample, batch_size=1, shuffle=False, num_workers=4) 
    test_data_loader_out_sample = DataLoader(noisy_mnist_test_out_sample, batch_size=1, shuffle=False, num_workers=4)    
    i, test_batch_in_sample = next(enumerate(test_data_loader_in_sample))
    j, test_batch_out_sample = next(enumerate(test_data_loader_out_sample))
    noisy_in_sample = test_batch_in_sample['image']
    noisy_out_sample = test_batch_out_sample['image']
    if use_gpu:
        noisy_in_sample = noisy_in_sample.cuda()
        noisy_out_sample = noisy_out_sample.cuda()
   
    tic = time.time()
    for epoch_idx in range(num_epoch):
        for i, batch in enumerate(data_loader):
            model.train()
            noisy_images = batch['image']                       
            net_input, mask = masker.mask(noisy_images, count_batch)
            if use_gpu:
                noisy_images = noisy_images.cuda()
                net_input = net_input.cuda()
                mask = mask.cuda()
            net_output = model(net_input)
            # only use the masked pixel to calculate loss.
            loss = loss_function(net_output*mask, noisy_images*mask)
            
            optimizer.zero_grad()
         
            loss.backward()
            
            optimizer.step()
            count_batch += 1
            print('number of batch:',count_batch)
            if i % 1 == 0:
                print("Loss (", i, "): \t", round(loss.item(), 8))
               

            if i == 100:
                break    
        
        if epoch_idx % 5 == 0:
            model.eval()
            # calculate the denoise result on test set.
            simple_output = model(noisy_in_sample)
            
            plot_tensors([noisy_in_sample[0], simple_output[0]],["Noisy Image", "Single Pass Inference"], plot=False,\
                         save=True, img_dir='babyUnet_denoise_in_sample/', img_name='Epoch_'+str(epoch_idx)) 
            simple_output = model(noisy_out_sample)
            plot_tensors([noisy_out_sample[0], simple_output[0]],["Noisy Image", "Single Pass Inference"], plot=False,\
                         save=True, img_dir='babyUnet_denoise_out_sample/', img_name='Epoch_'+str(epoch_idx))           
    
    
    # save the model
    torch.save(model.state_dict(),  'BabyUnet_denoise.pth')     
    torch.cuda.empty_cache()   
    toc = time.time()
    print('Run Time:{}s'.format(toc-tic))    
    #%% test the model on test image.          
    #noisy_mnist_test = XrayDataset('data/LowQ_digest_test', mode='test', tsfm=tsfm)            
#    test_data_loader = DataLoader(noisy_mnist_test, batch_size=32, shuffle=False, num_workers=4)     
#    i, test_batch = next(enumerate(test_data_loader))
#    noisy = test_batch['image']
#    model.eval()
#    # calculate the denoise result on test set.
#    simple_output = model(noisy)
#    #model.eval()
#    #invariant_output = masker.infer_full_image(noisy, model)
#    idx = 3
#    plot_tensors([noisy[idx], simple_output[idx]],\
#            ["Noisy Image", "Single Pass Inference"])
    
    
    
    
    
    