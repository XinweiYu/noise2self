#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 08:06:35 2019
# load the noise2self model and run it on test set
@author: yxw
"""

from __future__ import print_function, division
import torch
from models.babyunet import BabyUnet
from util_data import image_to_tensor
from util import plot_tensors




if __name__ == "__main__":
    
    use_cuda = True
    # load the model 
    model = BabyUnet()
    model.load_state_dict(torch.load('BabyUnet_denoise_old.pth')) # the file of stored model.
    model.cuda()
    model.eval()
    
    # load test image
    noisy_img_path = 'data/LowQ_digest_test/1.2.156.147522.44.410947.949.1.1.20190625143401_00.png'
    noisy = image_to_tensor(filepath=noisy_img_path, use_cuda=use_cuda)
    # calculate denoised image
    simple_output = model(noisy)
    # plot the image
    plot_tensors([noisy, simple_output],["Noisy Image", "Single Pass Inference"], plot=True)