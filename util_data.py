#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 07:09:54 2019
functions related to data load and data preprocess.
@author: yxw
"""
from PIL import Image
import torchvision



def crop_image(img, d=32):
    '''Make dimensions divisible by `d`'''

    new_size = (img.size[0] - img.size[0] % d, 
                img.size[1] - img.size[1] % d)

    bbox = [
            int((img.size[0] - new_size[0])/2), 
            int((img.size[1] - new_size[1])/2),
            int((img.size[0] + new_size[0])/2),
            int((img.size[1] + new_size[1])/2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped

def image_to_tensor(filepath, use_cuda=True):
    pil = Image.open(filepath)
    # crop the image so that it can be divided n times
    pil = crop_image(pil, d=64)
    pil_to_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    if use_cuda:
        tensor = pil_to_tensor(pil).cuda()
    else:
        tensor = pil_to_tensor(pil)
    return tensor.view([1]+list(tensor.shape))

def tensor_to_image(tensor, filename, save=True, use_cuda=True):
    tensor = tensor.view(tensor.shape[1:])
    if use_cuda:
        tensor = tensor.cpu()
    tensor_to_pil = torchvision.transforms.Compose([torchvision.transforms.ToPILImage()])
    pil = tensor_to_pil(tensor)
    if save:
        pil.save(filename)