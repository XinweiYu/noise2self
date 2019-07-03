import SimpleITK as sitk
import numpy as np
from PIL import Image
import os
import numpy.random as npr


def make_png(a, save_name):
    amin = np.min(a)
    amax = np.max(a)
    anorm = (np.clip(a, amin, amax)-amin)/(amax-amin)*255
    img = Image.fromarray(anorm.astype(np.uint8))
    img.save(save_name)

# # data to manully crop
# data_path = '/home/lyy/Documents/data/yc/20190626/to周老师/低信噪比消化道'
# fs = sorted(os.listdir(data_path))
# xmin = 137
# xmax = 844
# ymin = 4
# ymax = 984
# out_path = 'LR1'
# for f in fs:
#     itk = sitk.ReadImage(os.path.join(data_path,f))
#     arr = sitk.GetArrayFromImage(itk)
#     arr = arr[:,ymin:ymax,xmin:xmax]
#     for i, arr_ in enumerate(arr):
#         save_name = os.path.join(out_path, f.replace('.dcm','')+'_'+str(i).zfill(2)+'.npy')
#         np.save(save_name, arr_)
#         make_png(arr_, save_name.replace('.npy', '.png'))


# data_path = '/home/lyy/Documents/data/yc/20190626/to周老师/低信噪比腰椎'
# fs = sorted(os.listdir(data_path))
# crop_size = 23
# out_path = 'LR2'
# for f in fs:
#     itk = sitk.ReadImage(os.path.join(data_path,f))
#     arr = sitk.GetArrayFromImage(itk)
#     _, ysize, xsize = arr.shape
#     for i, arr_ in enumerate(arr):
#         ysum = np.sum(arr_,axis=0)
#         xsum = np.sum(arr_,axis=1)
#         x_valid = np.where(xsum>0)
#         y_valid = np.where(ysum>0)
#         xmin = np.min(x_valid)
#         xmax = np.max(x_valid)
#         ymin = np.min(y_valid)
#         ymax = np.max(y_valid)
#         arr_ = arr_[xmin:xmax,ymin+crop_size:ymax-crop_size]
#         save_name = os.path.join(out_path, f.replace('.dcm','')+'_'+str(i).zfill(2)+'.npy')
#         np.save(save_name, arr_)
#         make_png(arr_, save_name.replace('.npy', '.png'))

def preprocess_dcm_images(data_paths, out_dir, split=False, **kwargs):
    # This function preprcess dcm format medical images.
    # purpose1: transform format from dcm to png
    # purpose2: crop the image, only keep region of interest.
    # split is to split data into train and test set. 

    # data-path is the list of folders that has original image(format dcm)
    for data_path in data_paths:
        fs = sorted(os.listdir(data_path))
        crop_size = 0 # not in use now.
        # set the path to store images
        if split:
            # build train and test folder.
            out_dir_train = out_dir + '_train'
            out_dir_test = out_dir + '_test'
            if not os.path.exists(out_dir_train):
                os.makedirs(out_dir_train)
            if not os.path.exists(out_dir_test):
                os.makedirs(out_dir_test)           
        else:        
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
        for f in fs:
            itk = sitk.ReadImage(os.path.join(data_path,f))
            arr = sitk.GetArrayFromImage(itk)
            _, ysize, xsize = arr.shape
            for i, arr_ in enumerate(arr):
                ysum = np.sum(arr_, axis=0)
                xsum = np.sum(arr_, axis=1)
                # sometimes only region of interest has signal.(control by doctor)
                x_valid = np.where(xsum>0)
                y_valid = np.where(ysum>0)
                # get the axis limit for region of interest.
                xmin = np.min(x_valid)
                xmax = np.max(x_valid)
                ymin = np.min(y_valid)
                ymax = np.max(y_valid)
                # crop the image ?? crop_size=0, only keep region of interest.
                arr_ = arr_[xmin:xmax,ymin+crop_size:ymax-crop_size]
                
                if split:
                    if npr.rand() < kwargs['split_ratio']:
                        out_dir = out_dir_train
                    else:
                        out_dir = out_dir_test
                    
                save_name = os.path.join(out_dir, f.replace('.dcm','')+'_'+str(i).zfill(2)+'.npy')
                #np.save(save_name, arr_)
                make_png(arr_, save_name.replace('.npy', '.png'))
            
if __name__ == "__main__":
    # preprocess the image folder.
    #data_path = '/home/yxw/Desktop/transfer/data_customer/20190626/to周老师/低信噪比腰椎'
    data_path = ['/home/yxw/Desktop/transfer/data_customer/20190626/to周老师/低信噪比消化道']
    # Output folder.
    out_dir = 'LowQ_digest'
    #
    preprocess_dcm_images(data_path, out_dir, split=True, split_ratio=0.9)
    
    
    
    
    