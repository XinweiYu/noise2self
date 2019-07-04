import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import numpy as np
from PIL import Image
from torch.nn import MSELoss
import os


def jpg_to_tensor(filepath):
    pil = Image.open(filepath)
    # crop the image so that it can be divided n times
    pil = crop_image(pil, d=64)
    pil_to_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    if use_cuda:
        tensor = pil_to_tensor(pil).cuda()
    else:
        tensor = pil_to_tensor(pil)
    return tensor.view([1]+list(tensor.shape))

def tensor_to_jpg(tensor, filename):
    tensor = tensor.view(tensor.shape[1:])
    if use_cuda:
        tensor = tensor.cpu()
    tensor_to_pil = torchvision.transforms.Compose([torchvision.transforms.ToPILImage()])
    pil = tensor_to_pil(tensor)
    pil.save(filename)

#function which zeros out a random proportion of pixels from an image tensor.
def zero_out_pixels(tensor, prop=0.5):
    if use_cuda:
        mask = torch.rand([1]+[1] + list(tensor.shape[2:])).cuda()
    else:
        mask = torch.rand([1]+[1] + list(tensor.shape[2:]))
    mask[mask<prop] = 0
    mask[mask!=0] = 1
    mask = mask.repeat(1,3,1,1)
    deconstructed = tensor * mask
    return mask, deconstructed

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



class pixel_shuffle_hourglass(nn.Module):
    def __init__(self):
        super(pixel_shuffle_hourglass, self).__init__()
        self.d_conv_1 = nn.Conv2d(1, 8, 5, stride=2, padding=2)
        self.d_bn_1 = nn.BatchNorm2d(8)

        self.d_conv_2 = nn.Conv2d(8, 16, 5, stride=2, padding=2)
        self.d_bn_2 = nn.BatchNorm2d(16)

        self.d_conv_3 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.d_bn_3 = nn.BatchNorm2d(32)
        # s here stands for skip
        self.s_conv_3 = nn.Conv2d(32, 4, 5, stride=1, padding=2)

        self.d_conv_4 = nn.Conv2d(32, 64, 5, stride=2, padding=2)
        self.d_bn_4 = nn.BatchNorm2d(64)
        self.s_conv_4 = nn.Conv2d(64, 4, 5, stride=1, padding=2)

        self.d_conv_5 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.d_bn_5 = nn.BatchNorm2d(128)
        self.s_conv_5 = nn.Conv2d(128, 4, 5, stride=1, padding=2)

        self.d_conv_6 = nn.Conv2d(128, 256, 5, stride=2, padding=2)
        self.d_bn_6 = nn.BatchNorm2d(256)

        self.u_conv_5 = nn.Conv2d(68, 128, 5, stride=1, padding=2)
        self.u_bn_5 = nn.BatchNorm2d(128)

        self.u_conv_4 = nn.Conv2d(36, 64, 5, stride=1, padding=2)
        self.u_bn_4 = nn.BatchNorm2d(64)

        self.u_conv_3 = nn.Conv2d(20, 32, 5, stride=1, padding=2)
        self.u_bn_3 = nn.BatchNorm2d(32)

        self.u_conv_2 = nn.Conv2d(8, 16, 5, stride=1, padding=2)
        self.u_bn_2 = nn.BatchNorm2d(16)

        self.u_conv_1 = nn.Conv2d(4, 16, 5, stride=1, padding=2)
        self.u_bn_1 = nn.BatchNorm2d(16)

        self.out_conv = nn.Conv2d(4, 1, 5, stride=1, padding=2)
        self.out_bn = nn.BatchNorm2d(1)

        
    def forward(self, noise):
        down_1 = self.d_conv_1(noise)
        down_1 = self.d_bn_1(down_1)
        down_1 = F.leaky_relu(down_1)
        
        down_2 = self.d_conv_2(down_1)
        down_2 = self.d_bn_2(down_2)
        down_2 = F.leaky_relu(down_2)

        down_3 = self.d_conv_3(down_2)
        down_3 = self.d_bn_3(down_3)
        down_3 = F.leaky_relu(down_3)
        skip_3 = self.s_conv_3(down_3)

        down_4 = self.d_conv_4(down_3)
        down_4 = self.d_bn_4(down_4)
        down_4 = F.leaky_relu(down_4)
        skip_4 = self.s_conv_4(down_4)

        down_5 = self.d_conv_5(down_4)
        down_5 = self.d_bn_5(down_5)
        down_5 = F.leaky_relu(down_5)
        skip_5 = self.s_conv_5(down_5)

        down_6 = self.d_conv_6(down_5)
        down_6 = self.d_bn_6(down_6)
        down_6 = F.leaky_relu(down_6)   # 256 channels /2^6

        up_5 = F.pixel_shuffle(down_6, 2)   #64
        up_5 = torch.cat([up_5, skip_5], 1) #68
        up_5 = self.u_conv_5(up_5)
        up_5 = self.u_bn_5(up_5)
        up_5 = F.leaky_relu(up_5)

        up_4 = F.pixel_shuffle(up_5, 2)
        up_4 = torch.cat([up_4, skip_4], 1)
        up_4 = self.u_conv_4(up_4)
        up_4 = self.u_bn_4(up_4)
        up_4 = F.leaky_relu(up_4)

        up_3 = F.pixel_shuffle(up_4, 2)
        up_3 = torch.cat([up_3, skip_3], 1)
        up_3 = self.u_conv_3(up_3)
        up_3 = self.u_bn_3(up_3)
        up_3 = F.leaky_relu(up_3)

        up_2 = F.pixel_shuffle(up_3, 2)
        up_2 = self.u_conv_2(up_2)
        up_2 = self.u_bn_2(up_2)
        up_2 = F.leaky_relu(up_2)

        up_1 = F.pixel_shuffle(up_2, 2)
        up_1 = self.u_conv_1(up_1)
        up_1 = self.u_bn_1(up_1)
        up_1 = F.leaky_relu(up_1)

        out = F.pixel_shuffle(up_1, 2)
        out = self.out_conv(out)
        out = self.out_bn(out)
        out = F.sigmoid(out)
        return out

class deconv_hourglass(nn.Module):
    def __init__(self):
        super(deconv_hourglass, self).__init__()
        self.d_conv_1 = nn.Conv2d(1, 8, 5, stride=2, padding=2)
        self.d_bn_1 = nn.BatchNorm2d(8)

        self.d_conv_2 = nn.Conv2d(8, 16, 5, stride=2, padding=2)
        self.d_bn_2 = nn.BatchNorm2d(16)

        self.d_conv_3 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.d_bn_3 = nn.BatchNorm2d(32)
        self.s_conv_3 = nn.Conv2d(32, 4, 5, stride=1, padding=2)

        self.d_conv_4 = nn.Conv2d(32, 64, 5, stride=2, padding=2)
        self.d_bn_4 = nn.BatchNorm2d(64)
        self.s_conv_4 = nn.Conv2d(64, 4, 5, stride=1, padding=2)

        self.d_conv_5 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.d_bn_5 = nn.BatchNorm2d(128)
        self.s_conv_5 = nn.Conv2d(128, 4, 5, stride=1, padding=2)

        self.d_conv_6 = nn.Conv2d(128, 256, 5, stride=2, padding=2)
        self.d_bn_6 = nn.BatchNorm2d(256)

        self.u_deconv_5 = nn.ConvTranspose2d(256, 124, 4, stride=2, padding=1)
        self.u_bn_5 = nn.BatchNorm2d(128)

        self.u_deconv_4 = nn.ConvTranspose2d(128, 60, 4, stride=2, padding=1)
        self.u_bn_4 = nn.BatchNorm2d(64)

        self.u_deconv_3 = nn.ConvTranspose2d(64, 28, 4, stride=2, padding=1)
        self.u_bn_3 = nn.BatchNorm2d(32)

        self.u_deconv_2 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.u_bn_2 = nn.BatchNorm2d(16)

        self.u_deconv_2 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.u_bn_2 = nn.BatchNorm2d(16)

        self.u_deconv_1 = nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1)
        self.u_bn_1 = nn.BatchNorm2d(8)

        self.out_deconv = nn.ConvTranspose2d(8, 1, 4, stride=2, padding=1)        
        self.out_bn = nn.BatchNorm2d(1)

        
    def forward(self, noise):
        down_1 = self.d_conv_1(noise)
        down_1 = self.d_bn_1(down_1)
        down_1 = F.leaky_relu(down_1)
        
        down_2 = self.d_conv_2(down_1)
        down_2 = self.d_bn_2(down_2)
        down_2 = F.leaky_relu(down_2)

        down_3 = self.d_conv_3(down_2)
        down_3 = self.d_bn_3(down_3)
        down_3 = F.leaky_relu(down_3)
        skip_3 = self.s_conv_3(down_3)

        down_4 = self.d_conv_4(down_3)
        down_4 = self.d_bn_4(down_4)
        down_4 = F.leaky_relu(down_4)
        skip_4 = self.s_conv_4(down_4)

        down_5 = self.d_conv_5(down_4)
        down_5 = self.d_bn_5(down_5)
        down_5 = F.leaky_relu(down_5)
        skip_5 = self.s_conv_5(down_5)

        down_6 = self.d_conv_6(down_5)
        down_6 = self.d_bn_6(down_6)
        down_6 = F.leaky_relu(down_6)

        up_5 = self.u_deconv_5(down_6)
        up_5 = torch.cat([up_5, skip_5], 1)
        up_5 = self.u_bn_5(up_5)
        up_5 = F.leaky_relu(up_5)

        up_4 = self.u_deconv_4(up_5)
        up_4 = torch.cat([up_4, skip_4], 1)
        up_4 = self.u_bn_4(up_4)
        up_4 = F.leaky_relu(up_4)

        up_3 = self.u_deconv_3(up_4)
        up_3 = torch.cat([up_3, skip_3], 1)
        up_3 = self.u_bn_3(up_3)
        up_3 = F.leaky_relu(up_3)

        up_2 = self.u_deconv_2(up_3)
        up_2 = self.u_bn_2(up_2)
        up_2 = F.leaky_relu(up_2)

        up_1 = self.u_deconv_1(up_2)
        up_1 = self.u_bn_1(up_1)
        up_1 = F.leaky_relu(up_1)

        out = self.out_deconv(up_1)
        out = self.out_bn(out)
        out = F.sigmoid(out)

        return out

if __name__ == "__main__":
    
    use_cuda = True
    noisy_img_path = 'data/LowQ_digest_test/1.2.156.147522.44.410947.949.1.1.20190625143401_00.png'

    #standard deviation of added noise after each training set
    sigma = 1./30
    num_steps = 25001#25001
    save_frequency = 250  # frequency to save the images.
    output_dir = 'data/DIP_train_series/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #output_name = 'train_'
    #choose either 'pixel_shuffle' or 'deconv'
    method = 'pixel_shuffle'
    

    noisy_img = jpg_to_tensor(noisy_img_path)
    
    #mask, deconstructed = zero_out_pixels(truth)
    
    tensor_to_jpg(noisy_img, os.path.join(output_dir,'noisy.png'))
    
#    mask = Variable(mask)
#    deconstructed = Variable(deconstructed)
    
    if use_cuda:
        noisy_img = noisy_img.cuda()
        noise = Variable(torch.randn(noisy_img.shape).cuda())
    else:
        noise = Variable(torch.randn(noisy_img.shape))

    
    if method=='pixel_shuffle':
        net = pixel_shuffle_hourglass()
    elif method=='deconv':
        net = deconv_hourglass()
    
    if use_cuda:
        net.cuda()
        
    # optimizer    
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    # set loss function
    loss_function = MSELoss()
    
    
    save_img_ind = 0  # initialize the image index
    for step in range(num_steps):
        output = net(noise)
        #masked_output = output * mask
        optimizer.zero_grad()
        loss = loss_function(output, noisy_img)
        
        loss.backward()
        optimizer.step()
        print('At step {}, loss is {}'.format(step, loss.data.cpu()))
        if step % save_frequency == 0:
            tensor_to_jpg(output.data, os.path.join(output_dir,'train_{}.png'.format(save_img_ind)))
            save_img_ind += 1
        
        # induce noise in the input 
        if use_cuda:
            noise.data += sigma * torch.randn(noise.shape).cuda()
        else:
            noise.data += sigma * torch.randn(noise.shape)
    
    torch.cuda.empty_cache()
