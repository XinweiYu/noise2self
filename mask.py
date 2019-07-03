import numpy as np
import torch


class Masker():
    """Object for masking and demasking"""

    def __init__(self, width=3, mode='zero', infer_single_pass=False, include_mask_as_input=False):
        # grid_size is the size of the block.
        self.grid_size = width
        self.n_masks = width ** 2 # number of masks is the number of pixels in the block.

        self.mode = mode
        self.infer_single_pass = infer_single_pass
        self.include_mask_as_input = include_mask_as_input

    def mask(self, X, i):
        # infer 2d x and y from a single index.
        # i is the index for the current masked pixels.
        # X is the image.
        phasex = i % self.grid_size
        phasey = (i // self.grid_size) % self.grid_size
        # mask highlight the pixels(to be masked)
        mask = pixel_grid_mask(X[0, 0].shape, self.grid_size, phasex, phasey)
        mask = mask.to(X.device)
        
        mask_inv = torch.ones(mask.shape).to(X.device) - mask  # inverse of mask

        if self.mode == 'interpolate':
            # replace the pixels highlight in mask with the average of neighbors.
            masked = interpolate_mask(X, mask, mask_inv)
        elif self.mode == 'zero':
            # set the masked pixels to be zero.
            masked = X * mask_inv
        else:
            raise NotImplementedError
            
        if self.include_mask_as_input:
            #Concatenates the given sequence of seq tensors in the given dimension
            net_input = torch.cat((masked, mask.repeat(X.shape[0], 1, 1, 1)), dim=1) # what is the dimension of X ?
        else:
            net_input = masked

        return net_input, mask

    def __len__(self):
        return self.n_masks

    def infer_full_image(self, X, model):
        
        if self.infer_single_pass:
            if self.include_mask_as_input:
                #? add one more first layer of X to X?
                # just use the first channel as mask ???
                net_input = torch.cat((X, torch.zeros(X[:, 0:1].shape).to(X.device)), dim=1)
            else:
                net_input = X
            net_output = model(net_input)
            return net_output

        else:
            net_input, mask = self.mask(X, 0)
            net_output = model(net_input)

            acc_tensor = torch.zeros(net_output.shape).cpu()

            for i in range(self.n_masks):
                # each time only calculate the pixels that are masked.
                net_input, mask = self.mask(X, i)
                net_output = model(net_input)
                acc_tensor = acc_tensor + (net_output * mask).cpu()

            return acc_tensor


def pixel_grid_mask(shape, patch_size, phase_x, phase_y):
    A = torch.zeros(shape[-2:])
    for i in range(shape[-2]):
        for j in range(shape[-1]):
            # divide the whole image to blocks according to grid.
            # Then find the pixel to be masked in the image.
            if (i % patch_size == phase_x and j % patch_size == phase_y):
                A[i, j] = 1
    return torch.Tensor(A)


def interpolate_mask(tensor, mask, mask_inv):
    # tensor is the images.
    device = tensor.device

    mask = mask.to(device)

    kernel = np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], (0.5, 1.0, 0.5)])
    kernel = kernel[np.newaxis, np.newaxis, :, :]
    kernel = torch.Tensor(kernel).to(device)
    kernel = kernel / kernel.sum()  # normalization.
    # calculate the average for each pixels.
    filtered_tensor = torch.nn.functional.conv2d(tensor, kernel, stride=1, padding=1)

    return filtered_tensor * mask + tensor * mask_inv
