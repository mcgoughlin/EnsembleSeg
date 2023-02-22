# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 23:35:05 2022

@author: mcgoug01
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ovseg.networks.nfUNet import concat_attention, concat
from scipy.ndimage.filters import gaussian_filter
from ovseg.utils.torch_np_utils import check_type, maybe_add_channel_dim
import os
from skimage.measure import label
from skimage.transform import resize
from torch.nn.functional import interpolate
from scipy.ndimage.morphology import binary_fill_holes
from ovseg.utils.torch_morph import morph_cleaning
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    print('No tqdm found, using no pretty progressing bars')
    tqdm = lambda x: x
from time import sleep

os.environ['OV_DATA_BASE'] = "C:\\Users\\mcgoug01\\OneDrive - CRUK Cambridge Institute\\Python Scripts\\kits19\\ovseg_kits19data"

def get_padding(kernel_size):
    if isinstance(kernel_size, (list, tuple, np.ndarray)):
        return [(k - 1) // 2 for k in kernel_size]
    else:
        return (kernel_size - 1) // 2


def get_stride(kernel_size):
    if isinstance(kernel_size, (list, tuple, np.ndarray)):
        return [(k + 1)//2 for k in kernel_size]
    else:
        return (kernel_size + 1) // 2


# %%
class ConvNormNonlinBlock(nn.Module):

    def __init__(self, in_channels, out_channels, is_2d, kernel_size=3,
                 first_stride=1, conv_params=None, norm=None, norm_params=None,
                 nonlin_params=None, hid_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels if hid_channels is not None else out_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = get_padding(self.kernel_size)
        self.first_stride = first_stride
        self.is_2d = is_2d
        self.conv_params = conv_params
        self.norm_params = norm_params
        self.nonlin_params = nonlin_params

        if norm is None:
            norm = 'batch' if is_2d else 'inst'

        if self.conv_params is None:
            self.conv_params = {'bias': False}
        if self.nonlin_params is None:
            self.nonlin_params = {'negative_slope': 0.01, 'inplace': True}
        if self.norm_params is None:
            self.norm_params = {'affine': True}
        # init convolutions, normalisation and nonlinearities
        if self.is_2d:
            conv_fctn = nn.Conv2d
            if norm.lower().startswith('batch'):
                norm_fctn = nn.BatchNorm2d
            elif norm.lower().startswith('inst'):
                norm_fctn = nn.InstanceNorm2d
        else:
            conv_fctn = nn.Conv3d
            if norm.lower().startswith('batch'):
                norm_fctn = nn.BatchNorm3d
            elif norm.lower().startswith('inst'):
                norm_fctn = nn.InstanceNorm3d
        self.conv1 = conv_fctn(self.in_channels, self.hid_channels,
                               self.kernel_size, padding=self.padding,
                               stride=self.first_stride, **self.conv_params)
        self.conv2 = conv_fctn(self.hid_channels, self.out_channels,
                               self.kernel_size, padding=self.padding,
                               **self.conv_params)
        self.norm1 = norm_fctn(self.hid_channels, **self.norm_params)
        self.norm2 = norm_fctn(self.out_channels, **self.norm_params)

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        self.nonlin1 = nn.LeakyReLU(**self.nonlin_params)
        self.nonlin2 = nn.LeakyReLU(**self.nonlin_params)

    def forward(self, xb):
        xb = self.conv1(xb)
        xb = self.norm1(xb)
        xb = self.nonlin1(xb)
        xb = self.conv2(xb)
        xb = self.norm2(xb)
        xb = self.nonlin2(xb)
        return xb


# %% transposed convolutions
class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels, is_2d, kernel_size=2):
        super().__init__()
        if is_2d:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels,
                                           kernel_size, stride=kernel_size,
                                           bias=False)
        else:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels,
                                           kernel_size, stride=kernel_size,
                                           bias=False)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, xb):
        return self.conv(xb)


class UpLinear(nn.Module):

    def __init__(self, kernel_size, is_2d):
        
        if is_2d:
            self.up = nn.Upsample(scale_factor=kernel_size, mode='bilinear',
                                  align_corners=True)
        else:
            self.up = nn.Upsample(scale_factor=kernel_size, mode='trilinear',
                                  align_corners=True)
    def forward(self, xb):
        return self.up(xb)

# %% now simply the logits
class Logits(nn.Module):

    def __init__(self, in_channels, out_channels, is_2d, p_dropout=0):
        super().__init__()
        if is_2d:
            self.logits = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.dropout = nn.Dropout2d(p_dropout, inplace=True)
        else:
            self.logits = nn.Conv3d(in_channels, out_channels, 1, bias=False)
            self.dropout = nn.Dropout3d(p_dropout, inplace=True)
        nn.init.kaiming_normal_(self.logits.weight)

    def forward(self, xb):
        return self.dropout(self.logits(xb))

# %%
class res_skip(nn.Module):

    def forward(self, xb1, xb2):
        return xb1 + xb2

class param_res_skip(nn.Module):

    def __init__(self, in_channels, is_2d):
        super().__init__()

        if is_2d:
            self.a = nn.Parameter(torch.zeros((1, in_channels, 1, 1)))
        else:
            self.a = nn.Parameter(torch.zeros((1, in_channels, 1, 1, 1)))

    def forward(self, xb_up, xb_skip):
        return xb_up + self.a * xb_skip

class scaled_res_skip(nn.Module):

    def __init__(self):
        super().__init__()

        self.a = nn.Parameter(torch.zeros(()))
    def forward(self, xb_up, xb_skip):
        return xb_up + self.a * xb_skip

# %%
class UNet(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_sizes,
                 is_2d, filters=16, filters_max=384, n_pyramid_scales=None,
                 conv_params=None, norm=None, norm_params=None, nonlin_params=None,
                 kernel_sizes_up=None, skip_type='skip', use_trilinear_upsampling=False,
                 use_less_hid_channels_in_decoder=False, fac_skip_channels=1,
                 p_dropout_logits=0.0, stem_kernel_size=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.is_2d = is_2d
        self.n_stages = len(kernel_sizes)
        self.filters = filters
        self.filters_max = filters_max
        self.conv_params = conv_params
        self.norm = norm
        self.norm_params = norm_params
        self.nonlin_params = nonlin_params
        self.kernel_sizes_up = kernel_sizes_up if kernel_sizes_up is not None else kernel_sizes[:-1]
        assert skip_type in ['skip', 'self_attention', 'res_skip', 'param_res_skip', 
                             'scaled_res_skip']
        self.skip_type = skip_type
        self.use_trilinear_upsampling = use_trilinear_upsampling
        self.use_less_hid_channels_in_decoder = use_less_hid_channels_in_decoder
        assert fac_skip_channels <= 1 and fac_skip_channels > 0
        self.fac_skip_channels = fac_skip_channels
        self.p_dropout_logits = p_dropout_logits
        self.stem_kernel_size = stem_kernel_size
        
        # we double the amount of channels every downsampling step
        # up to a max of filters_max
        self.filters_list = [min([self.filters*2**i, self.filters_max])
                             for i in range(self.n_stages)]

        # first let's make the lists for the blocks on both pathes
        # number of input and output channels of the blocks on the contracting path
        if self.stem_kernel_size is None:
            self.in_channels_down_list = [self.in_channels] + self.filters_list[:-1]
        else:
            self.in_channels_down_list = [self.filters] + self.filters_list[:-1]
        self.out_channels_down_list = self.filters_list
        self.first_stride_list = [1] + [get_stride(ks) for ks in self.kernel_sizes[:-1]]

        # the number of channels we will feed forward
        self.skip_channels = [int(self.fac_skip_channels * ch)
                              for ch in self.out_channels_down_list[:-1]]

        # for the decoder this is more difficult and depend on other settings
        if self.skip_type in ['skip', 'self_attention']:
            self.in_channels_up_list = [2 * n_ch for n_ch in self.skip_channels]
        else:
            self.in_channels_up_list = [n_ch for n_ch in self.skip_channels]
        # if we do trilinear upsampling we have to make sure that the right number of channels is
        # outputed from the block below
        if self.use_trilinear_upsampling:
            self.out_channels_up_list = [self.filters // 2] + self.skip_channels[:-1]
            self.out_channels_down_list[-1] = self.skip_channels[-1]
        else:
            self.out_channels_up_list = self.filters_list

        if self.use_less_hid_channels_in_decoder and self.use_trilinear_upsampling:
            self.hid_channels_up_list = [(in_ch + out_ch) // 2 for in_ch, out_ch in
                                         zip(self.in_channels_up_list, self.out_channels_up_list)]
        else:
            self.hid_channels_up_list = self.filters_list[:-1]

        # determine how many scales on the upwars path with be connected to
        # a loss function
        if n_pyramid_scales is None:
            self.n_pyramid_scales = max([1, self.n_stages - 2])
        else:
            self.n_pyramid_scales = int(n_pyramid_scales)

        # now all the logits
        self.logits_in_list = self.out_channels_up_list[:self.n_pyramid_scales]

        # blocks on the contracting path
        self.blocks_down = []
        for in_channels, out_channels, kernel_size, first_stride in zip(self.in_channels_down_list,
                                                                        self.out_channels_down_list,
                                                                        self.kernel_sizes,
                                                                        self.first_stride_list):
            block = ConvNormNonlinBlock(in_channels=in_channels,
                                        out_channels=out_channels,
                                        is_2d=self.is_2d,
                                        kernel_size=kernel_size,
                                        first_stride=first_stride,
                                        conv_params=self.conv_params,
                                        norm=self.norm,
                                        norm_params=self.norm_params,
                                        nonlin_params=self.nonlin_params)
            self.blocks_down.append(block)

        # blocks on the upsampling path
        self.blocks_up = []
        for in_channels, out_channels, hid_channels, kernel_size in zip(self.in_channels_up_list,
                                                                        self.out_channels_up_list,
                                                                        self.hid_channels_up_list,
                                                                        self.kernel_sizes_up):
            block = ConvNormNonlinBlock(in_channels=in_channels,
                                        out_channels=out_channels,
                                        is_2d=self.is_2d,
                                        kernel_size=kernel_size,
                                        conv_params=self.conv_params,
                                        norm=self.norm,
                                        norm_params=self.norm_params,
                                        nonlin_params=self.nonlin_params,
                                        hid_channels=hid_channels)
            self.blocks_up.append(block)

        # upsaplings
        self.upsamplings = []
        if self.use_trilinear_upsampling:
            mode = 'bilinear' if self.is_2d else 'trilinear'
            for kernel_size in self.kernel_sizes[:-1]:         
                scaled_factor = tuple([(k+1)//2 for k in kernel_size])
                self.upsamplings.append(nn.Upsample(scale_factor=scaled_factor,
                                                    mode=mode,
                                                    align_corners=True))
        else:
            for in_channels, out_channels, kernel_size in zip(self.out_channels_up_list[1:],
                                                              self.skip_channels,
                                                              self.kernel_sizes):
                self.upsamplings.append(UpConv(in_channels=in_channels,
                                           out_channels=out_channels,
                                           is_2d=self.is_2d,
                                           kernel_size=get_stride(kernel_size)))
        # now the concats:
        self.concats = []
        if self.skip_type == 'self_attention':
            skip_fctn = lambda ch: concat_attention(ch, self.is_2d)
        elif self.skip_type == 'res_skip':
            skip_fctn = lambda ch: res_skip()
        elif self.skip_type == 'param_res_skip':
            skip_fctn = lambda ch: param_res_skip(ch, self.is_2d)
        elif self.skip_type == 'skip':
            skip_fctn = lambda ch: concat()
        elif self.skip_type == 'scaled_res_skip':
            skip_fctn = lambda ch: scaled_res_skip()
            
        for in_ch in self.skip_channels:
            self.concats.append(skip_fctn(in_ch))

        # logits
        self.all_logits = []
        for in_channels in self.logits_in_list:
            self.all_logits.append(Logits(in_channels=in_channels,
                                          out_channels=self.out_channels,
                                          is_2d=self.is_2d,
                                          p_dropout=self.p_dropout_logits))

        # now important let's turn everything into a module list
        self.blocks_down = nn.ModuleList(self.blocks_down)
        self.blocks_up = nn.ModuleList(self.blocks_up)
        self.upsamplings = nn.ModuleList(self.upsamplings)
        self.concats = nn.ModuleList(self.concats)
        self.all_logits = nn.ModuleList(self.all_logits)
        
        if self.stem_kernel_size is not None:
            if self.is_2d:
                self.stem = nn.Conv2d(self.in_channels,
                                      self.filters,
                                      self.stem_kernel_size,
                                      self.stem_kernel_size)
            else:
                self.stem = nn.Conv3d(self.in_channels,
                                      self.filters,
                                      self.stem_kernel_size,
                                      self.stem_kernel_size)
            self.final_up_conv = UpConv(self.filters,
                                        self.out_channels,
                                        self.is_2d,
                                        self.stem_kernel_size)                

    def forward(self, xb):
        
        if self.stem_kernel_size is not None:
            xb = self.stem(xb)
        
        # keep all out tensors from the contracting path
        xb_list = []
        logs_list = []
        # contracting path
        for block, skip_ch in zip(self.blocks_down, self.skip_channels):
            xb = block(xb)
            xb_list.append(xb[:, :skip_ch])

        # bottom block
        xb = self.blocks_down[-1](xb)

        # expanding path without logits
        for i in range(self.n_stages - 2, self.n_pyramid_scales-1, -1):
            xb = self.upsamplings[i](xb)
            xb = self.concats[i](xb, xb_list[i])
            del xb_list[i]
            xb = self.blocks_up[i](xb)

        # expanding path with logits
        for i in range(self.n_pyramid_scales - 1, -1, -1):
            xb = self.upsamplings[i](xb)
            xb = self.concats[i](xb, xb_list[i])
            del xb_list[i]
            xb = self.blocks_up[i](xb)
            logs = self.all_logits[i](xb)
            logs_list.append(logs)

        if self.stem_kernel_size is not None:
            logs_list.append(self.final_up_conv(xb))

        # as we iterate from bottom to top we have to flip the logits list
        return logs_list[::-1]

    def update_prg_trn(self, param_dict, h, indx=None):
        if 'p_dropout_logits' in param_dict:
            p = (1 - h) * param_dict['p_dropout_logits'][0] + h * param_dict['p_dropout_logits'][1]
            for l in self.all_logits:
                l.dropout.p = p
                
class SlidingWindowPrediction(object):

    def __init__(self, network, patch_size, batch_size=1, overlap=0.5, fp32=False,
                 patch_weight_type='gaussian', sigma_gaussian_weight=1/8, linear_min=0.1,
                 mode='flip'):

        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.network = network.to(self.dev)
        self.batch_size = batch_size
        self.overlap = overlap
        self.fp32 = fp32
        self.patch_weight_type = patch_weight_type
        self.sigma_gaussian_weight = sigma_gaussian_weight
        self.linear_min = linear_min
        self.mode = mode

        assert self.patch_weight_type.lower() in ['constant', 'gaussian', 'linear']
        assert self.mode.lower() in ['simple', 'flip']

        self._set_patch_size_and_weight(patch_size)
    

    def _set_patch_size_and_weight(self, patch_size): 
        # check and build up the patch weight
        # we can use a gaussian weighting since the predictions on the edge of the patch are less
        # reliable than the ones in the middle
        self.patch_size = np.array(patch_size).astype(int)
        if self.patch_weight_type.lower() == 'constant':
            self.patch_weight = np.ones(self.patch_size)
        elif self.patch_weight_type.lower() == 'gaussian':
            # we distrust the edge voxel the same in each direction regardless of the
            # patch size in that dimension

            # thanks to Fabian Isensee! I took this from his code:
            # https://github.com/MIC-DKFZ/nnUNet/blob/14992342919e63e4916c038b6dc2b050e2c62e3c/nnunet/network_architecture/neural_network.py#L250
            tmp = np.zeros(self.patch_size)
            center_coords = [i // 2 for i in self.patch_size]
            sigmas = [i * self.sigma_gaussian_weight for i in self.patch_size]
            tmp[tuple(center_coords)] = 1
            self.patch_weight = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
            self.patch_weight = self.patch_weight / np.max(self.patch_weight) * 1
            self.patch_weight = self.patch_weight.astype(np.float32)

            # self.patch_weight cannot be 0, otherwise we may end up with nans!
            self.patch_weight[self.patch_weight == 0] = np.min(
                self.patch_weight[self.patch_weight != 0])

        elif self.patch_weight_type.lower() == 'linear':
            lin_slopes = [np.linspace(self.linear_min, 1, s//2) for s in self.patch_size]
            hats = [np.concatenate([lin_slope, lin_slope[::-1]]) for lin_slope in lin_slopes]
            hats = [np.expand_dims(hat, [j for j in range(len(self.patch_size)) if j != i])
                    for i, hat in enumerate(hats)]

            self.patch_weight = np.ones(self.patch_size)
            for hat in hats:
                self.patch_weight *= hat

        self.patch_weight = self.patch_weight[np.newaxis]
        self.patch_weight = torch.from_numpy(self.patch_weight).to(self.dev).type(torch.float)

        # add an axis to the patch size and set is_2d
        if len(self.patch_size) == 2:
            self.is_2d = True
            self.patch_size = np.concatenate([[1], self.patch_size])
        elif len(self.patch_size) == 3:
            self.is_2d = False
        else:
            raise ValueError('patch_size must be of len 2 or 3 (for 2d and 3d networks).')
        


    def _get_xyz_list(self, shape, ROI=None):
        
        nz, nx, ny = shape
        
        if ROI is None:
            # not ROI is given take all coordinates
            ROI = torch.ones((1, nz, nx, ny)) > 0

        n_patches = np.ceil((np.array([nz, nx, ny]) - self.patch_size) / 
                            (self.overlap * self.patch_size)).astype(int) + 1

        # upper left corners of all patches
        if self.is_2d:
            z_list = np.arange(nz).astype(int).tolist()
        else:
            z_list = np.linspace(0, nz - self.patch_size[0], n_patches[0]).astype(int).tolist()
        x_list = np.linspace(0, nx - self.patch_size[1], n_patches[1]).astype(int).tolist()
        y_list = np.linspace(0, ny - self.patch_size[2], n_patches[2]).astype(int).tolist()
        
        zxy_list = []
        for z in z_list:
            for x in x_list:
                for y in y_list:
                    # we only predict the patch if the middle cube with half side length
                    # intersects the ROI
                    if self.is_2d:
                        z1, z2 = z, z+1
                    else:
                        z1, z2 = z+self.patch_size[0]//4, z+self.patch_size[0]*3//4
                    x1, x2 = x+self.patch_size[1]//4, x+self.patch_size[1]*3//4
                    y1, y2 = y+self.patch_size[2]//4, y+self.patch_size[2]*3//4
                    if ROI[0, z1:z2, x1:x2, y1:y2].any().item():
                        zxy_list.append((z, x, y))
                        
        return zxy_list

    def _sliding_window(self, volume, ROI=None):

        if not torch.is_tensor(volume):
            raise TypeError('Input must be torch tensor')
        if not len(volume.shape) == 4:
            raise ValueError('Volume must be a 4d tensor (incl channel axis)')

        # in case the volume is smaller than the patch size we pad it
        # and save the input size to crop again before returning
        shape_in = np.array(volume.shape)

        # %% possible padding of too small volumes
        pad = [0, self.patch_size[2] - shape_in[3], 0, self.patch_size[1] - shape_in[2],
               0, self.patch_size[0] - shape_in[1]]
        pad = np.maximum(pad, 0).tolist()
        volume = F.pad(volume, pad).type(torch.float)
        shape = volume.shape[1:]

        # %% reserve storage
        pred = torch.zeros((self.network.out_channels, *shape),
                           device=self.dev,
                           dtype=torch.float)
        # this is for the voxel where we have no prediction in the end
        # for each of those the method will return the (1,0,..,0) vector
        # pred[0] = 1
        ovlp = torch.zeros((1, *shape),
                           device=self.dev,
                           dtype=torch.float)
        
        # %% get all top left coordinates of patches
        zxy_list = self._get_xyz_list(shape, ROI)
        
        # introduce batch size
        # some people say that introducing a batch size at inference time makes it faster
        # I couldn't see that so far
        n_full_batches = len(zxy_list) // self.batch_size
        zxy_batched = [zxy_list[i * self.batch_size: (i + 1) * self.batch_size]
                       for i in range(n_full_batches)]

        if n_full_batches * self.batch_size < len(zxy_list):
            zxy_batched.append(zxy_list[n_full_batches * self.batch_size:])

        # %% now the magic!
        with torch.no_grad():
            for zxy_batch in zxy_batched:
                # crop
                batch = torch.stack([volume[:,
                                            z:z+self.patch_size[0],
                                            x:x+self.patch_size[1],
                                            y:y+self.patch_size[2]] for z, x, y in zxy_batch])

                # remove z axis if we have 2d prediction
                batch = batch[:, :, 0] if self.is_2d else batch
                # remember that the network is outputting a list of predictions for each scale
                if not self.fp32 and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        out = self.network(batch)[0]
                else:
                    out = self.network(batch)[0]

                # add z axis again maybe
                out = out.unsqueeze(2) if self.is_2d else out
                
                # update pred and overlap
                for i, (z, x, y) in enumerate(zxy_batch):
                    pred[:, z:z+self.patch_size[0], x:x+self.patch_size[1],
                         y:y+self.patch_size[2]] += F.softmax(out[i], 0) * self.patch_weight
                    ovlp[:, z:z+self.patch_size[0], x:x+self.patch_size[1],
                         y:y+self.patch_size[2]] += self.patch_weight

            # %% bring maybe back to old shape
            pred = pred[:, :shape_in[1], :shape_in[2], :shape_in[3]]
            ovlp = ovlp[:, :shape_in[1], :shape_in[2], :shape_in[3]]

            # set the prediction to background and prevent zero division where
            # we did not evaluate the network
            pred[0, ovlp[0] == 0] = 1
            ovlp[ovlp == 0] = 1

            # just to be sure
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
            return pred / ovlp

    def predict_volume(self, volume, ROI=None, mode=None):
        # evaluates the siliding window on this volume
        # predictions are returned as soft segmentations
        if mode is None:
            mode = self.mode

        if ROI is not None:
            ROI = maybe_add_channel_dim(ROI)

        self.network.eval()

        # check the type and bring to device
        is_np, _ = check_type(volume)
        if is_np:
            volume = torch.from_numpy(volume).to(self.dev)

        # check if inpt is 3d or 4d for the output
        volume = maybe_add_channel_dim(volume)

        if mode.lower() == 'simple':
            pred = self._predict_volume_simple(volume, ROI)
        elif mode.lower() == 'flip':
            pred = self._predict_volume_flip(volume, ROI)

        if is_np:
            pred = pred.cpu().numpy()

        return pred

    def __call__(self, volume, ROI=None, mode=None):
        return self.predict_volume(volume, ROI, mode)

    def _predict_volume_simple(self, volume, ROI=None):
        return self._sliding_window(volume, ROI)

    def _predict_volume_flip(self, volume, ROI=None):

        flip_z_list = [False] if self.is_2d else [False, True]
        
        if ROI is not None and isinstance(ROI, np.ndarray):
            ROI = torch.from_numpy(ROI)

        # collect all combinations of flipping
        flip_list = []
        for fz in flip_z_list:
            for fx in [False, True]:
                for fy in [False, True]:
                    flip_list.append((fz, fx, fy))

        # do the first one outside the loop for initialisation
        pred = self._sliding_window(volume, ROI=ROI)

        # now some flippings!
        sleep(0.05)
        for i in tqdm(range(len(flip_list[1:]))):
            sleep(0.01)
            f = flip_list[i]
            volume = self._flip_volume(volume, f)
            if ROI is not None:
                ROI = self._flip_volume(ROI, f)

            # predict flipped volume
            pred_flipped = self._sliding_window(volume, ROI)

            # flip back and update
            pred += self._flip_volume(pred_flipped, f)
            volume = self._flip_volume(volume, f)
            if ROI is not None:
                ROI = self._flip_volume(ROI, f)

        return pred / len(flip_list)

    def _flip_volume(self, volume, f):
        for i in range(3):
            if f[i]:
                volume = volume.flip(i+1)
        return volume


def get_2d_UNet(in_channels, out_channels, n_stages, filters=32):
    kernel_sizes = [3 for _ in range(n_stages)]
    return UNet(in_channels, out_channels, kernel_sizes, True)


def get_3d_UNet(in_channels, out_channels, n_stages, n_2d_blocks, filters=32):
    kernel_sizes = n_stages*[(3,3,3)]
    
    return UNet(in_channels, out_channels, kernel_sizes, False,filters_max=320,filters=filters,)

class SegmentationPostprocessing(object):

    def __init__(self,
                 apply_small_component_removing=False,
                 volume_thresholds=None,
                 remove_2d_comps=True,
                 remove_comps_by_volume=False,
                 mask_with_reg=False,
                 lb_classes=None,
                 use_fill_holes_2d=False,
                 use_fill_holes_3d=False,
                 keep_only_largest=False,
                 apply_morph_cleaning=False):
        self.apply_small_component_removing = apply_small_component_removing
        self.volume_thresholds = volume_thresholds
        self.remove_2d_comps = remove_2d_comps
        self.remove_comps_by_volume = remove_comps_by_volume
        self.mask_with_reg = mask_with_reg
        self.lb_classes = lb_classes
        self.apply_morph_cleaning = apply_morph_cleaning

        if self.apply_small_component_removing and \
                self.volume_thresholds is None:
            raise ValueError('No volume thresholds given.')
        if not isinstance(self.volume_thresholds, (list, tuple, np.ndarray)):
            self.volume_thresholds = [self.volume_thresholds]
        
        self.use_fill_holes_2d = use_fill_holes_2d
        self.use_fill_holes_3d = use_fill_holes_3d
        
        if self.lb_classes is not None:
            if isinstance(keep_only_largest, bool):
                
                self.keep_only_largest = len(self.lb_classes) * [keep_only_largest]
            
            elif isinstance(keep_only_largest, (tuple, list, np.ndarray)):
                
                assert len(keep_only_largest) == len(lb_classes)
                self.keep_only_largest = keep_only_largest
    
            else:
                raise TypeError('Received unexpected type for keep_only_largst '+str(type(keep_only_largest)))
        else:
            if not isinstance(keep_only_largest, (list, tuple, np.ndarray)):
                self.keep_only_largest = [keep_only_largest]

    def postprocess_volume(self, volume, reg=None, spacing=None, orig_shape=None):
        '''
        postprocess_volume(volume, orig_shape=None)
        Applies the following for post processing:
            - resizing to original voxel spacing (if given)
            - applying argmax to go from hard to soft labels
            - removing small connected components (if set to true)
        Parameters
        ----------
        volume : array tensor
            volume with soft segmentation/ output of the CNN
        orig_shape : len 3, optional
            if out_shape is given the volume is resized to original shape
            before any other postprocessing is done
        Returns
        -------
        postprocessed hard segmentation labels
        '''

        # first let's check if the input is right
        is_np, _ = check_type(volume)
        inpt_shape = np.array(volume.shape)
        if len(inpt_shape) != 4:
            raise ValueError('Expected 4d volume of shape '
                             '[n_channels, nx, ny, nz].')
        if self.mask_with_reg:
            if reg is None:
                raise ValueError('Trying to multiply the prediction with the reg of the '
                                 'previous stages, but no such array was given.')

            reg = maybe_add_channel_dim(reg)
        
        
        # first fun step: let's reshape to original size
        # before going to hard labels
        if orig_shape is not None:
            if np.any(orig_shape != inpt_shape):
                orig_shape = np.array(orig_shape)
                if torch.cuda.is_available():
                    with torch.no_grad():
                        if is_np:
                            volume = torch.from_numpy(volume).to('cuda').type(torch.float)
                        size = [int(s) for s in orig_shape]
                        volume = interpolate(volume.unsqueeze(0),
                                             size=size,
                                             mode='trilinear')[0]
                        if self.mask_with_reg:
                            if isinstance(reg, np.ndarray):
                                reg = torch.from_numpy(reg).to('cuda').type(torch.float)
                            reg = interpolate(reg.unsqueeze(0),
                                                   size=size,
                                                   mode='nearest')[0]
                else:
                    if not is_np:
                        volume = volume.cpu().numpy()
                    volume = np.stack([resize(volume[c], orig_shape, 1)
                                       for c in range(volume.shape[0])])
                    
                    if self.mask_with_reg:
                        if torch.is_tensor(reg):
                            reg = reg.cpu().numpy()
                        reg = np.stack([resize(reg[c], orig_shape, 0)
                                           for c in range(reg.shape[0])])

        # now change to CPU and numpy
        if torch.is_tensor(volume):
            volume = volume.cpu().numpy()
            
        volume = np.argmax(volume, 0).astype(np.float32)
        
        if self.apply_morph_cleaning:
            if not torch.is_tensor(volume):
                volume = torch.from_numpy(volume)
                if torch.cuda.is_available():
                    volume = volume.cuda()
            # this will work on GPU tensors
            volume = morph_cleaning(volume)

        if self.mask_with_reg:
            if torch.is_tensor(reg):
                reg = reg.cpu().numpy()

        if self.mask_with_reg:
            # now we're finally doing what we're asking the whole time about!
            volume *= reg[0]


        if self.apply_small_component_removing:
            # this can only be done on the CPU
            volume = self.remove_small_components(volume, spacing)

        volume = volume.astype(np.uint8)

        if self.lb_classes is not None:
            # now let's convert back from interger encoding to the classes
            volume_lb = np.zeros_like(volume)
            for i, c in enumerate(self.lb_classes):
                volume_lb[volume == i+1] = c
            volume = volume_lb
        
        # maybe we will the holes in the segmentations
        if self.use_fill_holes_3d:
            volume = self.fill_holes(volume, is_3d=True)
        elif self.use_fill_holes_2d:
            volume = self.fill_holes(volume, is_3d=False)
            
        # now we might keep only the largest component for some classes
        if np.any(self.keep_only_largest):
            volume = self.get_largest_component(volume)

        return volume

    def postprocess_data_tpl(self, data_tpl, prediction_key, reg=None):

        pred = data_tpl[prediction_key]

        spacing = data_tpl['spacing'] if 'spacing' in data_tpl else None

        if 'orig_shape' in data_tpl:
            # the data_tpl has preprocessed data.
            # predictions in both preprocessed and original shape will be added
            data_tpl[prediction_key] = self.postprocess_volume(pred,
                                                               reg,
                                                               spacing=spacing,
                                                               orig_shape=None)
            spacing = data_tpl['orig_spacing'] if 'orig_spacing' in data_tpl else None
            shape = data_tpl['orig_shape']
            data_tpl[prediction_key+'_orig_shape'] = self.postprocess_volume(pred,
                                                                             reg,
                                                                             spacing=spacing,
                                                                             orig_shape=shape)
        else:
            # in this case the data is not preprocessed
            orig_shape = data_tpl['image'].shape
            data_tpl[prediction_key] = self.postprocess_volume(pred,
                                                               reg,
                                                               spacing=spacing,
                                                               orig_shape=orig_shape)
        return data_tpl

    def remove_small_components(self, volume, spacing=None):
        if not isinstance(volume, np.ndarray):
            raise TypeError('Input must be np.ndarray')
        if not len(volume.shape) == 3:
            raise ValueError('Volume must be 3d array')

        if self.remove_comps_by_volume:
            if not isinstance(spacing,np.ndarray):
                raise ValueError('Spacing must be a list of length 3 to represent the spatial length '
                                 'of the voxel')

        # stores all coordinates of small components as 0 and rest as 1
        mask = np.ones_like(volume)

        num_classes = int(volume.max())
        if len(self.volume_thresholds) == 1:
            # if we only have one threshold we will apply it for all
            # lesion types
            thresholds = num_classes * self.volume_thresholds
        else:
            thresholds = self.volume_thresholds
            if len(self.volume_thresholds) < num_classes:
                raise ValueError('Less thresholds then fg classe given. '
                                 'Use either one threshold that is applied '
                                 'for all fg classes or')

        # we allow for different thresholds for the different lesions
        if self.remove_comps_by_volume:
            if self.remove_2d_comps:
                voxel_size = np.prod(spacing[1:])
            else:
                voxel_size = np.prod(spacing)
        else:
            # we remove the components by number of pixel
            voxel_size = 1
        
        if self.remove_2d_comps:
            for i, tr in enumerate(thresholds):
                
                for z in range(volume.shape[0]):
                    
                    components = label(volume[z] == i+1)
                    n_comps = components.max()
                    for j in range(1, n_comps + 1):
                        comp = components == j
                        if np.sum(comp) * voxel_size < tr :
                            mask[z][comp] = 0
        else:
            for i, tr in enumerate(thresholds):
                components = label(volume == i+1)
                n_comps = components.max()
                for j in range(1, n_comps + 1):
                    comp = components == j
                    if np.sum(comp) < tr * voxel_size:
                        mask[comp] = 0

        # done! The mask is 0 where all the undesired components are
        return mask * volume

    def fill_holes(self, volume, is_3d):
        
        if self.lb_classes is not None:
            for cl in self.lb_classes:
                
                if is_3d:
                    vol_filled = self.bin_fill_holes_3d((volume == cl).astype(volume.dtype))
                else:
                    vol_filled = self.bin_fill_holes_2d((volume == cl).astype(volume.dtype))
                
                volume[vol_filled > 0] = cl
            
            return volume
        else:
            lb_classes = list(range(1, volume.max()+1))
            for cl in lb_classes:
                
                if is_3d:
                    vol_filled = self.bin_fill_holes_3d((volume == cl).astype(volume.dtype))
                else:
                    vol_filled = self.bin_fill_holes_2d((volume == cl).astype(volume.dtype))
                
                volume[vol_filled > 0] = cl
            
            return volume

    def bin_fill_holes_2d(self, volume):
        
        assert len(volume.shape) == 3, 'expected 3d volume'
        
        return np.stack([binary_fill_holes(volume[z]) for z in range(volume.shape[0])], 0)

    def bin_fill_holes_3d(self, volume):
        
        assert len(volume.shape) == 3, 'expected 3d volume'
        
        return binary_fill_holes(volume)

    def get_largest_component(self, volume):
        
        
        if self.lb_classes is not None:
            for cl, keep in zip(self.lb_classes, self.keep_only_largest):
                
                if keep:
                    
                    largest = self.bin_get_largest_component((volume == cl).astype(volume.dtype))
                    
                    volume[volume == cl] == 0
                    volume[largest > 0] == cl
            
            return volume
        
        else:
            
            if not self.keep_only_largest[0]:
                return volume
            
            lb_classes = list(range(1, volume.max()))
            for cl in lb_classes:
                
                if keep:
                    
                    largest = self.bin_get_largest_component((volume == cl).astype(volume.dtype))
                    
                    volume[volume == cl] == 0
                    volume[largest > 0] == cl
            
            return volume

# %%
if __name__ == '__main__':
    gpu = torch.device('cuda')
    path = "/home/wcm23/SimStudy"
    path = os.path.join(path,"network_weights")
    sd = torch.load(path)
    
    net_3d = get_3d_UNet(1, 2, 4, 2)
    net_3d.load_state_dict(sd)
    net_3d = net_3d.to(gpu)
    
    a = SlidingWindowPrediction(net_3d,[96,96,96],batch_size=1,overlap=0.2)
    
    CT = np.load("C:\\Users\\mcgoug01\\Documents\\KiTS-00116.npy")
    pred = a(CT,mode='simple')
    
    
    
    post_process = SegmentationPostprocessing(apply_small_component_removing= True,lb_classes=[1],
                                              volume_thresholds = [200],
                                              remove_comps_by_volume=True,
                                              use_fill_holes_3d = True)
    seg = post_process.postprocess_volume(pred,spacing = np.array([2.75,0.8105,0.8105]))
    import matplotlib.pyplot as plt
    
    plt.subplot(211)
    plt.imshow(CT[0][5])
    plt.subplot(212)
    plt.imshow(seg[5])
    
    xb_3d = torch.randn((1, 1, 128, 128, 32), device=gpu)
    # xb_3d = torch.randn((1, 4, 96, 96, 96), device=gpu)
    print('3d')
    with torch.no_grad():
        yb_3d = net_3d(xb_3d)
    print('Output shapes:')
    for log in yb_3d:
        print(log.shape)
