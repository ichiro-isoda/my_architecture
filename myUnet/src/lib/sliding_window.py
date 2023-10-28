import numpy as np
import copy
import torch
from src.lib.utils import mirror_extension_image
from skimage import transform as tr
import torch.nn as nn

def sliding_window(x_batch,t,ndim,patchsize,model,gpu,resolution,loss=True):
    
    #================================
    #   calc stride and margin(sh)
    #================================

    ip_size = x_batch.shape
    stride = [int(psize/2) for psize in patchsize]
    sh = [int(st/2) for st in stride]


    #=======================================
    #  calc pad size
    #  window have to cover original size
    #   ip_size                         = segmentation area
    #   stride * stride_num + patchsize = from left-side to right-side of total window moved area
    #   -2 * sh                         = left and right side margin area
    #=======================================
    pad_size = []
    for axis in range(ndim):
        stride_num = 0
        while ip_size[axis] > stride[axis] * stride_num + patchsize[axis] - 2*sh[axis]: 
            stride_num += 1
        pad_size.append(stride[axis] * stride_num + patchsize[axis])
    pre_img = np.zeros(pad_size)

    if ndim == 2:
        x_batch = mirror_extension_image(image=x_batch, ndim=ndim, length=int(np.max(patchsize)))[patchsize[0]-sh[0]:patchsize[0]-sh[0]+pad_size[0], patchsize[1]-sh[1]:patchsize[1]-sh[1]+pad_size[1]]
        for y in range(0, pad_size[0]-stride[0], stride[0]):
            for x in range(0, pad_size[1]-stride[1], stride[1]):
                x_patch = torch.Tensor(np.expand_dims(x_batch[y:y+patchsize[0], x:x+patchsize[1]],0)) # add ch
                x_patch = torch.unsqueeze(x_patch, dim=0)  # add batch
                s_output = model(x=x_patch.to(gpu), t=None, seg=False)
                s_output = s_output.to('cpu').detach().numpy()
                pred = ((s_output[0][1] - s_output[0][0]) > 0) * 1
                # Add segmentation image
                pre_img[y:y+stride[0], x:x+stride[1]] += pred[sh[0]:-sh[0], sh[1]:-sh[1]]
    
        pred_img = pred_img[:ip_size[0], :ip_size[1]]
        
    elif ndim == 3:
        x_batch = mirror_extension_image(image=x_batch, ndim=ndim, length=int(np.max(patchsize)))[patchsize[0]-sh[0]:patchsize[0]-sh[0]+pad_size[0], patchsize[1]-sh[1]:patchsize[1]-sh[1]+pad_size[1], patchsize[2]-sh[2]:patchsize[2]-sh[2]+pad_size[2]]
        for z in range(0, pad_size[0]-stride[0], stride[0]):
            for y in range(0, pad_size[1]-stride[1], stride[1]):
                for x in range(0, pad_size[2]-stride[2], stride[2]):
                    x_patch = torch.Tensor(np.expand_dims(np.expand_dims(x_batch[z:z+patchsize[0], y:y+patchsize[1], x:x+patchsize[2]].astype(np.float32), axis=0),0)) # add ch
                    #x_patch = torch.unsqueeze(x_patch, dim=0)  # add batch
                    s_output = model(x=x_patch.to(gpu), t=None ,seg=False)
                    s_output = s_output.to('cpu').detach().numpy()
                    pred = ((s_output[0][1] - s_output[0][0]) > 0) * 1
                    # Add segmentation image
                    pre_img[z:z+stride[0], y:y+stride[1], x:x+stride[2]] += pred[sh[0]:-sh[0], sh[1]:-sh[1], sh[2]:-sh[2]]
                    
        pred_img = pred_img[0:ip_size[0], 0:ip_size[1], 0:ip_size[2]]

    if loss: 
        l = nn.functional.binary_cross_entropy(pred_img, t)
        pred_img = (pred_img > 0.5) * 1
        return l, pred_img
    else:
        pred_img = (pred_img > 0.5)*1
        return pred_img

