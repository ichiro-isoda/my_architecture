import numpy as np
import copy
import sys
def mirror_extension_image(image, ndim=3, length=10):
    # for 3D
    if ndim == 3:
        if image.ndim == 5:
            batch, ch, lz, ly, lx = image.shape
            exbox = np.array([np.pad(image[0, c], pad_width=length, mode='reflect') for c in range(ch)])
            return copy.deepcopy(exbox[:, :lz+length*2, :ly+length*2, :lx+length*2])
        elif image.ndim == 4:
            ch, lz, ly, lx = image.shape
            exbox = np.array([np.pad(image[c], pad_width=length, mode='reflect') for c in range(ch)])
            return copy.deepcopy(exbox[:lz+length*2, :ly+length*2, :lx+length*2])
        elif image.ndim == 3:
            lz, ly, lx = image.shape
            exbox = np.pad(image, pad_width=length, mode='reflect')
            return copy.deepcopy(exbox[:lz+length*2, :ly+length*2, :lx+length*2])

    # for 2D
    elif ndim == 2:
        if image.ndim == 4:
            batch, ch, ly, lx = image.shape
            exbox = np.array([np.pad(image[0, c], pad_width=length, mode='reflect') for c in range(ch)])
            return copy.deepcopy(exbox[:ly+length*2, :lx+length*2])
        elif image.ndim == 3:
            ch, ly, lx = image.shape
            exbox = np.array([np.pad(image[c], pad_width=length, mode='reflect') for c in range(ch)])
            return copy.deepcopy(exbox[:ly+length*2, :lx+length*2])
        elif image.ndim == 2:
            ly, lx = image.shape
            exbox = np.pad(image, pad_width=length, mode='reflect')
            return copy.deepcopy(exbox[:ly+length*2, :lx+length*2])

    else:
        print('Not corresponding to input image ndim in def mirror_extension_image()')
        sys.exit()
