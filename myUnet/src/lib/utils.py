import numpy as np
import copy
import sys
def mirror_extension_image(image, ndim=3, length=10):
    # for 3D
    if ndim == 3:
        if image.ndim == 5:
            _, _, lz, ly, lx = image.shape
            exbox = np.pad(image[0, 0], pad_width=length, mode='reflect')
            return copy.deepcopy(exbox[:lz+length*2, :ly+length*2, :lx+length*2].reshape(1, 1, lz+length*2, ly+length*2, lx+length*2))
        elif image.ndim == 4:
            _, lz, ly, lx = image.shape
            exbox = np.pad(image[0], pad_width=length, mode='reflect')
            return copy.deepcopy(exbox[:lz+length*2, :ly+length*2, :lx+length*2].reshape(1, lz+length*2, ly+length*2, lx+length*2))
        elif image.ndim == 3:
            lz, ly, lx = image.shape
            exbox = np.pad(image, pad_width=length, mode='reflect')
            return copy.deepcopy(exbox[:lz+length*2, :ly+length*2, :lx+length*2])

    # for 2D
    elif ndim == 2:
        if image.ndim == 4:
            _, _, ly, lx = image.shape
            exbox = np.pad(image[0, 0], pad_width=length, mode='reflect')
            return copy.deepcopy(exbox[:ly+length*2, :lx+length*2].reshape(1, 1, ly+length*2, lx+length*2))
        elif image.ndim == 3:
            _, ly, lx = image.shape
            exbox = np.pad(image[0], pad_width=length, mode='reflect')
            return copy.deepcopy(exbox[:ly+length*2, :lx+length*2].reshape(1, ly+length*2, lx+length*2))
        elif image.ndim == 2:
            ly, lx = image.shape
            exbox = np.pad(image, pad_width=length, mode='reflect')
            return copy.deepcopy(exbox[:ly+length*2, :lx+length*2])

    else:
        print('Not corresponding to input image ndim in def mirror_extension_image()')
        sys.exit()