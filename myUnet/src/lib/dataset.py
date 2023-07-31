import skimage.io as io
import skimage.transform as tr
from torch.utils.data import Dataset
import numpy as np
import sys
import os
import random

def get_dataset(args):
    train_dataset = PreprocessedDataset(
        root_path=args.root_path,
        gt_path=args.gt_path,
        split_list=args.split_list_train,
        train=True,
        model=args.model,
        arr_type=args.input_format,
        augmentation=True,
        scaling=args.scaling,
        resolution=eval(args.resolution),
        crop_size=eval(args.patch_size),
        ndim=args.ndim
    )
    validation_dataset = PreprocessedDataset(
        root_path=args.root_path,
        gt_path=args.gt_path,
        split_list=args.split_list_validation,
        train=False,
        model=args.model,
        arr_type=args.input_format,
        augmentation=False,
        scaling=args.scaling,
        resolution=eval(args.resolution),
        crop_size=eval(args.patch_size),
        ndim=args.ndim
    )
    return train_dataset, validation_dataset

class PreprocessedDataset(Dataset):

    def __init__(
        self,
        root_path,
        gt_path,
        split_list,
        train=True,
        arr_type='npz',
        augmentation=True,
        scaling='min-max',
        resolution=eval('(1.0, 1.0, 3.0)'),
        crop_size=eval('(128, 128, 128)'),
        ndim=3
        ):
        self.root_path = root_path
        self.gt_path = gt_path
        self.arr_type = arr_type
        self.augmentation = augmentation
        self.scaling = scaling
        self.resolution = resolution
        self.crop_size = crop_size
        self.train = train
        self.ndim = ndim

        with open(split_list, 'r') as f:
            self.img_path = f.read().split()

    def __len__(self):
        return len(self.img_path)

    def read_img(path, arr_type='npz'):
        """ read image array from path
        Args:
            path (str)          : path to directory which images are stored.
            arr_type (str)      : type of reading file {'npz','jpg','png','tif'}
        Returns:
            image (np.ndarray)  : image array
        """
        if arr_type == 'npz':
            image = np.load(path)['arr_0']
        elif arr_type in ('png', 'jpg'):
            image = io.imread(path, mode='L')
        elif arr_type == 'tif':
            image = io.imread(path)
        else:
            raise ValueError('invalid --input_type : {}'.format(arr_type))

        return image.astype(np.int32)
    def _get_image_2d(self, i):
        # =================================================
        #  method for reading and preprocessing raw image
        # =================================================
        image = self.read_img(os.path.join(self.root_path, 'raw', (self.img_path[i][0:-8]+ '.tif')), self.arr_type)
        #image = read_img(os.path.join(self.root_path, 'raw_fl', self.img_path[i]), self.arr_type)
        if len(image.shape) == 3:
            image = image.reshape(image.shape[1],image.shape[2])
        ip_size = (int(image.shape[0] * self.resolution[1]), int(image.shape[1] * self.resolution[1]))
        # resize
        image = tr.resize(image.astype(np.float32), ip_size, order=1, preserve_range=True)
        pad_size = np.max(np.array(self.crop_size) - np.array(ip_size))
        # padding
        if pad_size > 0:
            image = np.pad(image, pad_width=pad_size, mode='reflect')
        # scaling 
        if self.scaling == 'min-max':
            image = (image - image.min()) / (image.max() - image.min())
        elif self.scaling == 'z-scale':
            image = (image - image.mean()) / image.std()

        return image.astype(np.float32)

    def _get_label__(self, i):
        # ==========================
        #  method for reading gt img 
        # ==========================
        label_file = open(os.path.join(self.gtpath,'gt_volume', (self.img_path[i][0:-8]+ '.csv')))
        label = float(label_file.read())
        return label
    
    def augmentation_2d(image):
        aug_flag = random.randint(0, 3)
        image = np.rot90(image, k=aug_flag)
        return image

    def __getitem__(self, i):
        # =========================================
        #  this method is called in each iteration 
        # =========================================
        with open('reading_img.txt', 'a') as f:
            f.write(self.img_path[i])
            f.write('\n')
        if self.ndim == 2:
            image = self._get_image_2d(i)
            label = self._get_label__(i)
        elif self.ndim == 3:
            print("ndim=3 is not supprted yet")
            #image = self._get_image_3d(i)
            #label = self._get_label_3d(i)
        else:
            print('Error: ndim must be 2 or 3 dimensions')
        if self.train:
            if self.ndim == 2:
                x= self.augmentation_2d(image)
            elif self.ndim == 3:
                print("ndim=3 is not supported")
            return np.expand_dims(x.astype(np.float32), axis=0), label
        else:
            return image.astype(np.float32), label