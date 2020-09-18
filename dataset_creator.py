from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import noise
import torch
import os

PATH = '~/data/images'
for dirname, _, filenames in os.walk(PATH):
    for filename in filenames:
        print(os.path.join(dirname, filename))


class RiaeDataset(Dataset):
    def __init__(self, img_dir, pytorch=True):
        super().__init__()
        self.files = os.listdir(img_dir)
        self.pytorch = pytorch
        self.noise_dict = {'default': noise.default, 'speckle': noise.speckle, 'gauss': noise.gauss,
                           's_and_p': noise.s_and_p, 'poisson': noise.poisson, 'laplacian': noise.laplacian,
                           'invert_colors': noise.invert_colors, 'cartooning': noise.cartooning}

    def __repr__(self):
        s = 'Dataset class with {} files'.format(self.__len__())
        return s

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = torch.tensor(self.open_as_array(idx, invert=self.pytorch), dtype=torch.float32)
        y = torch.tensor(self.open_mask(idx, add_dims=False), dtype=torch.torch.int64)
        return x, y

    def open_as_array(self, idx, invert=False, noise_type='default'):
        raw_rgb = np.array(Image.open(self.files[idx]))
        if invert:
            raw_rgb = raw_rgb.transpose((2, 0, 1))

        raw_rgb = self.noise_dict[noise_type](raw_rgb)
        return raw_rgb / np.iinfo(raw_rgb.dtype).max  # Normalization

    def open_mask(self, idx, add_dims=False):
        raw_mask = np.array(Image.open(self.files[idx]))
        raw_mask = np.where(raw_mask == 255, 1, 0)
        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask

    def open_as_pil(self, idx):
        arr = 256 * self.open_as_array(idx)
        return Image.fromarray(arr.astype(np.uint8), 'RGB')
