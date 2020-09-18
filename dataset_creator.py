from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import itertools
import noise
import torch
import os


class RiaeDataset(Dataset):
    def __init__(self, img_dir, pytorch=False):
        super().__init__()
        self.img_dir = img_dir+'/'
        self.files = os.listdir(img_dir)
        self.pytorch = pytorch
        self.noise_dict = {0: noise.default, 1: noise.speckle, 2: noise.gauss, 3: noise.s_and_p, 4: noise.poisson,
                           5: noise.laplacian, 6: noise.invert_colors, 7: noise.cartooning, 8: noise.double_saturation}

        self.num_noises = len(self.noise_dict)
        noise_list = [[i for i in range(self.num_noises)] for _ in range(len(self.files))]
        self.noise_list = list(itertools.chain.from_iterable(noise_list))

    def __repr__(self):
        s = 'Dataset class with {} files'.format(self.__len__())
        return s

    def __len__(self):
        return len(self.files)*self.num_noises

    def __getitem__(self, idx):
        x = torch.tensor(self.open_as_array(idx, invert=self.pytorch), dtype=torch.float32)
        y = torch.tensor(self.open_gt(idx, add_dims=False), dtype=torch.torch.int64)
        return x, y

    def open_as_array(self, idx, invert=False):
        raw_rgb = np.array(Image.open(self.img_dir+self.files[int(idx/self.num_noises)]))
        if invert:
            raw_rgb = raw_rgb.transpose((2, 0, 1))
        noise_type = self.noise_list[idx]
        raw_rgb = self.noise_dict[noise_type](raw_rgb)
        return raw_rgb / 255  # Normalization

    def open_gt(self, idx, add_dims=False):
        raw_rgb = np.array(Image.open(self.img_dir+self.files[int(idx/self.num_noises)]))
        return np.expand_dims(raw_rgb, 0) if add_dims else raw_rgb

    def open_as_pil(self, idx):
        arr = 256 * self.open_as_array(idx)
        return Image.fromarray(arr.astype(np.uint8), 'RGB')
