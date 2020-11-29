from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os


class RivaeDataset(Dataset):
    def __init__(self, img_dir, pytorch=False):
        super().__init__()
        self.img_dir = f"{img_dir}/"
        self.files = os.listdir(img_dir)
        self.pytorch = pytorch

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((120, 100)),
        ])

    def __repr__(self):
        s = f"Dataset class with {self.__len__()} files"
        return s

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = self.transform(self.open_as_array(idx, invert=self.pytorch))
        y = self.transform(self.open_gt(idx, add_dims=False))
        return x, y

    def open_as_array(self, idx, invert=False):
        raw_rgb = np.array(Image.open(self.img_dir+self.files[idx]))
        if invert:
            raw_rgb = raw_rgb.transpose((2, 0, 1))
        return raw_rgb / 255  # Normalization

    def open_gt(self, idx, add_dims=False):
        raw_rgb = np.array(Image.open(self.img_dir+self.files[idx]))
        return np.expand_dims(raw_rgb, 0) if add_dims else raw_rgb

    def open_as_pil(self, idx):
        arr = 256 * self.open_as_array(idx)
        return Image.fromarray(arr.astype(np.uint8), 'RGB')
