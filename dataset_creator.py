from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os


class RiVAEDataset(Dataset):
    def __init__(self, img_dir, img_shape):
        super().__init__()
        self.img_dir = f"{img_dir}/"
        self.files = os.listdir(img_dir)

        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_shape[2], img_shape[3]))
        ])

    def __repr__(self):
        s = f"Dataset class with {self.__len__()} files"
        return s

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = self.trans(self.open_as_pil(idx))
        return x

    def open_as_array(self, idx):
        raw_rgb = np.array(Image.open(self.img_dir+self.files[idx]))
        return raw_rgb/255

    def open_as_pil(self, idx):
        arr = 255 * self.open_as_array(idx)
        return Image.fromarray(arr.astype(np.uint8), 'RGB')
