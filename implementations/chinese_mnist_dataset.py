import pandas as pd
import os

from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader

class ChineseMNISTDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.img_labels = pd.read_csv('../../data/chinese_mnist/chinese_mnist.csv')
        self.img_dir = '../../data/chinese_mnist/data/data'
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        suite_id,sample_id,code,value, _ = self.img_labels.iloc[idx]
        img_name = 'input_{0}_{1}_{2}.jpg'.format(suite_id,sample_id, code)
        img_path = os.path.join(self.img_dir, img_name)
        image = read_image(img_path)
        label = value
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

