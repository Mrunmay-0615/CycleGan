import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class HorseZebraDataset(Dataset):

    def __init__(self, root_horse, root_zebra, transform=None):
        super().__init__()
        self.root_horse = root_horse
        self.root_zebra = root_zebra

        self.horse_images = os.listdir(self.root_horse)
        self.zebra_images = os.listdir(self.root_zebra)

        self.length = max(len(self.horse_images), len(self.zebra_images))
        self.horse_len = len(self.horse_images)
        self.zebra_len = len(self.zebra_images)

        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        zebra_img = Image.open(os.path.join(self.root_zebra, self.zebra_images[index % self.zebra_len])).convert('RGB')
        horse_img = Image.open(os.path.join(self.root_horse, self.horse_images[index % self.horse_len])).convert('RGB')

        zebra_img = np.array(zebra_img)
        horse_img = np.array(horse_img)

        if self.transform:
            augmentations = self.transform(image=zebra_img, image0=horse_img)
            zebra_img = augmentations['image']
            horse_img = augmentations['image0']

        return zebra_img, horse_img

