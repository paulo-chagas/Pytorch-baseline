import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io
import cv2
from torchvision import transforms


class CustomDataset(Dataset):
    """Custom dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        class_index, class_label, inner_group, name = (
            self.annotations.iloc[index, 3],
            self.annotations.iloc[index, 2],
            self.annotations.iloc[index, 1],
            self.annotations.iloc[index, 0])
        img_path = os.path.join(self.root_dir,
                                class_label,
                                inner_group,
                                name)
        try:
            image = io.imread(img_path)
            # image = cv2.imread(img_path)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception:
            print('Error loading file {}'.format(img_path))
            sys.exit(1)

        y = torch.zeros(1).long()
        y = class_index

        if self.transform:
            image = self.transform(image=image)["image"]
            image = transforms.ToTensor()(image)

        return (image, y)


class GenericDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = sorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        try:
            # image = io.imread(img_loc)
            image = cv2.imread(img_loc)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception:
            print('Error loading file {}'.format(img_loc))
            sys.exit(1)
        if self.transform:
            image = self.transform(image=image)["image"]
            image = transforms.ToTensor()(image)
        return image
