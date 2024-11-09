#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torchvision
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")


class MyDataset(Dataset):
    def __init__(self, data_path, train=True):
        self.train = train
        self.data = torchvision.datasets.MNIST(root=data_path,
                                               train=self.train,
                                               transform=torchvision.transforms.ToTensor(),
                                               download=True)

        _, self.width, self.height = self.data[0][0].shape
        self.in_dim = self.width * self.height
        self.out_dim = len(self.data.classes)

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, idx):
        return self.data.__getitem__(idx)

    def get_features(self):
        return self.data.train_data if self.train else self.data.data.float()

    def get_labels(self):
        return self.data.train_labels if self.train else self.data.targets
