# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset
import warnings
import random

warnings.filterwarnings("ignore")

class UltraDataset(Dataset):
    def __init__(self, data_path, train=True):
        # Initialize dataset parameters
        self.train = train
        self.data = []
        self.n_user, self.m_item = 0, 0
        self.dataSize = 0
        self.constraint_mat = {}
        self.train_mat = None

        # Define file paths based on `train` flag
        # file_path = f"{data_path}/{'train.txt' if train else 'test.txt'}"
        file_path = data_path

        # Load data from the selected file
        self._load_data(file_path)

    def _load_data(self, file_path):
        uniqueUsers, items, users = [], [], []

        with open(file_path, 'r') as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    try:
                        item_list = [int(i) for i in l[1:]]
                    except:
                        item_list = []
                    uid = int(l[0])
                    uniqueUsers.append(uid)
                    users.extend([uid] * len(item_list))
                    items.extend(item_list)
                    try:
                        self.m_item = max(self.m_item, max(item_list, default=0))
                    except:
                        self.m_item = self.m_item
                    self.n_user = max(self.n_user, uid)
                    self.dataSize += len(item_list)

        # Convert to arrays and adjust matrix size
        uniqueUsers = np.array(uniqueUsers)
        users = np.array(users)
        items = np.array(items)
        self.n_user += 1
        self.m_item += 1

        # Populate self.data with user-item pairs
        self.data = list(zip(users, items))

        # Create sparse matrix if in training mode
        if self.train:
            self.train_mat = sp.dok_matrix((self.n_user, self.m_item), dtype=np.float32)
            for x in self.data:
                self.train_mat[x[0], x[1]] = 1.0

            # Calculate constraint matrix
            items_D = np.sum(self.train_mat, axis=0).reshape(-1)
            users_D = np.sum(self.train_mat, axis=1).reshape(-1)

            beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
            beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)

            self.constraint_mat = {
                "beta_uD": torch.from_numpy(beta_uD).reshape(-1),
                "beta_iD": torch.from_numpy(beta_iD).reshape(-1)
            }

            # Initialize mask and interacted items for training
            self.mask = torch.zeros(self.n_user, self.m_item)
            self.interacted_items = [[] for _ in range(self.n_user)]
            for (u, i) in self.train_data:
                self.mask[u][i] = -np.inf
                self.interacted_items[u].append(i)

    def __len__(self):
        return len(self.train_data) if self.train else len(self.data)

    def __getitem__(self, idx):
        if self.train:
            return self.train_data[idx]
        return self.data[idx]

    def get_train_matrix(self):
        return self.train_mat

    def get_constraint_matrix(self):
        return self.constraint_mat

    def get_user_item_counts(self):
        return self.n_user, self.m_item

    def get_interacted_items(self):
        if self.train:
            return self.interacted_items, self.mask
        else:
            return self.test_interacted_items, self.mask

    def get_test_ground_truth_list(self):
        if not self.train and not hasattr(self, 'test_ground_truth_list'):
            self.test_ground_truth_list = [[] for _ in range(self.n_user)]
            for (u, i) in self.data:
                self.test_ground_truth_list[u].append(i)
        return self.test_ground_truth_list
