#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import numpy as np
import torch
from loguru import logger

import pickle


def set_random_seed(seed, device):
    # for reproducibility (always not guaranteed in pytorch)
    # [1] https://pytorch.org/docs/stable/notes/randomness.html
    # [2] https://hoya012.github.io/blog/reproducible_pytorch/

    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if device == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


def log_param(param):
    for key, value in param.items():
        if type(value) is dict:
            for in_key, in_value in value.items():
                logger.info('{:20}:{:>50}'.format(in_key, '{}'.format(in_value)))
        else:
            logger.info('{:20}:{:>50}'.format(key, '{}'.format(value)))

def pload(path):
    with open(path, 'rb') as f:
        res = pickle.load(f)
    print('load path = {} object'.format(path))
    return res

def pstore(x, path):
    with open(path, 'wb') as f:
        pickle.dump(x, f)
    print('store object in path = {} ok'.format(path))

def get_ii_constraint_mat(train_mat, 
                          num_neighbors, 
                          ii_diaganol_zero=False):
    print('Computing \\Omega for the item-item graph...')
    A = train_mat.T.dot(train_mat) # I * I
    n_items = A.shape[0]
    res_mat = torch.zeros((n_items, num_neighbors))
    res_sim_mat = torch.zeros((n_items, num_neighbors))
    if ii_diaganol_zero:
        A[range(n_items), range(n_items)] = 0
    items_D = np.sum(A, axis=0).reshape(-1)
    users_D = np.sum(A, axis=1).reshape(-1)

    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)
    all_ii_constraint_mat = torch.from_numpy(beta_uD.dot(beta_iD))
    for i in range(n_items):
        row = all_ii_constraint_mat[i] * torch.from_numpy(A.getrow(i).toarray()[0])
        row_sims, row_idxs = torch.topk(row, num_neighbors)
        res_mat[i] = row_idxs
        res_sim_mat[i] = row_sims
        if i % 15000 == 0:
            print('i-i constraint matrix {} ok'.format(i))
    
    print('Computation \\Omega OK!')
    return res_mat.long(), res_sim_mat.float()