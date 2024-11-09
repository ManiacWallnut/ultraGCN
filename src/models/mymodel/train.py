#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from models.mymodel.model import MyModel
from tqdm import tqdm
from utils import log_param
from loguru import logger


class MyTrainer:
    def __init__(self, device, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device

    def train_with_hyper_param(self, train_data, hyper_param, verbose=False):

        batch_size = hyper_param['batch_size']
        epochs = hyper_param['epochs']
        learning_rate = hyper_param['learning_rate']

        data_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  drop_last=True)

        total_batches = len(data_loader)

        model = MyModel(self.in_dim, self.out_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        model.train()
        pbar = tqdm(range(epochs), leave=False, colour='green', desc='epoch')
        for epoch in pbar:
            avg_loss = 0
            for features, labels in tqdm(data_loader, leave=False, colour='red', desc='batch'):
                # send data to a running device (GPU or CPU)
                features = features.view(-1, self.in_dim).to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                loss = model(features, labels)

                loss.backward()
                optimizer.step()

                avg_loss += loss / total_batches

            if verbose:
                pbar.write('Epoch {:02}: {:.4} training loss'.format(epoch, loss.item()))

        pbar.close()

        return model
