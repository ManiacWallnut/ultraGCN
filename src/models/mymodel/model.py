#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch


class MyModel(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim):
        super(MyModel, self).__init__()

        # initialize variables
        self.in_dim = in_dim
        self.out_dim = out_dim

        # initialize layers
        self.linear = torch.nn.Linear(self.in_dim, self.out_dim, bias=True)
        torch.nn.init.normal_(self.linear.weight)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, features, labels):
        hypothesis = self.linear(features)
        loss = self.criterion(hypothesis, labels)

        return loss

    def predict(self, features):
        scores = self.linear(features)
        return torch.nn.functional.softmax(scores, dim=1)
