#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from models.ultragcnmodel.model import UltraGCN
from tqdm import tqdm
from utils import log_param
from loguru import logger
import time
import numpy as np
import csv
import os

from models.ultragcnmodel.eval import test
from torch.utils.tensorboard import SummaryWriter


class UltraGCNTrainer:
    def __init__(self, device):
        self.device = device
        self.recall_best_epoch,self.ndcg_best_epoch = 0, 0
        self.best_recall, self.best_ndcg = 0, 0
        self.best_metric = 0
        self.early_stop_count = 0
        self.early_stop = False

    def train_with_hyper_param(self, 
                               train_data,
                               hyper_param,
                               constraint_mat,
                               ii_constraint_mat,
                               ii_neighbor_mat,
                               early_stop_metric='recall', 
                               verbose=False):

        batch_size = hyper_param['batch_size']
        epochs = hyper_param['max_epoch']
        learning_rate = hyper_param['learning_rate']

        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=5)
        
        batches = len(train_loader.dataset) // hyper_param['batch_size']
        if len(train_loader.dataset) % hyper_param['batch_size'] != 0:
            batches += 1
        print('Total training batches = {}'.format(batches))

        if hyper_param['enable_tensorboard']:
            writer = SummaryWriter()
        
        ultragcn = UltraGCN(hyper_param, constraint_mat, ii_constraint_mat, ii_neighbor_mat).to(self.device)
        optimizer = torch.optim.Adam(ultragcn.parameters(), lr=learning_rate)

        for epoch in range(hyper_param['max_epoch']):
            avg_loss = 0
            ultragcn.train()
            start_time = time.time()

            # x: tensor:[users, pos_items]
            # for batch, x in tqdm(train_loader, leave=False, colour='red', desc='batch'):
            for batch, x in enumerate(train_loader):
                users, pos_items, neg_items = self.Sampling(x,
                                                            hyper_param['item_num'],
                                                            hyper_param['negative_num'],
                                                            hyper_param['interacted_items'],
                                                            hyper_param['sampling_sift_pos'])
                users = users.to(self.device)
                pos_items = pos_items.to(self.device)
                neg_items = neg_items.to(self.device)

                ultragcn.zero_grad()
                loss = ultragcn(users, pos_items, neg_items)
                if hyper_param['enable_tensorboard']:
                    writer.add_scalar('Loss/train', loss, epoch * batches + batch)
                loss.backward()
                optimizer.step()

                avg_loss += loss / batches

            train_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
            if hyper_param['enable_tensorboard']:
                writer.add_scalar('Loss/train', loss, epoch)

            # if verbose:
            #     pbar.write('Epoch {:02}: {:.4} training loss'.format(epoch, loss.item()))

            need_test = True
            if epoch < 50 and epoch % 5 != 0:
                need_test = False

            
            if need_test:
                start_time = time.time()
                test_loader = torch.utils.data.DataLoader(list(range(hyper_param['user_num'])),
                                                          batch_size=hyper_param['test_batch_size'],
                                                          shuffle=False,
                                                          num_workers=5)
                F1_score, Precision, Recall, NDCG = test(ultragcn,
                                                         test_loader,
                                                         hyper_param['test_ground_truth_list'],
                                                         hyper_param['mask'],
                                                         hyper_param['topk'],
                                                         hyper_param['user_num'])
                if hyper_param['enable_tensorboard']:
                    writer.add_scalar(f'Results/{early_stop_metric}@20', 
                                      Recall if early_stop_metric == 'recall' else NDCG, 
                                      epoch)
                test_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))

                print('The time for epoch {} is: train time = {}, test time = {}'.format(epoch, train_time, test_time))
                print("Loss = {:.4f}, F1-score: {:.4f} \t Precision: {:.4f}\t Recall: {:.4f}\tNDCG: {:.4f}".format(loss.item(), F1_score, Precision, Recall, NDCG))
                # print("Loss = {}, F1-score: {} \t Precision: {}\t Recall: {}\tNDCG: {}".format(loss.item(), F1_score, Precision, Recall, NDCG))

                csv_path = '../csv/exp_param.csv'.format()
                print('The results will save to {}'.format(csv_path))
                header = [
                    "Dataset",
                    "No"
                    "Item-Item N",
                    "Lambda",
                    "Gamma",
                    "Recall@20 Validation",
                    "Recall@20 Test",
                    "NDCG@20 Validation",
                    "NDCG@20 Test"
                ]

                dataset_name = hyper_param['dataset']
                if not os.path.exists(csv_path):
                    experiment_no = 1
                with open(csv_path, mode='r', encoding='utf-8') as file:
                    experiment_no = sum(1 for _ in file)
                item_item_n = hyper_param['ii_neighbor_num']
                lambda_val = hyper_param['lambda']
                gamma_val = hyper_param['gamma']
                if early_stop_metric == 'recall':
                    recall_val = Recall
                else:
                    recall_val = 0
                if early_stop_metric == 'ndcg':
                    ndcg_val = NDCG
                else:
                    ndcg_val = 0

                row = [
                    dataset_name,
                    experiment_no,
                    item_item_n,
                    lambda_val,
                    gamma_val,
                    recall_val,
                    ndcg_val
                ]

                # Write to CSV
                try:
                    # Check if the file exists
                    file_exists = os.path.exists(csv_path)
                    with open(csv_path, 
                              mode='a', 
                              newline='', 
                              encoding='utf-8') as file:
                        writer = csv.writer(file)
                        if not file_exists:
                            writer.writerow(header)
                        writer.writerow(row)
                    print(f'Results saved to {csv_path}')
                except Exception as e:
                    print(f'Error saving results to {csv_path}: {e}')

                if early_stop_metric == 'recall':
                    metric = Recall
                elif early_stop_metric == 'ndcg':
                    metric = NDCG
                else:
                    raise ValueError('The given metric is not supported...')

                if metric > self.best_metric:
                    self.best_metric = metric
                    self.best_epoch = epoch
                    self.early_stop_count = 0
                    torch.save(ultragcn.state_dict(), hyper_param['model_save_path'])
                else:
                    self.early_stop_count += 1
                    if self.early_stop_count >= hyper_param['early_stop_epoch']:
                        self.early_stop = True
            
            if self.early_stop:
                print('##########################################')
                print('Early stop is triggered at {} epochs.'.format(epoch))
                print('Results:')
                tested_metric = 'Recall' if early_stop_metric == 'recall' else 'NDCG'
                print('best epoch = {}, best {} = {}'.format(self.best_epoch, 
                                                             tested_metric,
                                                             self.best_metric))
                print('The best model is saved at {}'.format(hyper_param['model_save_path']))

                csv_path = '../csv/exp_param.csv'.format()
                print('The results will save to {}'.format(csv_path))
                header = [
                    "Dataset",
                    "No"
                    "Item-Item N",
                    "Lambda",
                    "Gamma",
                    "Recall@20 Validation",
                    "Recall@20 Test",
                    "NDCG@20 Validation",
                    "NDCG@20 Test"
                ]

                dataset_name = hyper_param['dataset']
                if not os.path.exists(csv_path):
                    experiment_no = 1
                with open(csv_path, mode='r', encoding='utf-8') as file:
                    experiment_no = sum(1 for _ in file)
                item_item_n = hyper_param['ii_neighbor_num']
                lambda_val = hyper_param['lambda']
                gamma_val = hyper_param['gamma']
                if early_stop_metric == 'recall':
                    recall_val = self.best_metric
                if early_stop_metric == 'ndcg':
                    ndcg_val = self.best_metric

                row = [
                    dataset_name,
                    experiment_no,
                    item_item_n,
                    lambda_val,
                    gamma_val,
                    recall_val,
                    ndcg_val
                ]

                # Write to CSV
                try:
                    # Check if the file exists
                    file_exists = os.path.exists(csv_path)
                    with open(csv_path, 
                              mode='a', 
                              newline='', 
                              encoding='utf-8') as file:
                        writer = csv.writer(file)
                        if not file_exists:
                            writer.writerow(header)
                        writer.writerow(row)
                    print(f'Results saved to {csv_path}')
                except Exception as e:
                    print(f'Error saving results to {csv_path}: {e}')
                break

        writer.flush()

        print('Training end!')

        return self.best_epoch, self.best_metric

    def Sampling(self,
                 pos_train_data, 
                 item_num, 
                 neg_ratio, 
                 interacted_items,
                 sampling_sift_pos):
        neg_candidates = np.arange(item_num)

        if sampling_sift_pos:
            neg_items = []
            for u in pos_train_data[0]:
                probs = np.ones(item_num)
                probs[interacted_items[u]] = 0
                probs /= np.sum(probs)

                u_neg_items = np.random.choice(neg_candidates,
                                               size=neg_ratio,
                                               p = probs,
                                               replace=True).reshape(1, -1)
                
                neg_items.append(u_neg_items)

            neg_items = np.concatenate(neg_items, axis=0)
        else:
            neg_items = np.random.choice(neg_candidates,
                                         (len(pos_train_data[0]), neg_ratio),
                                         replace=True)
            
        neg_items = torch.from_numpy(neg_items)

        # users, pos_items, neg_items
        return pos_train_data[0], pos_train_data[1], neg_items