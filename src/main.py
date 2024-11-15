#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import torch
import fire
from pathlib import Path
import configparser
import os
import numpy as np

from utils import set_random_seed
from data import MyDataset
from models.mymodel.train import MyTrainer
from models.mymodel.eval import MyEvaluator
from utils import log_param
from loguru import logger
from ultra_data import UltraDataset
from utils import pload, pstore, get_ii_constraint_mat
from models.ultragcnmodel.train import UltraGCNTrainer

def run_mymodel(device, train_data, test_data, hyper_param):
    trainer = MyTrainer(device=device, in_dim=train_data.in_dim, out_dim=train_data.out_dim)
    model = trainer.train_with_hyper_param(train_data=train_data, hyper_param=hyper_param, verbose=True)
    evaluator = MyEvaluator(device=device)
    accuracy = evaluator.evaluate(model, test_data)
    return accuracy

def run_ultragcn(device, 
                 train_data, 
                 hyper_param, 
                 constraint_mat, 
                 ii_constraint_mat, 
                 ii_neighbor_mat,
                 early_stop_metric):
    trainer = UltraGCNTrainer(device)
    best_epoch, best_metric = trainer.train_with_hyper_param(train_data=train_data, 
                                                             hyper_param=hyper_param, 
                                                             constraint_mat=constraint_mat,
                                                             ii_constraint_mat=ii_constraint_mat, 
                                                             ii_neighbor_mat=ii_neighbor_mat, 
                                                             early_stop_metric=early_stop_metric,
                                                             verbose=True)
    return best_epoch, best_metric

def main(config_file,
         ii_neighbor_num: int,
         gamma: float,
         lambda_: float,
         early_stop_metric: str='recall',
         tuning: bool=False):
    """
    Handle user arguments of UltraGCN

    :hyper_param config_file: The path of the configuration file
    :hyper_param ii_neighbor_num: number of selected neighbors for each item
    :hyper_param gamma: adjust the relative importance of item-item relationship
    :hyper_param lambda_: adjust the relative importance of user-item relationship
    :hyper_param early_stop_metric: the metric used for early stopping
    :hyper_param tuning: whether the model is in tuning mode
    """

    config = configparser.ConfigParser()
    config.read(config_file)

    logger.info("The main procedure has started with the following parameters:")
    gpu = config['Training']['gpu']
    device = torch.device('cuda:'+gpu if torch.cuda.is_available() else 'cpu')
    seed = config.getint('Training', 'seed')
    set_random_seed(seed=seed, device=device)

    hyper_param = {
        'model': config['Model']['model'],
        'seed': seed,
        'device': device
    }
    log_param(hyper_param)

    dataset = config['Training']['dataset']
    hyper_param['dataset'] = dataset
    data_path = config['Training']['file_path']
    
    if hyper_param['model'] == 'mymodel':
        train_data = MyDataset(data_path=data_path, train=True)
        test_data = MyDataset(data_path=data_path, train=False)
        logger.info("The datasets are loaded where their statistics are as follows:")
        logger.info("- # of training instances: {}".format(len(train_data)))
        logger.info("- # of test instances: {}".format(len(test_data)))
        hyper_param = {
            'batch_size': config.getint('Training', 'batch_size'),
            'max_epoch': config.getint('Model', 'max_epoch'),
            'learning_rate': config.getfloat('Training', 'learning_rate')
        }
        log_param(hyper_param)
        accuracy = run_mymodel(device=device, train_data=train_data, test_data=test_data, hyper_param=hyper_param)

    elif hyper_param['model'] == 'ultragcn':
        train_data = UltraDataset(data_path=data_path, train=True)
        test_data = UltraDataset(data_path=data_path, train=False)
        train_mat = train_data.get_train_matrix()
        user_num, item_num = train_data.get_user_item_counts()
        constraint_mat = train_data.get_constraint_matrix()
        hyper_param = {
            'user_num': user_num,
            'item_num': item_num,
            'batch_size': config.getint('Training', 'batch_size'),
            'max_epoch': config.getint('Model', 'max_epoch'),
            'learning_rate': config.getfloat('Training', 'learning_rate'),
            'embedding_dim': config.getint('Model', 'embedding_dim'),
            'ii_neighbor_num': ii_neighbor_num,
            'model_save_path': config['Model']['model_save_path'],
            'enable_tensorboard': config.getboolean('Model', 'enable_tensorboard'),
            'initial_weight': config.getfloat('Model', 'initial_weight'),
            'early_stop_epoch': config.getint('Training', 'early_stop_epoch'),
            'w1': config.getfloat('Training', 'w1'),
            'w2': config.getfloat('Training', 'w2'),
            'w3': config.getfloat('Training', 'w3'),
            'w4': config.getfloat('Training', 'w4'),
            'negative_num': config.getint('Training', 'negative_num'),
            'negative_weight': config.getfloat('Training', 'negative_weight'),
            'gamma': gamma,
            'lambda': lambda_,
            'sampling_sift_pos': config.getboolean('Training', 'sampling_sift_pos'),
            'test_batch_size': config.getint('Testing', 'test_batch_size'),
            'topk': config.getint('Testing', 'topk'),
            'constraint_mat': constraint_mat
        }
        if os.path.exists('../{}_ii_constraint_mat'.format(dataset)):
            ii_constraint_mat = pload('../{}_ii_constraint_mat'.format(dataset))
            ii_neighbor_mat = pload('../{}_ii_neighbor_mat'.format(dataset))
        else:
            ii_neighbor_mat, ii_constraint_mat = get_ii_constraint_mat(train_mat, hyper_param['ii_neighbor_num'])
            pstore(ii_neighbor_mat, '../{}_ii_neighbor_mat'.format(dataset))
            pstore(ii_constraint_mat, '../{}_ii_constraint_mat'.format(dataset))

        interacted_items, mask = train_data.get_interacted_items()
        hyper_param.update({'interacted_items': interacted_items, 
                            'mask': mask})

        if tuning:
            valid_ground_truth_list = train_data.get_valid_ground_truth_list()
            hyper_param['test_ground_truth_list'] = valid_ground_truth_list
        else:
            test_ground_truth_list = test_data.get_test_ground_truth_list()
            hyper_param['test_ground_truth_list'] = test_ground_truth_list

        best_epoch, best_metric = run_ultragcn(device=device, 
                                               train_data=train_data, 
                                               hyper_param=hyper_param, 
                                               constraint_mat=constraint_mat, 
                                               ii_constraint_mat=ii_constraint_mat, 
                                               ii_neighbor_mat=ii_neighbor_mat,
                                               early_stop_metric=early_stop_metric)

    else:
        logger.error("The given \"{}\" is not supported...".format(hyper_param['model']))
        return
    
    logger.info("The model has been trained. The best_epoch is {}, The best_metric is {}.".format(best_epoch, best_metric))

if __name__ == "__main__":
    fire.Fire(main)
