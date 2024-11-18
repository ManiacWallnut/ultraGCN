#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import torch
import fire
from pathlib import Path
import configparser
import os
import numpy as np
import csv

from utils import set_random_seed
from data import MyDataset
from models.mymodel.train import MyTrainer
from models.mymodel.eval import MyEvaluator
from utils import log_param
from loguru import logger
from ultra_data import UltraDataset
from utils import pload, pstore, get_ii_constraint_mat
from models.ultragcnmodel.train import UltraGCNTrainer
from models.ultragcnmodel.eval import test

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
    best_epoch, best_metric, model = trainer.train_with_hyper_param(train_data=train_data, 
                                                             hyper_param=hyper_param, 
                                                             constraint_mat=constraint_mat,
                                                             ii_constraint_mat=ii_constraint_mat, 
                                                             ii_neighbor_mat=ii_neighbor_mat, 
                                                             early_stop_metric=early_stop_metric)
    return best_epoch, best_metric, model

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

    # dataset name e.g., amazon_book
    dataset = config['Training']['dataset']

    # directory of the dataset
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
        if tuning:
            tuning_train_path = '../datasets/AmazonBooks_m1/tuning_train.txt'
            train_data = UltraDataset(data_path=tuning_train_path, train=True)
            tuning_validate_path = '../datasets/AmazonBooks_m1/tuning_validate.txt'
            test_data = UltraDataset(data_path=tuning_validate_path, train=False)
        else:
            train_path = '../datasets/AmazonBooks_m1/train.txt'
            train_data = UltraDataset(data_path=train_path, train=True)
            test_path = '../datasets/AmazonBooks_m1/test.txt'
            test_data = UltraDataset(data_path=test_path, train=False)

        # train_data = UltraDataset(data_path=data_path, train=True)
        # test_data = UltraDataset(data_path=data_path, train=False)
        train_mat = train_data.get_train_matrix()
        user_num, item_num = train_data.get_user_item_counts()
        constraint_mat = train_data.get_constraint_matrix()
        hyper_param = {
            'dataset': dataset,
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
            'constraint_mat': constraint_mat,
            'tuning': tuning
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

        test_ground_truth_list = test_data.get_test_ground_truth_list()
        hyper_param['test_ground_truth_list'] = test_ground_truth_list

        best_epoch, best_metric, trained_model = run_ultragcn(device=device, 
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

    # test
    final_test_data_path = '../datasets/AmazonBooks_m1/test.txt'
    final_test_data = UltraDataset(data_path=final_test_data_path, train=False)
    final_test_ground_truth_list = final_test_data.get_test_ground_truth_list()
    test_loader = torch.utils.data.DataLoader(list(range(hyper_param['user_num'])),
                                                        batch_size=hyper_param['test_batch_size'],
                                                        shuffle=False,
                                                        num_workers=5)
    F1_score, Precision, Recall, NDCG = test(trained_model,
                                            test_loader,
                                            final_test_ground_truth_list,
                                            hyper_param['mask'],
                                            hyper_param['topk'],
                                            hyper_param['user_num'])
    
    print('Results:')
    print('Hyper-param with ii_neighbor_num: {}, gamma: {}, lambda: {}'.format(ii_neighbor_num, gamma, lambda_))
    print('F1_score: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, NDCG: {:.4f}'.format(F1_score, Precision, Recall, NDCG))

    final_csv_path = '../csv/test_score.csv'
    if not os.path.exists(final_csv_path):
        with open(final_csv_path, 'w') as f:
            f.write('ii_neighbor_num, gamma, lambda, Recall, NDCG\n')

    header = [
        "ii_neighbor_num",
        "gamma",
        "lambda",
        "Recall@20 Test",
        "NDCG@20 Test"
    ]

    row = [
        ii_neighbor_num,
        gamma,
        lambda_,
        Recall,
        NDCG
    ]

    try:
        file_exists = os.path.isfile(final_csv_path)
        with open(final_csv_path, 'a', '', 'utf-8') as f:
            writer = csv.writer(f)
            if not file_exists or os.stat(final_csv_path).st_size == 0:
                writer.writerow(header)
            writer.writerow(row)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    fire.Fire(main)
