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

def run_ultragcn(device, train_data, hyper_param, ii_constraint_mat, ii_neighbor_mat):
    trainer = UltraGCNTrainer(device)
    model = trainer.train_with_hyper_param(train_data=train_data, hyper_param=hyper_param, ii_constraint_mat=ii_constraint_mat, ii_neighbor_mat=ii_neighbor_mat, verbose=True)
    return model

def main(config_file):
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
    # data_path = Path(__file__).parents[1].absolute().joinpath("datasets")
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
            'ii_neighbor_num': config.getint('Model', 'ii_neighbor_num'),
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
            'gamma': config.getfloat('Training', 'gamma'),
            'lambda': config.getfloat('Training', 'lambda'),
            'sampling_sift_pos': config.getboolean('Training', 'sampling_sift_pos'),
            'test_batch_size': config.getint('Testing', 'test_batch_size'),
            'topk': config.getint('Testing', 'topk'),
            'constraint_mat': constraint_mat
        }
        print(hyper_param['sampling_sift_pos'])
        if os.path.exists('../{}_ii_constraint_mat'.format(dataset)):
            ii_constraint_mat = pload('../{}_ii_constraint_mat'.format(dataset))
            ii_neighbor_mat = pload('../{}_ii_neighbor_mat'.format(dataset))
        else:
            ii_neighbor_mat, ii_constraint_mat = get_ii_constraint_mat(train_mat, hyper_param['ii_neighbor_num'])
            pstore(ii_neighbor_mat, '../{}_ii_neighbor_mat'.format(dataset))
            pstore(ii_constraint_mat, '../{}_ii_constraint_mat'.format(dataset))
        mask = torch.zeros(user_num, item_num)
        interacted_items = [[] for _ in range(user_num)]
        for (u, i) in train_data:
            mask[u][i] = -np.inf
            interacted_items[u].append(i)
        hyper_param.update({'interacted_items': interacted_items, 'mask': mask})
        test_ground_truth_list = [[] for _ in range(user_num)]
        for (u, i) in test_data:
            test_ground_truth_list[u].append(i)
        hyper_param['test_ground_truth_list'] = test_ground_truth_list
        # log_param(hyper_param)
        model = run_ultragcn(device=device, train_data=train_data, hyper_param=hyper_param, ii_constraint_mat=ii_constraint_mat, ii_neighbor_mat=ii_neighbor_mat)

    else:
        logger.error("The given \"{}\" is not supported...".format(param['model']))
        return

if __name__ == "__main__":
    fire.Fire(main)

# import sys
# import fire
# import torch
# from pathlib import Path
# import configparser
# import os
# import numpy as np

# from utils import set_random_seed
# from data import MyDataset
# from models.mymodel.train import MyTrainer
# from models.mymodel.eval import MyEvaluator
# from utils import log_param
# from loguru import logger
# from ultra_data import UltraDataset
# from utils import pload
# from utils import get_ii_constraint_mat
# from utils import pstore
# from models.ultragcnmodel.train import UltraGCNTrainer


# def run_mymodel(device, 
#                 train_data, 
#                 test_data, 
#                 hyper_param):
#     trainer = MyTrainer(device=device,
#                         in_dim=train_data.in_dim,
#                         out_dim=train_data.out_dim)

#     model = trainer.train_with_hyper_param(train_data=train_data,
#                                            hyper_param=hyper_param,
#                                            verbose=True)

#     evaluator = MyEvaluator(device=device)
#     accuracy = evaluator.evaluate(model, test_data)

#     return accuracy

# def run_ultragcn(device, 
#                  train_data, 
#                  hyper_param, 
#                  ii_constraint_mat,
#                  ii_neighbor_mat):
#     trainer = UltraGCNTrainer(device)

#     model = trainer.train_with_hyper_param(train_data=train_data,
#                                            hyper_param=hyper_param,
#                                            ii_constraint_mat=ii_constraint_mat,
#                                            ii_neighbor_mat=ii_neighbor_mat,
#                                            verbose=True)
    
#     return model


# def main(config_file):
#     """
#     Handle user arguments of ml-project-template

#     :param config_file: config file path
#     """

#     # Step 0. Initialization
#     config = configparser.ConfigParser()
#     config.read(config_file)

#     logger.info("The main procedure has started with the following parameters:")
#     gpu = config['Training']['gpu']
#     device = torch.device('cuda:'+gpu if torch.cuda.is_available() else 'cpu')
#     seed = config.getint('Training', 'seed') # have to implement in config file
#     set_random_seed(seed=seed, device=device)
#     param = dict()
#     model = config['Model']['model'] # have to implement in config file
#     param['model'] = model
#     param['seed'] = seed
#     param['device'] = device
#     log_param(param)

#     # Step 1. Load datasets
#     dataset = config['Training']['dataset'] # dataset name e.g., AmazonBooks_m1
#     data_param = dict()
#     data_param['dataset'] = dataset
#     data_path = Path(__file__).parents[1].absolute().joinpath("datasets")
#     if model == 'mymodel':
#         train_data = MyDataset(data_path=data_path, train=True)
#         test_data = MyDataset(data_path=data_path, train=False)
#         logger.info("The datasets are loaded where their statistics are as follows:")
#         logger.info("- # of training instances: {}".format(len(train_data)))
#         logger.info("- # of test instances: {}".format(len(test_data)))
#     elif model == 'ultragcn':
#         hyper_param = dict()
#         data_path.joinpath(dataset)
#         train_data = UltraDataset(data_path=data_path, train=True)
#         test_data = UltraDataset(data_path=data_path, train=False)
#         train_mat = train_data.get_train_matrix()
#         user_num, item_num = train_data.get_user_item_counts()
#         hyper_param['user_num'] = user_num
#         hyper_param['item_num'] = item_num
#         constraint_mat = train_data.get_constraint_matrix()
#         hyper_param['constraint_mat'] = constraint_mat
#         logger.info("The datasets are loaded where their statistics are as follows:")
#         logger.info("- # of training instances: {}".format(len(train_data)))
#         logger.info("- # of test instances: {}".format(len(test_data)))
#         logger.info("- train_mat: {}".format(train_mat))
#         logger.info("- user_num: {}".format(user_num))
#         logger.info("- item_num: {}".format(item_num))
#         logger.info("- constraint_mat: {}".format(constraint_mat))
#     else:
#         logger.error("The given \"{}\" is not supported...".format(model))
#         return
    

#     # Step 2. Run (train and evaluate) the specified model

#     logger.info("Training the model has begun with the following hyperparameters:")

#     if model == 'mymodel':
#         hyper_param['batch_size'] = config.getint('Training', 'batch_size')
#         hyper_param['epochs'] = config.getint('Model', 'max_epoch')
#         hyper_param['learning_rate'] = config.getfloat('Training', 'learning_rate')
#         log_param(hyper_param)
#         accuracy = run_mymodel(device=device,
#                                train_data=train_data,
#                                test_data=test_data,
#                                hyper_param=hyper_param)

#         # - If you want to add other model, then add an 'elif' statement with a new runnable function
#         #   such as 'run_my_model' to the below
#         # - If models' hyperparamters are varied, need to implement a function loading a configuration file
#     elif model == 'ultragcn':
#         hyper_param = dict()

#         ###############################################################################
#         # Model Parameters                                                            #
#         ###############################################################################
#         hyper_param['batch_size'] = config.getint('Training', 'batch_size')
#         hyper_param['epochs'] = config.getint('Model', 'max_epoch')
#         hyper_param['learning_rate'] = config.getfloat('Training', 'learning_rate')

#         embedding_dim = config.getint('Model', embedding_dim)
#         hyper_param['embedding_dim'] = embedding_dim
#         ii_neighbor_num = config.getint('Model', 'ii_neighbor_num')
#         hyper_param['ii_neighbor_num'] = ii_neighbor_num
#         model_save_path = config['Model']['model_save_path']
#         hyper_param['model_save_path'] = model_save_path
#         hyper_param['enable_tensorboard'] = config.getboolean('Model', 'enable_tensorboard')
#         initial_weight = config.getfloat('Model', 'initial_weight')
#         hyper_param['initial_weight'] = initial_weight

#         ###############################################################################
#         # Training Parameters                                                         #
#         ###############################################################################
#         early_stop_epoch = config.getint('Training', 'early_stop_epoch')
#         hyper_param['early_stop_epoch'] = early_stop_epoch
#         w1 = config.getfloat('Training', 'w1')
#         w2 = config.getfloat('Training', 'w2')
#         w3 = config.getfloat('Training', 'w3')
#         w4 = config.getfloat('Training', 'w4')
#         hyper_param['w1'] = w1
#         hyper_param['w2'] = w2
#         hyper_param['w3'] = w3
#         hyper_param['w4'] = w4
#         negative_num = config.getint('Training', 'negative_num')
#         negative_weight = config.getfloat('Training', 'negative_weight')
#         hyper_param['negative_num'] = negative_num
#         hyper_param['negative_weight'] = negative_weight

#         # gamma, lambda
#         gamma = config.getfloat('Training', 'gamma')
#         hyper_param['gamma'] = gamma
#         lambda_ = config.getfloat('Training', 'lambda')
#         hyper_param['lambda'] = lambda_

#         sampling_sift_pos = config.getboolean('Training', 'sampling_sift_pos')
#         hyper_param['sampling_sift_pos'] = sampling_sift_pos

#         ###############################################################################
#         # Testing Parameters                                                          #
#         ###############################################################################
#         test_batch_size = config.getint('Testing', 'test_batch_size')
#         hyper_param['test_batch_size'] = test_batch_size
#         topk = config.getint('Testing', 'topk')
#         hyper_param['topk'] = topk

#         ###############################################################################
#         # ii_constraint_mat, ii_neighbor_mat                                          #
#         ###############################################################################
#         ii_cons_mat_path = '../' + dataset + '_ii_constraint_mat'
#         ii_neigh_mat_path = '../' + dataset + '_ii_neighbor_mat'

#         if os.path.exists(ii_cons_mat_path):
#             ii_constraint_mat = pload(ii_cons_mat_path)
#             ii_neighbor_mat = pload(ii_neigh_mat_path)
#         else:
#             ii_neighbor_mat, ii_constraint_mat = get_ii_constraint_mat(train_mat, 
#                                                                        ii_neighbor_num)
#             pstore(ii_neighbor_mat, ii_neigh_mat_path)
#             pstore(ii_constraint_mat, ii_cons_mat_path)
        
#         # mask matrix for testing to accelarate testing speed
#         mask = torch.zeros(user_num, item_num)
#         interacted_items = [[] for _ in range(user_num)]
#         for (u, i) in train_data:
#             mask[u][i] = -np.inf
#             interacted_items[u].append(i)
#         hyper_param['interacted_items'] = interacted_items
#         hyper_param['mask'] = mask

#         # test user-item interaction, which is ground truth
#         test_ground_truth_list = [[] for _ in range(user_num)]
#         for (u, i) in test_data:
#             test_ground_truth_list[u].append(i)
#         hyper_param['test_ground_truth_list'] = test_ground_truth_list


#         log_param(hyper_param)
#         model = run_ultragcn(device=device,
#                              train_data=train_data,
#                              hyper_param=hyper_param,
#                              ii_constraint_mat=ii_constraint_mat,
#                              ii_neighbor_mat=ii_neighbor_mat)
#     else:
#         logger.error("The given \"{}\" is not supported...".format(model))
#         return

#     # # Step 3. Report and save the final results
#     # logger.info("The model has been trained. The test accuracy is {:.4}.".format(accuracy))


# if __name__ == "__main__":
#     sys.exit(fire.Fire(main))