#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import torch
from pathlib import Path
from utils import log_param
from utils import set_random_seed
from loguru import logger
from models.ultragcnmodel.train import UltraGCNTrainer
from ultra_data import UltraDataset
from tqdm import tqdm
import configparser
import numpy as np
from ultra_data import UltraDataset
from utils import pload, pstore, get_ii_constraint_mat
from models.ultragcnmodel.train import UltraGCNTrainer


# check the effect of the ii_neighbor_num
def main(config_file):
    # Step 0. Initialization
    config = configparser.ConfigParser()
    config.read(config_file)

    logger.info("Start the experiment for checking the effect of the ii_neighbor_num.")

    gpu = config['Training']['gpu']
    device = torch.device('cuda:'+gpu if torch.cuda.is_available() else 'cpu')
    seed = config.getint('Training', 'seed')
    set_random_seed(seed=seed, device=device)

    output_dir = Path(__file__).parents[2].absolute().joinpath("plots")
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir.joinpath("exp_hyper_param.tsv")

    hyper_param = {
        'model': config['Model']['model'],
        'seed': seed,
        'device': device
    }
    log_param(hyper_param)

    # Step 1. Load datasets
    dataset = config['Training']['dataset']
    data_path = config['Training']['file_path']
    
    train_data = UltraDataset(data_path=data_path, train=True)
    test_data = UltraDataset(data_path=data_path, train=False)
    train_mat = train_data.get_train_matrix()
    user_num, item_num = train_data.get_user_item_counts()
    constraint_mat = train_data.get_constraint_matrix()
    hyper_param.update({
        'user_num': user_num,
        'item_num': item_num,
        'batch_size': config.getint('Training', 'batch_size'),
        'max_epoch': config.getint('Model', 'max_epoch'),
        'learning_rate': config.getfloat('Training', 'learning_rate'),
        'embedding_dim': config.getint('Model', 'embedding_dim'),
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
    })
    
    # ii_neighbor_num 실험 목록 정의
    ii_neighbor_nums = [5, 10, 20, 50]
    
    # ii_constraint 및 neighbor 매트릭스 준비
    if os.path.exists('../{}_ii_constraint_mat'.format(dataset)):
        ii_constraint_mat = pload('../{}_ii_constraint_mat'.format(dataset))
        ii_neighbor_mat = pload('../{}_ii_neighbor_mat'.format(dataset))
    else:
        # 기본값 설정 후 첫 번째 ii_neighbor_num으로 초기화
        ii_neighbor_mat, ii_constraint_mat = get_ii_constraint_mat(train_mat, ii_neighbor_nums[0])
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

    # Step 2. 실험 수행
    trainer = UltraGCNTrainer(device)
    
    with open(output_path, "w") as output_file:
        output_file.write("ii_neighbor_num\tbest_epoch\tbest_recall\tbest_ndcg\n")
        pbar = tqdm(ii_neighbor_nums, leave=False, colour='blue', desc='ii_neighbor_num')
        for ii_neighbor_num in pbar:
            hyper_param['ii_neighbor_num'] = ii_neighbor_num  # ii_neighbor_num 설정
            # 매트릭스 업데이트
            ii_neighbor_mat, ii_constraint_mat = get_ii_constraint_mat(train_mat, ii_neighbor_num)

            best_epoch, best_recall, best_ndcg = trainer.train_with_hyper_param(
                train_data=train_data, 
                hyper_param=hyper_param, 
                constraint_mat=constraint_mat,
                ii_constraint_mat=ii_constraint_mat, 
                ii_neighbor_mat=ii_neighbor_mat, 
                verbose=True
            )

            # 결과 출력 및 기록
            pbar.write(f"ii_neighbor_num: {ii_neighbor_num}\tbest_epoch: {best_epoch}\tbest_recall: {best_recall:.4f}\tbest_ndcg: {best_ndcg:.4f}")
            output_file.write(f"{ii_neighbor_num}\t{best_epoch}\t{best_recall:.4f}\t{best_ndcg:.4f}\n")
        pbar.close()


if __name__ == "__main__":
    # config_file 인자 지정 없을 시 기본값 설정
    config_file = sys.argv[1]
    main(config_file)
