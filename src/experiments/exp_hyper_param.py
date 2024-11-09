#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch
from pathlib import Path
from utils import log_param
from utils import set_random_seed
from loguru import logger
from models.mymodel.train import MyTrainer
from models.mymodel.eval import MyEvaluator
from data import MyDataset
from tqdm import tqdm


# check the effect of the learning rate
def main():
    # Step 0. Initialization
    logger.info("Start the experiment for checking the effect of the learning rate.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 777
    set_random_seed(seed=seed, device=device)

    output_dir = Path(__file__).parents[2].absolute().joinpath("plots")
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir.joinpath("exp_hyper_param.tsv")

    param = dict()
    param['seed'] = seed
    param['device'] = device
    param['output_path'] = output_path
    log_param(param)

    # Step 1. Load datasets
    data_path = Path(__file__).parents[2].absolute().joinpath("datasets")
    train_data = MyDataset(data_path=data_path, train=True)
    test_data = MyDataset(data_path=data_path, train=False)
    logger.info("The datasets are loaded where their statistics are as follows:")
    logger.info("- # of training instances: {}".format(len(train_data)))
    logger.info("- # of test instances: {}".format(len(test_data)))

    hyper_param = dict()
    hyper_param['batch_size'] = 100
    hyper_param['epochs'] = 10
    log_param(hyper_param)

    # Step 2. Do experiment
    trainer = MyTrainer(device=device,
                        in_dim=train_data.in_dim,
                        out_dim=train_data.out_dim)
    evaluator = MyEvaluator(device=device)

    learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    with open(output_path, "w") as output_file:
        pbar = tqdm(learning_rates, leave=False, colour='blue', desc='lrate')
        for learning_rate in pbar:
            hyper_param['learning_rate'] = learning_rate
            model = trainer.train_with_hyper_param(train_data=train_data,
                                                   hyper_param=hyper_param,
                                                   verbose=False)
            test_accuracy = evaluator.evaluate(model, test_data)
            pbar.write("learning_rate: {:.4f}\ttest_accuracy: {:.4f}".format(learning_rate,
                                                                             test_accuracy))
            output_file.write("{}\t{}\n".format(learning_rate, test_accuracy))
        pbar.close()


if __name__ == "__main__":
    main()
