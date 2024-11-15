#!/bin/bash
#SBATCH -J hello_slurm
#SBATCH --gres=gpu:1
#SBATCH -o /home/sunuk/Research/UltraGCN/slurm_log/tmep.out
hostname
#conda activate UltraGCN

# 데이터셋 목록을 배열로 정의
item_item_N=(5 10 20 50)
gamma=(0.1 0.5 1 1.5 2 2.5 3 3.5)
lambda=(0.2 0.4 0.6 0.8 1 1.2 1.4)

# 각 데이터셋에 대해 실험을 반복
for item_item_N in "${item_item_N[@]}"
do
    for gamma in "${gamma[@]}"
    do
        for lambda in "${lambda[@]}"
        do
            python main.py --config_file ../config/ultragcn_amazonbooks_m1.ini --ii_neighbor_num $item_item_N --gamma $gamma --lambda_ $lambda --early_stop_metric recall --tuning False
            python main.py --config_file ../config/ultragcn_amazonbooks_m1.ini --ii_neighbor_num $item_item_N --gamma $gamma --lambda_ $lambda --early_stop_metric ndcg --tuning False
        done
    done
done