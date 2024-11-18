#!/bin/bash
#SBATCH -J hello_slurm
#SBATCH --gres=gpu:1
#SBATCH -o /home/sunuk/Research/UltraGCN/slurm_log/item_item_N.out
hostname
#conda activate UltraGCN

# 데이터셋 목록을 배열로 정의
item_item_N=(5 10 20 50)

# 각 데이터셋에 대해 실험을 반복
for item_item_N in "${item_item_N[@]}"
do
    python main.py --config_file ../config/ultragcn_amazonbooks_m1.ini --ii_neighbor_num $item_item_N --gamma 1e-4 --lambda_ 2.75 --early_stop_metric recall --tuning True
    python main.py --config_file ../config/ultragcn_amazonbooks_m1.ini --ii_neighbor_num $item_item_N --gamma 1e-4 --lambda_ 2.75 --early_stop_metric ndcg --tuning True
done