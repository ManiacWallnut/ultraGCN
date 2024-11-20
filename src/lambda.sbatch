#!/bin/bash
#SBATCH -J hello_slurm
#SBATCH --gres=gpu:1
#SBATCH -o /home/sunuk/Research/UltraGCN/slurm_log/lambda.out
hostname
#conda activate UltraGCN

# 데이터셋 목록을 배열로 정의
lambda=(0.1 0.5 1 1.5 2 2.5 3 3.5)

# 각 데이터셋에 대해 실험을 반복
for lambda in "${lambda[@]}"
do
    python main.py --config_file ../config/ultragcn_amazonbooks_m1.ini --ii_neighbor_num 10 --gamma 1e-4 --lambda_ $lambda --early_stop_metric recall --tuning True
    python main.py --config_file ../config/ultragcn_amazonbooks_m1.ini --ii_neighbor_num 10 --gamma 1e-4 --lambda_ $lambda --early_stop_metric ndcg --tuning True
done