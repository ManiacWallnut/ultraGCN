#!/bin/bash
#SBATCH -J hello_slurm
#SBATCH --gres=gpu:1

slurm_dir="../slurm_log/hyper_param"
mkdir -p $slurm_dir

#SBATCH -o /home/sunuk/Research/UltraGCN/slurm_log/hyper_param/experiment_ii_neighbor_num_param.out
hostname
#conda activate UltraGCN

echo "Starting experiment for ii_neighbor_num_param using config file ../../config/ultragcn_amazonbooks_m1_hyper_param.ini..." | tee -a $log_file
python experiments.exp_ii_neighbor_num.py --config_file ../../config/ultragcn_amazonbooks_m1_hyper_param.ini | tee -a $log_file
