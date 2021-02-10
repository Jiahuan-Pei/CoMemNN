#!/bin/bash

sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=$2$3
#SBATCH -o ./job_out/%x.train-%A.out
#SBATCH -e ./job_err/%x.train-%A.err
# SBATCH --nodelist=ilps-cn$1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c8
#SBATCH --mem=30G
#SBATCH --time=4-00:00:00

# Set-up the environment.
source ${HOME}/.bashrc
source activate tds_py37_pt

set PYTHONPATH=./
set OMP_NUM_THREADS=2
# Start the experiment.
python -m torch.distributed.launch --nproc_per_node=1 --master_port 29501 ./$2/Run.py --mode='train' --debug=0 --data_dir=/ivi/ilps/personal/jpei/TDS --model_name=$2 --exp_name=$3 $4
# $1_run_node $2_model_name $3_exp_setting $4_other_params
# Example 1: sh train.sh 101 CTDS _base $4_other_params
# Example 2: sh train.sh 106 CTDS _ng1000 --num_negative_samples=1000
# Example 3: sh train.sh 104 CTDS _ng1000_bs64 --num_negative_samples=1000 --train_batch_size=64
EOT