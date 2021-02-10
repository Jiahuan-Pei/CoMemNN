#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=$2$3
#SBATCH -o ./job_out/%x.test-%A.out
#SBATCH -e ./job_err/%x.test-%A.err
# SBATCH --nodelist=ilps-cn$1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c8
#SBATCH --mem=180G
#SBATCH --time=4-00:00:00

# Set-up the environment.
source ${HOME}/.bashrc
source activate tds_py37_pt

set PYTHONPATH=./
set OMP_NUM_THREADS=2
# Start the experiment.
python -m torch.distributed.launch --nproc_per_node=1 ./$2/Run.py --mode='test' --debug=0 --data_dir=/ivi/ilps/personal/jpei/TDS --model_name=$2 --exp_name=$3 $4
# Example 1: sh infer.sh 101 CTDS Base $4_other_params
# Example 2: sh infer.sh 101 CTDS _store_cpu_data --output_model_path=/ivi/ilps/personal/jpei/TDS/output/CTDS/model
EOT
