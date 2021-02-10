#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=$1
#SBATCH -o ./job_out/%x.pre-%A.out
#SBATCH -e ./job_err/%x.pre-%A.err
#SBATCH -p cpu
#SBATCH -c8
#SBATCH --mem=60G
#SBATCH --time=1-00:00:00

# run commands example:
# sinfo
# [$1: submodule_folder $2:hyper-parameters]
# sh pre.sh CTDS '--task=babi-small-t5 --profile_dropout_ratio=0.0 --neighbor_policy=k'

# Set-up the environment.
source ${HOME}/.bashrc
source activate tds_py37_pt

set PYTHONPATH=./
set OMP_NUM_THREADS=2
# Start the experiment.
python -u ./$1/Prepare_dataset.py --data_dir=/ivi/ilps/personal/jpei/TDS $2
EOT
