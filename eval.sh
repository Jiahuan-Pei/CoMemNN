#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=$1$2
#SBATCH -o ./job_out/%x.eval-%A.out
#SBATCH -e ./job_err/%x.eval-%A.err
#SBATCH -p cpu
#SBATCH -c8
#SBATCH --mem=50G
#SBATCH --time=4-00:00:00

# Set-up the environment.
source ${HOME}/.bashrc
source activate env_pytorch

set PYTHONPATH=./
set OMP_NUM_THREADS=2
# Start the experiment.
python ./$1/Run_Evaluation.py --eval_result=$1$2 --save_figure=1 --data_dir=/ivi/ilps/personal/jpei/TDS/
# sh eval.sh [$1_model_name] [$2_experiment_name]
# sh eval.sh CTDS _base
EOT
