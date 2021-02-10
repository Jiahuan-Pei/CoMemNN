#!/usr/bin/env bash
# creat a virtual environment
conda create -n tds_py37_pt python=3.7 anaconda
conda activate tds_py37_pt

conda install anaconda bcolz
conda install pytorch torchvision -c pytorch
# CPU version only
conda install faiss-cpu -c pytorch

pip install transformers whoosh

if [ ! -d job_out ]; then
    mkdir job_out job_err job_fig
fi