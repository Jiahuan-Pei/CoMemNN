#!/bin/bash
# sh log.sh $1_MODEL_NAME $2_JOB_NAME
# sh log.sh CTDS 421685

python ./$1/Run_Training_log.py $2
