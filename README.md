# CoMemNN

This is the source code of the paper "A Cooperative Memory Network for Personalized Task-oriented Dialogue Systems with Incomplete User Profiles", accepted by TheWebConf 2021.

Please contact me by email (j.pei@uva.nl) if you have any questions.

# Dataset
The original personalized TDS dataset is accessible using [this link](https://www.dropbox.com/s/4i9u4y24pt3paba/personalized-dialog-dataset.tar.gz?dl=1).
And the preprocessed dataset can be found at [this link](https://1drv.ms/u/s!AoSc7nKHsKfQarw_JvvhUsvHjrA?e=y7AekY).

# Examples of how to run .sh files with parameters
Parameters of train.sh and infer.sh are $1_run_node $2_model_name $3_exp_setting $4_other_params.

sh train.sh 104 CTDS _our_t5_pro1_g1_pre1_pdr0.0_k100all_ep250 '--use_profile=1 --use_neighbor=1 --use_preference=1 --profile_dropout_ratio=0.0 --bsl=0 --task=babi-small-t5 --neighbor_policy=k --epoches=250'
sh infer.sh 104 CTDS _our_t5_pro1_g1_pre1_pdr0.0_k100all_ep250 '--use_profile=1 --use_neighbor=1 --use_preference=1 --profile_dropout_ratio=0.0 --bsl=0 --task=babi-small-t5 --neighbor_policy=k --epoches=250'
