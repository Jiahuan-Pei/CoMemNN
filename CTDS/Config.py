#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Function : 
@License : Copyright(C), ILPS group, Univeristy of Amsterdam
@Author  : Jiahuan Pei
@Contact : j.pei@uva.nl
@Date: 2020-02-06
"""
import torch
from sys import platform
import os
import torch.backends.cudnn as cudnn
import argparse
from common.Utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
parser.add_argument("--mode", type=str, default='train', help='train or test')
parser.add_argument("--learn_mode", type=str, default='multiple') # --learn_mode=binary/--learn_mode=multiple
parser.add_argument("--data_dir", type=str, default='..')
parser.add_argument("--log_name", type=str, default='CTDS_origin_bs8_hop1.train-417341.out')
parser.add_argument("--model_name", type=str, default='CTDS') # must assign when run the code
parser.add_argument("--exp_name", type=str, default='') # tag different settings of a model
parser.add_argument("--output_model_path", type=str, default=None) # use only when to specify a model path
parser.add_argument("--task", type=str, default='babi-small-t5')
parser.add_argument("--debug", type=int, default=0) # to speed up implement during debugging phase
parser.add_argument("--bsl", type=int, default=0) # 1: run baseline; 0: run our model
parser.add_argument("--use_profile", type=int, default=1)
parser.add_argument("--use_preference", type=int, default=1)
parser.add_argument("--use_neighbor", type=int, default=1)
parser.add_argument("--use_loss_p", type=int, default=1) # use profile completion loss
parser.add_argument("--hops", type=int, default=3)
parser.add_argument("--k", type=int, default=100) # find k neighbor dialogue based on profile vector
parser.add_argument("--neighbor_policy", type=str, default='k') # h: use the head k; k: top k using knn; hk: head 2*k to rerank k using knn
parser.add_argument("--keep_attributes", type=str, default=None) # keep the values of some attributes; split by _
parser.add_argument("--num_negative_samples", type=int, default=-1) # -1: do not sample; ng10000_bs32
parser.add_argument("--num_infer_response", type=int, default=-1) # max 43862 = 43863 - 1 ; if10000_bs32
parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--test_batch_size", type=int, default=32)

parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--warmup", type=float, default=-1) # percentage of warm up steps
parser.add_argument("--max_grad_norm", type=int, default=10, help="Clip gradients to this norm.") # 1, 10.0, 40.0
parser.add_argument("--weight_decay", type=float, default=1e-5, help="Clip gradients to this norm.") # 1, 10.0, 40.0
parser.add_argument("--embedding_size", type=int, default=128)
parser.add_argument("--hidden_size", type=int, default=250) # memory_size = 250 in sota
parser.add_argument("--epoches", type=int, default=50)
parser.add_argument("--train_epoch_start", type=int, default=0)
parser.add_argument("--infer_epoch_start", type=int, default=190)
parser.add_argument("--accumulation_steps", type=int, default=1)
parser.add_argument("--num_histories", type=int, default=2)
parser.add_argument("--num_queries", type=int, default=1)
parser.add_argument("--sentence_len", type=int, default=20)
parser.add_argument("--max_profile_vector_size", type=int, default=-1)
parser.add_argument("--profile_dropout_ratio", type=float, default=0.0)
parser.add_argument("--pp_p", type=float, default=1.0,  help='profile prediction: profile weight')
parser.add_argument("--pp_np", type=float, default=1.0,  help='profile prediction: neighbor profile weight')
parser.add_argument("--pp_c", type=float, default=1.0,  help='profile prediction: context weight')
parser.add_argument("--pp_nc", type=float, default=1.0,  help='profile prediction: neighbor context weight')

args = parser.parse_args()

task = args.task
# data_dir = '/ivi/ilps/personal/jpei/TDS' if platform=='linux' else '/Users/pp/Code/TDS'
data_dir = args.data_dir
# task='babi-small-t5'
# data_dir='D:/Projects/TDS/'

src = '%s/datasets/personalized-dialog-dataset/%s' % (data_dir, task.split('-')[1])
# base_data_path = '%s/datasets' % data_dir
base_output_path = os.path.join(data_dir, 'output/%s%s/'%(args.model_name, args.exp_name))
output_model_path = os.path.join(base_output_path, 'model/') if not args.output_model_path else args.output_model_path # Model dir
output_result_path = os.path.join(base_output_path, 'result/') # Result dir

makedir_flag = (args.local_rank == None or args.local_rank==0)

if not os.path.exists(base_output_path) and makedir_flag:
    os.makedirs(base_output_path)

if not os.path.exists(output_model_path):
    if args.output_model_path==None and makedir_flag:
        os.makedirs(output_model_path)
    else:
        print('Err: do not exist this out_model_path=', args.output_model_path)

if not os.path.exists(output_result_path) and makedir_flag:
    os.makedirs(output_result_path)

if makedir_flag:
    print('base_data_path=', src)
    print('base_output_path=', base_output_path)
    print('output_model_path=', output_model_path)
    print('output_result_path=', output_result_path)
dir_path = os.path.dirname(os.path.realpath(__file__))

args.train_batch_size = 3 if args.debug else args.train_batch_size
args.test_batch_size = 3 if args.debug else args.test_batch_size
cut_data_index = 2 * args.train_batch_size if args.debug else None
embedding_size = 4 if args.debug else args.embedding_size
hidden_size = 4 if args.debug else args.hidden_size
epoches = 3 if args.debug else args.epoches
accumulation_steps = args.accumulation_steps
# train_size = args.train_size
num_histories = args.num_histories
num_queries = args.num_queries
sentence_len = args.sentence_len
k_neighbor = args.k

detected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.distributed.init_process_group(backend='NCCL', init_method='env://')

cudnn.enabled = True
cudnn.benchmark = False
cudnn.deterministic = True
if makedir_flag:
    print(torch.__version__)
    print(torch.version.cuda)
    print(cudnn.version())
init_seed(123456)

# define this when can not store all data
save_data_attributes = [
             'response_id', 'profile', 'incomplete_profile',
             'context', 'query',
             'neighbor_profile',
             # 'neighbor_context'
             ]

if __name__ == "__main__":
    pass