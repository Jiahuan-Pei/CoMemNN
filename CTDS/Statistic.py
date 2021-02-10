#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Function : 
@License : Copyright(C), ILPS group, Univeristy of Amsterdam
@Author  : Jiahuan Pei
@Contact : j.pei@uva.nl
@Date: 2020-04-02
"""
import pprint
import pandas as pd
import numpy as np

def cal_profile():

    _keys = ['gender', 'age', 'dietary', 'favorite']

    pp = pprint.PrettyPrinter(indent=4)
    dial_path = '/Users/pp/Code/TDS/datasets/personalized-dialog-dataset/full/personalized-dialog-task5-full-dialogs-trn.txt'
    # dial_path = '/Users/pp/Code/TDS/datasets/personalized-dialog-dataset/small/personalized-dialog-task5-full-dialogs-trn.txt'
    p_list = []
    p_dict = {} # profile: num
    total = 0
    with open(dial_path, 'r') as fr:
        for line in fr.readlines():
            if '\t' not in line and line[:2]=='1 ': # a profile line
                line = line.strip('\n')[2:]
                total += 1
                p_list.append(line)
                if line not in p_dict:
                    p_dict[line] = 1
                else:
                    p_dict[line] += 1
    print('Total number of profile:', total)
    df = pd.DataFrame([k.split()+[v] for k, v in p_dict.items()], columns= _keys + ['count'])
    for k in _keys:
        tmp_df = df.groupby([k]).sum()
        print(k, '#', len(tmp_df), '-' * 50)
        print(tmp_df/total*100)
    print('='*50)
    print('Number of profile types:', len(p_dict))
    # print(df)
    pp.pprint(p_dict)

def cal_kb():
    from CTDS.Prepare_dataset import _load_knowledge_base, src
    kb_namekey2value_dict, kb2id_dict, id2kb_dict = _load_knowledge_base(src)
    n_kb_name_key_value = len(kb_namekey2value_dict) # 2400
    n_keys = [len(v) for v in kb_namekey2value_dict.values()][0] # 12
    keys = list(list(kb_namekey2value_dict.values())[0].keys())
    # ['R_cuisine', 'R_location', 'R_price', 'R_rating', 'R_phone',
    # 'R_address', 'R_number', 'R_type', 'R_speciality',
    # 'R_social_media', 'R_parking', 'R_public_transport']

def statistic_sigma(score_list):
    diff_list = []
    for i in range(1, len(score_list)):
        diff = score_list[i]-score_list[i-1]
        diff_list.append(diff)
    return np.std(diff_list)

def PEA():
    table = {
        'CoMemNN':      [99.99, 99.93, 99.82, 99.83, 99.38, 98.98],

        'CoMemNN-PEL':  [85.71, 87.85, 91.34, 89.19, 90.04, 90.60],

        'CoMemNN-NP':   [99.87, 99.85, 99.24, 99.15, 99.13, 98.86],
        'CoMemNN-NP-CP':[98.89, 99.09, 99.16, 99.20, 99.14, 98.92],
        'CoMemNN-ND':   [99.72, 99.87, 99.80, 99.46, 98.72, 97.23],
        'CoMemNN-ND-CD':[99.99, 99.86, 99.68, 99.69, 99.19, 34.78],
        'CoMemNN-ND-NP':[99.09, 98.98, 97.95, 97.69, 97.06, 97.23]
    }

    for k, v in table.items():
        sigma = statistic_sigma(v)
        print('%s\t%.4f' % (k, sigma) )

def RSA():
    table = {
        'SOTAMemNN_Large':[97.49, 97.01, 96.05, 95.52, 95.40, 90.96, 90.50],
        'CoMemNN_Large':  [98.13, 97.94, 97.68, 97.53, 96.98, 96.63, 92.73],

        'SOTAMemNN':      [87.91, 86.11, 86.56, 85.79, 83.93, 84.08, 84.83],
        'CoMemNN':        [91.13, 89.90, 88.69, 87.80, 86.35, 84.83, 82.85],

        'CoMemNN-PEL':    [90.84, 90.29, 89.07, 87.18, 85.42, 80.54, 81.23],
        'CoMemNN-PEL-UPE':[87.91, 86.11, 86.56, 85.79, 83.93, 84.08, 84.83],

        'CoMemNN-NP':     [91.06, 91.23, 89.17, 85.26, 83.30, 82.10, 82.83],
        'CoMemNN-NP-CP':  [86.60, 86.10, 84.56, 83.53, 82.48, 81.95, 81.35],
        'CoMemNN-ND':     [90.91, 87.33, 89.06, 87.49, 86.59, 85.38, 85.41],
        'CoMemNN-ND-CD':  [87.70, 90.44, 85.79, 84.90, 83.56, 82.57, 85.38],
        'CoMemNN-ND-NP':  [90.04, 91.08, 89.23, 87.38, 85.76, 85.46, 85.41]
    }

    for k, v in table.items():
        sigma = statistic_sigma(v)
        print('%s\t%.4f' % (k, sigma) )


if __name__ == "__main__":
    # PEA()
    RSA()