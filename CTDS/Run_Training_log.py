#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Function : 
@License : Copyright(C), ILPS group, Univeristy of Amsterdam
@Author  : Jiahuan Pei
@Contact : j.pei@uva.nl
@Date: 2020-05-22
"""
import os, sys, re
import matplotlib.pyplot as plt
import pandas as pd

root_dir = os.path.dirname(os.path.dirname(__file__))
log_dir = os.path.join(root_dir, 'job_out')

def analyze_a_log(fname, fdir=log_dir, is_savefig=False):
    fpath = os.path.join(fdir, fname)
    print('log_path=', fpath)
    losses = {}
    with open(fpath, 'r') as fr:
        for line in fr.readlines():
            if line.startswith('Method'):
                g = re.match(r'.*Epoch\s+(\d+)\s+Batch\s+(\d+)\s+Loss\s+\[(\d+.\d+(e-\d+){0,1})\].*', line)
                if g:
                    epoch, batch, loss = int(g.group(1)), int(g.group(2)), float(g.group(3))
                    print(epoch, batch, loss)
                    if epoch not in losses:
                        losses[epoch] = [loss]
                    else:
                        losses[epoch].append(loss)
                else:
                    print('output file format errors...')
                    print(line)
                    return

    # print(losses)
    plot_losses = []
    for ep in losses:
        # step = 1/len(losses[ep])
        loss = 0
        for i, l in enumerate(losses[ep]):
            # x = ep + i * step
            # plot_losses.append((x, l))
            # print(x, l)
            loss+=l
        loss /= len(losses[ep])

        plot_losses.append((ep, loss))

    # print(plot_losses)
    df = pd.DataFrame(plot_losses, columns=['epoch', 'train']).set_index('epoch')
    # print(df)

    df.plot()
    # plt.title('Learning curve')
    plt.title(fname)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(range(0, len(losses)+1, 5))
    if is_savefig:
        exp, jobid, _ = fname.split('.')
        save_path = os.path.join(root_dir, 'job_fig/%s/' % exp)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        plt.savefig(os.path.join(save_path, '%s_ep%s.png' % (jobid, len(losses))))
    else:
        plt.show()

def summary_all_log(fdir=log_dir):
    for fname in os.listdir(fdir):
        if '.out' in fname:
            analyze_a_log(fname, fdir=fdir, is_savefig=True)

def summary_list_of_log(pid_list, fdir=log_dir):
    if pid_list[0] == 'all':  # sys.argv[1] == all
        summary_all_log()
        return
    log_file_list = [f for f in os.listdir(fdir) for pid in pid_list if '-%s.' % pid in f]
    for log_name in log_file_list:
        if '.out' in log_name:
            analyze_a_log(log_name, fdir, is_savefig=True)
    return

if __name__ == "__main__":
    # input a list of job_ids
    summary_list_of_log(sys.argv[1:], fdir=log_dir)
    # analyze_a_log('CTDS_origin_bs8_hop1.train-417341.out')
    # line = 'Method train Epoch 2 Batch  501 Loss  [9.16799] Time  150.53529167175293 Learning rate  [0.000221625]'
    # line = 'Method train Epoch 2 Batch  501 Loss  [9.16799035621807e-05] Time  150.53529167175293 Learning rate  [0.000221625]'
    # g = re.match(r'.*Epoch\s+(\d+)\s+Batch\s+(\d+)\s+Loss\s+\[(\d+.\d+(e-\d+){0,1})\].*', line)
    # if g:
    #     epoch, batch, loss = int(g.group(1)), int(g.group(2)), float(g.group(3))
    #     print(epoch, batch, loss)