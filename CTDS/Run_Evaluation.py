import os
import argparse
import torch
import pandas as pd
import re
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn import metrics
from tqdm import tqdm
import numpy as np

root_dir = os.path.dirname(os.path.dirname(__file__))

def eval_from_files(eval_result, data_dir, save_figure=1, bsl=0):
    output_path=os.path.join(data_dir, 'output/'+eval_result+'/result/')

    print('-'*50)
    print('Evaluating test set...%s' % output_path)
    all_label, all_pred, all_acc = {}, {}, {}
    all_profile, all_incomplete_profile, all_pred_profile, all_profile_acc = {}, {}, {}, {}

    files = sorted(os.listdir(output_path))
    for file in files:
        if '.png' in file:
            os.remove(os.path.join(output_path, file))
            continue
        # g = re.search(r'(\w+).(\d+).(None|\d)', file)
        # mode, epoch, local_rank = g.group(1), g.group(2), g.group(3)
        mode, epoch, local_rank = file.split('.')
        epoch = int(epoch)
        # [(data, output[Batch, Candidate])]
        if mode not in all_label:
            all_label[mode] = {}
            all_pred[mode] = {}
            all_acc[mode] = {}
            if bsl == 0:
                all_profile[mode], all_incomplete_profile[mode], \
                all_pred_profile[mode], all_profile_acc[mode] = {}, {}, {}, {}
        if epoch not in all_label[mode]:
            all_label[mode][epoch] = []
            all_pred[mode][epoch] = []
            all_acc[mode][epoch] = 0

            if bsl == 0:
                all_profile[mode][epoch], all_incomplete_profile[mode][epoch], \
                all_pred_profile[mode][epoch], all_profile_acc[mode][epoch] = [], [], [], 0

        # print('Load start', file)
        list_output = torch.load(os.path.join(output_path, file), map_location=torch.device('cpu'))
        # print('Load:', file)
        for golden_data, pred_output in tqdm(list_output, disable=True):  # each output from each gpu

            # golden_response_id, golden_profile_vec, golden_incomplete_profile_vec = golden_data
            try:
                golden_response_id, golden_profile_vec, golden_incomplete_profile_vec = golden_data['response_id'], golden_data['profile'], golden_data['incomplete_profile']
                pred_response_id, pred_profile_vec, pred_profile_vec_prob = pred_output['pred_response_id'], pred_output['pred_profile'], pred_output['pred_profile_prob']
                # print(golden_data)
                # print(pred_output)
                # exit(0)
            except:
                print(golden_data)
                print(pred_output)
                exit(0)
            all_label[mode][epoch].append(golden_response_id)
            all_pred[mode][epoch].append(pred_response_id)
            if bsl == 0:
                all_profile[mode][epoch].append(golden_profile_vec)
                all_incomplete_profile[mode][epoch].append(golden_incomplete_profile_vec)
                all_pred_profile[mode][epoch].append(pred_profile_vec)
        # do your evaluation for each output file from each epoch here

    # pprint('target', all_label)
    # pprint('prediction', all_pred)

    # for ep in range(len(all_label['test'])):
    #     target = all_label['test'][str(ep)]
    #     prediction = all_pred['test'][str(ep)]
    #     print('target=', target)
    #     print('prediction=', prediction)

    for eval_mode in all_label:
        for ep in all_label[eval_mode]:
            # evaluation of dialogue response selection
            acc = metrics.accuracy_score(y_true=torch.cat(all_label[eval_mode][ep]),
                                         y_pred=torch.cat(all_pred[eval_mode][ep]))
            all_acc[eval_mode][ep] = acc * 100
            # print('%s %s acc=%s %%' % (eval_mode, ep, acc * 100))
            # evaluation of profile prediction
            if bsl == 0:
                profile = torch.cat(all_profile[eval_mode][ep]).reshape(-1)
                incomplete_profile = torch.cat(all_incomplete_profile[eval_mode][ep]).reshape(-1)
                pred_profile = torch.cat(all_pred_profile[eval_mode][ep]).reshape(-1)
                dropped_positions = profile-incomplete_profile # only count the value that dropped
                correct = (profile*dropped_positions).long()&(pred_profile*dropped_positions).long()
                if dropped_positions.sum()==0:
                    profile_acc = 1.0
                else:
                    profile_acc = correct.sum()/dropped_positions.sum()
                all_profile_acc[eval_mode][ep] = profile_acc * 100

    # pprint(all_acc)
    df = pd.DataFrame(all_acc).sort_index()
    # pprint(df)
    # df['dev'] = [0.533333, 0.933333]
    # df['test'] = [0.333333, 0.73333333]
    pprint(df)

    max_dev_ep = df['dev'].idxmax()
    max_dev_acc = df['dev'].max()
    test_acc = df['test'][df['dev'].idxmax()]
    test_profile_acc = all_profile_acc['test'][max_dev_ep] if bsl==0 else 0

    print('Summary: Epoch=%s, Dev_Acc=%.4f %%, Test_Acc=%.4f %%, Test_Profile_Acc%.4f %%' %
          (max_dev_ep, max_dev_acc, test_acc, test_profile_acc))

    df.plot()
    # plt.title('Learning curve')
    plt.title(eval_result)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    # plt.xticks(range(len(df) + 1))
    plt.xticks(np.arange(0, len(df) + 1, 10))
    if save_figure:
        fig_path = os.path.join(root_dir, 'job_fig/' + eval_result)
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)

        plt.savefig(os.path.join(fig_path,
                                 'eval_ep%s_dev%.4f_test%.4f_profile%.4f.png' %
                                 (max_dev_ep, max_dev_acc, test_acc, test_profile_acc)))
    else:
        plt.show()


if __name__ == '__main__':

    # detected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='/Users/pp/Code/TDS/') # must use absolute path
    parser.add_argument("--eval_result", type=str, default='CTDS') # eval_result=[model_name]_[exp_name]
    parser.add_argument("--save_figure", type=int, default=1)
    parser.add_argument("--bsl", type=int, default=0)
    args = parser.parse_args()
    print(args)
    eval_from_files(args.eval_result, args.data_dir, args.save_figure, args.bsl)

    # result_file format: dev/test.epoch_id.local_rank



