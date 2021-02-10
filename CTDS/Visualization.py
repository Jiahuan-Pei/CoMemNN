#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Function : 
@License : Copyright(C), ILPS group, Univeristy of Amsterdam
@Author  : Jiahuan Pei
@Contact : j.pei@uva.nl
@Date: 2020-09-23
"""
'''
test_list_output = [(golden_data, pred_output)...] is small batch of output data; counted # 636
golden_data = (data['response_id'], data['profile'], data['incomplete_profile'])
pred_output 
'''

import torch
import argparse
from CTDS.Profile import vector2profile, vector2probdict, pred_profile_to_onehot_vec
import textdistance
import seaborn as sns
import matplotlib.pyplot as plt
from CTDS.Utils import babi_tokenizer
import pandas as pd

def main():
    output_epoch_list = [
        [
            ('CTDS_bsl_t5_pro1_g1_pre1_pdr0.0_k100all_ep250', '247'),
            ('CTDS_our_t5_pro1_g1_pre1_pdr0.0_k100all_ep250', '239'),
            'profile discard ratio=0%',
            4
        ],  # test acc 91.1324 / 87.9118
        [
            ('CTDS_bsl_t5_pro1_g1_pre1_pdr0.5_k100all_ep250', '214'),
            ('CTDS_our_t5_pro1_g1_pre1_pdr0.5_k100all_ep250', '244'),
            'profile discard ratio=50%',
            4
        ],  # test acc 87.8033 / 85.7862
        # [
        #     ('CTDS_bsl_t5_pro1_g1_pre1_pdr0.7_k100all_ep250', '209'),
        #     ('CTDS_our_t5_pro1_g1_pre1_pdr0.7_k100all_ep250', '244'),
        #     'profile discard ratio=70%',
        #     4
        # ],  # test acc 86.3484 / 84.0107
        # [('CTDS_bsl', '1'), ('CTDS_our', '1'), 1]  # debug test acc [(bsl_folder, bsl_epoch), (our_folder, our_epoch), distribute_num]
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='../output')
    args = parser.parse_args()
    # load dictionaries
    dict_dir = '../datasets/personalized-dialog-dataset/small/babi-small-t5-k'
    candidates_path = '%s/candidates.pkl' % dict_dir
    _, _, id2response_dict = torch.load(candidates_path)
    vocab_path = '%s/vocab.pkl' % dict_dir
    _, id2vocab = torch.load(vocab_path)

    fw_good = open('../output/case_study_good', 'w')
    fw_bad = open('../output/case_study_bad', 'w')

    fig, axes = plt.subplots(1, len(output_epoch_list), figsize=(12, 6), sharex=True, sharey=True, dpi=300)
    for i, (bsl, our, title, num) in enumerate(output_epoch_list):
        print(title)
        bsl_result_path = '%s/%s/result/test.%s'%(args.output_dir, bsl[0], bsl[1])
        our_result_path = '%s/%s/result/test.%s'%(args.output_dir, our[0], our[1])
        compare(bsl_result_path, our_result_path, axes[i], fw_good, fw_bad, id2vocab, id2response_dict, num)

        axes[i].tick_params(labelsize=20)
        axes[i].set_ylabel(ylabel='Probability', fontsize=20)
        # axes[i].set_xlabel(xlabel='Similarity', fontsize=20)
        axes[i].set_title(title, fontsize=24)
    axes[0].legend(fontsize=20, loc='upper left') # put legend in the first subplot
    fig.text(0.5, 0.03, 'Similarity', ha='center', fontsize=20)
    # fig.text(-0.1, 0.5, 'Count', va='center', rotation='vertical')
    plt.xlabel(' ', fontsize=20)
    # plt.xlim(-0.05, 1.05)
    # plt.ylim(0, 750)
    plt.tight_layout()
    # plt.show()
    fig.savefig('../job_fig/example_distribution.png', dpi=400)
    fw_good.close()
    fw_bad.close()

def tokenizer(text, type=None):
    if type is None:
        return text
    elif type == 'babi':
        return babi_tokenizer(text)
    elif type == 'word':
        result = babi_tokenizer(text)
        for t in babi_tokenizer(text):
            if '_' in t:
                result += t.split('_')
        return result

def load_result_file(result_path, distribute_num=4):
    test_list_output = []
    if distribute_num==1:
        test_list_output = torch.load('%s.None' % (result_path),
                                        map_location=torch.device('cpu'))
    else:
        for i in range(distribute_num):
            test_list_output_i = torch.load('%s.%s' % (result_path, i),
                                            map_location=torch.device('cpu'))
            test_list_output.extend(test_list_output_i)
            # print(test_list_output_i[0])
    return test_list_output

def single_compare(bsl, our, j, label, count, id2vocab, id2response_dict):
    # golden data
    golden_context = [[id2vocab[w.item()] for w in sent if w != 0] for sent in our[0]['context'][j] if
                      not all(sent == 0)]
    golden_query = [id2vocab[w.item()] for w in our[0]['query'][0] if w != 0]
    golden_response = id2response_dict[our[0]['response_id'][j].item()]
    golden_profile = vector2profile(our[0]['profile'][j])
    golden_incomplete_profile = vector2profile(our[0]['incomplete_profile'][j])
    # prediction
    our_pred_response = id2response_dict[our[1]['pred_response_id'][j].item()]
    # our_pred_profile = vector2profile(our[1]['pred_profile'][j])
    our_pred_profile = vector2profile(
        pred_profile_to_onehot_vec(our[1]['pred_profile_prob'][j].unsqueeze(0)).squeeze(0))
    our_pred_profile_prob = vector2probdict(our[1]['pred_profile_prob'][j])
    bsl_pred_response = id2response_dict[bsl[1]['pred_response_id'][j].item()]
    # similarity
    # similarity = round(textdistance.hamming.normalized_similarity(tokenizer(bsl_pred_response, type=None), tokenizer(our_pred_response, type=None)), 3)
    similarity = round(textdistance.jaccard(tokenizer(bsl_pred_response, type='word'), tokenizer(our_pred_response, type='word')), 3)
    # similarity = round(textdistance.cosine(bsl_pred_response, our_pred_response), 3)
    print_str = ''
    if len(golden_incomplete_profile) < len(golden_profile) and len(golden_incomplete_profile)!=0:
        print_str = '%s %s %s\n' % (count, '=' * 50, label)
        print_str += 'CONTEXT:\n'
        for sent in golden_context: #[-5:]:  # avoid too much context
            print_str += '%s: %s\n' % (''.join(sent[-2:]), ' '.join(sent[:-2]))
            # if sent[-2] == '$kb' and (sent[0] in our_pred_response or sent[0] in bsl_pred_response): # if restaurant mentioned in the response
            #         pass
            # else:
            #     print_str += '%s: %s\n' % (''.join(sent[-2:]), ' '.join(sent[:-2]))
        print_str += 'QUERY: %s\n' % ' '.join(golden_query)
        print_str += 'INCOMPLETE:%s\n COMPLETE:%s\n PRED:%s\n PRED_PROB:%s\n' % (golden_incomplete_profile, golden_profile, our_pred_profile, our_pred_profile_prob)
        print_str += 'GOLD: %s\n BSL: %s\n OUR: %s\n' % (golden_response, bsl_pred_response, our_pred_response)
        print_str += 'SIM: %s\n' % similarity
        print(print_str)
        # if label == 'GOOD':
        #     fw_good.write(print_str)
        # elif label == 'BAD':
        #     fw_bad.write(print_str)
    return similarity, print_str

def compare(bsl_result_path, our_result_path, axi, fw_good, fw_bad, id2vocab, id2response_dict, num=4):
    # load results
    bsl_test_list_output = load_result_file(bsl_result_path, num)
    our_test_list_output = load_result_file(our_result_path, num)
    count_positive = 0
    count_negative = 0
    count_all = 0
    positive_similarities = []
    negative_similarities = []
    for i, (bsl, our) in enumerate(zip(bsl_test_list_output, our_test_list_output)):
        count_all += len(our[0]['response_id'])
        if all(bsl[0]['response_id']==our[0]['response_id']): # the golden response ids are correct
            positive_candidate_list = (our[1]['pred_response_id'] == our[0]['response_id'])*(bsl[1]['pred_response_id'] != our[0]['response_id'])
            negetive_candidate_list = (our[1]['pred_response_id'] != our[0]['response_id'])*(bsl[1]['pred_response_id'] == our[0]['response_id'])
            for j, flag in enumerate(positive_candidate_list):
                if flag:
                    count_positive += 1
                    # our_pred_response = id2response_dict[our[1]['pred_response_id'][j].item()]
                    # bsl_pred_response = id2response_dict[bsl[1]['pred_response_id'][j].item()]
                    # info = para_model(['%s&%s' % (our_pred_response, bsl_pred_response)])
                    # print('para=', info)
                    similarity, print_str=single_compare(bsl, our, j, 'GOOD', count_positive, id2vocab, id2response_dict)
                    positive_similarities.append(similarity)
                    fw_good.write(print_str)

            for j, flag in enumerate(negetive_candidate_list):
                if flag:
                    count_negative += 1
                    similarity, print_str = single_compare(bsl, our, j, 'BAD', count_negative, id2vocab, id2response_dict)
                    negative_similarities.append(similarity)
                    fw_bad.write(print_str)

    print('Positive/Negative/All: %s/%s/%s' % (count_positive, count_negative, count_all))

    sns.histplot(positive_similarities, label='good cases', color='deeppink', kde=True, stat='probability', ax=axi)  # , cumulative=False
    sns.histplot(negative_similarities, label='bad cases', color='blue', kde=True, stat='probability', ax=axi)  # , cumulative=False
    return


def examples():
    fig = plt.figure(figsize=(12, 2))
    # #814
    good_profile = [('female', -0.2105417549610138), ('male', 0.8970237374305725), ('elderly', 0.10984566062688828), ('middle-aged', 0.1174253299832344), ('young', 1.0489683151245117), ('non-veg', 0.9569934606552124), ('veg', 0.13070961833000183), ('biryani', 0.15655657649040222), ('curry', 0.11156253516674042), ('english_breakfast', 0.0037590116262435913), ('fish_and_chips', 0.007639534771442413), ('omlette', -0.04427327588200569), ('paella', 0.000394284725189209), ('pasta', 0.054128989577293396), ('pizza', 0.042931899428367615), ('ratatouille', -0.010720670223236084), ('risotto', -0.0017945729196071625), ('shepherds_pie', 0.04447142779827118), ('souffle', -0.10022856295108795), ('tapas', -0.08267875760793686), ('tart', 0.011044062674045563), ('tikka', 0.7374579906463623)]
    sorted_good_profile = sorted(good_profile, key=lambda tup: tup[1], reverse=True)
    print(sorted_good_profile)
    # #51
    bad_profile = [('female', 0.8523349761962891), ('male', -0.06383176147937775), ('elderly', 0.9815521240234375), ('middle-aged', -0.021564707159996033), ('young', 0.1105080097913742), ('non-veg', 0.03246179223060608), ('veg', 0.8400934338569641), ('biryani', -0.029429510235786438), ('curry', -0.11573213338851929), ('english_breakfast', -0.08266189694404602), ('fish_and_chips', -0.023211002349853516), ('omlette', 1.104180097579956), ('paella', -0.07349763810634613), ('pasta', -0.015307266265153885), ('pizza', 0.04076734185218811), ('ratatouille', 0.05320534110069275), ('risotto', 0.03178118169307709), ('shepherds_pie', 0.026960216462612152), ('souffle', -0.0012018196284770966), ('tapas', -0.0009939372539520264), ('tart', -0.0941290333867073), ('tikka', 0.003862038254737854)]
    sorted_bad_profile = sorted(bad_profile, key= lambda tup: tup[1], reverse=True)
    print(sorted_bad_profile)
    gp = pd.DataFrame(good_profile, columns=['attribute', 'u1']).set_index('attribute')
    bp = pd.DataFrame(bad_profile, columns=['attribute', 'u2']).set_index('attribute')
    df = pd.concat([gp, bp], axis=1)
    normalized_df = (df - df.min()) / (df.max() - df.min())
    sns.heatmap(data=normalized_df.T.round(2), square=True, linewidths=0.01, cmap="Blues", cbar=False, annot=True, annot_kws={"size": 11})  # , cbar_kws={"orientation": "horizontal"}
    plt.xlabel('')
    plt.yticks(
        rotation=0,
        horizontalalignment='right',
        fontweight='light',
        fontsize=11
    )
    plt.xticks(
        rotation=25,
        horizontalalignment='right',
        fontweight='light',
        fontsize=11
    )
    # plt.tight_layout(pad=0.01)
    # plt.show()
    plt.savefig("../job_fig/case_user_profiles.png", bbox_inches='tight', pad_inches=0.01)

    return

if __name__ == "__main__":
    # main()
    examples()