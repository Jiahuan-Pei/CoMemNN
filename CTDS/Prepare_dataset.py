#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Function : 
@License : Copyright(C), ILPS group, Univeristy of Amsterdam
@Author  : Jiahuan Pei
@Contact : j.pei@uva.nl
@Date: 2020-01-30
"""
import sys
sys.path.append('./')
import torch
from common.Constants import *
from common.Utils import *
from common.KNN import *
from CTDS.Config import *
from CTDS.Profile import Profile, _keys
import os
import re
from collections import defaultdict
from typing import List, Dict, Any
import pickle
from tqdm import tqdm
from datetime import datetime
from CTDS.CTDSDataset import *
from CTDS.Utils import *

kb_keys = ['R_cuisine', 'R_location', 'R_price', 'R_rating', 'R_phone',
           'R_address', 'R_number', 'R_type', 'R_speciality',
           'R_social_media', 'R_parking', 'R_public_transport']

class Sample:
    def __init__(self, sample_id, context, query, response_id, response, profile, profile_kb):
        """
        The :class:`Sample` represents a data sample.
        context : list
            The conversation context represented by a list of utterances (a list of tokens).
            Example:
            [
                ['good morning', $u, #1, []
                ['hello maam how can i help you', $s, #2, []
                ['may i have table in expensive french restaurant', $u, #3, []
                ['resto_madrid_expensive_french_3stars_2 R_cuisine french', $kb, #4, [kb_id=1234]
            ]
        """
        self.sample_id= sample_id

        self.context = context
        self.query = query
        self.response_id = response_id
        self.response = response

        self.profile = Profile(profile)
        self.profile_kb = profile_kb

        self.neighbor_samples=[]
        self.lambda_indices=[]


def _set_neighbor_dialogues_to_samples(train_samples, samples, k, vknn=None):
    print('find neighbor samples...')
    if not vknn:
        d = len(train_samples[0].profile.vector)
        vector_to_index = torch.tensor([s.profile.vector for s in train_samples], dtype=torch.float32) # [#samples, d]
        vknn = Vector_KNN(d)
        vknn.index(vector_to_index)
    for i, sample in enumerate(samples):
        # search & get a list of index
        sample_index_list = vknn.search(torch.tensor(sample.profile.vector, dtype=torch.float32).unsqueeze(0), 3*k).squeeze(0)
        # get a list of sample_id without same dialouge_id
        sample_id_list = [train_samples[sample_index].sample_id for sample_index in sample_index_list \
                          if train_samples[sample_index].sample_id.split('_')[0]!=sample.sample_id.split('_')[0]]
        sample.neighbor_samples = sample_id_list[:k]
        if len(sample.neighbor_samples)!=k:
            print('ERR: increase n in n*k!')
            exit(0)
    return vknn

def bsl_set_neighbor_dialogues_to_samples(train_samples, samples, k=args.k):
    sample_index_list_dict = defaultdict(list)
    for i, sample in enumerate(train_samples):
        sample_index_list_dict[sample.profile.id].append(sample.sample_id)

    for i, sample in enumerate(samples):
        # search & get a list of index
        sample_index_list = sample_index_list_dict[sample.profile.id]
        # get a list of sample_id without same dialouge_id
        sample_id_list = [sample_id for sample_id in sample_index_list \
                          if sample_id.split('_')[0] != sample.sample_id.split('_')[0]]
        sample.neighbor_samples = sample_id_list[:k]
    return

def bsl_plus_knn_set_neighbor_dialogues_to_samples(train_samples, samples, k, vknn=None):
    if not vknn:
        d = len(train_samples[0].profile.vector)
        vector_to_index = torch.tensor([s.profile.vector for s in train_samples], dtype=torch.float32) # [#samples, d]
        vknn = Vector_KNN(d)
        vknn.index(vector_to_index)

    sample_index_list_dict = defaultdict(list)
    for i, sample in enumerate(train_samples):
        sample_index_list_dict[sample.profile.id].append(sample.sample_id)

    for i, sample in enumerate(samples):
        # get a list of index
        local_sample_id_list = sample_index_list_dict[sample.profile.id]
        # ranked list of index -- 2*k
        ranked_index_list = vknn.search(torch.tensor(sample.profile.vector, dtype=torch.float32).unsqueeze(0), 3*k).squeeze(0)
        ranked_sample_id_list = [train_samples[i].sample_id for i in ranked_index_list if train_samples[i].sample_id in local_sample_id_list]
        # get a list of sample_id without same dialouge_id
        sample_id_list = [sample_id for sample_id in ranked_sample_id_list \
                          if sample_id.split('_')[0] != sample.sample_id.split('_')[0]]
        sample.neighbor_samples = sample_id_list[:k]
        if len(sample.neighbor_samples)!=k:
            print('ERR: increase n in n*k!', len(sample.neighbor_samples))
            exit(0)
    return

def extract_kb_entry_from_str(sentence, kb_namekey2value_dict):
    pattern = re.compile(r'[a-zA-Z0-9_]+_[a-zA-Z0-9_]+') # match strings, e.g., "resto_paris_moderate_british_1stars_2", "R_speciality", "fish_and_chips"
    entry_str_list = pattern.findall(sentence)
    entry_tuple_list = []
    for entry in entry_str_list:
        if entry in kb_namekey2value_dict.keys():
            entry_tuple_list.append((entry, None, None)) # name, key, value
        else:
            for key in kb_keys:
                if key[1:] in entry:
                    name = re.sub(key[1:],'', entry)
                    entry_tuple_list.append((name, key, entry)) # name, key, value
                    break
    return entry_tuple_list


def _load_samples(src: str, task: str, mode: str, kb_namekey2value_dict: dict, response2id_dict: dict) -> List[Sample]:
    """Loads the conversations in a specific mode (train, dev, test)."""
    fp = _get_conversations_fp(src, task, mode)

    result = []

    conversation_context = []
    kb_tuple_list = [] # list of (name, key, value) in the kb lines
    # q = [0] * 5
    # r = [0] * 5
    dialogue_count = 1
    sample_count = 0
    for line_id, line in enumerate(_load_lines(fp)):
        if not line:  # <line is empty> means a conversation is ended
            # result[-1].neighbor_context_accessory = [
            #     tokens for tokens in conversation_context[1:-2] if '_' not in ''.join(tokens)]
            conversation_context.clear()
            kb_tuple_list.clear()
            dialogue_count+=1
        else:
            turn_id, line = line.split(' ', 1)
            if '\t' in line:  # this line is an question-answer pair
                # i.e. ['male', 'middle-aged', 'non-veg', 'fish_and_chips']
                profile = conversation_context[0][0].split()
                profile_kb = conversation_context[0][-1]

                query, response = line.split('\t')

                # should_use_kb = any(i in query for i in ['direction', 'contact'])
                #!!!!!!!!!!!!!!!!!!!!!
                sample_id = '%s_%s_%s_%s' % (sample_count, dialogue_count, turn_id, line_id+1)
                sample = Sample(
                    sample_id=sample_id,
                    context=conversation_context[1:], # remove profile
                    query=query,
                    response_id=response2id_dict[response],
                    response=response,
                    profile=profile,
                    profile_kb=profile_kb
                )
                sample_count += 1

                # assume: at most 1 entry_name
                # add current query & response to next context
                if '_' in sample.query:
                    q_kb_entry_tuple_list = extract_kb_entry_from_str(sample.query, kb_namekey2value_dict)
                    conversation_context.append([sample.query, '$u', '#' + turn_id, q_kb_entry_tuple_list])
                else:
                    conversation_context.append([sample.query, '$u', '#'+turn_id, []])

                if '_' in sample.response and 'api_call' not in sample.response:
                    r_kb_entry_tuple_list = extract_kb_entry_from_str(sample.response, kb_namekey2value_dict)
                    conversation_context.append([sample.response, '$r', '#'+turn_id, r_kb_entry_tuple_list])
                else:
                    conversation_context.append([sample.response, '$r', '#' + turn_id, []])

                result.append(sample)
            else:  # this line is a knowledge base entity or profile k-v description
                if turn_id=='1':
                    conversation_context.append([line, '$pf', '#'+turn_id, []])
                else: # kb line;
                    # optional 1: kb lines to tuples, as the last element of api_call system response or profile
                    # multiple kb lines will be added to api_call sample, taken as $r
                    # [r for r in result if len(r.context) and 'api_call' in r.context[-1][0] and r.context[-1][-1]]
                    # api_call is the previous turn of kb facts
                    if len(conversation_context) and 'api_call' in conversation_context[-1][0]:
                        conversation_context[-1][-1].append(tuple(line.split() + ['$kb', '#'+turn_id]))
                    else:
                        # task 3 & 4: do not have api_call, then kb will be added to profile, taken as $pf
                        # print('conversation_context[-1]=', conversation_context[-1])
                        conversation_context[-1][-1].append(tuple(line.split() + ['$kb', '#'+turn_id]))  # $pf
                        # conversation_context[-1][-1].append(tuple(line.split() + ['$kb', '#'+turn_id]))  # $pf
                    # option 2: kb lines added to context
                    # conversation_context.append([line, '$kb', '#'+turn_id, kb_tuple_list])  # add $kb to context

    # result.sort(key=lambda x: len(x.context), reverse=True)
    return result



#-------------------------------------------------------------------------

def _load_knowledge_base(src: str):
    kb2id_dict = dict()
    id2kb_dict = dict()
    kb_vocab = []

    fp = src
    while os.path.basename(fp) != 'personalized-dialog-dataset':
        fp = os.path.dirname(fp)

    fp = os.path.join(fp, 'personalized-dialog-kb-all.txt')
    result = defaultdict(dict)

    for i, line in enumerate(_load_lines(fp)):
        # line: restaurant_name[\s]key[\t]value
        restaurant_name, key, value = re.split('\s', line)[1:]
        kb_vocab.extend([restaurant_name, key, value])
        result[restaurant_name][key] = value
        kb2id_dict[line.replace('\t',' ')] = i
        id2kb_dict[i] = line.replace('\t',' ')
    return result, kb2id_dict, id2kb_dict, sorted(list(set(kb_vocab)))

def _load_responses(src: str, task: str, kb_namekey2value_dict):
    response2id_dict = dict()
    id2response_dict = dict()

    if any(i in task for i in ['small', 'full']):
        fp = os.path.join(src, os.path.pardir)
    else:
        fp = src

    fp = os.path.join(fp, 'personalized-dialog-candidates.txt')
    candidates = []

    for i, line in enumerate(_load_lines(fp)):
        line = line.strip().split(' ', 1)[1]
        candidates.append((line, extract_kb_entry_from_str(line, kb_namekey2value_dict)))
        response2id_dict[line] = i
        id2response_dict[i] = line

    return candidates, response2id_dict, id2response_dict

def _load_lines(fp: str) -> List[str]:
    lines = []

    with open(fp) as f:
        for line in f:
            lines.append(line.strip())

    return lines

def _get_conversations_fp(src, task, mode):
    task_index = task[-1]
    if task_index not in '12345':
        task_index = '5'

    def _filename_filter(filename: str):
        _result = [
            filename.startswith('personalized-dialog-task{}-'.format(task_index)),
            'OOV' not in filename,
            mode in filename]
        return all(i for i in _result)

    fp = next(i for i in os.listdir(src) if _filename_filter(i))
    fp = os.path.join(src, fp)
    return fp

#---------------------------------------------------

def _build_meta_data(all_data: List[Sample], candidates: List[str], tokenizer):
    extra_info_posfixes = ['_phone', '_social_media', '_parking', '_public_transport']
    candidates_txt = [c[0] for c in candidates]
    extra_info_mask = [[1 if postfix in text else 0 for text in candidates_txt] for postfix in extra_info_posfixes]

    profile_vector_size = set(len(sample.profile.vector) for sample in all_data)
    assert len(profile_vector_size) == 1
    profile_vector_size = list(profile_vector_size)[0]

    max_context_length = max(len(sample.context) for sample in all_data)
    mean_context_length = sum(len(sample.context) for sample in all_data) / len(all_data)

    max_context_kb_length = 0
    # max_contexts = []
    for sample in all_data:
        context_list = []
        for c in sample.context:
            context_list.append(c[0])
            if 'api_call' in c[0]:
                context_list.extend([' '.join(t) for t in c[-1]])
        if len(context_list) > max_context_kb_length:
            max_context_kb_length = len(context_list)

        # if len(context_list) > 200 and 're welcome' in sample.response:
        #     max_contexts.append(sample)

    # there should be no mentions, or a fixed count of mentions of candidates
    assert len(set(len(sample.lambda_indices) for sample in all_data)) <= 2
    lambda_indices_size = max(len(sample.lambda_indices) for sample in all_data)
    lambda_indices_size = max(lambda_indices_size, 1)  # avoiding zero

    def split_token_len(s):
        return len(s.split())
    top_n_long_context = sorted(set([sent[0] for sample in all_data for sent in sample.context]), key=split_token_len, reverse=True)[:1000]
    max_sentence_length = max([len(tokenizer(s)) for s in top_n_long_context])
    top_n_long_query = sorted(set([sample.query for sample in all_data]), key=split_token_len, reverse=True)[:1000]
    max_question_length = max(len(tokenizer(s)) for s in top_n_long_query)
    max_sentence_length = max(max_sentence_length, max_question_length)

    min_k_neighbor = min([len(sample.neighbor_samples) for sample in all_data])

    meta_data = {
        'min_k_neighbor': min_k_neighbor,
        'max_sentence_length': max_sentence_length,
        'max_context_length': max_context_length,
        'max_context_kb_length': max_context_kb_length,
        'mean_context_length': mean_context_length,
        'candidates_size': len(candidates),
        'profile_vector_size': profile_vector_size,
        'lambda_indices_size': lambda_indices_size,
        'extra_info_mask': extra_info_mask
    }

    print(meta_data)

    return meta_data


#---------------------------------------------------
def _set_lambda_indices_to_samples(samples: List[Sample], candidates: List[str], knowledge_base):
    for sample in samples:
        mentions = _get_lambda_mention_indices(sample, candidates, knowledge_base)
        sample.lambda_indices = mentions[:]

# get a index list of candidates that mention the restaurant_name in the golden response (also must in knowledge_base)
def _get_lambda_mention_indices(sample: Sample, candidates: List[str], knowledge_base):
    entity_postfixes = ['_phone', '_social_media', '_parking', '_public_transport', '_address']

    mentions = []

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    should_use_kb = any(i in sample.query for i in ['direction', 'contact'])
    if should_use_kb:
        restaurant_name = babi_tokenizer(sample.response)[-1]
        for postfix in entity_postfixes:
            restaurant_name = restaurant_name.replace(postfix, '')
        assert restaurant_name in knowledge_base

        for i, c in enumerate(candidates):
            if restaurant_name in c[0]:
                mentions.append(i)

    return mentions


def build_babi_vocab(samples, candidates, tokenizer):
    dyn_vocab2id = dict({PAD_WORD: 0, BOS_WORD: 1, UNK_WORD: 2, EOS_WORD: 3, SEP_WORD: 4, CLS_WORD: 5, MASK_WORD: 6})
    dyn_id2vocab = dict({0: PAD_WORD, 1: BOS_WORD, 2: UNK_WORD, 3: EOS_WORD, 4: SEP_WORD, 5: CLS_WORD, 6: MASK_WORD})

    special_words = ['$pf', '#1', '$kb', '$u', '$r']
    for w in special_words:
        dyn_vocab2id[w] = len(dyn_vocab2id)
        dyn_id2vocab[len(dyn_id2vocab)] = w

    for sample in samples:
        for k, v in sample.profile.profile.items():
            if k not in dyn_vocab2id:
                dyn_vocab2id[k] = len(dyn_vocab2id)
                dyn_id2vocab[len(dyn_id2vocab)] = k
            if v not in dyn_vocab2id:
                dyn_vocab2id[v] = len(dyn_vocab2id)
                dyn_id2vocab[len(dyn_id2vocab)] = v

        for sent in sample.context:
            # sent [context_txt, $u, #3, [kb_tuple_list]]
            for w in tokenizer(sent[0])+sent[1:3]:
                if w not in dyn_vocab2id:
                    dyn_vocab2id[w] = len(dyn_vocab2id)
                    dyn_id2vocab[len(dyn_id2vocab)] = w

            # kb tuples
            if  'api_call' in sent[0] and len(sent[-1])>0:
                turn_id = int(sent[2][1])
                for kb_tuple in sent[-1]:
                    for w in tokenizer(' '.join(kb_tuple)):
                        # print('@', w)
                        if w not in dyn_vocab2id:
                            dyn_vocab2id[w] = len(dyn_vocab2id)
                            dyn_id2vocab[len(dyn_id2vocab)] = w

        # TODO: add profile_kb to vocab
        if sample.profile_kb:
            for kb_tuple in sample.profile_kb:
                for w in tokenizer(' '.join(kb_tuple)):
                    # print('@', w)
                    if w not in dyn_vocab2id:
                        dyn_vocab2id[w] = len(dyn_vocab2id)
                        dyn_id2vocab[len(dyn_id2vocab)] = w

        for w in tokenizer(sample.query):
            if w not in dyn_vocab2id:
                dyn_vocab2id[w] = len(dyn_vocab2id)
                dyn_id2vocab[len(dyn_id2vocab)] = w

        for w in tokenizer(sample.response):
            if w not in dyn_vocab2id:
                dyn_vocab2id[w] = len(dyn_vocab2id)
                dyn_id2vocab[len(dyn_id2vocab)] = w

    for sent in candidates:
        for w in tokenizer(sent[0]):
            if w not in dyn_vocab2id:
                dyn_vocab2id[w] = len(dyn_vocab2id)
                dyn_id2vocab[len(dyn_id2vocab)] = w

    return dyn_vocab2id, dyn_id2vocab

def main_logic():
    policy = args.neighbor_policy
    task_dir = '%s/%s-%s' % (src, task, policy)

    drop_attr = ''
    if args.keep_attributes is not None:
        for k in _keys:
            if k not in args.keep_attributes:
                drop_attr += '_%s' % k


    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    print('prepare knowledge base...')
    kbs_path = '%s/kbs.pkl' % task_dir
    if os.path.exists(kbs_path):
        kb_namekey2value_dict, kb2id_dict, id2kb_dict, kb_vocab = torch.load(kbs_path)
    else:
        kb_namekey2value_dict, kb2id_dict, id2kb_dict, kb_vocab = _load_knowledge_base(src)
        torch.save((kb_namekey2value_dict, kb2id_dict, id2kb_dict, kb_vocab), kbs_path)

    # _, vocab2id, id2vocab = babi_tokenizer(extra_vocab=kb_vocab)

    print('prepare response candidates...')
    # (candidates, response2id_dict, id2response_dict)
    # candidates (txt, list_of_kb_tuples)
    candidates_path = '%s/candidates.pkl' % task_dir
    if os.path.exists(candidates_path):
        candidates, response2id_dict, id2response_dict = torch.load(candidates_path)
    else:
        candidates, response2id_dict, id2response_dict = _load_responses(src, task, kb_namekey2value_dict)
        torch.save((candidates, response2id_dict, id2response_dict), candidates_path)

    print('prepare train_samples...')
    train_samples_path='%s/train.pkl' % task_dir
    if os.path.exists(train_samples_path):
        train_samples = torch.load(train_samples_path)
        vknn = None
    else:
        train_samples = _load_samples(src, task, 'trn', kb_namekey2value_dict, response2id_dict)
        _set_lambda_indices_to_samples(train_samples, candidates, kb_namekey2value_dict)
        if policy == 'h':
            bsl_set_neighbor_dialogues_to_samples(train_samples, train_samples)
        elif policy == 'k':
            vknn = _set_neighbor_dialogues_to_samples(train_samples, train_samples, k_neighbor)
        elif policy == 'hk':
            vknn = bsl_plus_knn_set_neighbor_dialogues_to_samples(train_samples, train_samples, k_neighbor)
        else:
            print('Do not have this policy to find neighbors...%s'%policy)
            exit(0)
        torch.save(train_samples, train_samples_path)

    print('prepare dev_samples...')
    dev_samples_path='%s/dev.pkl' % task_dir
    if os.path.exists(dev_samples_path):
        dev_samples = torch.load(dev_samples_path)
    else:
        dev_samples = _load_samples(src, task, 'dev', kb_namekey2value_dict, response2id_dict)
        _set_lambda_indices_to_samples(dev_samples, candidates, kb_namekey2value_dict)
        if policy == 'h':
            bsl_set_neighbor_dialogues_to_samples(train_samples, dev_samples)
        elif policy == 'k':
            _set_neighbor_dialogues_to_samples(train_samples, dev_samples, k_neighbor, vknn)
        elif policy == 'hk':
            bsl_plus_knn_set_neighbor_dialogues_to_samples(train_samples, dev_samples, k_neighbor, vknn)
        else:
            print('Do not have this policy to find neighbors...%s'%policy)
            exit(0)
        torch.save(dev_samples, dev_samples_path)

    print('prepare test_samples...')
    test_samples_path = '%s/test.pkl' % task_dir
    if os.path.exists(test_samples_path):
        test_samples = torch.load(test_samples_path)
    else:
        test_samples = _load_samples(src, task, 'tst', kb_namekey2value_dict, response2id_dict)
        _set_lambda_indices_to_samples(test_samples, candidates, kb_namekey2value_dict)
        if policy == 'h':
            bsl_set_neighbor_dialogues_to_samples(train_samples, test_samples)
        elif policy == 'k':
            _set_neighbor_dialogues_to_samples(train_samples, test_samples, k_neighbor, vknn)
        elif policy == 'hk':
            bsl_plus_knn_set_neighbor_dialogues_to_samples(train_samples, test_samples, k_neighbor, vknn)
        else:
            print('Do not have this policy to find neighbors...%s'%policy)
            exit(0)
        torch.save(test_samples, test_samples_path)

    print('prepare vocabulary...')
    vocab_path = '%s/vocab.pkl' % task_dir
    tokenizer = babi_tokenizer
    if os.path.exists(vocab_path):
        vocab2id, id2vocab = torch.load(vocab_path)
    else:
        vocab2id, id2vocab = build_babi_vocab(train_samples+dev_samples+test_samples, candidates, tokenizer)
        torch.save((vocab2id, id2vocab), vocab_path)
    print('vocabulary size=', len(vocab2id))

    print('prepare meta data and compute global objects...')
    meta_data_path = '%s/meta.pkl' % task_dir
    if os.path.exists(meta_data_path):
        meta_data = torch.load(meta_data_path)
    else:
        meta_data = _build_meta_data(train_samples+dev_samples+test_samples, candidates, tokenizer)
        torch.save(meta_data, meta_data_path)


    print('prepare train dataset objects...')
    train_dataset_path = '%s/train.ctds-%s%s.pkl' % (task_dir, args.profile_dropout_ratio, drop_attr)
    candidate_tensor_path = '%s/candidate.ctds.pkl' % task_dir
    if os.path.exists(train_dataset_path):
        train_sample_tensor = torch.load(train_dataset_path)
        candidate_tensor = torch.load(candidate_tensor_path)
    else:
        train_dataset = CTDSDataset(train_samples, candidates, meta_data, tokenizer, vocab2id, id2vocab, train_sample_tensor=None)
        candidate_tensor = train_dataset.candidate_tensor
        torch.save(candidate_tensor, candidate_tensor_path)

        train_sample_tensor = train_dataset.sample_tensor
        torch.save(train_sample_tensor, train_dataset_path)

    print('prepare dev dataset objects...')
    dev_dataset_path='%s/dev.ctds-%s%s.pkl' % (task_dir, args.profile_dropout_ratio, drop_attr)
    if os.path.exists(dev_dataset_path):
        dev_sample_tensor = torch.load(dev_dataset_path)
    else:
        dev_dataset = CTDSDataset(dev_samples, candidates, meta_data, tokenizer, vocab2id, id2vocab, train_sample_tensor=train_sample_tensor)
        dev_sample_tensor = dev_dataset.sample_tensor
        torch.save(dev_sample_tensor, dev_dataset_path)

    print('prepare test dataset objects...')
    test_dataset_path='%s/test.ctds-%s%s.pkl' % (task_dir, args.profile_dropout_ratio, drop_attr)
    if os.path.exists(test_dataset_path):
        test_sample_tensor = torch.load(test_dataset_path)
    else:
        test_dataset = CTDSDataset(test_samples, candidates, meta_data, tokenizer, vocab2id, id2vocab, train_sample_tensor=train_sample_tensor)
        test_sample_tensor = test_dataset.sample_tensor
        torch.save(test_sample_tensor, test_dataset_path)


if __name__ == "__main__":
    start = datetime.now()
    main_logic()
    end = datetime.now()
    print('run time:%.2f mins'% ((end-start).seconds/60))
