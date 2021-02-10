from transformers import *
from common.Constants import *
import re
import torch
import torch.nn.functional as F

def one_hot(indices, depth):
    '''
    :param indices: [Batch, N_Label_Nums]
    :param depth: Max_Label_Num
    :return: [Batch, Max_Label_Num], each element indicate how many times the corresponding label mentioned in indices[i, :]
    '''
    revised_indices = indices.masked_fill(indices.eq(-1), 0).long() # fill 0 when eq to -1
    revised_indices = F.one_hot(revised_indices, depth) # [3, 4, C]
    mask = torch.zeros_like(indices).masked_fill(indices.ne(-1), 1) # [3, 4]
    return revised_indices * mask.unsqueeze(-1)

def babi_tokenizer(sentence):
    """Gets the tokens of a sentence including punctuation (except for the last one).

    Parameters
    ----------
    sentence : str
        The raw text of a sentence.

    Returns
    -------
    list of str
        A list of tokens in the sentence.

    Examples
    --------
    >>> babi_tokenizer('Bob dropped the apple. Where is the apple?')
    ['bob', 'dropped', 'apple', '.', 'where', 'is', 'apple']
    >>> babi_tokenizer('$u #1 [UNK] hi_you_ki <SLI> how are you non-veg')
    ['$u', '#1', '[UNK]', 'hi_you_ki', '<SLI>', 'how', 'are', 'you', 'non-veg']
    """
    _STOP_WORDS = {"a", "an", "the"}
    sentence = sentence.lower()
    if sentence == '<silence>':
        return [sentence]
    result = [x.strip() for x in re.split('([^A-Za-z0-9-_(<\w+>|\[\w+\]|#\d+|$\w+)]+)', sentence) if
              x.strip() and x.strip() not in _STOP_WORDS]
    if not result:
        result = ['<silence>']
    if result[-1] in '.!?':
        result = result[:-1]
    # pp added: to test if it improves Task 4
    for r in result:
        if '_' in r: # entity
            result[:-1].extend(r.split('_')) # add the constitute of name_key_value entities at the end of a sentence
        else:
            pass
    return result

def babi_bert_tokenizer(uncased=True, extra_vocab=None):
    if uncased:
        t = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True) # 30522 words by default
        # m = BertModel.from_pretrained("bert-base-uncased")
    else:
        t = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)
        # m = BertModel.from_pretrained("bert-base-cased")

    special_tokens_dict = {'additional_special_tokens':
                               ['<silence>', 'api_call', 'non-veg'] + extra_vocab
                           }
    if special_tokens_dict!=None:
        # print('demo@1@', t.vocab_size, len(t)) # 30522 30522 | 30522 30522
        t.add_special_tokens(special_tokens_dict)
        # print('demo@2@', t.vocab_size, len(t)) # 30522 30527 | 30522 44965 (added 14443)
        # t.vocab.update(t.added_tokens_encoder)
        # print('demo@3@', t.vocab_size, len(t)) # 30527 30532 | 44965 59408
        # t.ids_to_tokens.update(t.added_tokens_decoder)
        # print('demo@4@', t.vocab_size, len(t)) #30527 30532  | 44965 59408

    def tokenize_(seq):
        if seq == '<silence>':
            return [seq]
        return t.tokenize(seq)

    # print('@@@', t.vocab_size, len(t.vocab), len(t))
    # return tokenize_, {**t.vocab, **t.added_tokens_encoder}, {**t.ids_to_tokens, **t.added_tokens_decoder}
    return tokenize_, {**t.vocab, **t.added_tokens_encoder}, {**t.ids_to_tokens, **t.added_tokens_decoder}

def babi_bert_tokenizer_test_demo():
    kb_vocab = ['R_type', 'resto_paris_moderate_italian_2stars_2']
    tokenizer, vocab2id, id2vocab = babi_tokenizer(extra_vocab=kb_vocab)
    print('vocab_len=', len(vocab2id))
    sentences = [
        '76 what food are you looking for [UNK]',
        '<SILENCE>',
        '[CLS] [UNK]',
        '69 resto_paris_moderate_italian_2stars_2 R_type non-veg',
        'api_call spanish london eight expensive',
        'actually i would prefer for eight people	i shall modify your reservation is there any other change',
        # 'here is the information you asked for resto_bombay_moderate_spanish_6stars_1_address resto_bombay_moderate_spanish_6stars_1_parking'
         ]

    for s in sentences:
        print('-' * 10, s)
        # must be lower case expect the bert special tokens, e.g., [UNK]
        s = ' '.join([w.lower() if not re.match('\[\w+\]', w) else w for w in s.split()])
        for w in tokenizer(s):
            print('%s(%s)' % (w, vocab2id[w]), end='  ')
        print('')

# babi_tokenizer_test_demo()