from torch.utils.data import Dataset
from common.Utils import *
from common.Constants import *
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from .Config import makedir_flag, args
from .Dropout_profile import profile_dropout

class CTDSDataset(Dataset):
    def __init__(self, samples, candidates, meta, tokenizer, vocab2id, id2vocab, num_negative_samples=-1,n=1E10, sample_tensor=None, train_sample_tensor=None):
        super(CTDSDataset, self).__init__()
        self.meta=meta
        self.tokenizer=tokenizer
        self.num_negative_samples=num_negative_samples

        if sample_tensor is None:
            self.samples = samples
            self.vocab2id = vocab2id
            self.id2vocab = id2vocab
            self.candidates = candidates
            self.candidate_tensor = None
            self.n = n
            self.sample_tensor = []
            self.train_sample_tensor = train_sample_tensor
            self.load()
        else:
            self.samples = samples
            self.candidates = candidates
            self.sample_tensor = sample_tensor
            self.train_sample_tensor = train_sample_tensor

        self.len=len(self.sample_tensor)

        if makedir_flag:
            print('data size: ', self.len)
            for k, v in self.meta.items():
                if k != 'extra_info_mask':
                    print('%s: %s' % (k, v))
            print('candidate size:', len(self.candidates))

    def tokens2ids(self, list_of_tokens):
        return [self.vocab2id.get(w) if w in self.vocab2id else self.vocab2id[UNK_WORD] for w in list_of_tokens]

    def load(self):
        sent_len = self.meta['max_sentence_length'] + 2 # plus two ($role, #turn_id)
        num_context = self.meta['max_context_length']
        # k_neighbor = min(self.meta['min_k_neighbor'], args.k)
        # k_neighbor = 100
        response_sent=[]
        for r in tqdm(self.candidates, desc='response', disable=True):
            r_sent = self.tokenizer(r[0])[-sent_len:]
            for w in r_sent:
                if w not in self.vocab2id:
                    print('<UNK>', w)
            if len(r_sent) < sent_len:
                r_sent = r_sent + [PAD_WORD] * (sent_len - len(r_sent))
            # r_sent = [CLS_WORD]+r_sent
            response = torch.tensor([self.vocab2id.get(w, self.vocab2id[UNK_WORD]) for w in r_sent], requires_grad=False).long()
            response_sent.append(response)
        self.candidate_tensor = torch.stack(response_sent)

        for id in tqdm(range(len(self.samples)), desc='sample', disable=True):
            sample = self.samples[id]

            id_tensor = torch.tensor([id]).long()
            # sample_id_tensor = torch.tensor([sample.sample_id]).long()

            neighbor_ids_list = []

            for neighbor_sample_id in sample.neighbor_samples:
                # sample_id = '%s_%s_%s_%s' % (sample_count, dialogue_count, turn_id, line_id+1)
                neighbor_id = int(neighbor_sample_id.split('_')[0])
                if self.train_sample_tensor is None: # for train data
                    if neighbor_sample_id == self.samples[neighbor_id].sample_id:
                        neighbor_ids_list.append(neighbor_id)
                    else:
                        print('err!!!')
                else: # for dev and test data
                    if neighbor_id == self.train_sample_tensor[neighbor_id][0]:
                        neighbor_ids_list.append(neighbor_id)
                    else:
                        print('err!!!')
            neighbor_ids_tensor = torch.tensor(neighbor_ids_list).long()


            context_list = []
            # TODO: add kb if profile_kb is not None
            if sample.profile_kb:
                context_list.extend([' '.join(fact) for fact in sample.profile_kb])
            for c in sample.context:
                context_list.append(' '.join(c[0:-1]))  # plus $role and #turn_id at the end
                if 'api_call' in c[0] and len(c[-1])>0: # add kb to context
                    context_list.extend([' '.join(t) for t in c[-1]])

            context_sent=[]
            for c in context_list:
                c_sent=self.tokenizer(c)[-sent_len:]
                if len(c_sent)<sent_len:
                    c_sent=c_sent+[PAD_WORD]*(sent_len-len(c_sent))
                # c_sent = [CLS_WORD] + c_sent
                context_sent.append(torch.tensor([self.vocab2id.get(w, self.vocab2id[UNK_WORD]) for w in c_sent]).long())
                for w in c_sent:
                    if w not in self.vocab2id:
                        print('<UNK>', w)
            # pad if less than num_context
            while len(context_sent)<num_context:
                context_sent = [torch.tensor([0]*sent_len).long()] + context_sent
            # cut off if test/dev has longer context
            context_sent = context_sent[-num_context:]
            context_sent_tensor=torch.stack(context_sent)


            query = self.tokenizer(sample.query)[-sent_len:]
            if len(query) < sent_len:
                query = query + [PAD_WORD] * (sent_len - len(query))
            # query = [CLS_WORD] + query
            query_sent_tensor=torch.tensor([self.vocab2id.get(w, self.vocab2id[UNK_WORD]) for w in query]).long()
            for w in query:
                if w not in self.vocab2id:
                    print('<UNK2>', w)
            response_id_tensor=torch.tensor([sample.response_id]).long()

            profile_tensor=torch.tensor(sample.profile.vector).long()

            ### dropout profile
            if 'k' not in args.neighbor_policy: # do not use knn, then must keep gender, age
                keep_attr_list = ['gender', 'age']
            else:
                if args.keep_attributes is None:
                    keep_attr_list = None
                else:
                    keep_attr_list = args.keep_attributes.split('_')
            incomplete_profile_tensor = profile_dropout(profile_tensor, profile_dropout_ratio=args.profile_dropout_ratio, keep_attributes=keep_attr_list)

            lambda_indices = sample.lambda_indices if len(sample.lambda_indices) > 0 else [-1] * self.meta['lambda_indices_size']
            lambda_indices_tensor=torch.tensor(lambda_indices).long()

            # self.sample_tensor.append([id_tensor, sample_id_tensor, sample_id_tensor, neighbor_ids_tensor, profile_tensor, incomplete_profile_tensor, context_sent_tensor, query_sent_tensor, response_id_tensor, lambda_indices_tensor])
            self.sample_tensor.append([id_tensor,
                                       # sample_id_tensor,
                                       neighbor_ids_tensor,
                                       profile_tensor,
                                       incomplete_profile_tensor,
                                       context_sent_tensor,
                                       query_sent_tensor,
                                       response_id_tensor,
                                       lambda_indices_tensor])
            # print(sample.neighbor_samples)

            if id>=self.n:
                break

    def __getitem__(self, index): # simple logic for gpu use; currently compute on cpu
        neighbor_context_list = []
        neighbor_profile_list = []
        '''
        # 0:id, 1:sample_id, 2:neighbor_ids, 3: profile, 4: incomplete_profile, 5: context, 6:query, 7:response_id, 8:lambda_indices, 9:neighbor_profile, 10:neighbor_context
        for neighbor_id in self.sample_tensor[index][2]: # 2:neighbor_ids
            try:
                neighbor_context = self.train_sample_tensor[neighbor_id][5] # 5: context
                neighbor_context_list.append(neighbor_context)

                neighbor_profile = self.train_sample_tensor[neighbor_id][4] # 4: incomplete_profile
                neighbor_profile_list.append(neighbor_profile)
            except:
                print('neighbor_id err!')
                pass
        '''
        # id, neighbor_ids, 2: profile, 3: incomplete_profile, 4: context, query, response_id, lambda_indices, neighbor_profile, neighbor_context
        for neighbor_id in self.sample_tensor[index][1]:
            try:
                neighbor_context = self.train_sample_tensor[neighbor_id][4]
                neighbor_context_list.append(neighbor_context)

                # neighbor_profile = self.train_sample_tensor[neighbor_id][2]
                neighbor_profile = self.train_sample_tensor[neighbor_id][3]
                neighbor_profile_list.append(neighbor_profile)
            except:
                print('neighbor_id err!')
                pass
        # neighbor_context_list = neighbor_context_list[:1000]
        neighbor_context_tensor = torch.stack(neighbor_context_list)  # concatenate all neighbor context to 1
        neighbor_profile_tensor = torch.stack(neighbor_profile_list)
        return self.sample_tensor[index] + [neighbor_profile_tensor, neighbor_context_tensor]

    def __len__(self):
        return self.len

def collate_fn(data): # combine batch
    # id, sample_id, neighbor_ids, profile, incomplete_profile, context, query, response_id, lambda_indices, neighbor_profile, neighbor_context = zip(*data)
    id, neighbor_ids, profile, incomplete_profile, context, query, response_id, lambda_indices, neighbor_profile, neighbor_context = zip(*data)

    return {
            'id': torch.cat(id),
            # 'sample_id': torch.cat(sample_id),d
            'profile': torch.stack(profile),
            'incomplete_profile': torch.stack(incomplete_profile),
            'neighbor_profile': torch.stack(neighbor_profile),
            'context': torch.stack(context),
            'neighbor_context': torch.stack(neighbor_context),
            'query': torch.stack(query),
            # 'response': torch.stack(response),
            'response_id': torch.cat(response_id),
            'lambda_indices': torch.stack(lambda_indices)
            }