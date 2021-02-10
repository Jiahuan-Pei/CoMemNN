import torch.nn as nn
from common.Utils import *
from common.DotAttention import *
from CTDS.Config import *
from CTDS.Utils import one_hot
from CTDS.Profile import pred_profile_to_onehot_vec

class CTDS(nn.Module):
    def __init__(self, hidden_size, vocab2id, id2vocab, candidate_tensor, meta_data, dropout=0.1):
        super(CTDS, self).__init__()

        self.hidden_size=hidden_size
        self.id2vocab=id2vocab
        self.vocab_size=len(id2vocab)
        self.vocab2id=vocab2id
        self.candidate_tensor=candidate_tensor
        self.meta = meta_data
        if args.max_profile_vector_size != -1: # pad profile if less than max_profile_vector_size
            self.profile_vector_size = args.max_profile_vector_size
        else:
            self.profile_vector_size = self.meta['profile_vector_size']

        self.token_emb = nn.Embedding(self.vocab_size, hidden_size, padding_idx=0) # [44965, 4]

        self.attn = DotAttention()
        # self.neighbor_attn = self.attn
        self.neighbor_attn = DotAttention()
        self.neighbor_profile_attn = DotAttention()
        # self.highway=Highway(2*hidden_size, hidden_size)

        self.linear1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear3 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear4 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_p = nn.Linear(hidden_size, self.profile_vector_size, bias=False)

        self.profile_emb = nn.Linear(self.profile_vector_size, hidden_size, bias=False)  # [22, 4]
        self.neighbor_profile_emb = nn.Linear(self.profile_vector_size, hidden_size, bias=False)  # [22, 4]
        self.preference_emb = nn.Linear(self.profile_vector_size, len(self.meta['extra_info_mask']), bias=False)  # [18, 4]

        self.layer_norm = nn.LayerNorm(self.meta['candidates_size'])
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        with torch.no_grad():
            print('reset zero padding')
            self.token_emb.weight[0].fill_(0) # 0-th is padding, should be reset as zero tensor

    def seq_encode(self, token_seq):
        batch_size, num_seq, seq_len = token_seq.size()
        token_seq = token_seq.reshape(-1, seq_len) # [3, 4, 20] -> [12, 20]
        seq_mask = token_seq.ne(0).detach() # True if element is not 0: PAD

        seq_enc = self.token_emb(token_seq) # [12, 20] -> [12, 20, 4]
        seq_enc = seq_enc*seq_mask.unsqueeze(-1)

        seq_enc = seq_enc.reshape(batch_size, num_seq, seq_len, -1)  # [12, 20, 4] -> [3, 4, 20, 4]
        seq_mask = seq_mask.reshape(batch_size, num_seq, seq_len)  # [3, 4, 20]

        return seq_enc, seq_mask

    def preference_embedding(self, profile, lambda_indices, response_enc):
        extra_info_scores = self.preference_emb(profile.float())  # [3, 4]
        scores_on_candidates = torch.mm(extra_info_scores, torch.tensor(self.meta['extra_info_mask'], device=detected_device).float())  # [3, C]
        one_hot_lambda_indices = one_hot(lambda_indices, self.meta['candidates_size'])  # [3, L, C]
        mentions_mask = one_hot_lambda_indices.sum(axis=1) #
        preference_bias = scores_on_candidates * mentions_mask
        return preference_bias

    def context_memory_reading(self, q, c):
        '''
        :param q: [B, 1, H]
        :param context_enc: [B, N, H]
        :return: c [B, 1, H]
        '''
        # batch_size = context_enc.size(0)

        # context_enc = context_enc.reshape(batch_size, -1, context_enc.size(3))  # [3, 4, 20, 4] -> [3, 80, 4]

        c, _, _ = self.attn(q, c, c)  # [3, 1, 4]
        # qc = self.highway(torch.cat([q, c], dim=-1))  # [3, 1, 4]
        # qc = q + c
        return c

    def neighbor_context_memory_reading(self, q, c):
        '''
        :param q: [B, 1, H]
        :param context_enc: [B, N, H]
        :return: c [B, 1, H]
        '''
        # batch_size = context_enc.size(0)

        # context_enc = context_enc.reshape(batch_size, -1, context_enc.size(3))  # [3, 4, 20, 4] -> [3, 80, 4]

        c, _, _ = self.neighbor_attn(q, c, c)  # [3, 1, 4]
        # qc = self.highway(torch.cat([q, c], dim=-1))  # [3, 1, 4]
        # qc = q + c
        return c

    def neighbor_profile_encode(self, neighbor_profile):
        '''
        :param neighbor_profile: [B, N, D]
        :return: c [B, N, H]
        '''
        batch_size, num_neighbor_profile, profile_vector_size= neighbor_profile.size()
        neighbor_profile_enc_list = []
        for i in range(num_neighbor_profile):
            neighbor_profile_enc_list.append(self.neighbor_profile_emb(neighbor_profile[:, i, :].float()))
        return torch.stack(neighbor_profile_enc_list, dim=1)

    def neighbor_profile_memory_reading(self, profile_enc, neighbor_profile_enc):
        '''
        :param profile: [B, 1, H]
        :param neighbor_profile: [B, N, H]
        :return: neighbor_profile [B, 1, H]
        '''
        neighbor_profile_enc, _, _ = self.neighbor_profile_attn(profile_enc, neighbor_profile_enc, neighbor_profile_enc)  # [3, 1, 4]
        return neighbor_profile_enc

    def profile_updating(self, incomplete_profile_memory, context_memory, neighbor_context_memory, neighbor_profile_memory, w1=args.pp_p, w2=args.pp_np, w3=args.pp_c, w4=args.pp_nc, use_neighbor=args.use_neighbor): # neighbor_N * profile_size
        # incomplete_profile_enc [B, H], context_memory [B, 1, H], neighbor_context_memory [B, 1, H], neighbor_profile_memory [B, 1, H]
        if use_neighbor==1:
            p = w1*incomplete_profile_memory + w2*neighbor_profile_memory.squeeze(1) + w3*context_memory.squeeze(1) + w4*neighbor_context_memory.squeeze(1)
        else:
            p = w1 * incomplete_profile_memory + w3*context_memory.squeeze(1)
        pred_profile_enc = self.linear_p(p)
        return pred_profile_enc

    # DRS module
    # ''' DRS: v4
    def dialogue_response_selection(self, profile, incomplete_profile, neighbor_profile, context, neighbor_context, query, response, lambda_indices):
        # profile[3, 22], context[3, 4, 20], query[3, 1, 20], response[1, C, 20], neighbor_context[3, N, 20]

        batch_size = query.size(0)
        hidden_size = self.hidden_size
        _, num_response, response_len = response.size()  # 3, C, 20
        _, num_context, context_len = context.size() # [3, 4, 20, 4]
        _, k, num_neighbor_context, neighbor_context_len = neighbor_context.size()  # [B, k, N, SenLen]

        ## profile
        incomplete_profile_enc = self.profile_emb(incomplete_profile.float())
        pred_profile = None
        pred_profile_enc = incomplete_profile_enc

        ## query
        query_enc, query_mask = self.seq_encode(query)  # [3, 1, 20, 4]
        query_enc = query_enc.squeeze(1)  # [3, 20, 4]
        query_mask = query_mask.squeeze(1)  # [3, 20]
        q = universal_sentence_embedding(query_enc, query_mask).unsqueeze(1)  # sum [3, 1, 4]

        ## context
        context_enc, context_mask = self.seq_encode(context) # [3, k*4, 20, 4]
        c = universal_sentence_embedding(context_enc.reshape(batch_size*num_context, context_len, hidden_size),
                                         context_mask.reshape(batch_size*num_context, -1)).unsqueeze(1) # sum
        c = c.reshape(batch_size, num_context, hidden_size) # [3, 4, 4]

        ## neighbor context
        # num_global_context = min(1000, k*num_neighbor_context)
        num_global_context = k*num_neighbor_context
        neighbor_context = neighbor_context.reshape(batch_size, -1, neighbor_context_len)[:, :num_global_context, :]
        neighbor_context_enc, neighbor_context_mask = self.seq_encode(neighbor_context)  # [3, 4, 20, 4]
        neighbor_c = universal_sentence_embedding(neighbor_context_enc.reshape(batch_size*num_global_context, neighbor_context_len, hidden_size),
                                                neighbor_context_mask.reshape(batch_size*num_global_context, -1)).unsqueeze(1) # sum
        neighbor_c = neighbor_c.reshape(batch_size, num_global_context, hidden_size) # [3, 4, 4]

        ## response
        response_enc, response_mask = self.seq_encode(response)  # [1, C, 20, 4]

        qs = [q]  # [3, 1, 4]

        if args.use_neighbor:
            q = qs[0] # note that here we should use the original question
            neighbor_c = self.neighbor_context_memory_reading(q, neighbor_c)

        for i in range(args.hops):
            q = qs[-1]

            # [3, 1, 4]
            c = self.context_memory_reading(q, c)
            q = q + c
            # q = self.highway(torch.cat([q, c], dim=-1))
            q = self.linear1(q)  # [3, 1, 4]

            if args.use_profile:
                ## neighbor profile
                neighbor_profile_enc = self.neighbor_profile_encode(neighbor_profile)
                neighbor_profile_enc = self.neighbor_profile_memory_reading(incomplete_profile_enc.unsqueeze(1),
                                                                            neighbor_profile_enc)

                # profile update
                # pred_profile = self.profile_updating(incomplete_profile_enc, c, neighbor_c, neighbor_profile_enc)
                pred_profile = self.profile_updating(pred_profile_enc, c, neighbor_c, neighbor_profile_enc)
                pred_profile_enc = self.profile_emb(pred_profile.float())
                q = q + pred_profile_enc.unsqueeze(1)
                q = self.linear2(q)  # [3, 1, 4]

            qs.append(q)


        if args.use_neighbor:
            q = qs[-1] + neighbor_c
            q = self.linear3(q)
            qs.append(q)

        # response_enc [1, C, 20, 4], response_mask [1, C, 20] --> r [1, C, 4]
        r = universal_sentence_embedding(response_enc.reshape(-1, response_len, hidden_size),
                                         response_mask.reshape(-1, response_len))#.reshape(batch_size, num_response, hidden_size)
        r = r.expand(batch_size, -1, -1) # r [3, C, 4]
        r = self.linear4(r)  # [3, C, 4]

        q = qs[-1]  # [3, 1, 4]
        logits = torch.bmm(r, q.transpose(1, 2)).squeeze(-1)  # [3, C, 4], [3, 1, 4] -> [3, C]
        # F.softmax(logits, dim=1)

        if args.use_profile:
            revised_mask = torch.sigmoid(torch.bmm(r, pred_profile_enc.unsqueeze(1).transpose(1, 2)).squeeze(-1))  # [3, C, 4], [3, 1, 4] -> [3, C, 1] -> [3, C]
            logits = logits * revised_mask

        if args.use_preference:
            preference_bias = self.preference_embedding(incomplete_profile, lambda_indices, response_enc)
            # if torch.sum(preference_bias).item() > 0:
            #     print('<preference>'*3, torch.sum(preference_bias).item())
            logits = logits + preference_bias

        # logits = self.layer_norm(logits)
        # logits = self.dropout(logits)

        return logits, pred_profile

    def do_train(self, data, check_txt_data=False):
        # print(data)
        losses = []

        profile = data['profile']
        incomplete_profile = data['incomplete_profile']
        neighbor_profile = data['neighbor_profile'] # [B, k, ProfileLen]
        if profile.size(1) < self.profile_vector_size:
            profile = torch.cat((profile.float(),torch.zeros(profile.size(0), self.profile_vector_size-profile.size(1), device=detected_device)), dim=-1)
            neighbor_profile = torch.cat((neighbor_profile.float(),
                                          torch.zeros(neighbor_profile.size(0), neighbor_profile.size(1), self.profile_vector_size-neighbor_profile.size(2), device=detected_device)), dim=-1)

        context = data['context'] # [B, N, SenLen]
        neighbor_context = data['neighbor_context'] # [B, k, N, SenLen]
        query = data['query'].unsqueeze(1)
        response = self.candidate_tensor.unsqueeze(0)#.expand(query.size(0), -1, -1)
        response = response.cuda() if torch.cuda.is_available() else response
        response_id = data['response_id']
        lambda_indices = data['lambda_indices']


        # if check_txt_data:
        #     profile_txt = []
        #     context_txt = [[' '.join([self.id2vocab[idx.item()] for idx in sent]) for sent in a_context] for a_context in context]
        #     query_txt = [[' '.join([self.id2vocab[idx.item()] for idx in sent]) for sent in a_query] for a_query in query]
        #     response_idx = [response[i][rid] for i, rid in enumerate(response_id)]
        #     response_txt = [' '.join([self.id2vocab[idx.item()] for idx in a_response]) for a_response in response_idx]
        #     sample_id = data['sample_id']
        # [3, C]
        prediction, pred_profile = self.dialogue_response_selection(profile, incomplete_profile, neighbor_profile, context, neighbor_context, query, response, lambda_indices)

        # multiple classification
        if args.learn_mode == 'multiple':
            if args.num_negative_samples == -1:
                target = response_id
            else:
                target = torch.zeros_like(response_id)
            # loss_fct = nn.CrossEntropyLoss() # loss3
            # loss_sel = loss_fct(prediction, response_id)
            loss_sel = F.cross_entropy(prediction, target) # loss of drs

            if args.use_loss_p:
                loss_p = F.mse_loss(pred_profile, profile.float())  # profile completion loss
            else:
                loss_p = 0.0

            losses.append(loss_sel+loss_p)
            # if makedir_flag:
            #     print('prediction_num=%s' % len(prediction)) # number of train_batch_size
            #     for i, (p, t) in enumerate(zip(prediction, target)):
            #         p = F.softmax(p, dim=0)
            #         print('i=%s' %i, '#'*50)
            #         print('prediction_id=%s, target_id=%s' % (p.argmax().item(), t.item()))
            #         print('prediction[prediction_id]=%s, prediction[target_id]=%s' % (p[p.argmax()].item(), p[t].item()))
                #     print('prediction=%s\ntarget=%s' % (p, t))

        elif args.learn_mode == 'binary':  # binary classification
            # target [B, C], each element is an one-hot vector
            target = torch.zeros_like(prediction)
            weight = torch.ones_like(prediction)*10e-3
            lambda_positive = 0.9
            if args.num_negative_samples == -1:
                for i in response_id:
                    target[:, i] = 1
                    weight[:, i] = lambda_positive
            else:
                target[:, 0] = 1
                weight[:, 0] = lambda_positive

            loss_sel = F.binary_cross_entropy_with_logits(prediction, target, weight=weight)
            losses.append(loss_sel)

            # if makedir_flag:
            #     print('prediction_num=%s' % len(prediction)) # number of train_batch_size
            #     for i, (p, t) in enumerate(zip(prediction, target)):
            #         p = F.softmax(p, dim=1)
            #         print('i=%s' %i, '#'*50)
            #         print('prediction_id=%s, target_id=%s' % (p.argmax().item(), t.argmax().item()))
            #         print('prediction[prediction_id]=%s, prediction[target_id]=%s' % (p[p.argmax()].item(), p[t.argmax()].item()))
                #     print('prediction=%s\ntarget=%s' % (p, t))
        else:
            pass
        return losses

    # def pred_profile_to_onehot_vec(self, pred_profile):
    #     profile_len = pred_profile.size(1)
    #     v = torch.zeros_like(pred_profile)
    #     v[:, 0:2][:, pred_profile[:, 0:2].argmax(dim=1)] = 1  # gender, 0, 1
    #     v[:, 2:5][:, pred_profile[:, 2:5].argmax(dim=1)] = 1  # age, 2, 3, 4
    #     if profile_len > 5:
    #         v[:, 5:7][:, pred_profile[:, 5:7].argmax(dim=1)] = 1  # dietary, 5, 6
    #         v[:, 7:][:, pred_profile[:, 7:].argmax(dim=1)] = 1    # favorite, 7-21
    #     return v

    def do_infer(self, data):
        profile = data['profile']
        incomplete_profile = data['incomplete_profile']
        neighbor_profile = data['neighbor_profile']
        if profile.size(1) < self.profile_vector_size:
            profile = torch.cat((profile.float(),torch.zeros(profile.size(0), self.profile_vector_size-profile.size(1), device=detected_device)), dim=-1)
            neighbor_profile = torch.cat((neighbor_profile.float(),
                                          torch.zeros(neighbor_profile.size(0), neighbor_profile.size(1), self.profile_vector_size-neighbor_profile.size(2), device=detected_device)), dim=-1)

        context = data['context']
        neighbor_context = data['neighbor_context']  # [B, k*N, SenLen]
        query = data['query'].unsqueeze(1)
        response = self.candidate_tensor.unsqueeze(0)#.expand(query.size(0), -1, -1)
        response = response.cuda() if torch.cuda.is_available() else response
        response_id = data['response_id']
        lambda_indices = data['lambda_indices']

        # print(data)
        if args.num_infer_response == -1:
            prediction, pred_profile = self.dialogue_response_selection(profile, incomplete_profile, neighbor_profile, context, neighbor_context, query, response, lambda_indices)
        else:
            ps = []
            pps = []
            i = 0
            step = args.num_infer_response
            # 43863 candidates decomposed
            while i < response.size(1):
                p, pred_p = self.dialogue_response_selection(profile, incomplete_profile, neighbor_profile, context, neighbor_context, query, response[:, i:i + step], lambda_indices)
                ps.append(p)
                pps.append(pred_p)
                i += step
                if i > response.size(1):
                    i = response.size(1)
            # 43863 candidates composed
            prediction = torch.cat(ps, dim=1)
            pred_profile = torch.cat(pps, dim=1)

        # if makedir_flag:
        #     print('prediction_id=', prediction.argmax(dim=1))
        #     print('all_target=', response_id)
        #     print('all_prediction=', prediction)
        pred_profile_label = pred_profile_to_onehot_vec(pred_profile)
        pred_output = {}
        pred_output['pred_response_id'] = prediction.argmax(dim=1).cpu()
        pred_output['pred_profile'] = pred_profile_label.cpu()
        pred_output['pred_profile_prob'] = pred_profile.cpu()
        return pred_output

    def find_neighbors(self, data, k=100, policy='head'):
        if policy == 'head':
            data['neighbor_profile'] = data['neighbor_profile'][:, :k, :] # [B, k, ProfileLen]
            data['neighbor_context'] = data['neighbor_context'][:, :k, :] # [B, k , SenLen]
        elif policy == 'knn':
            from common.KNN import Vector_KNN
            d = data['profile'].size(1)
            neighbor_profile_list = []
            neighbor_context_list = []

            for i in range(data['profile'].size(0)):
                vector_to_index = data['neighbor_profile'][i].float()  # [#samples, d]
                vknn = Vector_KNN(d)
                vknn.index(vector_to_index)
                # search & get a list of index
                sample_index_list = vknn.search(data['incomplete_profile'][i].float().unsqueeze(0), k).squeeze(0)
                # get a list of sample_id without same dialouge_id
                new_neighbor_profile = torch.stack([data['neighbor_profile'][i][sample_index] for sample_index in sample_index_list])
                new_neighbor_context = torch.stack([data['neighbor_context'][i][sample_index] for sample_index in sample_index_list])
                neighbor_profile_list.append(new_neighbor_profile)
                neighbor_context_list.append(new_neighbor_context)
            data['neighbor_profile'] = torch.stack(neighbor_profile_list)
            data['neighbor_context'] = torch.stack(neighbor_context_list)
        return data

    def forward(self, data, method='train'):
        # data = self.find_neighbors(data, k=args.k, policy=args.neighbor_policy)
        if method == 'train':
            return self.do_train(data)
        elif method == 'infer':
            return self.do_infer(data)


