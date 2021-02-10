import sys
sys.path.append('./')
from torch import optim
from common.CumulativeTrainer import *
from CTDS.CTDSDataset import *
if args.bsl:
    from CTDS.Model_bsl import *
else:
    from CTDS.Model import *
from CTDS.Utils import *
from CTDS.Prepare_dataset import Sample
from CTDS.Run_Evaluation import eval_from_files
from datetime import datetime
from CTDS.Profile import Profile, _keys

def train(args):
    batch_size=args.train_batch_size
    ratio = args.profile_dropout_ratio
    policy = args.neighbor_policy
    task_dir = '%s/%s-%s' % (src, task, policy)
    drop_attr = ''
    if args.keep_attributes is not None:
        for k in _keys:
            if k not in args.keep_attributes:
                drop_attr += '_%s' % k

    _, _, _, kb_vocab = torch.load('%s/kbs.pkl' % task_dir)
    candidates = torch.load('%s/candidates.pkl' % task_dir)
    candidate_tensor = torch.load('%s/candidate.ctds.pkl' % task_dir)
    # candidate_tensor = candidate_tensor.cuda() if torch.cuda.is_available() else candidate_tensor
    train_samples = torch.load('%s/train.pkl' % task_dir)
    train_sample_tensor = torch.load('%s/train.ctds-%s%s.pkl' % (task_dir, ratio, drop_attr))
    meta_data = torch.load('%s/meta.pkl' % task_dir)
    vocab2id, id2vocab = torch.load('%s/vocab.pkl' % task_dir)
    tokenizer = babi_tokenizer

    print('Item size', len(vocab2id))

    train_dataset = CTDSDataset(train_samples[:cut_data_index], candidates, meta_data, tokenizer, vocab2id, id2vocab, sample_tensor=train_sample_tensor[:cut_data_index], train_sample_tensor=train_sample_tensor)

    if args.train_epoch_start>0: # load a model and continue to train
        file = os.path.join(output_model_path, str(args.train_epoch_start) + '.pkl')

        if os.path.exists(file):
            model = CTDS(hidden_size, vocab2id, id2vocab, candidate_tensor, meta_data)
            model.load_state_dict(torch.load(file, map_location='cpu'))
        else:
            print('ERR: do not have %s' % args.train_epoch_start)

    else:
        model = CTDS(hidden_size, vocab2id, id2vocab, candidate_tensor, meta_data)
        init_params(model)

    train_size = len(train_dataset)
    model_bp_count=(epoches*train_size)/(4*batch_size*accumulation_steps) # global_batch_step
    model_optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # model_optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # model_optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    if args.warmup > 0:
        model_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(model_optimizer, round(args.warmup*model_bp_count), int(model_bp_count) + 100)
    else:
        model_scheduler = None
    model_trainer = CumulativeTrainer(model, tokenizer, None, args.local_rank, 4, accumulation_steps=accumulation_steps, max_grad_norm=args.max_grad_norm, save_data_attributes=save_data_attributes)

    for i in range(args.train_epoch_start, epoches):
        model_trainer.train_epoch('train', train_dataset, collate_fn, batch_size, i, model_optimizer, model_scheduler)
        model_trainer.serialize(i, output_path=output_model_path)

def test(args):
    batch_size = args.test_batch_size
    ratio = args.profile_dropout_ratio
    policy = args.neighbor_policy
    task_dir = '%s/%s-%s' % (src, task, policy)
    drop_attr = ''
    if args.keep_attributes is not None:
        for k in _keys:
            if k not in args.keep_attributes:
                drop_attr += '_%s' % k

    _, _, _, kb_vocab = torch.load('%s/kbs.pkl' % task_dir)
    candidates = torch.load('%s/candidates.pkl' % task_dir)
    candidate_tensor = torch.load('%s/candidate.ctds.pkl' % task_dir)
    train_samples = torch.load('%s/train.pkl' % task_dir)
    dev_samples = torch.load('%s/dev.pkl' % task_dir)
    test_samples = torch.load('%s/test.pkl' % task_dir)
    meta_data = torch.load('%s/meta.pkl' % task_dir)
    vocab2id, id2vocab = torch.load('%s/vocab.pkl' % task_dir)
    tokenizer = babi_tokenizer
    print('Item size', len(vocab2id))
    train_sample_tensor = torch.load('%s/train.ctds-%s%s.pkl' % (task_dir, ratio, drop_attr))
    dev_sample_tensor = torch.load('%s/dev.ctds-%s%s.pkl' % (task_dir, ratio, drop_attr))
    test_sample_tensor = torch.load('%s/test.ctds-%s%s.pkl' % (task_dir, ratio, drop_attr))
    dev_dataset = CTDSDataset(dev_samples[:cut_data_index], candidates, meta_data, tokenizer, vocab2id, id2vocab, sample_tensor=dev_sample_tensor[:cut_data_index], train_sample_tensor=train_sample_tensor)
    test_dataset = CTDSDataset(test_samples[:cut_data_index], candidates, meta_data, tokenizer, vocab2id, id2vocab, sample_tensor=test_sample_tensor[:cut_data_index], train_sample_tensor=train_sample_tensor)

    for i in range(args.infer_epoch_start, epoches):
        print('epoch', i)
        file = os.path.join(output_model_path, str(i) + '.pkl')

        if os.path.exists(file):
            model = CTDS(hidden_size, vocab2id, id2vocab, candidate_tensor, meta_data)
            model.load_state_dict(torch.load(file, map_location='cpu'))

            model_trainer = CumulativeTrainer(model, tokenizer, None, args.local_rank, 4, accumulation_steps=accumulation_steps, max_grad_norm=args.max_grad_norm, save_data_attributes=save_data_attributes)

            # Dev infer # dev_list_output[0][1].tolist()
            dev_list_output=model_trainer.predict('infer', dev_dataset, collate_fn, batch_size)
            #save result, note each GPU process will save a separate file
            # save_dev = [[batch_data[0][0], batch_data[0][1], batch_data[0][2], batch_data[1][0], batch_data[1][1]] for batch_data in dev_list_output]
            # save_dev = [[batch_data[0]['response_id'], batch_data[1]] for batch_data in dev_list_output]
            torch.save(dev_list_output, os.path.join(output_result_path, 'dev.%s.%s'%(i, args.local_rank)))

            # Test infer
            test_list_output = model_trainer.predict('infer', test_dataset, collate_fn, batch_size)
            # save_test = [[batch_data[0][0], batch_data[0][1], batch_data[1][0], batch_data[1][1]] for batch_data in test_list_output]
            # save_test = [[batch_data[0]['response_id'], batch_data[1]] for batch_data in test_list_output]
            torch.save(test_list_output, os.path.join(output_result_path, 'test.%s.%s'%(i, args.local_rank)))

if __name__ == '__main__':
    if makedir_flag:
        print(args)

    start = datetime.now()

    if args.mode=='test':
        test(args)
        if makedir_flag:
            eval_from_files('%s%s' % (args.model_name, args.exp_name), args.data_dir, bsl=args.bsl)

    elif args.mode=='train':
        train(args)
        # analyze_a_log(args.log_name, is_savefig=True)
        # test(args)

    end = datetime.now()
    print('run time:%.2f mins'% ((end-start).seconds/60))