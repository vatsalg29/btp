
import argparse
import json
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from dataset import Dictionary, VQAFeatureDataset
import base_model
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--input', type=str, required=True, help='input model path')
    parser.add_argument('--output', type=str, required=True, help='output file name prefix, will append .json')
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()
    return args


def get_question(q, dataloader):
    str = []
    dictionary = dataloader.dataset.dictionary
    for i in range(q.size(0)):
        str.append(dictionary.idx2word[q[i]] if q[i] < len(dictionary.idx2word) else '_')
    return ' '.join(str)


def get_answer(p, dataloader):
    _m, idx = p.max(0)
    return dataloader.dataset.answer_dict.idx2word(idx.item())


@torch.no_grad()
def get_logits(model, dataloader):
    N = len(dataloader.dataset)
    M = dataloader.dataset.answer_dict.num_vocab
    pred = torch.FloatTensor(N, M).zero_()
    qIds = torch.IntTensor(N).zero_()
    idx = 0
    for batch in tqdm(iter(dataloader)):
        v,b,q = batch['features'],batch['spatial'],batch['ques']
        q_id = batch['ques_id']
        batch_size = v.size(0)
        v = Variable(v).cuda()
        b = Variable(b).cuda()
        q = Variable(q.type(torch.LongTensor)).cuda()
        logits, q_emb = model(v, b, q)
        pred[idx:idx+batch_size,:].copy_(logits.data)
        qIds[idx:idx+batch_size].copy_(q_id)
        idx += batch_size
    return pred, qIds


def make_json(logits, qIds, dataloader):
    utils.assert_eq(logits.size(0), len(qIds))
    results = []
    for i in tqdm(range(logits.size(0))):
        result = {}
        result['question_id'] = qIds[i].item()
        result['answer'] = get_answer(logits[i], dataloader)
        results.append(result)
    return results

if __name__ == '__main__':
    args = parse_args()

    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    eval_dset = VQAFeatureDataset(args.split, dictionary)

    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(eval_dset, args.num_hid).cuda()    
#     model.w_emb.init_embedding('data/vqa2.0_glove.6B.300d.txt.npy')
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=False, num_workers=15)

    def process(args, model, eval_loader):
        model_path = args.input
    
        print('loading %s' % model_path)
        model_data = torch.load(model_path)

        model = nn.DataParallel(model).cuda()
        model.load_state_dict(model_data.get('model_state', model_data))

        model.train(False)

        logits, qIds = get_logits(model, eval_loader)
        results = make_json(logits, qIds, eval_loader)
 
        with open(args.output+'.json', 'w') as f:
            json.dump(results, f)

    process(args, model, eval_loader)
    print("Done!")
