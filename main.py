import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary, VQAFeatureDataset
from question_consistency import Qgen
import base_model
from train import train
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--input', type=str,default=None)
    parser.add_argument('--output', type=str, default='saved_models/exp0')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--use_cycle', action='store_true', help='To train with cycle')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    
    output = 'saved_models/' + str(args.seed)
    print('Store path: ' + output)

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    train_dset = VQAFeatureDataset('train', dictionary)
    eval_dset = VQAFeatureDataset('val', dictionary)
    batch_size = args.batch_size
    
    ans_size = train_dset.answer_dict.num_vocab
    ques_size = train_dset.vocab_dict.num_vocab
    
    print(ans_size, ques_size)

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda()    
    model.w_emb.init_embedding('data/vqa2.0_glove.6B.300d.txt.npy')
    model = nn.DataParallel(model).cuda()
    
    optim = None
    epoch = 0

    # load snapshot
    if args.input is not None:
        print('loading %s' % args.input)
        model_data = torch.load(args.input)
        model.load_state_dict(model_data.get('model_state', model_data))
        optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()))
        optim.load_state_dict(model_data.get('optimizer_state', model_data))
        epoch = model_data['epoch']

    q_gen=None
    if args.use_cycle:
        q_gen = Qgen(embed_path = 'data/vqa2.0_glove.6B.300d.txt.npy', ans_vocab_size = ans_size, question_vocab_size = ques_size)
        q_gen = nn.DataParallel(q_gen).cuda()
    
    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=15)
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=15)
    train(model, train_loader, eval_loader, args.epochs, output, optim, epoch, q_gen, args.use_cycle)