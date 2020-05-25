"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np

from dataset import Dictionary, VQAFeatureDataset
import base_model
import utils
from utils import trim_collate
# from dataset import tfidf_from_questions

from question_consistency import Qgen

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='vqa', help='vqa or flickr')
    parser.add_argument('--epochs', type=int, default=13)
    parser.add_argument('--num_hid', type=int, default=1280)
    parser.add_argument('--model', type=str, default='ban')
    parser.add_argument('--op', type=str, default='c')
    parser.add_argument('--gamma', type=int, default=8, help='glimpse')
    parser.add_argument('--use_both', action='store_true', help='use both train/val datasets to train?')
    parser.add_argument('--use_vg', action='store_true', help='use visual genome dataset to train?')
    parser.add_argument('--tfidf', action='store_false', help='tfidf word embedding?')
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default='saved_models/ban')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1204, help='random seed')
    parser.add_argument('--use_cycle', action='store_true', help='To train with cycle')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    if args.task == 'vqa':
        from train import train
        dict_path = 'data/dictionary.pkl'
        dictionary = Dictionary.load_from_file(dict_path)
        train_dset = VQAFeatureDataset('train', dictionary)
        val_dset = VQAFeatureDataset('val', dictionary)
        w_emb_path = 'data/vqa2.0_glove.6B.300d.txt.npy'

    utils.create_dir(args.output)
    logger = utils.Logger(os.path.join(args.output, 'args.txt'))
    logger.write(args.__repr__())

    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args.num_hid, args.op, args.gamma, args.task).cuda()


    tfidf = None
    weights = None

    model.w_emb.init_embedding(w_emb_path, tfidf, weights)

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
        epoch = model_data['epoch'] + 1

    if args.task == 'vqa':
        if args.use_both: # use train & val splits to optimize
            if args.use_vg: # use a portion of Visual Genome dataset
                vg_dsets = [
                    VisualGenomeFeatureDataset('train', \
                        train_dset.features, train_dset.spatials, dictionary, adaptive=True, pos_boxes=train_dset.pos_boxes),
                    VisualGenomeFeatureDataset('val', \
                        val_dset.features, val_dset.spatials, dictionary, adaptive=True, pos_boxes=val_dset.pos_boxes)]
                trainval_dset = ConcatDataset([train_dset, val_dset]+vg_dsets)
            else:
                trainval_dset = ConcatDataset([train_dset, val_dset])
            train_loader = DataLoader(trainval_dset, batch_size, shuffle=True, num_workers=1, collate_fn=utils.trim_collate)
            eval_loader = None
        else:
            train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=15, collate_fn=utils.trim_collate)
            eval_loader = DataLoader(val_dset, batch_size, shuffle=False, num_workers=15, collate_fn=utils.trim_collate)

    elif args.task == 'flickr':
        train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=15, collate_fn=utils.trim_collate)
        eval_loader = DataLoader(val_dset, batch_size, shuffle=False, num_workers=15, collate_fn=utils.trim_collate)

    q_gen=None
    if args.use_cycle:
        q_gen = Qgen(embed_path = 'data/vqa2.0_glove.6B.300d.txt.npy')
        q_gen = nn.DataParallel(q_gen).cuda()
    
    train(model, train_loader, eval_loader, args.epochs, args.output, optim, epoch, q_gen, args.use_cycle)