from __future__ import print_function
import os
import json
import pickle
import numpy as np
import utils
import h5py
import torch
from torch.utils.data import Dataset

import text_processing

def compute_answer_scores(answers, num_of_answers, unk_idx):
    scores = np.zeros((num_of_answers), np.float32)
    for answer in set(answers):
        if answer == unk_idx:
            scores[answer] = 0
        else:
            answer_count = answers.count(answer)
            scores[answer] = min(np.float32(answer_count)*0.3, 1)
    return scores

class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx[w])
        return tokens

    def dump_to_file(self, path):
        pickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = pickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer):
    if answer is not None:
        answer.pop('image_id')
        answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'answer'      : answer}
    return entry


def _load_dataset(dataroot, name, img_id2val):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    
    
    if name != 'test':
        question_path = os.path.join(dataroot, 'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
        questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
        answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
        answers = pickle.load(open(answer_path, 'rb'))
        answers = sorted(answers, key=lambda x: x['question_id'])

        utils.assert_eq(len(questions), len(answers))
        entries = []
        for question, answer in zip(questions, answers):
            utils.assert_eq(question['question_id'], answer['question_id'])
            utils.assert_eq(question['image_id'], answer['image_id'])
            img_id = question['image_id']
            entries.append(_create_entry(img_id2val[img_id], question, answer))
            
    else:
        question_path = os.path.join(dataroot, 'cons/questions.json') ###### change later as required
        questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
        entries = []
        for question in questions:
            img_id = question['image_id']
            entries.append(_create_entry(img_id2val[img_id], question, None))

    return entries


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary, dataroot='data'):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'val', 'test']
        
        self.vocab_dict = text_processing.VocabDict('/home1/BTP/pg_aa_1/vocabulary_vqa.txt')
        self.answer_dict = text_processing.VocabDict('/home1/BTP/pg_aa_1/btp/data/answers_vqa_larger.txt')
        
        if name != 'test':
            imdb = np.load('/home1/BTP/pg_aa_1/btp/data/imdb_imps/imdb_' + name + '2014.npy',allow_pickle=True)
            self.imdb = imdb
            self.map = {}
            for i in range(1,len(self.imdb)):
                self.map[self.imdb[i]['question_id']] = i

        self.dictionary = dictionary

        if name != 'test':
            self.img_id2idx = pickle.load(
                open(os.path.join(dataroot, '%s36_imgid2idx.pkl' % name), 'rb'))
            print('loading features from h5 file')
            h5_path = os.path.join(dataroot, '%s36.hdf5' % name)
            
        else:
            self.img_id2idx = pickle.load(
                open(os.path.join(dataroot, 'val36_imgid2idx.pkl'), 'rb'))
            print('loading features from h5 file')
            h5_path = os.path.join(dataroot, 'val36.hdf5')
            
        with h5py.File(h5_path, 'r') as hf:
            self.features = np.array(hf.get('image_features'))
            self.spatials = np.array(hf.get('spatial_features'))

        self.entries = _load_dataset(dataroot, name, self.img_id2idx)

#         self.tokenize()
        self.tensorize()
        self.v_dim = self.features.size(2)
        self.s_dim = self.spatials.size(2)      
                

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.vocab_dict.num_vocab] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        self.features = torch.from_numpy(self.features)
        self.spatials = torch.from_numpy(self.spatials)

        for entry in self.entries:
#             question = torch.from_numpy(np.array(entry['q_token']))
#             entry['q_token'] = question

            answer = entry['answer']
            if answer is not None:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        features = self.features[entry['image']]
        spatials = self.spatials[entry['image']]
        
        q_id = entry['question_id']
        image_id = entry['image_id']
        answer = entry['answer']
        
        ############# Load input question ################
        input_seq = np.zeros((14), np.int32)
        sentence = entry['question']
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        question_inds = (
            [self.vocab_dict.word2idx(w) for w in words])
        seq_length = len(question_inds)
        read_len = min(seq_length, 14)
        input_seq[:read_len] = question_inds[:read_len]
        
        sample = dict(ques = input_seq)
        sample['spatial'] = spatials
        sample['features'] = features
        sample['image_id'] = image_id
        sample['ques_id'] = q_id
        
        
        
        if answer is not None:
            imdb_idx = self.map[q_id]
            iminfo = self.imdb[imdb_idx]
            valid_answers = iminfo['valid_answers']
            answer_r = np.random.choice(valid_answers)

            ######### Load implied question ###########
            implied_seq = np.zeros((14), np.int32)
            imp_idx = np.random.choice(len(iminfo['qa_tokens'][answer_r]))
            imp_ques_tokens = iminfo['qa_tokens'][answer_r][imp_idx]
            imp_question_inds = (
                    [self.vocab_dict.word2idx(w) for w in imp_ques_tokens])
            imp_seq_length = len(imp_question_inds)
            imp_read_len = min(imp_seq_length, 14)
            implied_seq[:imp_read_len] = imp_question_inds[:imp_read_len]


            ########### Load implied answer ###########
            imp_answer = None
            imp_valid_answers_idx = np.zeros((10), np.int32)
            imp_valid_answers_idx.fill(-1)
            imp_answer_scores = np.zeros(self.answer_dict.num_vocab, np.float32)
            imp_answer = iminfo['qa_answers'][answer_r][imp_idx]
            imp_answer_idx = self.answer_dict.word2idx(imp_answer)
            imp_valid_answers_idx.fill(imp_answer_idx)
            imp_valid_answers = iminfo['valid_answers'].copy() #bug fix :p
            for i in range(0,len(imp_valid_answers)):
                imp_valid_answers[i] = imp_answer

            ans_idx = ([self.answer_dict.word2idx(ans) for ans in imp_valid_answers])
            imp_answer_scores = (compute_answer_scores(ans_idx, self.answer_dict.num_vocab, self.answer_dict.UNK_idx))
            imp_answer_final = torch.from_numpy(np.array(imp_answer_scores))
            
            ########### Load original answer ############
            valid_answers_idx = np.zeros((10), np.int32)
            valid_answers_idx.fill(-1)
            answer_scores = np.zeros(self.answer_dict.num_vocab, np.float32)
            valid_answers_idx[:len(valid_answers)] = ([self.answer_dict.word2idx(ans) for ans in valid_answers])
            ans_idx = ([self.answer_dict.word2idx(ans) for ans in valid_answers])
            answer_scores = (compute_answer_scores(ans_idx,self.answer_dict.num_vocab,self.answer_dict.UNK_idx))
            answer_final = torch.from_numpy(np.array(answer_scores))

             ############# Load implication type ####################
            imp_type = None
            if 'imp_type' in iminfo:
                imp_type = torch.from_numpy(np.array(iminfo['imp_type'][answer_r][imp_idx], np.float32))
            else:
                imp_type = torch.from_numpy(np.array([1,0,0], np.float32))

            ########### Load Flag ########################
            imp_flag = None
            if 'is_imps' in iminfo:
                imp_flag = iminfo['is_imps']
            else:
                imp_flag = True

#             labels = answer['labels']
#             scores = answer['scores']
#             target = torch.zeros(self.answer_dict.num_vocab)
#             if labels is not None:
#                 target.scatter_(0, labels, scores)


            sample['target'] = answer_final
            sample['imp_ques'] = implied_seq
            sample['imp_len'] = imp_seq_length
            sample['imp_ans'] = imp_answer_final
            sample['imp_type'] = imp_type
            sample['imp_flag'] = imp_flag

        
#         return features, spatials, input_seq, answer_final
        return sample

    def __len__(self):
        return len(self.entries)
