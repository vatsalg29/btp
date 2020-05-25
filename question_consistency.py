import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Qgen(nn.Module):
    def __init__(self, **kwargs):
        super(Qgen, self).__init__()
        embed_size = kwargs.get('embed_size', 300)
        hidden_size = kwargs.get('hidden_size', 512)
        ans_embed_hidden_size = kwargs.get('ans_embed_hidden_size', 1000)
        question_embed_size = kwargs.get('question_embed_size', 1024)
        
        # Add 2 for <start> and <end>
        n_ans = kwargs.get('ans_vocab_size')
        vocab_size = kwargs.get('question_vocab_size') + 2
        embed_path = kwargs.get('embed_path',"data/vqa2.0_glove.6B.300d.txt.npy")

        self.embed = nn.Embedding(vocab_size,embed_size,scale_grad_by_freq=False)
        embed_init = np.load(embed_path)
       
        #initialize start and end at extremes
        se_init = np.zeros([2, embed_size])
        se_init[0][1] = 1.0
        se_init[1][1:] = -1.0
        embed_init = np.concatenate([embed_init, se_init], 0)
        self.embed.weight.data.copy_(torch.from_numpy(embed_init))
        
        self.a_embed = nn.Sequential(nn.ReLU(),
                                     nn.Linear(n_ans, ans_embed_hidden_size),
                                     nn.ReLU(),
                                     nn.Linear(ans_embed_hidden_size, embed_size),
                                     nn.ReLU())
        
        self.imp_a_embed = nn.Linear(3, embed_size)

        self.loss_fn = nn.CrossEntropyLoss()
        self.lstm = nn.LSTM(embed_size, hidden_size, 1, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.linear2 = nn.Linear(question_embed_size,embed_size)
        
        self.max_seg_length = 14
        self.start_idx = vocab_size - 2
        self.end_idx = vocab_size - 1

    def compute_loss(self, pred, gt):
        return self.loss_fn(pred, gt)

    def fuse_features(self, q, a, imp_ans):
        
        a = a.to(self.a_embed[1].weight.device)
        
        answer_embedding = self.a_embed(a) 
        question_embedding = self.linear2(q)
        imp_ans_embedding = self.imp_a_embed(imp_ans)
        
        mixed_feat = imp_ans_embedding + answer_embedding + question_embedding
#         mixed_feat += torch.randn(mixed_feat.size()).cuda() #uncomment for noise
        
        mixed_feat = mixed_feat.unsqueeze(1)
        return mixed_feat

    def forward(self, q, a, imp_knob, captions,lengths):
        
        mixed_feat = self.fuse_features(q, a, imp_knob)
        
        captions = captions.to(self.embed.weight.device)
                
        # Add <end> token to captions
        for i in range(len(captions)):
            if lengths[i] <= self.max_seg_length -1:
                captions[i][lengths[i]] = self.end_idx
            else:
                captions[i][self.max_seg_length -1] = self.end_idx

        # Add <start> token to captions
        start_vector = torch.ones([len(captions), 1]).to(captions.device).long() * self.start_idx
        captions = torch.cat([start_vector, captions], 1)

        # Add 2 element to the lengths
        # Only needed if manually appending start and end vectors for original vocabulary
        lengths += 2
                
        s_lengths, indices = torch.sort(lengths, descending=True)
        s_lengths = s_lengths.cpu().numpy().tolist()
        s_lengths = [s if s < self.max_seg_length else self.max_seg_length for s in s_lengths]
        
        captions = captions[indices]
        mixed_feat = mixed_feat[indices]
        
        embeddings = self.embed(captions)
        embeddings = torch.cat([mixed_feat, embeddings], 1)
        packed = pack_padded_sequence(embeddings, s_lengths, batch_first=True) 
        target_tuple = pack_padded_sequence(captions, s_lengths, batch_first=True)
        targets = target_tuple[0]
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        
        loss = self.compute_loss(outputs, targets)
        
        sampled_ids = self.sample(q, a, imp_knob)
        
        return loss,sampled_ids

    def sample(self, q, a, imp_knob, states=None):
        sampled_ids = []
        inputs = self.fuse_features(q, a, imp_knob)

        """ 
        To introduce noise in inference
        bs = features.shape[0]
        states = (torch.Tensor(1, bs, 512).normal_(0, 1.1).cuda(),
                  torch.Tensor(1, bs, 512).uniform_(0.5, -0.5).cuda())
        """

        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))    # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)               # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                 # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)        # sampled_ids: (batch_size, max_seq_length)

        return sampled_ids