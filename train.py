"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import os
import time
import itertools
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from pdb import set_trace
import gc


def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction=reduction)
    if reduction == 'mean':
        loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def obtain_vocabs():
    q_vocab_path = '/home1/BTP/pg_aa_1/vocabulary_vqa.txt'
    a_vocab_path = '/home1/BTP/pg_aa_1/btp/data/answers_vqa_larger.txt'

    q_vocab = [l.rstrip() for l in tuple(open(q_vocab_path))]
    q_vocab.extend(["<start>", "<end>"])
    a_vocab = [l.rstrip() for l in tuple(open(a_vocab_path))]
    return q_vocab, a_vocab

def train(model, train_loader, eval_loader, num_epochs, output, opt=None, s_epoch=0, q_gen = None, cycle=False):
    
    lr_default = 1e-3 if eval_loader is not None else 7e-4
    lr_decay_step = 2
    lr_decay_rate = .25
    lr_decay_epochs = range(10,20,lr_decay_step) if eval_loader is not None else range(10,20,lr_decay_step)
    gradual_warmup_steps = [0.5 * lr_default, 1.0 * lr_default, 1.5 * lr_default, 2.0 * lr_default]
    
    saving_epoch = 3
    grad_clip = .25

    utils.create_dir(output)
    
    if cycle:
        print("Training with cycle activated")
        params = [{'params': filter(lambda p: p.requires_grad,model.parameters()), 'lr': lr_default},
                  {'params': q_gen.parameters(),'lr': 0.0005}]
    else:
        print("Training default model")
        params = [{'params': filter(lambda p: p.requires_grad,model.parameters()),
                   'lr': lr_default}]
    
    optim = torch.optim.Adamax(params) if opt is None else opt
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0
    q_vocab,a_vocab = obtain_vocabs()
    
    utils.print_model(model, logger)
    logger.write('optim: adamax lr=%.4f, decay_step=%d, decay_rate=%.2f, grad_clip=%.2f' % \
        (lr_default, lr_decay_step, lr_decay_rate, grad_clip))
    total_iter = s_epoch*len(train_loader)
    
    for epoch in range(s_epoch, num_epochs):
        total_loss = 0
        train_score = 0
        total_norm = 0
        count_norm = 0
        t = time.time()
        model.train()
        if q_gen is not None:
            q_gen.train()
        N = len(train_loader.dataset)
        if epoch < len(gradual_warmup_steps):
            optim.param_groups[0]['lr'] = gradual_warmup_steps[epoch]
            logger.write('gradual warmup lr: %.4f' % optim.param_groups[0]['lr'])
        elif epoch in lr_decay_epochs:
            optim.param_groups[0]['lr'] *= lr_decay_rate
            logger.write('decreased lr: %.4f' % optim.param_groups[0]['lr'])
        else:
            logger.write('lr: %.4f' % optim.param_groups[0]['lr'])
        
        
        for i, batch in enumerate(train_loader):
            optim.zero_grad()
            total_iter+=1
            
            v,b,q,a = batch['features'],batch['spatial'],batch['ques'],batch['target']     
            v = Variable(v).cuda()
            b = Variable(b).cuda()
            q = Variable(q.type(torch.LongTensor)).cuda()
            a = Variable(a).cuda()
            
            pred, att, q_emb = model(v, b, q)
            loss = instance_bce_with_logits(pred, a)
            batch_score = compute_score_with_logits(pred, a.data).sum()
            
            
            if cycle:
                imp_q,imp_len,knob,flag = batch['imp_ques'],batch['imp_len'],batch['imp_type'],batch['imp_flag']
                knob = Variable(knob).cuda()
                imp_q = Variable(imp_q.type(torch.LongTensor)).cuda()

                new_idx = [i for i in range(0,len(flag)) if flag[i]]
                qc_loss,cycle_loss,l_b = 0,0,1

                if len(new_idx):
                    q_embed, logits = q_emb.clone().detach()[new_idx], pred.clone().detach()[new_idx]
                    knob, imp_q, imp_len = knob[new_idx], imp_q[new_idx], imp_len[new_idx]

                    qc_loss,sample_ids = q_gen(q_embed,logits,knob,imp_q,imp_len)
                    loss+= 0.5*qc_loss

                if total_iter == 5500:
                    print("Cycle Activated")
                if total_iter > 5500 : #late activation = 5500

                    imp_a, cycle_v, cycle_b = batch['imp_ans'][new_idx], v.clone()[new_idx], b.clone()[new_idx]
                    generated_questions = sample_ids.clone()
                    # Preprocess to remove start and end
                    generated_questions[generated_questions == len(q_vocab)-2] = 0
                    generated_questions[generated_questions == len(q_vocab)-1] = 0

                    # First letter cannot be unk
                    generated_questions = torch.cat(
                        [
                            generated_questions.narrow(
                                1, 1, generated_questions.shape[1] - 1
                            ),
                            generated_questions.narrow(1, 0, 1),
                        ],
                        1,
                    )

                    # Gating Mechanism
                    detached_g_q = generated_questions.clone().detach()
                    detached_g_emb = q_gen.module.embed(detached_g_q).sum(1)

                    detached_o_q = imp_q.detach()
                    detached_o_emb = q_gen.module.embed(detached_o_q).sum(1)

                    cosine_similarity = F.cosine_similarity(detached_g_emb, detached_o_emb)

                    allowed_indices = (cosine_similarity > 0.9) #threshold = 0.9
                    l_b = allowed_indices.sum().cpu().item()
                    #print("Allowed Batches {}".format(l_b))

                    #run vqa on generated ques

                    cycle_pred,_,_ = model(cycle_v,cycle_b,generated_questions)
                    if allowed_indices.sum() > 0:
                        cycle_loss += instance_bce_with_logits(cycle_pred[allowed_indices], imp_a[allowed_indices].cuda())
                        loss+= 1.5*cycle_loss

                if total_iter%200 == 0:
                    print("Train: loss - %.4f, qc_loss - %.4f, cycle_loss - %.4f, batch_score - %.4f" 
                          % ( loss, qc_loss, cycle_loss, batch_score/256))

            
            loss.backward()
            total_norm += nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            count_norm += 1
            optim.step()
            
            total_loss += loss.item() * v.size(0)
            train_score += batch_score.item()

        total_loss /= N
        train_score = 100 * train_score / N
        if None != eval_loader:
            model.eval()
            if q_gen is not None:
                q_gen.eval()
            eval_score, bound, entropy = evaluate(model, q_gen, eval_loader, epoch, output, cycle)
            
        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, norm: %.4f, score: %.2f' % (total_loss, total_norm/count_norm, train_score))
        if eval_loader is not None:
            logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

        if eval_loader is not None and entropy is not None:
            info = ''
            for i in range(entropy.size(0)):
                info = info + ' %.2f' % entropy[i]
            logger.write('\tentropy: ' + info)

        model_path = os.path.join(output, 'model_epoch%d.pth' % epoch)
        utils.save_model(model_path, model, epoch, optim)
        
        if eval_score > best_eval_score:
            m_path = os.path.join(output, 'best_model.pth')
            torch.save(model.state_dict(), m_path)
            best_eval_score = eval_score


@torch.no_grad()
def evaluate(model, q_gen, dataloader, epoch, output, cycle):
    score = 0
    upper_bound = 0
    num_data = 0
    entropy = None
    if hasattr(model.module, 'glimpse'):
        entropy = torch.Tensor(model.module.glimpse).zero_().cuda()
        
    # Make dict to store generated questions
    gq_dict = {"annotations": [], "ques_answers": []}
    vocab, ans_vocab = obtain_vocabs()
    
    def store_questions(logits, batch, sampled_ids, new_idx):
        
        pred_ans = torch.argmax(logits.detach().cpu(),dim=1).data.cpu().numpy()
        imp_ans = torch.argmax(batch["imp_ans"][new_idx],dim=1).data.cpu().numpy()
        orig_ans = torch.argmax(batch["target"][new_idx],dim=1).data.cpu().numpy()

        orig_q = batch["ques"][new_idx].data.cpu().numpy()
        orig_q = [[vocab[idx] for idx in orig_q[j]] for j in range(len(orig_q))]
        
        sampled_ids = sampled_ids.data.cpu().numpy()
        gen_imp = [[vocab[idx] for idx in sampled_ids[j]] for j in range(len(sampled_ids))]
        
        gt_q = batch["imp_ques"][new_idx].data.cpu().numpy()
        gt_imp = [[vocab[idx] for idx in gt_q[j]] for j in range(len(gt_q))]

        images = batch["image_id"][new_idx]
        qids =  batch['ques_id'][new_idx]
        
        for jdx, (img,qid,oq,oa, pa) in enumerate(zip(images, qids, orig_q, orig_ans, pred_ans)):
            gq_dict["ques_answers"] += [{"image_id": int(img), "orig_ques": " ".join(oq), "orig_ans": ans_vocab[oa], "ques_id": int(qid), "pred_ans": ans_vocab[pa]}]

        for jdx, (img, ia, q, gtq) in enumerate(zip(images, imp_ans, gen_imp, gt_imp)):
            gq_dict["annotations"] += [{"image_id": int(img), "imp_ans": ans_vocab[ia], "gen_ques": " ".join(q), "gt_gen_ques": " ".join(gtq)}]
            
    
    for i, batch in enumerate(dataloader):
        
        v,b,q,a = batch['features'],batch['spatial'],batch['ques'],batch['target']
        v = Variable(v).cuda()
        b = Variable(b).cuda()
        q = Variable(q.type(torch.LongTensor)).cuda()
        
        pred, att, q_emb = model(v, b, q)
        
        batch_score = compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score.item()
        upper_bound += (a.max(1)[0]).sum().item()
        num_data += pred.size(0)
        if att is not None and 0 < model.module.glimpse:
            entropy += calc_entropy(att.data)[:model.module.glimpse]
            
        if cycle:
            imp_q,imp_len,knob,flag = batch['imp_ques'],batch['imp_len'],batch['imp_type'],batch['imp_flag']    
            knob = Variable(knob).cuda()
            imp_q = Variable(imp_q.type(torch.LongTensor)).cuda()

            new_idx = [i for i in range(0,len(flag)) if flag[i]]
            if len(new_idx):
                q_emb, pred, knob, imp_q, imp_len = q_emb[new_idx], pred[new_idx], knob[new_idx], imp_q[new_idx], imp_len[new_idx]
                _ ,sample_ids = q_gen(q_emb,pred,knob,imp_q,imp_len)

                store_questions(pred, batch, sample_ids, new_idx)
                
    if cycle:
        np.save(os.path.join(output, "gq_{}.npy".format(epoch)), np.array(gq_dict))


    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)

    if entropy is not None:
        entropy = entropy / len(dataloader.dataset)
    
    gc.collect()
    return score, upper_bound, entropy

def calc_entropy(att): # size(att) = [b x g x v x q]
    sizes = att.size()
    eps = 1e-8
    p = att.view(-1, sizes[1], sizes[2] * sizes[3])
    return (-p * (p+eps).log()).sum(2).sum(0) # g