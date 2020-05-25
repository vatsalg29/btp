import os
import time
import numpy as np
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
import torch.nn.functional as F
from pdb import set_trace
import gc

def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
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

def train(model, train_loader, eval_loader, num_epochs, output, opt, ep, q_gen=None, cycle=False):
    utils.create_dir(output)
    
    if cycle:
        print("Training with cycle activated")
        params = [{'params': model.parameters()},
                  {'params': q_gen.parameters(),'lr': 0.0005}]
    else:
        print("Training default model")
        params = [{'params': model.parameters()}]

    optim = torch.optim.Adamax(params) if opt is None else opt
    
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    q_vocab,a_vocab = obtain_vocabs()
    
    best_eval_score = 0
    total_iter = ep*len(train_loader)
    for epoch in range(ep,num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()
        
        model.train()
        if q_gen is not None:
            q_gen.train()
        
        for i, batch in enumerate(train_loader):
            optim.zero_grad()
            total_iter+=1
            
            v,b,q,a = batch['features'],batch['spatial'],batch['ques'],batch['target']           
            
            v = Variable(v).cuda()
            b = Variable(b).cuda()
            q = Variable(q.type(torch.LongTensor)).cuda()
            a = Variable(a).cuda()

            pred,q_emb = model(v, b, q)
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

                    cycle_pred,_ = model(cycle_v,cycle_b,generated_questions)
                    if allowed_indices.sum() > 0:
                        cycle_loss += instance_bce_with_logits(cycle_pred[allowed_indices], imp_a[allowed_indices].cuda())
                        loss+= 1.5*cycle_loss

                if total_iter%200 == 0:
                    print("Train: loss - %.4f, qc_loss - %.4f, cycle_loss - %.4f, batch_score - %.4f" 
                          % ( loss, qc_loss, cycle_loss, batch_score/256))
                       
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()
            
            total_loss += loss.item() * v.size(0)
            train_score += batch_score

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)
        model.eval()
        if q_gen is not None:
            q_gen.eval()
        eval_score, bound = evaluate(model, q_gen, eval_loader,epoch,output, cycle)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
        logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
        
        m_path = os.path.join(output, 'model_epoch%d.pth' % epoch)
        torch.save(model.state_dict(), m_path)

        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score

@torch.no_grad()
def evaluate(model, q_gen, dataloader, epoch, output, cycle):
    score = 0
    upper_bound = 0
    num_data = 0
    
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
            
    
    for batch in iter(dataloader):
        
        v,b,q,a = batch['features'],batch['spatial'],batch['ques'],batch['target']
        v = Variable(v).cuda()
        b = Variable(b).cuda()
        q = Variable(q.type(torch.LongTensor)).cuda()
        
        pred, q_emb = model(v, b, q)
        
        batch_score = compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score
        upper_bound += (a.max(1)[0]).sum()
        num_data += pred.size(0)
        
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

    gc.collect()
    return score, upper_bound