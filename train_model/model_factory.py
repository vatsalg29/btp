# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from global_variables.global_variables import *
from top_down_bottom_up.top_down_bottom_up_model import (vqa_multi_modal_model,
     vqa_multi_modal_with_qc_cycle, vqa_multi_modal_with_fpqc_cycle)
from top_down_bottom_up.image_attention import build_image_attention_module
from top_down_bottom_up.classifier import build_classifier
from top_down_bottom_up.question_embeding import build_question_encoding_module
from top_down_bottom_up.image_embedding import image_embedding
from top_down_bottom_up.multi_modal_combine import build_modal_combine_module
from top_down_bottom_up.intermediate_layer import inter_layer
from top_down_bottom_up.image_feature_encoding \
    import build_image_feature_encoding
from cycle_consistency.failure_predictor import build_failure_prediction_module
from cycle_consistency.question_consistency import \
    build_question_consistency_module


import torch.nn as nn
import torch

# from butd.attention import Attention, NewAttention
# from butd.language_model import WordEmbedding, QuestionEmbedding
# from butd.classifier import SimpleClassifier
# from butd.fc import FCNet

# from ban.attention import BiAttention
# from ban.language_model import WordEmbedding, QuestionEmbedding
# from ban.classifier import SimpleClassifier
# from ban.fc import FCNet
# from ban.bc import BCNet
# from ban.counting import Counter

from top_down_bottom_up.top_down_bottom_up_model import butd_with_qc_cycle, ban_with_qc_cycle, butd_model, ban_model

def get_two_layer(img_dim):
    return inter_layer(img_dim, 2)


def prepare_model(num_vocab_txt, num_choices, **model_config):
    image_feat_dim = model_config['image_feat_dim']

    # generate the list of question embedding models
    ques_embeding_models_list = model_config['question_embedding']
    question_embeding_models = nn.ModuleList()
    final_question_embeding_dim = 0
    for ques_embeding_model in ques_embeding_models_list:
        ques_model_key = ques_embeding_model['method']
        ques_model_par = ques_embeding_model['par']
        tmp_model = build_question_encoding_module(ques_model_key,
                                                   ques_model_par,
                                                   num_vocab=num_vocab_txt)

        question_embeding_models.append(tmp_model)
        final_question_embeding_dim += tmp_model.text_out_dim

    image_feature_encode_list = nn.ModuleList()
    for image_feat_model_par in model_config['image_feature_encoding']:
        image_feat_model = build_image_feature_encoding(
            image_feat_model_par['method'],
            image_feat_model_par['par'],
            image_feat_dim)
        image_feature_encode_list.append(image_feat_model)
        image_feat_dim = image_feat_model.out_dim

    # generate the list of image attention models
    image_emdedding_models_list = nn.ModuleList()
    num_image_feat = model_config['num_image_feat']
    final_image_embedding_dim = 0
    for i_image in range(num_image_feat):
        image_emdedding_models = nn.ModuleList()
        image_att_model_list = model_config['image_embedding_models']

        for image_att_model in image_att_model_list:
            image_att_model_par = image_att_model
            tmp_img_att_model = build_image_attention_module(
                image_att_model_par,
                image_dim=image_feat_dim,
                ques_dim=final_question_embeding_dim)

            tmp_img_model = image_embedding(tmp_img_att_model)
            final_image_embedding_dim += tmp_img_model.out_dim
            image_emdedding_models.append(tmp_img_model)
        image_emdedding_models_list.append(image_emdedding_models)

    final_image_embedding_dim *= image_feat_dim

    inter_model = None

    # parse multi-modal combination after image-embedding & question-embedding
    multi_modal_combine = build_modal_combine_module(
        model_config['modal_combine']['method'],
        model_config['modal_combine']['par'],
        final_image_embedding_dim,
        final_question_embeding_dim)

    joint_embedding_dim = multi_modal_combine.out_dim
    # generate the classifier
    classifier = build_classifier(
        model_config['classifier']['method'],
        model_config['classifier']['par'],
        in_dim=joint_embedding_dim,
        out_dim=num_choices)

    ############# BUTD Model ####################
#     num_hid = 1024
#     w_emb = WordEmbedding(num_vocab_txt, 300, 0.0)
#     q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
#     v_att = NewAttention(2048, q_emb.num_hid, num_hid)
#     q_net = FCNet([q_emb.num_hid, num_hid])
#     v_net = FCNet([2048, num_hid])
#     classifier = SimpleClassifier(num_hid, num_hid * 2, num_choices, 0.5)
    
    ############### BAN Model ######################
#     num_hid = 1280
#     gamma = 8
#     op = 'c'
#     w_emb = WordEmbedding(num_vocab_txt, 300, .0, op)
#     q_emb = QuestionEmbedding(300 if 'c' not in op else 600, num_hid, 1, False, .0)
#     v_att = BiAttention(2048, num_hid, num_hid, gamma)
#     b_net = []
#     q_prj = []
#     c_prj = []
#     objects = 36  # minimum number of boxes, can be 36 for our approach
#     for i in range(gamma):
#         b_net.append(BCNet(2048, num_hid, num_hid, None, k=1))
#         q_prj.append(FCNet([num_hid, num_hid], '', .2))
#         c_prj.append(FCNet([objects + 1, num_hid], 'ReLU', .0))
#     classifier = SimpleClassifier(num_hid, num_hid * 2, num_choices, .5)
#     counter = Counter(objects)

    ################ QC or FP Module ###################
    is_failure_prediction = model_config.get('failure_predictor', {}).get('hidden_1', 0)
    is_question_consistency = model_config.get('question_consistency', {}).get('hidden_size', 0)
    print("Question consistency is" + str(is_question_consistency) + str(is_failure_prediction))

    if is_question_consistency and not is_failure_prediction:

        question_consistency_model = build_question_consistency_module(**model_config['question_consistency'])
        skip_thought = model_config['question_consistency'].get('skip_thought', False)
        decode_question = model_config['question_consistency'].get('decode_question', False)
        attended = model_config['question_consistency'].get('attended', False)
        model_class = vqa_multi_modal_with_qc_cycle
#         model_class = butd_with_qc_cycle
#         model_class = ban_with_qc_cycle

        my_model = model_class(image_emdedding_models_list, question_embeding_models,
                               multi_modal_combine, classifier, image_feature_encode_list,
                               inter_model, question_consistency_model, skip_thought,
                               decode_question, attended)
        
        ##### Call for BUTD #######
#         my_model = model_class(w_emb, q_emb, v_att, q_net, v_net, classifier, question_consistency_model, skip_thought,
#                                decode_question, attended)
        
        ###### Call for BAN #########
#         my_model = model_class(w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, op, gamma, question_consistency_model,
#                                 skip_thought, decode_question, attended)

    elif is_question_consistency and is_failure_prediction:
        feat_combine = model_config['failure_predictor']['feat_combine']
        if feat_combine == 'iqa':
            input_size = joint_embedding_dim + model_config['failure_predictor']['answer_hidden_size']
        elif feat_combine == 'iq':
            input_size = joint_embedding_dim
        else:  
            raise NotImplementedError('feat combine of type {} is not implemented'.format(feat_combine))

        failure_predictor = build_failure_prediction_module(input_size, **model_config['failure_predictor'])

        skip_thought = model_config['question_consistency'].get('skip_thought', False)
        decode_question = model_config['question_consistency'].get('decode_question', False)
        attended = model_config['question_consistency'].get('attended', False)
        question_consistency_model = build_question_consistency_module(**model_config['question_consistency'])

        model_class = vqa_multi_modal_with_fpqc_cycle

        my_model = model_class(image_emdedding_models_list, question_embeding_models,
                               multi_modal_combine, classifier, image_feature_encode_list,
                               inter_model, failure_predictor, question_consistency_model,
                               skip_thought, decode_question, attended)

    else:
        my_model = vqa_multi_modal_model(image_emdedding_models_list, question_embeding_models,
                                         multi_modal_combine, classifier, image_feature_encode_list, inter_model)
#         my_model = butd_model(w_emb, q_emb, v_att, q_net, v_net, classifier)
#         my_model = ban_model(w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, op, gamma)

    if use_cuda:
        my_model = my_model.cuda()

    if torch.cuda.device_count() > 1:
        my_model = nn.DataParallel(my_model)

    return my_model
