#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger
import torch
import torch.nn as nn
import torch.nn.functional as nn_function

from base.dataloader.event_data_utils import FEATURE_INDEX_MAP
from base.detector.event_detector_cnn import CNNEventDetector
from base.detector.event_detector_dmcnn import DMCNNEventDetector
from base.detector.event_detector_rnn import RNNEventDetector
from base.pt4nlp import ConvolutionEmbedding, select_position_rnn_hidden, GradReverseLayer
from base.pt4nlp import Embeddings, Constants, RelativePositionEmbedding, MLPClassifier
from base.pt4nlp import mask_util, UNK, get_default_dict, convert_to_long_tensor, align_batch
from base.utils.utils import get_feature_flag


class EmbeddingLayer(nn.Module):

    def __init__(self, word_dict, args, feature_dict_list=None):
        super(EmbeddingLayer, self).__init__()
        self.feat_vec_size = args.feat_vec_size
        self.trigger_mask = args.trigger_mask
        encoder_feature_size = 0

        char_dict = get_default_dict(char=True)
        for word in word_dict.word2index.keys():
            for char in word:
                char_dict.add(char)

        if args.char:
            self.char_embedding = ConvolutionEmbedding(feat_vec_size=50,
                                                       feat_dict=char_dict,
                                                       hidden_size=100)
            encoder_feature_size += self.char_embedding.output_size
        else:
            self.char_embedding = None

        self.feature_flag = get_feature_flag(args)

        if args.rp:
            self.rp_embedding = RelativePositionEmbedding(embedding_size=self.feat_vec_size)
            print("init %s with uniform (-1, 1)" % self.rp_embedding.embedding)
            self.rp_embedding.embedding.emb_luts[0].weight.data.uniform_(-1, 1)
            self.rp_embedding.embedding.emb_luts[0].weight.data[Constants.PAD].zero_()
            encoder_feature_size += self.rp_embedding.output_size
        else:
            self.rp_embedding = None

        self.embeddings = Embeddings(word_vec_size=args.word_vec_size, dicts=word_dict,
                                     feature_dicts=feature_dict_list,
                                     feature_dims=[self.feat_vec_size] * (len(self.feature_flag) - 1))

        encoder_feature_size += self.embeddings.output_size
        self.feature_num = len(self.embeddings.emb_luts)

        if len(self.embeddings.emb_luts) > 1:
            for emb_index in range(1, len(self.embeddings.emb_luts)):
                emb_table = self.embeddings.emb_luts[emb_index]
                emb_table.weight.data.uniform_(-1, 1)
                emb_table.weight.data[Constants.PAD].zero_()

        self.dropout = nn.Dropout(p=args.emb_dropout)

        self.output_size = encoder_feature_size

    def forward(self, feature, positions, lengths):
        """
        :param feature:   (batch, max_len, n)
        :param positions: (batch, )
        :param lengths:   (batch, )
        :return:
            (batch, label_num)
        """
        # discrete_feature, seq_feature, classifier feature, text

        discrete_feature, seq_feature, _, text = feature[:4]

        if len(self.feature_flag) != len(FEATURE_INDEX_MAP):
            discrete_feature = torch.stack([discrete_feature[:, :, flag_index] for flag_index in self.feature_flag],
                                           2).detach()

        if self.trigger_mask:
            position_mask = mask_util.position2mask(positions, discrete_feature.size(1), byte=True)
            discrete_feature[:, :, 0].masked_fill_(position_mask, UNK)

        word_embeddings = self.embeddings.forward(discrete_feature)

        to_cat_embedding_list = []

        if self.char_embedding:
            char_input_list = [
                convert_to_long_tensor(self.char_embedding.feat_dict.convert_to_index(t, Constants.UNK_WORD))
                for t in text]
            char_input = align_batch(char_input_list).to(discrete_feature.device)
            char_embedding = self.char_embedding.forward(char_input, lengths)
            to_cat_embedding_list += [char_embedding]

        if self.rp_embedding:
            relative_embedding = self.rp_embedding.forward(positions, lengths)
            to_cat_embedding_list += [relative_embedding]

        if to_cat_embedding_list:
            word_embeddings = torch.cat([word_embeddings] + to_cat_embedding_list, 2)

        return self.dropout.forward(word_embeddings)


class EncoderLayer(nn.Module):

    def __init__(self, encoder_input_size, args, model):
        super(EncoderLayer, self).__init__()
        self.model = model.lower()
        if model.lower() == 'dmcnn':
            self.event_detector = DMCNNEventDetector(opt=args, encoder_input_size=encoder_input_size,
                                                     word_vec_size=args.word_vec_size)
        elif model.lower() == 'cnn':
            self.event_detector = CNNEventDetector(opt=args, encoder_input_size=encoder_input_size,
                                                   word_vec_size=args.word_vec_size)
        elif model.lower() == 'rnn':
            self.event_detector = RNNEventDetector(opt=args, encoder_input_size=encoder_input_size,
                                                   word_vec_size=args.word_vec_size)
        else:
            raise NotImplementedError

        self.output_size = self.event_detector.output_size

    def forward(self, embeddings, positions, lengths, mask=None):
        if self.model == 'debug':
            return self.event_detector.forward(embeddings, positions, lengths, mask)
        return self.event_detector.forward(embeddings, positions, lengths)

    def analysis_context_weight(self, embeddings, positions, lengths):
        return self.event_detector.analysis_context_weight(embeddings, positions, lengths)


class SingleLabelLossLayer(nn.Module):

    def __init__(self, args, label_dict):
        super(SingleLabelLossLayer, self).__init__()
        self.loss_layer = self.get_loss_layer(args, label_dict)

    def forward(self, pred_score, target):
        """
        :param pred_score: (Batch, Label_size)
        :param target:     (Batch, Label_size)
        :return:
        """
        # all 0 is "Other"
        target_index = torch.max(target, 1)[1]
        return self.loss_layer.forward(pred_score, target=target_index)

    @staticmethod
    def infer_prob(pred_score):
        return nn_function.softmax(pred_score, 1)

    @staticmethod
    def get_loss_layer(args, label_dict):
        if args.neg_scale is not None:
            loss_weight = torch.ones(label_dict.size())
            loss_weight[label_dict.word2index['other']] = args.neg_scale
        else:
            loss_weight = torch.ones(label_dict.size())

        if args.loss == 'nll':
            loss_layer = nn.CrossEntropyLoss(weight=loss_weight)
        else:
            raise NotImplementedError("No %s Loss" % args.loss)

        return loss_layer


class EventDetector(nn.Module):

    def __init__(self, word_dict, label_dict, args, feature_dict_list=None):
        super(EventDetector, self).__init__()
        self.trigger_mask = args.trigger_mask
        self.args = args

        if self.args.adv_target == 'word':
            self.unify_dim = 300
            self.lexi_dim = args.word_vec_size
        else:
            self.unify_dim = 512
            self.lexi_dim = 1024

        self.lexi_embedding_layer = EmbeddingLayer(word_dict, args)
        self.lexi_spec_embedding_layer = EmbeddingLayer(word_dict, args, feature_dict_list=feature_dict_list)
        self.lexi_free_embedding_layer = EmbeddingLayer(word_dict, args, feature_dict_list=feature_dict_list)

        self.lexical_specific_layer = EncoderLayer(self.lexi_spec_embedding_layer.output_size, args=args, model='rnn')
        self.lexical_free_layer = EncoderLayer(self.lexi_free_embedding_layer.output_size, args=args, model='dmcnn')

        self.specific_to_unify = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.lexical_specific_layer.output_size, self.unify_dim),
            nn.LeakyReLU()
        )
        self.free_to_unify = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.lexical_free_layer.output_size, self.unify_dim),
            nn.LeakyReLU()
        )
        self.lexi_to_unify = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.lexi_dim, self.unify_dim),
            nn.LeakyReLU()
        )

        self.multi_gate = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.unify_dim, self.unify_dim),
            nn.LeakyReLU(),
            nn.Linear(self.unify_dim, self.unify_dim),
        )

        classifier_feature_num = self.unify_dim

        self.classifier = MLPClassifier(input_dim=classifier_feature_num,
                                        output_dim=label_dict.size(),
                                        dropout=args.mlp_dropout,
                                        hidden_sizes=args.mlp_hiddens,
                                        bn=args.mlp_bn,
                                        act=args.mlp_act,
                                        )

        self.adv_target = args.adv_target
        self.adv_num = args.adv_num
        self.adv_weight = args.adv_weight
        self.enhance_weight = args.enhance_weight

        self.gradient_reversal_layer = GradReverseLayer(lamb=args.adv_weight)
        self.gradient_scale_layer = GradReverseLayer(lamb=-args.enhance_weight)

        if args.adv_weight == 0 and args.enhance_weight == 0:
            self.gradient_reversal_layer = None
            self.adv_classifier = None
        else:

            self.adv_classifier = nn.Sequential(
                nn.Linear(in_features=self.unify_dim + self.lexi_dim,
                          out_features=500),
                nn.Tanh(),
                nn.Linear(in_features=500,
                          out_features=1),
                nn.Sigmoid())

        self.pre_train_adv = True

        self.loss_layer = SingleLabelLossLayer(args, label_dict)

        self.loss_type = args.loss

    def load_specific_model(self, model):
        self.lexi_spec_embedding_layer.load_state_dict(model.lexi_spec_embedding_layer.state_dict())
        self.lexical_specific_layer.load_state_dict(model.lexical_specific_layer.state_dict())
        self.specific_to_unify.load_state_dict(model.specific_to_unify.state_dict())

    def load_free_model(self, model):
        self.lexi_free_embedding_layer.load_state_dict(model.lexi_free_embedding_layer.state_dict())
        self.lexical_free_layer.load_state_dict(model.lexical_free_layer.state_dict())
        self.free_to_unify.load_state_dict(model.free_to_unify.state_dict())

    def load_pre_train_model(self, model, embedding=True, encoder=True, fusion_layer=True, classifier=True):
        if embedding:
            print('Load Embedding Module')
            self.lexi_spec_embedding_layer.load_state_dict(model.lexi_spec_embedding_layer.state_dict())
            self.lexi_free_embedding_layer.load_state_dict(model.lexi_free_embedding_layer.state_dict())
        if encoder:
            print('Load Encoder Module')
            self.lexical_free_layer.load_state_dict(model.lexical_free_layer.state_dict())
            self.lexical_specific_layer.load_state_dict(model.lexical_specific_layer.state_dict())
        if fusion_layer:
            print('Load Fusion Module')
            self.specific_to_unify.load_state_dict(model.specific_to_unify.state_dict())
            self.free_to_unify.load_state_dict(model.free_to_unify.state_dict())
            self.lexi_to_unify.load_state_dict(model.lexi_to_unify.state_dict())
            self.multi_gate.load_state_dict(model.multi_gate.state_dict())
        if classifier:
            print('Load Classifier Module')
            self.classifier.load_state_dict(model.classifier.state_dict())

    def get_event_detector_params(self):
        return self.event_detector.parameters()

    def get_lexi_feature(self, feature, positions):
        if self.adv_target == 'word':
            lexi_feature = select_position_rnn_hidden(feature[0], positions)[:, 0].unsqueeze(-1)
            lexi_feature = self.lexi_embedding_layer.embeddings.forward(lexi_feature).squeeze(1).detach()
        elif self.adv_target == 'elmo':
            lexi_feature = feature[2]
        else:
            raise NotImplementedError
        return lexi_feature

    def get_adv_feature(self, feature, hidden_states, positions):
        if self.adv_target == 'word':
            adv_feature = select_position_rnn_hidden(feature[0], positions)[:, 0].unsqueeze(-1)
            adv_feature = self.lexi_embedding_layer.embeddings.forward(adv_feature).squeeze(1).detach()
        elif self.adv_target == 'elmo':
            adv_feature = select_position_rnn_hidden(feature[1], positions)
        elif self.adv_target == 'hidden':
            adv_feature = select_position_rnn_hidden(hidden_states, positions)
        else:
            raise NotImplementedError
        return adv_feature

    def get_adv_neg_feature(self, feature, hidden_states, lengths):
        neg_position = (lengths.new_zeros(lengths.size()).float().uniform_() * lengths.float()).floor().long().detach()
        return self.get_adv_feature(feature, hidden_states, neg_position)

    def get_adv_pred(self, feature, hidden_states, positions, lengths, to_adv_feature, enhance=False):
        if enhance:
            adv_classifier_feature = self.gradient_scale_layer.forward(to_adv_feature)
        else:
            adv_classifier_feature = self.gradient_reversal_layer.forward(to_adv_feature)

        pred_list = list()
        gold_list = list()

        positive_feature = self.get_adv_feature(feature, hidden_states, positions)
        positive_adv_pred = self.adv_classifier.forward(torch.cat([positive_feature, adv_classifier_feature], 1))
        pred_list += [positive_adv_pred]
        gold_list += [feature[0].new_ones(positions.size(0)).float()]

        for i in range(self.adv_num):
            negative_feature = self.get_adv_neg_feature(feature, hidden_states, lengths)
            negative_adv_pred = self.adv_classifier.forward(torch.cat([negative_feature, adv_classifier_feature], 1))
            pred_list += [negative_adv_pred]
            gold_list += [feature[0].new_zeros(positions.size(0)).float()]
        return gold_list, pred_list

    def get_adv_loss(self, feature, hidden_states, positions, lengths, to_adv_feature, enhance=False):

        if enhance:
            adv_classifier_feature = self.gradient_scale_layer.forward(to_adv_feature)
        else:
            adv_classifier_feature = self.gradient_reversal_layer.forward(to_adv_feature)

        positive_feature = self.get_adv_feature(feature, hidden_states, positions)
        positive_adv_pred = self.adv_classifier.forward(torch.cat([positive_feature, adv_classifier_feature], 1))
        positive_adv_loss = nn.functional.binary_cross_entropy(positive_adv_pred.squeeze(1),
                                                               feature[0].new_ones(positions.size(0)).float())
        adv_loss = positive_adv_loss
        for i in range(self.adv_num):
            negative_feature = self.get_adv_neg_feature(feature, hidden_states, lengths)
            negative_adv_pred = self.adv_classifier.forward(torch.cat([negative_feature, adv_classifier_feature], 1))
            negative_adv_loss = nn.functional.binary_cross_entropy(negative_adv_pred.squeeze(1),
                                                                   feature[0].new_zeros(positions.size(0)).float())
            adv_loss += negative_adv_loss

        return adv_loss

    def _forward(self, feature, positions, lengths):
        """
        :param feature:   (batch, max_len, 1)
        :param positions: (batch, )
        :param lengths:   (batch, )
        :return:
            (batch, label_num)
        """
        lexi_spec_embeddings = self.lexi_spec_embedding_layer.forward(feature, positions, lengths)
        lexi_free_embeddings = self.lexi_free_embedding_layer.forward(feature, positions, lengths)

        lexi_spec_feature = self.lexical_specific_layer.forward(embeddings=lexi_spec_embeddings,
                                                                positions=positions,
                                                                lengths=lengths,
                                                                mask=feature[-1])
        lexi_free_feature = self.lexical_free_layer.forward(embeddings=lexi_free_embeddings,
                                                            positions=positions,
                                                            lengths=lengths,
                                                            mask=feature[-1])
        lexi_feature = self.get_lexi_feature(feature, positions)

        if not self.args.full_model_update:
            lexi_spec_feature = lexi_spec_feature.detach()
            lexi_free_feature = lexi_free_feature.detach()

        lexi_spec_uni = self.specific_to_unify(lexi_spec_feature)
        lexi_free_uni = self.free_to_unify(lexi_free_feature)
        lexi_uni = self.lexi_to_unify(lexi_feature)

        # for rd_fuse_rw or rg_fuse_rw, comment the other one is fine.
        gate1 = self.multi_gate.forward(lexi_spec_uni)
        gate2 = self.multi_gate.forward(lexi_free_uni)
        gate3 = self.multi_gate.forward(lexi_uni)
        gate = torch.stack([gate1, gate2, gate3], 2)
        gate = nn_function.softmax(gate, 2)
        classifier_feature = gate[:, :, 0] * lexi_spec_uni + gate[:, :, 1] * lexi_free_uni + gate[:, :, 2] * lexi_uni

        pred = self.classifier.forward(classifier_feature)

        return pred, lexi_spec_uni, lexi_free_uni, lexi_spec_embeddings

    def forward(self, feature, positions, lengths):
        """
        :param feature:   (batch, max_len, 1)
        :param positions: (batch, )
        :param lengths:   (batch, )
        :return:
            (batch, label_num)
        """
        pred, _, _, _ = self._forward(feature, positions, lengths)
        return pred

    def forward_with_adv(self, feature, positions, lengths):
        """
        :param feature:   (batch, max_len, 1)
        :param positions: (batch, )
        :param lengths:   (batch, )
        :return:
            (batch, label_num)
        """
        pred, lexi_spec_uni, lexi_free_uni, embeddings = self._forward(feature, positions, lengths)
        hidden_states, _ = self.lexical_specific_layer.event_detector.rnn_feature_extractor.rnn_forward(embeddings,
                                                                                                        lengths)
        return pred, lexi_spec_uni, lexi_free_uni, hidden_states

    def context_analysis(self, feature, positions, lengths):
        """
        :param feature:   (batch, max_len, 1)
        :param positions: (batch, )
        :param lengths:   (batch, )
        :return:
            (batch, label_num)
        """
        embeddings = self.embedding_layer.forward(feature, positions, lengths)

        context_weight = self.encoder_layer.analysis_context_weight(embeddings=embeddings, positions=positions,
                                                                    lengths=lengths)

        return context_weight

    def eval_enhance_adv(self, feature, positions, lengths):
        pred, map_lexi_specific, map_lexi_free, hidden_states = self.forward_with_adv(feature, positions, lengths)
        enhance_gold, enhance_pred = self.get_adv_pred(feature, hidden_states, positions, lengths, map_lexi_specific,
                                                       enhance=True)
        adv_gold, adv_pred = self.get_adv_pred(feature, hidden_states, positions, lengths, map_lexi_free,
                                               enhance=False)
        return enhance_gold, enhance_pred, adv_gold, adv_pred

    def loss(self, feature, positions, lengths, label):
        if self.adv_weight > 0 or self.enhance_weight > 0:
            pred, map_lexi_specific, map_lexi_free, hidden_states = self.forward_with_adv(feature, positions, lengths)
            classifier_loss = self.loss_layer.forward(pred, label)
            if self.pre_train_adv:
                enhance_loss = self.get_adv_loss(feature, hidden_states, positions, lengths, map_lexi_specific.detach(),
                                                 enhance=True)
                adversarial_loss = self.get_adv_loss(feature, hidden_states, positions, lengths, map_lexi_free.detach(),
                                                     enhance=False)
                return enhance_loss + adversarial_loss, [0, enhance_loss.item(), adversarial_loss.item()]
            else:
                enhance_loss = self.get_adv_loss(feature, hidden_states, positions, lengths, map_lexi_specific,
                                                 enhance=True)
                adversarial_loss = self.get_adv_loss(feature, hidden_states, positions, lengths, map_lexi_free,
                                                     enhance=False)
                return classifier_loss + enhance_loss + adversarial_loss, [classifier_loss.item(),
                                                                           enhance_loss.item(),
                                                                           adversarial_loss.item()]
        else:
            pred = self.forward(feature, positions, lengths)

            classifier_loss = self.loss_layer.forward(pred, label)
            return classifier_loss, [classifier_loss.item(), 0]

    def infer_prob(self, feature, positions, lengths):
        predict_score = self.forward(feature, positions, lengths)
        return self.loss_layer.infer_prob(predict_score)
