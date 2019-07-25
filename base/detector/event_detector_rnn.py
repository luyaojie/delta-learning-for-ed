#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger
import torch.nn as nn

from ..feature_extractor import RNNEventFeatureExtractor


class RNNEventDetector(nn.Module):

    def __init__(self, opt, encoder_input_size, word_vec_size):
        super(RNNEventDetector, self).__init__()

        self.encoder_input_size = encoder_input_size
        self.word_vec_size = word_vec_size

        self.rnn_feature_extractor = RNNEventFeatureExtractor(self.encoder_input_size, hidden_size=opt.rnn_size,
                                                              num_layers=opt.rnn_layer, rnn_type=opt.rnn_type,
                                                              brnn=True, dropout=opt.rnn_dropout,
                                                              context=opt.rnn_context,
                                                              rnn_cat=opt.rnn_cat)

        # self.lexi_feature_extractor = LexicalFeatureExtractor(input_size=self.word_vec_size, lexi_win=1)

        classifier_feature_num = self.rnn_feature_extractor.output_size

        self.output_size = classifier_feature_num

    def get_event_detect_feature(self, embeddings, positions, lengths):
        """
        :param embeddings: (batch, max_len, word_vec_size)
        :param positions:  (batch, )
        :param lengths:    (batch, )
        :return:           (batch, feature)
        """

        rnn_input_embedding = embeddings

        rnn_feature = self.rnn_feature_extractor.forward(x=rnn_input_embedding, position=positions, lengths=lengths)
        # lexi_feature = self.lexi_feature_extractor.forward(embeddings=embeddings, position=positions, length=lengths)

        # event_detect_feature = torch.cat([cnn_feature, lexi_feature], 1)
        event_detect_feature = rnn_feature

        return event_detect_feature

    def forward(self, embeddings, positions, lengths):
        """
        :param embeddings: (batch, max_len, word_vec_size)
        :param positions:  (batch, )
        :param lengths:    (batch, )
        :return:           (batch, label_num)
        """
        event_detect_feature = self.get_event_detect_feature(embeddings, positions, lengths)
        return event_detect_feature

    def analysis_context_weight(self, embeddings, positions, lengths):
        """
        :param embeddings: (batch, max_len, word_vec_size)
        :param positions:  (batch, )
        :param lengths:    (batch, )
        :return:           (batch, label_num)
        """
        context_weight = self.rnn_feature_extractor.analysis_context_weight(embeddings, positions, lengths)
        return context_weight
