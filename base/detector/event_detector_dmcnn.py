#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger
import torch
import torch.nn as nn

from ..feature_extractor import LexicalFeatureExtractor, CNNEventFeatureExtractor


class DMCNNEventDetector(nn.Module):

    def __init__(self, opt, encoder_input_size, word_vec_size):
        super(DMCNNEventDetector, self).__init__()

        self.encoder_input_size = encoder_input_size
        self.word_vec_size = word_vec_size

        self.cnn_feature_extractor = CNNEventFeatureExtractor(input_size=encoder_input_size,
                                                              act=opt.cnn_act,
                                                              hidden_size=opt.cnn_hidden_size,
                                                              cnn_size=opt.cnn_filter,
                                                              bn=opt.cnn_bn,
                                                              dropout=opt.cnn_dropout)
        if opt.lexi_wind >= 0:
            self.lexi_feature_extractor = LexicalFeatureExtractor(
                input_size=self.word_vec_size, lexi_win=opt.lexi_wind)
            classifier_feature_num = self.lexi_feature_extractor.output_size + self.cnn_feature_extractor.output_size
        else:
            self.lexi_feature_extractor = None
            classifier_feature_num = self.cnn_feature_extractor.output_size

        self.output_size = classifier_feature_num

    def get_event_detect_feature(self, embeddings, positions, lengths):
        """
        :param embeddings: (batch, max_len, word_vec_size)
        :param positions:  (batch, )
        :param lengths:    (batch, )
        :return:           (batch, feature)
        """
        cnn_input_embedding = embeddings

        cnn_feature = self.cnn_feature_extractor.forward(x=cnn_input_embedding, position=positions, lengths=lengths)
        if self.lexi_feature_extractor is not None:
            lexi_feature = self.lexi_feature_extractor.forward(embeddings=embeddings[:, :, :self.word_vec_size],
                                                               position=positions, length=lengths)

            event_detect_feature = torch.cat([cnn_feature, lexi_feature], 1)
        else:
            event_detect_feature = cnn_feature

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
        context_weight = self.cnn_feature_extractor.analysis_context_weight(embeddings, positions, lengths)
        return context_weight
