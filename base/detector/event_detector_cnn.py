#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger

import torch.nn as nn

from ..feature_extractor import CNNEventFeatureExtractor


class CNNEventDetector(nn.Module):

    def __init__(self, opt, encoder_input_size, word_vec_size):
        super(CNNEventDetector, self).__init__()

        self.encoder_input_size = encoder_input_size
        self.word_vec_size = word_vec_size

        self.cnn_feature_extractor = CNNEventFeatureExtractor(input_size=encoder_input_size,
                                                              act=opt.cnn_act,
                                                              hidden_size=opt.cnn_hidden_size,
                                                              cnn_size=opt.cnn_filter,
                                                              bn=opt.cnn_bn,
                                                              dropout=opt.cnn_dropout,
                                                              multi_pooling=False)

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

        return cnn_feature

    def forward(self, embeddings, positions, lengths):
        """
        :param embeddings: (batch, max_len, word_vec_size)
        :param positions:  (batch, )
        :param lengths:    (batch, )
        :return:           (batch, label_num)
        """
        event_detect_feature = self.get_event_detect_feature(embeddings, positions, lengths)
        return event_detect_feature
