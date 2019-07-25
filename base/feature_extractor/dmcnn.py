#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger
import torch.nn as nn
from ..pt4nlp import MultiSizeMultiPoolingCNNEncoder, MultiSizeCNNEncoder


class CNNEventFeatureExtractor(nn.Module):

    def __init__(self, input_size,
                 multi_pooling=True, hidden_size=300,
                 cnn_size=[3], pooling_type='max',
                 dropout=0, act='iden', bn=False, bias=True):
        super(CNNEventFeatureExtractor, self).__init__()
        self.multi_pooling = multi_pooling

        if self.multi_pooling:
            self.encoder = MultiSizeMultiPoolingCNNEncoder(input_size,
                                                           hidden_size=hidden_size,
                                                           window_size=[int(ws) for ws in cnn_size],
                                                           pooling_type=pooling_type,
                                                           dropout=dropout,
                                                           bias=bias,
                                                           split_point_number=1)
        else:
            self.encoder = MultiSizeCNNEncoder(input_size,
                                               hidden_size=hidden_size,
                                               window_size=[int(ws) for ws in cnn_size],
                                               pooling_type=pooling_type,
                                               dropout=dropout,
                                               bias=bias)

        if act.lower() != 'iden':
            self.act_function = getattr(nn, act)()
        else:
            self.act_function = None

        if bn:
            self.bn_layer = nn.BatchNorm1d(self.encoder.output_size)
        else:
            self.bn_layer = None

        self.output_size = self.encoder.output_size

    def forward(self, x, position=None, lengths=None):
        if self.multi_pooling:
            sentence_embedding = self.encoder.forward(x, position=position, lengths=lengths)
        else:
            sentence_embedding = self.encoder.forward(x, lengths=lengths)
        if self.bn_layer is not None:
            sentence_embedding = self.bn_layer(sentence_embedding)

        if self.act_function is not None:
            sentence_embedding = self.act_function(sentence_embedding)

        return sentence_embedding
