#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger
import torch
import torch.nn as nn

from ..pt4nlp import RNNEncoder, get_attention, select_position_rnn_hidden, mask_util


class RNNEventFeatureExtractor(nn.Module):

    def __init__(self, input_size, rnn_type='LSTM', brnn=True, hidden_size=168, dropout=0.2, num_layers=1,
                 context='none', rnn_cat=False):
        super(RNNEventFeatureExtractor, self).__init__()
        self.input_size = input_size
        self.rnn_type = rnn_type
        self.brnn = brnn
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layers = num_layers

        self.encoder = RNNEncoder(input_size=self.input_size,
                                  hidden_size=self.hidden_size,
                                  dropout=self.dropout,
                                  num_layers=self.num_layers,
                                  brnn=self.brnn,
                                  rnn_type=self.rnn_type)
        self.context = context
        self.rnn_cat = rnn_cat

        if self.context == 'none':
            self.attention = None
            self.output_size = self.encoder.output_size
        else:
            self.attention = get_attention(self.context)(self.encoder.output_size, self.encoder.output_size)
            self.output_size = self.encoder.output_size * 2 if self.rnn_cat else self.encoder.output_size

    def rnn_forward(self, x, lengths=None):
        """
        :param x:        (batch, len, input_size)
        :param lengths:  (batch, )
        :return:
            (batch, len, hidden_size)
        """
        batch_size, max_len, dim_size = x.size()
        assert dim_size == self.input_size
        outputs, last_hidden = self.encoder.forward(x, lengths)
        return outputs, last_hidden

    def forward(self, x, position, lengths=None):
        """
        :param x:        (batch, len, input_size)
        :param position: (batch, )
        :param lengths:  (batch, )
        :return:
        """
        # (batch, len, num_directions * cell_size)
        outputs, last_hidden = self.rnn_forward(x, lengths)

        # (batch, len, hidden_size)
        #   -> (batch, hidden_size)
        hidden = select_position_rnn_hidden(outputs, position)

        if self.attention is None:
            return hidden
        else:
            if self.rnn_cat:
                position_mask = mask_util.position2mask(position, x.size(1), byte=True, negation=True)
                length_mask = mask_util.lengths2mask(lengths, max_length=x.size(1), byte=True, negation=False)
                context, weight = self.attention.forward_mask(hidden, outputs, mask=position_mask & length_mask)
                return torch.cat([hidden, context], 1)
            else:
                context, weight = self.attention.forward(hidden, outputs, lengths)
                return context

    def analysis_context_weight(self, x, position, lengths=None):
        """
        :param x:        (batch, len, input_size)
        :param position: (batch, )
        :param lengths:  (batch, )
        :return:
        """
        if self.attention is None:
            weight = mask_util.position2mask(position, x.size(1))
            return weight

        # (batch, len, num_directions * cell_size)
        outputs, last_hidden = self.rnn_forward(x, lengths)

        # (batch, len, hidden_size)
        #   -> (batch, hidden_size)
        hidden = select_position_rnn_hidden(outputs, position)

        if self.rnn_cat:
            position_mask = mask_util.position2mask(position, x.size(1), byte=True, negation=True)
            length_mask = mask_util.lengths2mask(lengths, max_length=x.size(1), byte=True, negation=False)
            context, weight = self.attention.forward_mask(hidden, outputs, mask=position_mask & length_mask)
            return weight + (1 - position_mask.float())
        else:
            context, weight = self.attention.forward(hidden, outputs, lengths)
            return weight
