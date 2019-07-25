#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger
import torch
import torch.nn as nn
from torch.autograd import Variable

from .mask_util import relative_postition2mask, lengths2mask
from .pooling import get_pooling, _big_number


class CNNEncoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=168,
                 window_size=3,
                 pooling_type='max',
                 padding=True,
                 dropout=0.5,
                 bias=True):
        super(CNNEncoder, self).__init__()
        # Define Parameter
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.pooling_type = pooling_type if isinstance(pooling_type, list) else [pooling_type]
        self.padding = padding
        self.dropout = dropout
        self.bias = bias

        # Define Layer
        # (N, Cin, Hin, Win)
        # In NLP, Hin is length, Win is Word Embedding Size
        self.conv_layer = nn.Conv2d(in_channels=1,
                                    out_channels=hidden_size,
                                    kernel_size=(self.window_size, self.input_size),
                                    # window_size-1 padding for length
                                    # zero padding for word dim
                                    padding=(self.window_size - 1, 0) if padding else 0,
                                    bias=bias)
        self.dropout_layer = nn.Dropout(self.dropout)

        self.init_model()

        self.output_size = self.hidden_size * len(self.pooling_type)

    def init_model(self):
        for name, param in self.conv_layer.named_parameters():
            if param.data.dim() >= 2:
                print("Init cnn weight %s(%s) with %s" % (name, param.data.size(), "xavier_uniform"))
                nn.init.xavier_uniform_(param)

    def forward_conv(self, inputs):
        """
        :param inputs: batch x len x input_size
        :return:
                if padding is False:
                    batch x len - window_size + 1 x hidden_size
                if padding is True
                    batch x len + window_size - 1 x hidden_size
        """
        # (batch x len x input_size) -> (batch x 1 x len x input_size)
        inputs = torch.unsqueeze(inputs, 1)
        # (batch x 1 x len x input_size) -> (batch x hidden_size x new_len x 1)
        _temp = self.conv_layer(inputs)
        # (batch x hidden_size x new_len x 1)
        # -> (batch x hidden_size x new_len)
        # -> (batch x new_len x hidden_size)
        _temp.squeeze_(3)
        return torch.transpose(_temp, 1, 2)

    def forward(self, inputs, lengths=None):
        """
        :param inputs: batch x len x input_size
        :param lengths: batch
        :return: batch x hidden_size
        """
        dp_input = self.dropout_layer(inputs)
        conv_result = self.forward_conv(dp_input)
        if lengths is not None:
            if self.padding:
                lengths = lengths + (self.window_size - 1)
            else:
                lengths = lengths - (self.window_size - 1)
        pooling_result = [get_pooling(conv_result, pooling_type=pooling_type, lengths=lengths)
                          for pooling_type in self.pooling_type]
        return torch.cat(pooling_result, 1)

    def pooling_index(self, inputs, lengths=None):
        """
        :param inputs: batch x len x input_size
        :param lengths: batch
        :return: batch x hidden_size
        """
        dp_input = self.dropout_layer(inputs)
        conv_result = self.forward_conv(dp_input)
        if lengths is not None:
            if self.padding:
                lengths = lengths + (self.window_size - 1)
            else:
                lengths = lengths - (self.window_size - 1)
        mask = lengths2mask(lengths, conv_result.size()[1])
        pooling_index = torch.max(conv_result * mask[:, :, None] - _big_number * (1 - mask[:, :, None]), 1)[1]
        if lengths is not None:
            if self.padding:
                pooling_index = pooling_index - (self.window_size - 1)
            else:
                pooling_index = pooling_index + (self.window_size - 1)
        return pooling_index


class MultiSizeCNNEncoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=168,
                 window_size=[3, 4, 5],
                 pooling_type='max',
                 padding=True,
                 dropout=0.5,
                 bias=True):
        super(MultiSizeCNNEncoder, self).__init__()
        # Define Parameter
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.pooling_type = pooling_type
        self.padding = padding
        self.dropout = dropout
        self.bias = bias

        # Define Layer
        # (N, Cin, Hin, Win)
        # In NLP, Hin is length, Win is Word Embedding Size
        self.conv_layer = nn.ModuleList([CNNEncoder(input_size,
                                                    hidden_size=self.hidden_size,
                                                    window_size=win_size,
                                                    pooling_type=self.pooling_type,
                                                    padding=self.padding,
                                                    dropout=self.dropout,
                                                    bias=self.bias)
                                         for win_size in self.window_size])

        self.init_model()

        self.output_size = self.hidden_size * len(self.conv_layer)

    def init_model(self):
        pass

    def forward(self, inputs, lengths=None):
        """
        :param inputs: batch x len x input_size
        :param lengths: batch
        :return: batch x hidden_size
        """
        conv_results = [conv(inputs, lengths) for conv in self.conv_layer]
        return torch.cat(conv_results, 1)


class MultiPoolingCNNEncoder(CNNEncoder):
    def __init__(self,
                 input_size,
                 hidden_size=168,
                 window_size=3,
                 pooling_type='max',
                 dropout=0.5,
                 bias=True,
                 split_point_number=1):
        CNNEncoder.__init__(self,
                            input_size=input_size,
                            hidden_size=hidden_size,
                            window_size=window_size,
                            pooling_type=pooling_type,
                            padding=True,
                            dropout=dropout,
                            bias=bias)

        self.split_point_number = split_point_number

        self.output_size = self.hidden_size * (self.split_point_number + 1)

    def init_model(self):
        for name, param in self.conv_layer.named_parameters():
            '''
            if param.data.dim() == 4:
                f1, f2, f3, f4 = param.data.size()
                fan_in = f2 * f3 * f4 * 1.
                fan_out = f1 * f3 * f4 / 100.
                low_high_range = math.sqrt(6.0 / (fan_in +fan_out ))
                print(fan_in)
                print(fan_out)
                print("Init %s with %s" % (name, low_high_range))
                param.data.uniform_(-low_high_range, low_high_range)
            '''
            if param.data.dim() >= 2:
                print("Init %s with %s" % (name, "xavier_uniform"))
                nn.init.xavier_uniform_(param)

    def forward(self, inputs, position, lengths=None):
        """
        :param inputs: batch x len x input_size
        :param position: batch x split number
        :param lengths: batch
        :return: batch x hidden_size
        """
        batch_size = inputs.size(0)

        if position.dim() == 1:
            position = position.unsqueeze(1)

        split_number = position.size(1)
        assert split_number == self.split_point_number

        if lengths is None:
            max_length = inputs.size(1)
        else:
            max_length = torch.max(lengths).item()
        if self.padding:
            max_length += self.window_size - 1
            position = position + self.window_size - 1

        dp_input = self.dropout_layer(inputs)
        conv_result = self.forward_conv(dp_input)

        if lengths is not None:
            if self.padding:
                lengths = lengths + (self.window_size - 1)
            else:
                lengths = lengths - (self.window_size - 1)

        split_positions = [p.squeeze(0) for p in position.t().split(1)]
        zero_start = Variable(inputs.data.new(batch_size).fill_(0)).long()
        length_end = lengths if lengths is not None else Variable(inputs.data.new(batch_size).fill_(max_length)).long()

        mask_list = list()
        left_positions = [zero_start] + split_positions
        right_positions = split_positions + [length_end]
        for left, right in zip(left_positions,
                               right_positions):
            mask_list.append(relative_postition2mask(left, right, max_length))

        if lengths is not None:
            mask_all = torch.sum(torch.stack(mask_list))
            assert int(mask_all.item()) == int(mask_all.item())

        pooling_results = [get_pooling(conv_result, pooling_type=pooling_type, mask=mask)
                           for pooling_type in self.pooling_type
                           for mask in mask_list]
        return torch.cat(pooling_results, dim=1)


class MultiSizeMultiPoolingCNNEncoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=168,
                 window_size=[3, 4, 5],
                 pooling_type='max',
                 padding=True,
                 dropout=0.5,
                 bias=True,
                 split_point_number=1):
        super(MultiSizeMultiPoolingCNNEncoder, self).__init__()
        # Define Parameter
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.pooling_type = pooling_type
        self.padding = padding
        self.dropout = dropout
        self.bias = bias
        self.split_point_number = split_point_number

        # Define Layer
        # (N, Cin, Hin, Win)
        # In NLP, Hin is length, Win is Word Embedding Size
        self.conv_layer = nn.ModuleList([MultiPoolingCNNEncoder(input_size,
                                                                hidden_size=self.hidden_size,
                                                                window_size=win_size,
                                                                pooling_type=self.pooling_type,
                                                                dropout=self.dropout,
                                                                bias=self.bias,
                                                                split_point_number=self.split_point_number)
                                         for win_size in self.window_size])

        self.init_model()

        self.output_size = sum([conv.output_size for conv in self.conv_layer])

    def init_model(self):
        pass

    def forward(self, inputs, position, lengths=None):
        """
        :param inputs: batch x len x input_size
        :param inputs: batch x n
        :param lengths: batch
        :return: batch x hidden_size
        """
        conv_results = [conv(inputs, position, lengths) for conv in self.conv_layer]
        return torch.cat(conv_results, 1)
