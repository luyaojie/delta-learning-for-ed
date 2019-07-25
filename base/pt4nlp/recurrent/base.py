#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger
import torch
import torch.nn as nn


def get_rnn(rnn_type):
    if rnn_type == "lstm":
        rnn = nn.LSTM
    elif rnn_type == "gru":
        rnn = nn.GRU
    elif rnn_type == "rnn":
        rnn = nn.RNN
    else:
        raise NotImplementedError("RNN Type: LSTM GRU RNN")
    return rnn


def get_rnn_cell(rnn_type):
    if rnn_type == "lstm":
        rnn_cell = nn.LSTMCell
    elif rnn_type == "gru":
        rnn_cell = nn.GRUCell
    elif rnn_type == "rnn":
        rnn_cell = nn.RNNCell
    else:
        raise NotImplementedError("RNN Type: LSTM GRU RNN")
    return rnn_cell


def select_position_rnn_hidden(hidden_states, position):
    """
    :param hidden_states: (batch, len, hidden_size)
    :param position:      (batch, )
    :return:
        (batch, hidden_size)
    """
    batch_size, max_len, dim_size = hidden_states.size()
    max_position = torch.max(position)
    assert max_len > max_position.item()
    position = position.unsqueeze(-1).unsqueeze(-1)
    position = position.expand(batch_size, 1, dim_size)
    hidden = torch.gather(hidden_states, 1, position).squeeze(1)
    return hidden
