#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger
import torch
import torch.nn as nn
from torch.autograd import Variable
from .base import get_rnn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class RNNEncoder(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size=168,
                 num_layers=1,
                 dropout=0.2,
                 brnn=True,
                 rnn_type="LSTM",
                 multi_layer_hidden='last',
                 bias=True):
        super(RNNEncoder, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = brnn
        self.cell_size = hidden_size // 2 if self.bidirectional else hidden_size
        self.hidden_size = self.cell_size * 2 if self.bidirectional else self.cell_size
        self.rnn_type = rnn_type.lower()
        self.multi_layer_hidden = multi_layer_hidden
        self.bias = bias

        if self.multi_layer_hidden == 'last':
            self.output_size = self.hidden_size
        else:
            raise NotImplementedError

        rnn = get_rnn(self.rnn_type)

        self.rnn = rnn(input_size=self.input_size,
                       hidden_size=self.cell_size,
                       num_layers=self.num_layers,
                       dropout=self.dropout,
                       bidirectional=self.bidirectional,
                       batch_first=True,
                       bias=self.bias)

        self.init_model()

    def init_model(self):
        for name, weight in self.rnn.named_parameters():
            if weight.data.dim() == 2:
                print("Init rnn weight %s(%s) with %s" % (name, weight.data.size(), "orthogonal"))
                nn.init.orthogonal_(weight)

    def get_h0_state(self, x):

        batch_size = x.size()[0]
        state_shape = self.num_layers * 2 if self.bidirectional else self.num_layers, batch_size, self.cell_size

        if self.rnn_type == 'lstm':
            if isinstance(x, Variable):
                h0 = c0 = Variable(x.data.new(*state_shape).zero_()).float()
            else:
                h0 = c0 = Variable(x.new(*state_shape).zero_()).float()
            return h0, c0
        else:
            if isinstance(x, Variable):
                h0 = Variable(x.data.new(*state_shape).zero_()).float()
            else:
                h0 = Variable(x.new(*state_shape).zero_()).float()
            return h0

    def forward(self, x, lengths=None, hidden=None):
        """
        :param x:  (batch, max_len, input_size)
        :param lengths: (batch, )
        :param hidden:  (layer_num * direction, batch_size, cell_size)
                        tuple for LSTM
        :return:
            outputs:     (batch, len, num_directions * hidden_size)
            last_hidden: (batch, hidden_size * num_directions)
        """
        batch_size = x.size()[0]

        if hidden is None:
            h0_state = self.get_h0_state(x)
        else:
            h0_state = hidden

        if lengths is not None:
            sort_lengths, sort_index = torch.sort(lengths, dim=0, descending=True)
            _, raw_index = torch.sort(sort_index, dim=0)

            x = torch.index_select(x, 0, sort_index)

            packed_inputs = pack_padded_sequence(x, sort_lengths.data.tolist(), batch_first=True)

            # # ht for GRU RNN, (ht, ct) for LSTM
            # ht_state (num_layers * num_directions, batch, cell_size)
            outputs, ht_state = self.rnn(packed_inputs, h0_state)

            outputs = pad_packed_sequence(outputs, batch_first=True)[0]

            outputs = torch.index_select(outputs, 0, raw_index)

            if self.rnn_type == 'lstm':
                ht_state = torch.index_select(ht_state[0], 1, raw_index), torch.index_select(ht_state[1], 1, raw_index)
            else:
                ht_state = torch.index_select(ht_state, 1, raw_index)
        else:
            # # ht for GRU RNN, (ht, ct) for LSTM
            outputs, ht_state = self.rnn(x, h0_state)

        if self.rnn_type == "lstm":
            # (num_layers * num_directions, batch, hidden_size), (num_layers * num_directions, batch, hidden_size)
            ht = ht_state[0]
        else:
            # (num_layers * num_directions, batch, hidden_size)
            ht = ht_state

        if self.multi_layer_hidden == 'last':
            # (num_layers * num_directions, batch, hidden_size)
            # -> Last Layer (batch, num_directions * hidden_size)
            if self.bidirectional:
                last_hidden = ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)
            else:
                last_hidden = ht[-1]
        else:
            raise NotImplementedError

        return outputs, last_hidden

    def forward_step(self, x, hidden):
        """

        :param x: (batch, 1, input_size)
        :param hidden: (num_directions * num_layers, batch, cell_size)
                       lstm tuple
        :return:
            (batch, cell_size * num_directions)
            (num_directions * num_layers, batch, cell_size)
            LSTM tuple
        """
        # output (batch, 1, cell_size * num_directions)
        # h_t    (num_directions * num_layers, batch, cell_size)
        #        LSTM tuple
        output, h_t = self.rnn(x, hidden)
        output = output.squeeze(1)
        return output, h_t


class PadBasedRNNEncoder(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size=168,
                 num_layers=1,
                 dropout=0.2,
                 brnn=True,
                 rnn_type="LSTM",
                 multi_layer_hidden='last',
                 bias=True):
        super(PadBasedRNNEncoder, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = brnn
        self.cell_size = hidden_size // 2 if self.bidirectional else hidden_size
        self.hidden_size = self.cell_size * 2 if self.bidirectional else self.cell_size
        self.rnn_type = rnn_type.lower()
        self.multi_layer_hidden = multi_layer_hidden
        self.bias = bias

        if self.multi_layer_hidden == 'concatenate':
            self.output_size = self.hidden_size * self.num_layers
        else:
            self.output_size = self.hidden_size

        rnn = get_rnn(self.rnn_type)

        self.rnns = nn.ModuleList()

        input_size = self.input_size

        for layer_i in range(self.num_layers):
            rnn_i = rnn(input_size=input_size,
                        hidden_size=self.cell_size,
                        dropout=self.dropout,
                        bidirectional=self.bidirectional,
                        batch_first=True,
                        bias=self.bias)
            self.rnns.append(rnn_i)
            input_size = hidden_size

        self.init_model()

    def init_model(self):
        for weight in self.rnns.parameters():
            if weight.data.dim() == 2:
                nn.init.orthogonal(weight)

    def get_h0_state(self, x):

        batch_size = x.size()[0]
        state_shape = self.num_layers * 2 if self.bidirectional else self.num_layers, batch_size, self.cell_size

        if self.rnn_type == 'lstm':
            if isinstance(x, Variable):
                h0 = c0 = Variable(x.data.new(*state_shape).zero_()).float()
            else:
                h0 = c0 = Variable(x.new(*state_shape).zero_()).float()
            return h0, c0
        else:
            if isinstance(x, Variable):
                h0 = Variable(x.data.new(*state_shape).zero_()).float()
            else:
                h0 = Variable(x.new(*state_shape).zero_()).float()
            return h0

    def _forward_one_layer(self, layer_i, x, lengths=None, hidden=None):
        """
        :param x:  (batch, max_len, input_size)
        :param lengths: (batch, )
        :param hidden:  (direction, batch_size, cell_size)
                        tuple for LSTM
        :return:
            outputs:     (batch, seq_len, cell_size * num_directions)
            last_hidden: (num_directions, batch, cell_size)
                          tuple for LSTM
        """
        if hidden is None:
            h0_state = self.get_h0_state(x)
        else:
            h0_state = hidden

        if lengths is not None:
            sort_lengths, sort_index = torch.sort(lengths, dim=0, descending=True)
            _, raw_index = torch.sort(sort_index, dim=0)

            x = torch.index_select(x, 0, sort_index)

            packed_inputs = pack_padded_sequence(x, sort_lengths.data.tolist(), batch_first=True)

            # (batch, seq_len, hidden_size * num_directions)
            # # ht for GRU RNN, (ht, ct) for LSTM
            # ht_state (num_layers * num_directions, batch, cell_size)
            outputs, ht_state = self.rnns[layer_i](packed_inputs, h0_state)

            outputs = pad_packed_sequence(outputs, batch_first=True)[0]

            outputs = torch.index_select(outputs, 0, raw_index)

            if self.rnn_type == 'lstm':
                ht_state = torch.index_select(ht_state[0], 1, raw_index), torch.index_select(ht_state[1], 1, raw_index)
            else:
                ht_state = torch.index_select(ht_state, 1, raw_index)
        else:
            # # ht for GRU RNN, (ht, ct) for LSTM
            outputs, ht_state = self.rnn(x, h0_state)

        return outputs, ht_state

    def forward(self, x, lengths=None, hidden=None):
        """
        :param x:  (batch, max_len, input_size)
        :param lengths: (batch, )
        :param hidden:  (direction * layer_num, batch_size, cell_size)
                        tuple for LSTM
        :return:
            outputs:     multi_layer_hidden 'concatenate' -> (batch, len, cell_size * num_directions * num_layer)
                         multi_layer_hidden 'last' -> (batch, len, cell_size * num_directions)
            last_hidden: multi_layer_hidden 'concatenate' -> (batch, cell_size * num_directions * num_layer)
                         multi_layer_hidden 'last' -> (batch, cell_size * num_directions)
        """
        batch_size = x.size()[0]
        num_direction = 2 if self.bidirectional else 1

        if hidden is None:
            h0_state = self.get_h0_state(x)
        else:
            h0_state = hidden

        # list[(batch, seq_len, cell_size * num_directions)]
        outputs_list = list()
        # list[(num_directions, batch, cell_size)]
        last_hidden_state_list = list()

        input_data = x

        for i in range(self.num_layers):
            hidden_start, hidden_end = i * num_direction, (i + 1) * num_direction

            if self.rnn_type == 'lstm':
                h0_state_i = h0_state[0][hidden_start:hidden_end], h0_state[1][hidden_start:hidden_end]
            else:
                h0_state_i = h0_state[hidden_start:hidden_end]

            # outputs:     (batch, seq_len, cell_size * num_directions)
            # last_hidden: (num_directions, batch, cell_size)
            outputs, ht_state = self._forward_one_layer(i, input_data, lengths, h0_state_i)

            outputs_list.append(outputs)

            # (num_directions, batch, cell_size) -> (batch, cell_size * num_directions)
            if self.rnn_type == 'lstm':
                ht_state = ht_state[0]
            last_hidden_state_list.append(ht_state.transpose(0, 1).contiguous().view(batch_size, -1))

            input_data = outputs

        if self.multi_layer_hidden == 'concatenate':
            # list[(batch, seq_len, cell_size * num_directions)] * layer_num
            #   -> (batch, seq_len, cell_size * num_directions * layer_num)
            outputs = torch.cat(outputs_list, dim=2)
            last_hidden = torch.cat(last_hidden_state_list, dim=1)
        elif self.multi_layer_hidden == 'last':
            outputs = outputs_list[-1]
            last_hidden = last_hidden_state_list[-1]
        else:
            raise NotImplementedError

        return outputs, last_hidden

    def forward_step(self, x, hidden):
        """
        :param x: (batch, 1, input_size)
        :param hidden: (num_layers * num_directions, batch, cell_size)
                       lstm tuple
        :return:
            (batch, cell_size * num_directions * num_layer)
            (num_layers * num_directions, batch, cell_size)
            LSTM tuple
        """
        # output_list: list[(batch, cell_size * num_directions)]
        # hidden_list: list[(num_directions, batch, cell_size)]
        output_list = list()
        hidden_list = list()
        num_direction = 2 if self.bidirectional else 1

        input_data = x

        for i in range(self.num_layers):
            hidden_start, hidden_end = i * num_direction, (i + 1) * num_direction

            if self.rnn_type == 'lstm':
                h0_state_i = hidden[0][hidden_start:hidden_end], hidden[1][hidden_start:hidden_end]
            else:
                h0_state_i = hidden[hidden_start:hidden_end]

            # outputs:     (batch, seq_len, cell_size * num_directions)
            # last_hidden: (num_directions, batch, cell_size)
            output, h_t = self.rnns[i].forward(input_data, h0_state_i)

            output_list.append(output)
            hidden_list.append(h_t)

            input_data = output

        # hidden_list: list[(num_directions, batch, cell_size)]
        #   -> (num_layers * num_directions, batch, cell_size)
        if self.rnn_type == 'lstm':
            h_t, c_t = zip(*hidden_list)
            h_t = torch.cat(h_t, 0), torch.cat(c_t, 0)
        else:
            h_t = torch.cat(hidden_list, 0)

        if self.multi_layer_hidden == 'concatenate':
            # output_list: list[(batch, seq_len, cell_size * num_directions)]
            #   -> (batch, cell_size * num_directions * num_layer)
            output = torch.cat(output_list, 2).squeeze(1)
        elif self.multi_layer_hidden == 'last':
            # output_list: list[(batch, seq_len, cell_size * num_directions)]
            #   -> (batch, seq_len, cell_size * num_directions)
            output = output_list[-1]
        else:
            raise NotImplementedError

        # (batch, seq_len, cell_size * num_directions)
        #   -> (batch, cell_size * num_directions)
        output = output.squeeze(1)

        return output, h_t
