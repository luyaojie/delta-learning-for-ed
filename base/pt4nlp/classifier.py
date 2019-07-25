# -*- coding:utf-8 -*-
# Created by Roger
from collections import OrderedDict

import torch
import torch.nn as nn


class SoftmaxClassifier(nn.Module):
    """
    Applies the Softmax function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range (0,1) and sum to 1

    Softmax is defined as
    :math:`f_i(x) = exp(x_i) / sum_j exp(x_j)`

    Shape:
        - Input: :math:`(N, L)`
        - Output: :math:`(N, L)`
    """

    def __init__(self):
        super(SoftmaxClassifier, self).__init__()

        self.classifier = nn.Softmax()

    def forward(self, x):
        """
        :param x: `(N, L)`
        :return: `(N, L)`
        """
        assert x.dim() == 2, 'Softmax requires a 2D tensor as input'
        return self.classifier.forward(x)

    def __repr__(self):
        return self.__class__.__name__ + ' ()'

    def predict_prob(self, x):
        """
        :param x: `(N, L)`
        :return: `(N, L)`
        """
        return self.forward(x)

    @staticmethod
    def prob2label(self, prob):
        """
        :param prob: `(N, L)`
        :return: `(N, L)`
        """
        _, pred_label = torch.max(prob, 1)
        return pred_label

    def predict(self, x, prob=True):
        return self.predict_prob(x) if prob else self.predict_label(x)


class MLPClassifier(nn.Module):

    def __init__(self, input_dim, output_dim, dropout=0, hidden_sizes=None, act='Iden', bn=False):
        """
        input_dim -> hidden_sizes[0] -> ... -> hidden_sizes[-1] -> output
        :param input_dim:
        :param output_dim:
        :param dropout:
        :param hidden_sizes: [hidden_size] * layer_num
        :param act:          Only Need when len(hidden_sizes) > 0
        """
        super(MLPClassifier, self).__init__()

        out_component = OrderedDict()

        last_layer_dim = input_dim

        if hidden_sizes is not None and len(hidden_sizes) > 0:
            for layer_index, hidden_size in enumerate(hidden_sizes):
                out_component['mlp_layer_%s_hidden_dropout' % layer_index] = nn.Dropout(dropout)
                out_component['mlp_layer_%s_hidden_linear' % layer_index] = nn.Linear(last_layer_dim, hidden_size)
                torch.nn.init.xavier_uniform_(out_component['mlp_layer_%s_hidden_linear' % layer_index].weight)
                print("Init %s.weight (%s) with %s" % (out_component['mlp_layer_%s_hidden_linear' % layer_index],
                                                       out_component[
                                                           'mlp_layer_%s_hidden_linear' % layer_index].weight.size(),
                                                       "xavier_uniform"))
                if bn:
                    out_component['mlp_layer_%s_hidden_bn' % layer_index] = nn.BatchNorm1d(hidden_size)
                if act.lower() != 'iden':
                    out_component['mlp_layer_%s_hidden_act' % layer_index] = getattr(nn, act)()
                last_layer_dim = hidden_size

        out_component['dropout'] = nn.Dropout(dropout)
        out_component['output'] = nn.Linear(last_layer_dim, output_dim)

        self.encoder = nn.Sequential(out_component)
        self.output_size = output_dim

    def forward(self, x):
        return self.encoder.forward(x)
