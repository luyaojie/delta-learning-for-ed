# -*- coding:utf-8 -*-
# Author: Roger
import abc
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mask_util import lengths2mask


class WordSeqAttentionModel(nn.Module):
    __metaclass__ = abc.ABCMeta

    def __init__(self, input_size, seq_size):
        super(WordSeqAttentionModel, self).__init__()
        self.input_size = input_size
        self.output_size = seq_size
        self.seq_size = seq_size

    @abc.abstractmethod
    def _score(self, x, seq):
        """
        Using through attention function
        :param x:
        :param seq:
        :return:
        """
        pass

    def attention(self, x, seq, lengths=None, log_softmax=False):
        """
        :param x: (batch, dim, )
        :param seq: (batch, length, dim, )
        :param lengths: (batch, )
        :param log_softmax: using log_softmax to normalize
        :return: weight: (batch, length)
        """
        # Check Size
        batch_size, input_size = x.size()
        seq_batch_size, max_len, seq_size = seq.size()
        assert batch_size == seq_batch_size

        score = self._score(x, seq)

        if lengths is not None:
            mask = lengths2mask(lengths, max_len, byte=True, negation=True)
            score.masked_fill_(mask, float("-inf"))

        if isinstance(self, TanhBilinearWordSeqAttention):
            score = F.tanh(score)

        if log_softmax:
            weight = F.log_softmax(score, dim=1)
        else:
            weight = F.softmax(score, dim=1)

        return weight

    def attention_mask(self, x, seq, mask=None, log_softmax=False):
        """
        :param x: (batch, dim, )
        :param seq: (batch, length, dim, )
        :param mask: (batch, )
                True is Use
                False is not Use
                mask -> [0, 0, 1, 1]
                weight -> [0.5, 0.5, 0, 0]
        :param log_softmax: using log_softmax to normalize
        :return: weight: (batch, length)
        """
        # Check Size
        batch_size, input_size = x.size()
        seq_batch_size, max_len, seq_size = seq.size()
        assert batch_size == seq_batch_size

        score = self._score(x, seq)

        if mask is not None:
            score.masked_fill_(1 - mask, float("-inf"))

        if isinstance(self, TanhBilinearWordSeqAttention):
            score = F.tanh(score)

        if log_softmax:
            weight = F.log_softmax(score, dim=1)
        else:
            weight = F.softmax(score, dim=1)

        return weight

    def forward(self, x, seq, lengths):
        """
        :param x: (batch, dim, )
        :param seq: (batch, length, dim, )
        :param lengths: (batch, )
        :return: hidden: (batch, dim)
                 weight: (batch, length)
        """
        # (batch, length)
        weight = self.attention(x, seq, lengths)
        # (batch, 1, length) bmm (batch, length, dim) -> (batch, 1, dim) -> (batch, dim)
        return torch.bmm(weight[:, None, :], seq).squeeze(1), weight

    def forward_mask(self, x, seq, mask):
        """
        :param x: (batch, dim, )
        :param seq: (batch, length, dim, )
        :param mask: (batch, length)
                True is Use
                False is not Use
                mask -> [0, 0, 1, 1]
                weight -> [0.5, 0.5, 0, 0]
        :return: hidden: (batch, dim)
                 weight: (batch, length)
        """
        # (batch, length)
        weight = self.attention_mask(x, seq, mask)
        # (batch, 1, length) bmm (batch, length, dim) -> (batch, 1, dim) -> (batch, dim)
        return torch.bmm(weight[:, None, :], seq).squeeze(1), weight

    def check_size(self, x, seq):
        batch_size, input_size = x.size()
        seq_batch_size, max_len, seq_size = seq.size()
        assert batch_size == seq_batch_size
        assert input_size == self.input_size
        assert seq_size == self.seq_size

    @staticmethod
    def expand_x(x, max_len):
        """
        :param x: (batch, input_size)
        :param max_len: scalar
        :return:  (batch * max_len, input_size)
        """
        batch_size, input_size = x.size()
        return torch.unsqueeze(x, 1).expand(batch_size, max_len, input_size).contiguous().view(batch_size * max_len, -1)

    @staticmethod
    def pack_seq(seq):
        """
        :param seq: (batch_size, max_len, seq_size)
        :return: (batch_size * max_len, seq_size)
        """
        return seq.view(seq.size(0) * seq.size(1), -1)


class DotWordSeqAttention(WordSeqAttentionModel):
    """
    Effective Approaches to Attention-based Neural Machine Translation
    Minh-Thang Luong, Hieu Pham, and Christopher D. Manning
    In Proceedings of EMNLP 2015
    http://aclweb.org/anthology/D/D15/D15-1166.pdf
    """

    def __init__(self, input_size, seq_size):
        super(DotWordSeqAttention, self).__init__(input_size=input_size, seq_size=seq_size)
        assert input_size == seq_size

    def _score(self, x, seq):
        """
        :param x: (batch, dim)
        :param seq: (batch, length, dim)
        :return: weight: (batch, length)
        """
        self.check_size(x, seq)

        # (batch, dim) -> (batch, dim, 1)
        _x = torch.unsqueeze(x, -1)

        # (batch, length, dim) dot (batch, dim, 1) -> (batch, length, 1)
        score = torch.bmm(seq, _x)

        # (batch, length, 1) -> (batch, length)
        score = torch.squeeze(score, -1)

        return score


class BilinearWordSeqAttention(WordSeqAttentionModel):
    """
    Effective Approaches to Attention-based Neural Machine Translation
    Minh-Thang Luong, Hieu Pham, and Christopher D. Manning
    In Proceedings of EMNLP 2015
    http://aclweb.org/anthology/D/D15/D15-1166.pdf
    """

    def __init__(self, input_size, seq_size):
        super(BilinearWordSeqAttention, self).__init__(input_size=input_size, seq_size=seq_size)
        self.linear = nn.Linear(input_size, seq_size)

    def _score(self, x, seq):
        """
        :param x: (batch, word_dim)
        :param seq: (batch, length, seq_dim)
        :return: score: (batch, length, )
        """
        self.check_size(x, seq)

        # (batch, word_dim) -> (batch, seq_dim)
        x_w = self.linear(x)
        #  (batch, length, seq_dim) batch mm (batch, seq_dim, 1) -> (batch, length)
        score = seq.bmm(x_w.unsqueeze(2)).squeeze(2)

        return score


class TanhBilinearWordSeqAttention(WordSeqAttentionModel):
    """
    Effective Approaches to Attention-based Neural Machine Translation
    Minh-Thang Luong, Hieu Pham, and Christopher D. Manning
    In Proceedings of EMNLP 2015
    http://aclweb.org/anthology/D/D15/D15-1166.pdf
    """

    def __init__(self, input_size, seq_size):
        super(TanhBilinearWordSeqAttention, self).__init__(input_size=input_size, seq_size=seq_size)
        self.linear = nn.Linear(input_size, seq_size)
        self.bias = nn.Parameter(torch.Tensor([1]))
        self.nonlinear = nn.Tanh()

    def _score(self, x, seq):
        """
        :param x: (batch, word_dim)
        :param seq: (batch, length, seq_dim)
        :return: score: (batch, length, )
        """
        self.check_size(x, seq)

        # (batch, word_dim) -> (batch, seq_dim)
        x_w = self.linear(x)
        #  (batch, length, seq_dim) batch mm (batch, seq_dim, 1) -> (batch, length)
        score = seq.bmm(x_w.unsqueeze(2)).squeeze(2) + self.bias

        return score


class ConcatWordSeqAttention(WordSeqAttentionModel):
    """
    Effective Approaches to Attention-based Neural Machine Translation
    Minh-Thang Luong, Hieu Pham, and Christopher D. Manning
    In Proceedings of EMNLP 2015
    http://aclweb.org/anthology/D/D15/D15-1166.pdf
    """

    def __init__(self, input_size, seq_size):
        super(ConcatWordSeqAttention, self).__init__(input_size=input_size, seq_size=seq_size)
        # (word_dim + seq_dim)
        self.layer = nn.Linear(input_size + seq_size, 1, bias=False)

    def _score(self, x, seq):
        """
        :param x: (batch, word_dim)
        :param seq: (batch, length, seq_dim)
        :return: score: (batch, length, )
        """
        batch_size, max_len = seq.size()[:2]
        self.check_size(x, seq)

        # (batch, word_dim) -> (batch * max_len, word_dim)
        _x = self.expand_x(x, max_len=seq.size(1))

        # (batch, max_len, seq_dim) -> (batch * max_len, seq_dim)
        _seq = self.pack_seq(seq)

        # (batch * max_len, word_dim) (batch * max_len, seq_dim) -> (batch * max_len, word_dim + seq_dim)
        to_input = torch.cat([_x, _seq], 1)

        # (batch * max_len, word_dim + seq_dim) -> (batch * max_len, 1) -> (batch * max_len, ) -> (batch, max_len)
        score = self.layer.forward(to_input).view(batch_size, max_len)

        return torch.nn.functional.tanh(score)


class MLPWordSeqAttention(WordSeqAttentionModel):
    """
    Neural Machine Translation By Jointly Learning To Align and Translate
    Dzmitry Bahdanau, KyungHyun Cho, and Yoshua Bengio
    In Proceedings of ICLR 2015
    http://arxiv.org/abs/1409.0473v3
    """

    def __init__(self, input_size, seq_size, hidden_size=None, activation="Tanh", bias=False):
        super(MLPWordSeqAttention, self).__init__(input_size=input_size, seq_size=seq_size)
        self.bias = bias
        self.hidden_size = hidden_size
        if hidden_size is None:
            hidden_size = int((input_size + seq_size) / 2)
        component = OrderedDict()
        component['layer1'] = nn.Linear(input_size + seq_size, hidden_size, bias=bias)
        component['act'] = getattr(nn, activation)()
        component['layer2'] = nn.Linear(hidden_size, 1, bias=bias)
        self.layer = nn.Sequential(component)

    def _score(self, x, seq):
        """
        :param x: (batch, word_dim)
        :param seq: (batch, length, seq_dim)
        :return: score: (batch, length, )
        """
        batch_size, max_len = seq.size()[:2]
        self.check_size(x, seq)

        # (batch, word_dim) -> (batch * max_len, word_dim)
        _x = self.expand_x(x, max_len=seq.size(1))

        # (batch, max_len, seq_dim) -> (batch * max_len, seq_dim)
        _seq = self.pack_seq(seq)

        # (batch * max_len, word_dim) (batch * max_len, seq_dim) -> (batch * max_len, word_dim + seq_dim)
        to_input = torch.cat([_x, _seq], 1)

        # (batch * max_len, word_dim + seq_dim)
        #   -> (batch * max_len, 1)
        #       -> (batch * max_len, )
        #           -> (batch, max_len)
        score = self.layer.forward(to_input).view(batch_size, max_len)

        return score


class DotMLPWordSeqAttention(WordSeqAttentionModel):
    """
    WebQA: A Chinese Open-Domain Factoid Question Answering Dataset
    Peng Li, Wei Li, Zhengyan He, Xuguang Wang, Ying Cao, Jie Zhou, and Wei Xu
    http://arxiv.org/abs/1607.06275
    """

    def __init__(self, input_size, seq_size, activation="Tanh", bias=False):
        super(DotMLPWordSeqAttention, self).__init__(input_size=input_size, seq_size=seq_size)
        self.bias = bias
        component = OrderedDict()
        component['layer1'] = nn.Linear(seq_size, input_size, bias=bias)
        component['act'] = getattr(nn, activation)()
        self.layer = nn.Sequential(component)

    def _score(self, x, seq):
        """
        :param x: (batch, word_dim)
        :param seq: (batch, length, seq_dim)
        :return: score: (batch, length, )
        """
        batch_size, max_len = seq.size()[:2]
        self.check_size(x, seq)

        # (batch, word_dim) -> (batch * max_len, word_dim)
        _x = self.expand_x(x, max_len=seq.size(1))

        # (batch, max_len, seq_dim) -> (batch * max_len, seq_dim)
        _seq = self.pack_seq(seq)

        # (batch * max_len, seq_dim) -> (batch * max_len, word_dim)
        _seq_output = self.layer(_seq)

        # (batch * max_len, word_dim) * (batch * max_len, word_dim)
        #   -> (batch * max_len, word_dim)
        #       -> (batch * max_len, 1)
        #           -> (batch, max_len)
        score = torch.sum(_x * _seq_output, 1).view(batch_size, max_len)

        return score


ATTENTION_TYPE = ['dot', 'bilinear', 'general', 'mlp', 'dotmlp', 'tanhbilinear']


def get_attention(type_name):
    attention_map = {'dot': 'DotWordSeqAttention'.lower(),
                     'bilinear': 'BilinearWordSeqAttention'.lower(),
                     'general': 'ConcatWordSeqAttention'.lower(),
                     'mlp': 'MLPWordSeqAttention'.lower(),
                     'dotmlp': 'DotMLPWordSeqAttention'.lower(),
                     'tanhbilinear': 'TanhBilinearWordSeqAttention'.lower(),

                     'DotWordSeqAttention'.lower(): 'DotWordSeqAttention'.lower(),
                     'BilinearWordSeqAttention'.lower(): 'BilinearWordSeqAttention'.lower(),
                     'ConcatWordSeqAttention'.lower(): 'ConcatWordSeqAttention'.lower(),
                     'MLPWordSeqAttention'.lower(): 'MLPWordSeqAttention'.lower(),
                     'DotMLPWordSeqAttention'.lower(): 'DotMLPWordSeqAttention'.lower(),
                     'TanhBilinearWordSeqAttention': 'TanhBilinearWordSeqAttention'.lower(),
                     }

    if attention_map[type_name] == 'DotWordSeqAttention'.lower():
        return DotWordSeqAttention
    elif attention_map[type_name] == 'BilinearWordSeqAttention'.lower():
        return BilinearWordSeqAttention
    elif attention_map[type_name] == 'ConcatWordSeqAttention'.lower():
        return ConcatWordSeqAttention
    elif attention_map[type_name] == 'MLPWordSeqAttention'.lower():
        return MLPWordSeqAttention
    elif attention_map[type_name] == 'DotMLPWordSeqAttention'.lower():
        return DotMLPWordSeqAttention
    elif attention_map[type_name] == 'TanhBilinearWordSeqAttention'.lower():
        return TanhBilinearWordSeqAttention
    else:
        raise NotImplementedError
