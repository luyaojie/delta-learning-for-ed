#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger
import torch

from .mask_util import lengths2mask

_big_number = 10000


def mask_mean_pooling(inputs, mask):
    """
    :param inputs:
    :param mask: 1 is using, 0 is not using
    :return:
    """
    return torch.sum(inputs, 1) / torch.sum(mask, 1)[:, None].float()


def mask_sum_pooling(inputs, mask):
    """
    :param inputs:
    :param mask: 1 is using, 0 is not using
    :return:
    """
    return torch.sum(inputs * mask[:, :, None], 1)


def mask_max_pooling(inputs, mask):
    """
    :param inputs:
    :param mask: 1 is using, 0 is not using
    :return:
    """
    return torch.max(inputs * mask[:, :, None] - _big_number * (1 - mask[:, :, None]), 1)[0]


def mask_min_pooling(inputs, mask):
    """
    :param inputs:
    :param mask: 1 is using, 0 is not using
    :return:
    """
    return torch.min(inputs * mask[:, :, None] + _big_number * (1 - mask[:, :, None]), 1)[0]


def mean_pooling(inputs):
    return torch.sum(inputs, 1)


def sum_pooling(inputs):
    return torch.sum(inputs, 1)


def max_pooling(inputs):
    return torch.max(inputs, 1)[0]


def min_pooling(inputs):
    return torch.min(inputs, 1)[0]


def get_pooling(inputs, pooling_type='mean', lengths=None, mask=None):
    if mask is not None:
        if pooling_type == 'mean':
            return mask_mean_pooling(inputs, mask)
        elif pooling_type == 'max':
            return mask_max_pooling(inputs, mask)
        elif pooling_type == 'min':
            return mask_min_pooling(inputs, mask)
        elif pooling_type == 'sum':
            return mask_sum_pooling(inputs, mask)
        else:
            raise NotImplementedError
    elif lengths is not None:
        mask = lengths2mask(lengths, inputs.size()[1])
        if pooling_type == 'mean':
            return mask_mean_pooling(inputs, mask)
        elif pooling_type == 'max':
            return mask_max_pooling(inputs, mask)
        elif pooling_type == 'min':
            return mask_min_pooling(inputs, mask)
        elif pooling_type == 'sum':
            return mask_sum_pooling(inputs, mask)
        else:
            raise NotImplementedError
    else:
        if pooling_type == 'mean':
            return mean_pooling(inputs)
        elif pooling_type == 'max':
            return max_pooling(inputs)
        elif pooling_type == 'min':
            return min_pooling(inputs)
        elif pooling_type == 'sum':
            return sum_pooling(inputs)
        else:
            raise NotImplementedError
