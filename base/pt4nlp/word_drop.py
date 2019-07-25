#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger
import torch

from .Constants import PAD, UNK


def word_drop_(x, replace_index=UNK, pad_index=PAD, dropout=0.05):
    """
    :param x:               (Batch, Length)
    :param replace_index:   Index for Replace
    :param pad_index:       Pad Index
    :param dropout:         Dropout Rate
    :return:
        Indexes after Word Drop
        Example: replace_index = 1
        [[ 4,  2,  4,  2],
         [ 5,  2,  3,  2]]
         ->
        [[ 4,  1,  4,  2],
         [ 5,  2,  1,  2]]
    """
    drop_position_rate = (x != pad_index).to(x.device, dtype=torch.float32) * dropout

    drop_position = torch.bernoulli(drop_position_rate).byte()

    x.masked_fill_(drop_position, replace_index)

    return x
