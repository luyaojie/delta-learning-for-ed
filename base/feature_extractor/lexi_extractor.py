#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger
import torch
import torch.nn as nn


class LexicalFeatureExtractor(nn.Module):

    def __init__(self, input_size, lexi_win=1):
        super(LexicalFeatureExtractor, self).__init__()
        assert lexi_win >= 0
        self.input_size = input_size
        self.lexi_win = lexi_win
        self.output_size = input_size * (2 * lexi_win + 1)

    def forward(self, embeddings, position, length):
        """
        :param embeddings:  (batch, max_len, dim)
        :param position:    (batch, )
        :param length:      (batch, )
        :return:
            (batch, 2 * lexi_win + 1, dim)
        """
        select_embeddings = list()
        for index in range(-self.lexi_win, self.lexi_win + 1):
            select_position = torch.clamp(position + index, min=0)
            select_position = torch.min(select_position, length - 1)
            select_position = select_position.view(position.size(0), 1, 1).expand(embeddings.size(0), 1,
                                                                                  embeddings.size(2))
            select_embeddings.append(torch.gather(embeddings, 1, select_position).squeeze(1))
        return torch.cat(select_embeddings, 1)
