#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger
import torch
import torch.nn as nn

from .Constants import PAD_WORD, PAD
from .embedding import Embeddings
from .utils import get_default_dict
from .mask_util import lengths2mask


class RelativePositionEmbedding(nn.Module):

    def __init__(self, embedding_size=5, max_length=100, position_dict=None):
        super(RelativePositionEmbedding, self).__init__()
        self.max_length = max_length
        self.relative_position_dict = RelativePositionEmbedding.get_position_dictionary(max_length, position_dict)
        self.embedding = Embeddings(word_vec_size=embedding_size, dicts=self.relative_position_dict)
        self.output_size = self.embedding.output_size

    @staticmethod
    def get_position_dictionary(max_length=100, position_dict=None):
        if position_dict is None:
            position_dict = get_default_dict(special_symbol=False)
            position_dict.add_specials([PAD_WORD], [PAD])

        for number in range(-max_length, max_length + 1):
            position_dict.add_special(number)

        return position_dict

    def get_relative_position(self, positions, lengths):
        """
        :param positions:   (batch, )
        :param lengths:     (batch, )
        :return:
        """
        # Get relative_positions (batch, max_length)
        # relative_positions = torch.range(0, torch.max(lengths).item() - 1, dtype=torch.long).to(lengths.device)
        # torch range -> torch.arange, because torch 0.5 will remove range
        relative_positions = torch.arange(0, torch.max(lengths), dtype=torch.long).to(lengths.device)
        relative_positions = relative_positions.expand(positions.size(0), relative_positions.size(0))

        relative_positions = relative_positions + self.max_length - positions[:, None] + 1
        relative_positions = torch.clamp(relative_positions, min=1, max=2 * self.max_length + 1)

        # Pad excluding index
        pad_position = lengths2mask(lengths, torch.max(lengths).item(), byte=True, negation=True)
        relative_positions.masked_fill_(pad_position, PAD)
        return relative_positions.detach()

    def forward(self, positions, lengths):
        """
        :param positions:   (batch, )
        :param lengths:     (batch, )
        :return:
            (batch, length, embedding_size)
        """
        relative_position = self.get_relative_position(positions, lengths)
        rp_embedding = self.embedding.forward(relative_position)
        return rp_embedding
