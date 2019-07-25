#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger
from .Constants import *
from .attention import DotMLPWordSeqAttention, get_attention
from .attention import DotWordSeqAttention, BilinearWordSeqAttention, ConcatWordSeqAttention, MLPWordSeqAttention
from .classifier import SoftmaxClassifier, MLPClassifier
from .convolution import CNNEncoder, MultiSizeCNNEncoder, MultiPoolingCNNEncoder, MultiSizeMultiPoolingCNNEncoder
from .dictionary import Dictionary, CharDictionary
from .early_stopper import EarlyStopper
from .embedding import Embeddings, ConvolutionEmbedding
from .gradient_reversal import GradReverse, grad_reverse, GradReverseLayer
from .optimizer import Optimizer
from .position import RelativePositionEmbedding
from .recurrent import RNNEncoder, PadBasedRNNEncoder, select_position_rnn_hidden
from .utils import clip_weight_norm, get_default_dict, convert_to_long_tensor, to_variable, to_variable_as, load_model
from .utils import save_word2vec_format, align_batch

__all__ = ["Dictionary", "CharDictionary",
           "get_default_dict", "convert_to_long_tensor",
           "to_variable", "load_model", "to_variable_as",
           "align_batch",
           "PAD", "UNK", "BOS", "EOS", "PAD_WORD", "UNK_WORD", "BOS_WORD", "EOS_WORD",
           "Embeddings", "ConvolutionEmbedding",
           "CNNEncoder", "MultiSizeCNNEncoder", "MultiPoolingCNNEncoder", "MultiSizeMultiPoolingCNNEncoder",
           "RNNEncoder", "PadBasedRNNEncoder", "select_position_rnn_hidden",
           "SoftmaxClassifier", "MLPClassifier",
           "DotWordSeqAttention", "BilinearWordSeqAttention", "ConcatWordSeqAttention", "MLPWordSeqAttention",
           "DotMLPWordSeqAttention", "get_attention",
           "clip_weight_norm", "save_word2vec_format",
           "Optimizer",
           "RelativePositionEmbedding",
           "EarlyStopper",
           "GradReverse", "grad_reverse", "GradReverseLayer"]
