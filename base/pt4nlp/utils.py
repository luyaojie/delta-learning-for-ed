#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger
from __future__ import absolute_import

import sys
from builtins import range

import numpy as np
import torch

REAL = np.float32
if sys.version_info[0] >= 3:
    unicode = str


def to_unicode(text, encoding='utf8', errors='strict'):
    """Convert a string (bytestring in `encoding` or unicode), to unicode.
    :param text:
    :param encoding:
    :param errors: errors can be 'strict', 'replace' or 'ignore' and defaults to 'strict'.
    """
    if isinstance(text, unicode):
        return text
    return unicode(text, encoding, errors=errors)


def any2utf8(text, encoding='utf8', errors='strict'):
    """Convert a string (unicode or bytestring in `encoding`), to bytestring in utf8."""
    if isinstance(text, unicode):
        return text.encode('utf8')
    # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
    return unicode(text, encoding, errors=errors).encode('utf8')


def aeq(*args):
    base = args[0]
    for a in args[1:]:
        assert a == base, str(args)


def load_word2vec_format(filename, word_idx, binary=False, normalize=False,
                         encoding='utf8', unicode_errors='ignore'):
    """
    refer to gensim
    load Word Embeddings
    If you trained the C model using non-utf8 encoding for words, specify that
    encoding in `encoding`.
    :param filename :
    :param word_idx :
    :param binary   : a boolean indicating whether the data is in binary word2vec format.
    :param normalize:
    :param encoding :
    :param unicode_errors: errors can be∂ 'strict', 'replace' or 'ignore' and defaults to 'strict'.
    """
    vocab = set()
    print("loading word embedding from %s" % filename)
    with open(filename, 'rb') as fin:
        header = to_unicode(fin.readline(), encoding=encoding)
        vocab_size, vector_size = map(int, header.split())  # throws for invalid file format
        word_matrix = torch.zeros(len(word_idx), vector_size)

        def add_word(_word, _weights):
            if _word not in word_idx:
                return
            vocab.add(_word)
            word_matrix[word_idx[_word]] = _weights

        if binary:
            binary_len = np.dtype(np.float32).itemsize * vector_size
            for _ in range(vocab_size):
                # mixed text and binary: read text first, then binary
                word = []
                while True:
                    ch = fin.read(1)
                    if ch == b' ':
                        break
                    if ch != b'\n':  # ignore newlines in front of words (some binary files have)
                        word.append(ch)
                word = to_unicode(b''.join(word), encoding=encoding, errors=unicode_errors)
                weights = torch.from_numpy(np.fromstring(fin.read(binary_len), dtype=REAL))
                add_word(word, weights)
        else:
            for line_no, line in enumerate(fin):
                parts = to_unicode(line.rstrip(), encoding=encoding, errors=unicode_errors).split(" ")
                if len(parts) != vector_size + 1:
                    raise ValueError("invalid vector on line %s (is this really the text format?)" % line_no)
                word, weights = parts[0], list(map(REAL, parts[1:]))
                add_word(word, weights)
    if word_idx is not None:
        assert (len(word_idx), vector_size) == word_matrix.size()
    if normalize:
        # each row normalize to 1
        word_matrix = torch.renorm(word_matrix, 2, 0, 1)
    print("loaded %d words pre-trained from %s with %d" % (len(vocab), filename, vector_size))
    return word_matrix, vector_size, vocab


def save_word2vec_format(word_idx, word_matrix, filename, binary=False):
    # word_matrix[0] for mask in embedding.py
    assert len(word_idx) == word_matrix.shape[0]
    with open(filename, 'wb') as fout:
        fout.write(any2utf8("%s %s\n" % (len(word_idx), word_matrix.shape[1])))
        for word in word_idx.keys():
            _embedding = word_matrix[word_idx[word]].astype(REAL)
            if binary:
                fout.write(any2utf8(word) + b" " + _embedding.tostring())
            else:
                fout.write(any2utf8("%s %s\n" % (word, ' '.join("%f" % val for val in _embedding))))


def clip_weight_norm(model, max_norm, norm_type=2,
                     row_norm_params=None, col_norm_params=None):
    """Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        model: nn.Module
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        row_norm_params (list): using row norm
        col_norm_params (list): using col norm

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    for name, param in model.named_parameters():
        if len(param.size()) != 2:
            continue
        if row_norm_params and name in row_norm_params:
            row_norm = torch.norm(param.data, norm_type, 1)
            desired_norm = torch.clamp(row_norm, 0, np.sqrt(max_norm))
            scale = desired_norm / (row_norm + 1e-7)
            param.data = scale[:, None] * param.data
        elif col_norm_params and name in col_norm_params:
            col_norm = torch.norm(param.data, norm_type, 0)
            desired_norm = torch.clamp(col_norm, 0, np.sqrt(max_norm))
            scale = desired_norm / (col_norm + 1e-7)
            param.data *= scale


def align_batch(features, default_pad=0):
    """
    :param features: List[(len1, feature_size), (len2, feature_size), (len2, feature_size), ...]
    :param default_pad: Default Pad Index is 0
    :return:
        (batch, max_len, feature_size)
    """
    batch_size = len(features)
    max_length = max([feature.size(0) for feature in features])
    if len(features[0].size()) == 2:
        feature_size = features[0].size(1)
        aligned_feature = features[0].new(batch_size, max_length, feature_size).fill_(default_pad)
        for index, feature in enumerate(features):
            length = feature.size(0)
            aligned_feature[index, :, :].narrow(0, 0, length).copy_(feature)
    else:
        aligned_feature = features[0].new(batch_size, max_length).fill_(default_pad)
        for index, feature in enumerate(features):
            length = feature.size(0)
            aligned_feature[index, :].narrow(0, 0, length).copy_(feature)
    return aligned_feature


def get_default_dict(special_symbol=True, lower=True, char=False):
    from . import Constants
    if char:
        from .dictionary import CharDictionary
        word_d = CharDictionary(lower)
    else:
        from .dictionary import Dictionary
        word_d = Dictionary(lower)
    if special_symbol:
        word_d.add_specials([Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD],
                            [Constants.PAD, Constants.UNK, Constants.BOS, Constants.EOS])
    return word_d


def convert_to_long_tensor(x):
    return torch.LongTensor(x)


def to_variable(v, volatile=False, device=-1):
    from torch.autograd import Variable
    if device >= 0:
        return Variable(v).cuda(device)
    else:
        return Variable(v)


def to_variable_as(v, m):
    from torch.autograd import Variable
    return Variable(v).to(m.device)


def load_model(model_file, device=-1):
    print("Load model from %s ..." % model_file)
    if device >= 0:
        model = torch.load(model_file, map_location=lambda storage, loc: storage.cuda(device))
    else:
        model = torch.load(model_file, map_location=lambda storage, loc: storage.cpu())
    return model


def save_model(model, model_file):
    print("Save model to %s ..." % model_file)
    torch.save(model, model_file)
