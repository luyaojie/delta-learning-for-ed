#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import torch


def max_infer_scores(scores):
    """
    Infer Label with scores
    :param scores: (batch, label_size)
    :return:
    """
    return torch.max(scores, 1)[1].unsqueeze(1).tolist()


def max_infer_label(model, batch):
    """
    :param model:
    :param batch:
    :return: [[label1, label2], [label1,], ...] batch size
    """
    predict = model.forward(batch.feature, positions=batch.position, lengths=batch.length)
    return max_infer_scores(predict)


def model_predict(model, data, batch_size=64):
    """
    :param model:
    :param data:
    :param batch_size:
    :return:
        valid_predict_results: [(doc_id, start, length, label), ...]
        predict_results: [(doc_id, start, length, token, label), ...]
    """
    predict_results = list()
    model.eval()
    torch.set_grad_enabled(False)
    for batch in data.next_val_batch(batch_size):
        # predict = model.forward(batch.feature, positions=batch.position, lengths=batch.length)
        # predict_label = torch.max(predict, 1)[1].tolist()
        predict_label = max_infer_label(model, batch)
        batch.pred = predict_label
        batch_predict = data.batch2pred(batch)
        # (doc_id, start, length, token, label)
        predict_results += batch_predict
    valid_predict_results = [(d[0], d[1], d[2], d[4]) for d in predict_results]
    return valid_predict_results, predict_results
