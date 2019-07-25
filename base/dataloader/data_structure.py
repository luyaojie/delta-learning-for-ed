#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger


class EventDetectBatch:
    def __init__(self, feature, position, label, length, iden=None, pred=None):
        """
        :param feature:  discrete_feature, seq_feature, classifier feature, text
            discrete_feature (batch, event_len, feature_size)
            seq_feature      (batch, event_len, feature_size)
            classifier feature (batch, feature_size)
            text [batch[w1, w2, ], ...]
            mask:     (batch, length)
                1 is use
                but excluding 0
        :param position: (batch, )
        :param length:   (batch, )
        :param label:    (batch, )
        :param iden:     [(docid, sentid, tokenid), ...]
        """
        self.feature = feature
        self.position = position
        self.length = length
        self.label = label
        self.ident = iden
        self.pred = pred


class EventData:
    def __init__(self, folder_name, prefix_name):
        self.golden_dat = "%s/%s.golden.dat" % (folder_name, prefix_name)
        self.sents_dat = "%s/%s.sents.dat" % (folder_name, prefix_name)
        self.ids_dat = "%s/%s.ids.dat" % (folder_name, prefix_name)
        self.trigger_dat = "%s/%s.trigger.dat" % (folder_name, prefix_name)
        self.elmo_dat = "%s/%s.elmo.dat" % (folder_name, prefix_name)
