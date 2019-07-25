#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger

import math
import random as pyrandom
import string
import torch
from nltk.corpus import stopwords
from six import iteritems

from .data_structure import EventDetectBatch
from .event_data_utils import *
from ..pt4nlp import convert_to_long_tensor, to_variable, Constants, get_default_dict, align_batch

OTHER_NAME = "other"
pos_filter_set = ['N', 'V', 'J']
english_stopwords = set(stopwords.words('english'))


class FeatureMapper:
    """
    Feature Mapper for ELMos
    """

    def __init__(self, elmo_filename, device=-1):
        self.device = device
        self.feature_index_mapper, self.feature_matrix = self.load_feature_matrix(elmo_filename)
        self.pad_index = self.feature_index_mapper['PAD']
        if device >= 0:
            self.feature_matrix = self.feature_matrix.cuda(device)

    def convert_to_feature(self, token_keys):
        token_indexes = [self.feature_index_mapper[token_key] for token_key in token_keys]
        return self.feature_matrix[token_indexes].detach()

    def convert_to_seq_feature(self, seq_token_keys, lengths):
        token_indexes = list()
        max_length = max(lengths)
        for token_keys, length in zip(seq_token_keys, lengths):
            temp_indexes = [self.feature_index_mapper[token_key] for token_key in token_keys]
            temp_indexes += [self.pad_index] * (max_length - length)
            token_indexes += [temp_indexes]
        return self.feature_matrix[convert_to_long_tensor(token_indexes)].detach()

    @staticmethod
    def load_feature_matrix(filename):
        import h5py
        import time
        start = time.time()
        d = h5py.File(filename, 'r')
        elmo_index_mapper = dict()
        elmo_matrix = torch.Tensor(d['feature'])
        elmo_matrix = torch.cat([elmo_matrix, torch.zeros(1, elmo_matrix.size(1))])
        for index, key in enumerate(d['token_key']):
            if len(key) == 3:
                doc_id, sent_id, token_id = key
                elmo_index_mapper[(doc_id.decode('utf8'), int(sent_id), int(token_id))] = index
            elif len(key) == 2:
                doc_id, sent_id = key
                elmo_index_mapper[(doc_id.decode('utf8'), int(sent_id))] = index
            else:
                raise NotImplementedError
        elmo_index_mapper['PAD'] = len(elmo_index_mapper)
        print("Load Elmo Matrix (%s, %s) using %ds" % (elmo_matrix.size(0), elmo_matrix.size(1), time.time() - start))
        return elmo_index_mapper, elmo_matrix


class EventDetectionCorpus(object):

    def __init__(self,
                 event_data,
                 word_dictionary, label_dictionary, feature_dictionary_list,
                 max_length=300,
                 device=-1,
                 trigger_window=-1,
                 random=True,
                 neg_ratio=14,
                 fix_neg=False,
                 word_dropout=0.,
                 pos_filter=False,
                 neg_from_global=False,
                 neg_sample_seed=3435,
                 label_multi_token=False,
                 elmo=False,
                 sentence_feature=False, ):

        self.word_dictionary = word_dictionary
        self.label_dictionary = label_dictionary
        self.feature_dictionary_list = feature_dictionary_list

        self.device = device
        self.trigger_window = trigger_window if trigger_window > 0 else max_length
        self.max_length = max_length
        self.random = random
        self.neg_ratio = neg_ratio
        self.word_dropout = word_dropout
        self.fix_neg = fix_neg
        self.pos_filter = pos_filter
        self.neg_from_global = neg_from_global
        if neg_sample_seed == 0:
            self.neg_sample_seed = 0
        else:
            self.neg_sample_seed = neg_sample_seed if neg_sample_seed > 0 else pyrandom.randint(1, 5000)
        self.random = pyrandom.Random(self.neg_sample_seed)
        print("Neg Sample Seed: %s." % self.neg_sample_seed)

        if elmo:
            self.elmo_mapper = FeatureMapper(event_data.elmo_dat, device=self.device)
        else:
            self.elmo_mapper = None

        if sentence_feature:
            self.sentence_mapper = self.load_sentence_feature(event_data.sentence_feature)
        else:
            self.sentence_mapper = None

        data = self.load_data_file(event_data.ids_dat,
                                   # Label Multi-Token is trigger_label_file is not None
                                   trigger_label_file=event_data.trigger_dat if label_multi_token else None,
                                   label_dict=label_dictionary,
                                   word_dict=word_dictionary,
                                   feature_dict_list=feature_dictionary_list,
                                   max_length=self.max_length,
                                   pos_filter=self.pos_filter)

        self.gold_list = load_trigger_gold_list(event_data.golden_dat)

        self.event_data, self.non_event_data, self.ids_data, self.sent_data = data
        self.data = self.sample_data()

    def clear_data(self):
        self.elmo_mapper = None
        self.event_data = None
        self.non_event_data = None
        self.ids_data = None
        self.sent_data = None
        self.data = None

    def cpu(self):
        self.device = -1

    def cuda(self, device):
        assert device >= 0
        self.device = device

    @property
    def event_data_size(self):
        return len(self.event_data)

    @property
    def nonevent_data_size(self):
        return len(self.non_event_data)

    @staticmethod
    def load_ids_file(filename):
        """
        :param filename: ids file name
        # type is list
        :return: (docid, senid, tokenid) -> start, length, type, token
        """
        token_dict, sent_dict = load_ids_file(filename)

        # check sent
        to_del_list = list()
        sent_keys = sent_dict.keys()
        for sent_key in sent_keys:
            token_set = sent_dict[sent_key]
            if max(token_set) + 1 != len(token_set):
                sys.stderr.write("[WARNING] max token (%s) != length (%s): %s, %s\n" % (max(token_set) + 1,
                                                                                        len(token_set),
                                                                                        sent_key[0],
                                                                                        sent_key[1]))
                to_del_list += [sent_key]
        # Del Sentence
        for to_del in to_del_list:
            del sent_dict[to_del]

        pos_sent_set = set()
        for token_key in token_dict:
            token = token_dict[token_key]
            if token['tri_type'] != 'other':
                pos_sent_set.add((token_key[0], token_key[1]))

        return token_dict, sent_dict, pos_sent_set

    @staticmethod
    def load_pre_dict(filename, word_dict=None, feature_dict_list=None):
        """
        :param filename:   ids file name
        :param word_dict:   Dictionary
        :param feature_dict_list: List[Dictionary]
            Default:
                1: 'entity',
                2: 'pos',
                3: 'lemma',
                4: 'deprole',
                5: 'dephead',
        """
        if word_dict is None:
            word_dict = get_default_dict()
        if feature_dict_list is None:
            feature_dict_list = [get_default_dict(lower=False) for _ in range(len(FEATURE_INDEX_MAP) - 1)]

        with codecs.open(filename, 'r', 'utf8') as fin:
            lines = fin.read().split('\n')
            # for line in fin:
            for line_no, line in enumerate(lines):
                if len(line) == 0:
                    continue
                line_no += 1
                att = line.strip().split('\t')
                if len(att) != TOKEN_ATT_NUM:
                    sys.stderr.write("[WARNING] Line %s: %s\n" % (line_no, line.split()))
                    continue

                token = att_to_token(att)

                word_dict.add(token['token'])

                for feature_index in range(len(FEATURE_INDEX_MAP) - 1):
                    feature_dict_list[feature_index].add(token[INDEX_FEATURE_MAP[feature_index + 1]])
        return word_dict, feature_dict_list

    @staticmethod
    def load_label_dictionary(label2id_file, label_dict=None):
        if label_dict is None:
            label_dict = get_default_dict(special_symbol=False, lower=False)

        with codecs.open(label2id_file, 'r', 'utf8') as fin:
            for line in fin:
                label, index = line.strip().split()
                label_dict.add_special(key=label, idx=int(index))

        return label_dict

    @staticmethod
    def load_data_file(ids_file, label_dict, word_dict, feature_dict_list,
                       trigger_label_file=None,
                       min_length=2, max_length=300, pos_filter=False,
                       remove_stop_word=False):

        if trigger_label_file:
            # Label Multi-Token
            trigger_label_dict = load_trigger_dat(trigger_label_file)
            label_using_trigger_data = True
            print("Label load from %s ..." % trigger_label_file)
        else:
            trigger_label_dict = dict()
            label_using_trigger_data = False

        pos_data = list()
        neg_data = list()

        sentence_data = dict()
        ids_data, sent_dict, posi_sent_set = EventDetectionCorpus.load_ids_file(ids_file)

        escape_count = 0
        sentence_count = 0

        unk_num = 0
        all_num = 0
        pos_num = 0

        for sent_key in sent_dict:

            sentence_count += 1
            docid, sentid = sent_key

            sent_length = len(sent_dict[sent_key])

            if sent_length > max_length or sent_length < min_length:
                escape_count += 1
                continue

            valid_token_position = list()
            if remove_stop_word:
                for tokenid in range(sent_length):
                    token_str = ids_data[(docid, sentid, tokenid)]['token']
                    if token_str not in english_stopwords:
                        valid_token_position += [tokenid]
            else:
                valid_token_position = range(sent_length)

            # Count Sentence Word
            sentence_token_str = [ids_data[(docid, sentid, tokenid)]['token'] for tokenid in valid_token_position]
            unk_num += sentence_token_str.count(Constants.UNK_WORD)
            all_num += len(sentence_token_str)
            sentence_token = convert_to_long_tensor(
                word_dict.convert_to_index(sentence_token_str, unk_word=Constants.UNK_WORD))

            sentence_data[sent_key] = [torch.stack([sentence_token], 1), sentence_token_str]

            for token_position, token_id in enumerate(valid_token_position):
                candidate = ids_data[(docid, sentid, token_id)]
                if candidate['token'] in string.punctuation:
                    continue
                if pos_filter and candidate['pos_tag'][0] not in pos_filter_set:
                    continue

                ident = (docid, sentid, token_id)
                label_id = torch.zeros(label_dict.size()).long()
                if label_using_trigger_data:
                    # Label Multi-Token
                    # Label using Trigger Data
                    # label: 1 -> [0, 1, 0, ..., label size]
                    # other label is all zero vector
                    if ident in trigger_label_dict:
                        label = trigger_label_dict[ident]
                    else:
                        label = OTHER_NAME
                    if label == OTHER_NAME:
                        _data = [token_position, label_id, sent_length, ident]
                        neg_data.append(_data)
                    else:
                        for l in label:
                            label_id[label_dict.lookup(l, default=0)] = 1
                        _data = [token_position, label_id, sent_length, ident]
                        pos_num += 1
                        pos_data.append(_data)
                else:
                    # Label from ids data
                    # label: 1 -> [0, 1, 0, ..., label size]
                    # label: 0 -> [1, 0, 0, ..., label size]
                    if candidate['tri_type'] == OTHER_NAME:
                        label_id[label_dict.lookup(OTHER_NAME, default=0)] = 1
                        _data = [token_position, label_id, sent_length, ident]
                        neg_data.append(_data)
                    else:
                        for label in candidate['tri_type'].split(';'):
                            label_id[label_dict.lookup(label, default=0)] = 1
                        _data = [token_position, label_id, sent_length, ident]
                        pos_num += 1
                        pos_data.append(_data)

        print("Word UNK/ALL: %s/%s = %.2f%%, positive: %d" % (unk_num, all_num, float(unk_num) / all_num * 100.,
                                                              pos_num))
        print("Pos: %d, Neg: %d, Load Sentence: %s, Escape: %d, Pos Sent %d." % (len(pos_data), len(neg_data),
                                                                                 sentence_count, escape_count,
                                                                                 len(posi_sent_set)))

        return pos_data, neg_data, ids_data, sentence_data

    def sample_data(self):
        if self.neg_ratio > 0:
            sample_size = int(min(self.event_data_size * self.neg_ratio, self.nonevent_data_size))
            neg_data = self.random.sample(self.non_event_data, sample_size)
            print("Sample Pos: %d, Neg: %d" % (len(self.event_data), len(neg_data)))
            return self.event_data + neg_data
        else:
            return self.event_data + self.non_event_data

    def batchify(self, sent_data, data, volatile=False):
        # data list [token_position, label_id, length, ident]
        # lengths = convert_to_long_tensor([d[2] for d in data])
        # _, sort_index = torch.sort(lengths, dim=0, descending=True)
        # data = [data[i] for i in sort_index]
        position, label, _, ids = zip(*data)

        position = list(position)

        # Cut Sentence with Trigger Windows
        raw_features_with_text = [sent_data[iden[:2]] for iden in ids]

        raw_features, raw_text = zip(*raw_features_with_text)

        features, texts, lengths = list(), list(), list()
        context_ids = list()
        for i in range(len(raw_features)):
            safe_start = max(0, position[i] - self.trigger_window)
            safe_end = min(raw_features[i].size(0), position[i] + self.trigger_window)
            features.append(raw_features[i][safe_start:safe_end, :])
            position[i] = position[i] - safe_start
            texts.append(raw_text[i][safe_start:safe_end])
            lengths.append(features[-1].size(0))
            context_ids.append([(ids[i][0], ids[i][1], token_i) for token_i in range(safe_start, safe_end)])

        lengths = convert_to_long_tensor(lengths)
        lengths, sort_index = torch.sort(lengths, dim=0, descending=True)
        data = [data[i] for i in sort_index]
        _, label, _, ids = zip(*data)

        features = [features[i] for i in sort_index]
        texts = [texts[i] for i in sort_index]
        position = [position[i] for i in sort_index]
        context_ids = [context_ids[i] for i in sort_index]

        classifier_feature_list = list()

        # Load ELMO Feature
        if self.elmo_mapper:
            classifier_feature_elmo = self.elmo_mapper.convert_to_feature(ids)
            seq_feature_elmo = self.elmo_mapper.convert_to_seq_feature(context_ids, lengths=lengths.tolist())
            classifier_feature_list += [classifier_feature_elmo]
        else:
            classifier_feature_elmo = None
            seq_feature_elmo = None

        if classifier_feature_list:
            classifier_feature = torch.cat(classifier_feature_list, 1)
            classifier_feature = to_variable(classifier_feature, volatile=volatile, device=self.device)
        else:
            classifier_feature = None

        features = align_batch(features, default_pad=Constants.PAD)

        features = to_variable(features, volatile=volatile, device=self.device)
        # label = to_variable(convert_to_long_tensor(label), volatile=volatile, device=self.device)
        label = to_variable(torch.stack(label, 0), volatile=volatile, device=self.device)
        position = to_variable(convert_to_long_tensor(position), volatile=volatile, device=self.device)
        lengths = to_variable(lengths, volatile=volatile, device=self.device)

        return EventDetectBatch(feature=(features, seq_feature_elmo, classifier_feature, texts),
                                position=position,
                                label=label,
                                length=lengths,
                                iden=ids,
                                pred=None,
                                )

    def next_batch(self, batch_size):
        if self.neg_ratio == 0:
            # evaluate
            data = self.event_data + self.non_event_data
            num_batch = int(math.ceil((len(data) / float(batch_size))))
            random_indexes = range(num_batch)
        else:
            if not self.fix_neg:
                self.data = self.sample_data()
            num_batch = int(math.ceil(len(self.data) / float(batch_size)))

            data = [self.data[index] for index in torch.randperm(len(self.data))]

            random_indexes = torch.randperm(num_batch)

        for index, i in enumerate(random_indexes):
            start, end = i * batch_size, (i + 1) * batch_size
            batch = self.batchify(self.sent_data, data[start:end])

            if self.word_dropout > 0:
                drop_position_rate = (batch.feature[0][:, :, 0] != Constants.PAD).float() * self.word_dropout
                drop_position = torch.bernoulli(drop_position_rate).byte().unsqueeze(-1).expand_as(batch.feature[0])
                batch.feature[0].masked_fill_(drop_position, Constants.UNK)

            yield batch

    def next_val_batch(self, batch_size):
        # evaluate
        data = self.event_data + self.non_event_data
        num_batch = int(math.ceil((len(data) / float(batch_size))))
        for i in range(num_batch):
            start, end = i * batch_size, (i + 1) * batch_size
            batch = self.batchify(self.sent_data, data[start:end], volatile=True)
            yield batch

    def next_predict_batch(self, data, sent_data, batch_size):
        num_batch = int(math.ceil((len(data) / float(batch_size))))
        for i in range(num_batch):
            start, end = i * batch_size, (i + 1) * batch_size
            batch = self.batchify(sent_data, data[start:end], volatile=True)
            yield batch

    def batch2pred(self, batch):
        pred_list = list()

        assert batch.pred is not None
        for iden, pred in zip(batch.ident, batch.pred):
            doc_id, sent_id, token_id = iden
            ids_dict = self.ids_data[iden]
            for pred_label in pred:
                pred_list += [(doc_id, ids_dict['start'], ids_dict['length'], ids_dict['token'],
                               self.label_dictionary.index2word[pred_label]
                               )]
        return pred_list
