#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger
# VERSION v1.2
# V1.1
#   add load_trigger_bio_dat, FEATURE_INDEX_MAP, INDEX_FEATURE_MAP
# V1.2 2018-09-03
#   add load_coref, load_filler, load_event_mention_data, att_to_filler, att_to_event_mention
# V1.3 2018-09-10
#   add load_trigger_dat
# V1.4 2018-10-31
#   token level entity use first label

import codecs
import sys
from collections import defaultdict

POSTAG_INDEX = 0
NERTAG_INDEX = 1
TOKEN_ATT_NUM = 13
ENTITY_ATT_NUM = 6
ARG_ATT_NUM = 7
FEATURE_NUM = 2

FEATURE_INDEX_MAP = {
    'token': 0,
    'entity': 1,
    'pos_tag': 2,
    'lemma': 3,
    'deprole': 4,
    'dephead': 5,
}

INDEX_FEATURE_MAP = {
    0: 'token',
    1: 'entity',
    2: 'pos_tag',
    3: 'lemma',
    4: 'deprole',
    5: 'dephead',
}


def att_to_token(att):
    return {'docid': att[0],
            'sentid': int(att[1]),
            'tokenid': int(att[2]),
            'start': int(att[3]),
            'length': int(att[4]),
            'tri_type': att[5],
            'token': att[6],
            'pos_tag': att[7],
            'entity': att[8].split(';')[0],
            'stanford_ner': att[9],
            'dephead': int(att[10]),
            'deprole': att[11],
            'lemma': att[12], }


def att_to_entity(att):
    return {'docid': att[0],
            'entity_id': att[1],
            'entity_type': att[2],
            'tokens': eval(att[3]),
            'start': int(att[4]),
            'length': int(att[5]),
            }


def att_to_event_mention(att):
    return {'docid': att[0],
            'evm_id': att[1],
            'tri_type': att[2],
            'tokens': eval(att[3]),
            'start': int(att[4]),
            'length': int(att[5]),
            }


def att_to_filler(att):
    return {'docid': att[0],
            'filler_id': att[1],
            'filler_type': att[2],
            'nom_time': att[3],
            'filler_value': att[4],
            }


def load_ids_file(ids_filename):
    """
    :param ids_filename: IDS Data Filename -> str
    :return:
        dict: (docid, sentid, tokenid) -> Token
        dict: (docid, sentid) -> token list
    """
    # (docid, sentid, tokenid) -> Token
    token_dict = dict()
    multi_label_trigger_num, all_trigger_num = 0, 0
    # (docid, sentid) -> token list
    sent_list_dict = defaultdict(list)
    with codecs.open(ids_filename, 'r', 'utf8') as fin:
        line_no = 0
        for line in fin:
            line_no += 1
            att = line.strip().split('\t')
            if len(att) != TOKEN_ATT_NUM:
                sys.stderr.write("[WARNING] Line %s: %s\n" % (line_no, line.split()))
                continue

            # Get Token from att
            token = att_to_token(att)

            if token['tri_type'] != 'other':
                all_trigger_num += 1

            if ';' in token['tri_type']:
                # print(token)
                multi_label_trigger_num += 1

            # Add Token to token_dict
            key = (token['docid'], token['sentid'], token['tokenid'])
            token_dict[key] = token
            sent_list_dict[(token['docid'], token['sentid'])] += [token['tokenid']]
    for key in sent_list_dict:
        sent_list_dict[key].sort()

    print("Multi-Label Trigger %s/%s" % (multi_label_trigger_num, all_trigger_num))
    return token_dict, sent_list_dict


def load_entity_file(entity_filename):
    """
    :param entity_filename: Entity Data Filename -> str
    :return:
        doc_entity dict: DocId -> [EntityId1, EntityId2, ...]
        token_entity dict: (docid, sentid, tokenid) -> [EntityId1, EntityId2, ...]
        entity_dict  dict: EntityId -> Entity dict{}
    """
    # DocId -> [EntityId1, EntityId2, ...]
    doc_entity = defaultdict(list)
    # Token -> [EntityId1, EntityId2, ...]
    token_entity = defaultdict(list)
    # EntityId -> Entity
    # EntityId -> Entity dict{}
    entity_dict = dict()

    none_token_num = 0

    with open(entity_filename, 'r') as fin:
        line_no = 0
        for line in fin:
            line_no += 1
            att = line.strip().split('\t')
            if len(att) != ENTITY_ATT_NUM:
                sys.stderr.write("[WARNING] Line %s: %s\n" % (line_no, line.split()))
                continue

            # Get Entity from att
            entity = att_to_entity(att)

            if not entity['tokens']:
                none_token_num += 1
                # sys.stderr.write('%s\n' % str(entity))
                continue

            # Add Entity to entity_dict
            entity_dict[entity['entity_id']] = entity

            # Add Entity to doc_entity
            doc_entity[entity['docid']] += [entity['entity_id']]

            # Add Entity to token_entity
            for sentid, tokenid in entity['tokens']:
                key = (entity['docid'], sentid, tokenid)
                token_entity[key] += [entity['entity_id']]

    print("None Tokens Entity: %s" % none_token_num)
    return doc_entity, token_entity, entity_dict


def find_head_word(entity, token_dict):
    doc_id = entity['docid']

    if len(entity['tokens']) == 1:
        sent_id, token_id = entity['tokens'][0]
        return doc_id, sent_id, token_id

    for token in reversed(entity['tokens']):
        sent_id, token_id = token
        if token_dict[(doc_id, sent_id, token_id)]['pos_tag'][0] == 'N':
            return doc_id, sent_id, token_id

    sent_id, token_id = entity['tokens'][-1]

    return doc_id, sent_id, token_id


def check_head_word(entity_dict, token_dict):
    for entity_id in entity_dict:
        entity = entity_dict[entity_id]
        docid = entity['docid']
        entity_tokens = entity['tokens']
        if len(entity_tokens) > 1:
            tokens = [token_dict[(docid, key[0], key[1])]['token'] for key in entity_tokens]
            if entity['entity_type'].startswith('Time') or entity['entity_type'].startswith('Numeric'):
                continue
            head_word = find_head_word(entity, token_dict)
            print(' '.join(tokens), entity['entity_type'], token_dict[head_word]['token'])


def load_trigger_bio_dat(filename):
    """
    CNN_ENG_20030607_170312.6       CNN_ENG_20030607_170312.6-EV6-2 Justice:Charge-Indict   [(36, 11)]      2575    7
    :param filename:
    :return:
    label with bio
        (doc_id, sent_id, token_id) -> [label, ..., ]
    """
    token_label_dict = defaultdict(list)
    with open(filename) as fin:
        for line in fin:
            att = line.strip().split('\t')
            doc_id = att[0]
            label = att[2]
            token_ids = eval(att[3])
            for index, (sent_id, token_id) in enumerate(token_ids):
                suffix = '-B' if index == 0 else '-I'
                token_label_dict[(doc_id, sent_id, token_id)] += [label + suffix]
    return token_label_dict


def load_trigger_dat(filename):
    """
    CNN_ENG_20030607_170312.6       CNN_ENG_20030607_170312.6-EV6-2 Justice:Charge-Indict   [(36, 11)]      2575    7
    :param filename:
    :return:
    label without bio
        (doc_id, sent_id, token_id) -> [label, ..., ]
    """
    token_label_dict = defaultdict(list)
    with open(filename) as fin:
        for line in fin:
            att = line.strip().split('\t')
            doc_id = att[0]
            label = att[2]
            token_ids = eval(att[3])
            for index, (sent_id, token_id) in enumerate(token_ids):
                token_label_dict[(doc_id, sent_id, token_id)] += [label]
    return token_label_dict


def load_trigger_gold_list(filename):
    gold_list = list()
    with open(filename, 'r') as fin:
        for line in fin:
            att = line.strip().split('\t')
            gold_list += [[att[0], int(att[1]), int(att[2]), att[4]]]
    return gold_list


def split_pos_neg(sent_list_dict, token_dict, min_len=2, max_len=80):
    """
    Split Sentences and Tokens to Pos/Neg
    :param sent_list_dict:  (doc_id, sent_id) -> token list
    :param token_dict:      (doc_id, sent_id, token_id) -> Token
    :param min_len:
    :param max_len:
    :return:
    """
    pos_sentence_set, neg_sentence_set = set(), set()
    pos_token_set, neg_token_set = set(), set()
    for sent_id in sent_list_dict:
        event_sentence = False
        if len(sent_list_dict[sent_id]) > max_len or len(sent_list_dict[sent_id]) < min_len:
            continue
        for token_id in sent_list_dict[sent_id]:
            token = token_dict[(sent_id[0], sent_id[1], token_id)]
            if 'other' in token['tri_type']:
                neg_token_set.add((sent_id[0], sent_id[1], token_id))
            else:
                pos_token_set.add((sent_id[0], sent_id[1], token_id))
                event_sentence = True
        if event_sentence:
            pos_sentence_set.add(sent_id)
        else:
            neg_sentence_set.add(sent_id)
    return pos_sentence_set, neg_sentence_set, pos_token_set, neg_token_set


def get_candidate_entity_set(token_key, token_dict, token_entity, sent_window=0):
    """
    :param token_key:       (doc_id, sent_id, token_id)
    :param token_dict:      (doc_id, sent_id, token_id) -> Token
    :param token_entity:    (doc_id, sent_id, token_id) -> [EntityId1, EntityId2, ...]
    :param sent_window:     Sentence Window Size
    :return:
    """
    doc_id, sent_id, token_id = token_key
    entity_ids_set = set()
    for candidate_sent_id in range(sent_id - sent_window, sent_id + sent_window + 1):

        # Check Valid Sentence
        if (doc_id, candidate_sent_id, 0) not in token_dict:
            continue

        candidate_token_id = 0
        # Loop on Valid Token
        while (doc_id, candidate_sent_id, candidate_token_id) in token_dict:
            if (doc_id, candidate_sent_id, candidate_token_id) in token_entity:
                for entity in token_entity[(doc_id, candidate_sent_id, candidate_token_id)]:
                    entity_ids_set.add(entity)
            candidate_token_id += 1

    return entity_ids_set


def get_candidate_trigger_entity_pair(token_dict, token_entity, label_set=None, sent_window=0):
    """
    :param token_dict:      (doc_id, sent_id, token_id) -> Token
    :param token_entity:    (doc_id, sent_id, token_id) -> [EntityId1, EntityId2, ...]
    :param label_set:       Valid Label Set
    :param sent_window:     Sentence Window Size
    :return:
        [[token key, entity_id, tri_type, 'other']]
    """
    data_list = list()
    token_keys = sorted(token_dict.keys())
    for token_key in token_keys:
        token = token_dict[token_key]

        # Skip Non-Trigger
        if token['tri_type'] == 'other':
            continue

        # Handle Multi-Label Trigger
        for tri_type in token['tri_type'].split(';'):

            # Skip Label not Required
            if (label_set is not None) and (tri_type not in label_set):
                break

            candidate_entity_ids = get_candidate_entity_set(token_key, token_dict, token_entity, sent_window)

            for entity_id in candidate_entity_ids:
                data_list += [[token_key, entity_id, tri_type, 'other']]
    return data_list


def get_span(token_dict, start, end):
    """
    :param token_dict:  (doc_id, sent_id, token_id) -> Token
    :param start:       (doc_id, sent_id, token_id)
    :param end:         (doc_id, sent_id, token_id)
    :return:
    """
    assert start[0] == end[0]
    assert start[1] <= end[1]
    if start[1] == end[1]:
        assert start[2] <= end[2]
    start_token = token_dict[(start[0], start[1], start[2])]
    end_token = token_dict[(end[0], end[1], end[2])]
    span_start = start_token['start']
    span_len = end_token['start'] - start_token['start'] + end_token['length']
    return span_start, span_len


def get_span_text(token_dict, start, end):
    assert start[0] == end[0]
    assert start[1] <= end[1]
    if start[1] == end[1]:
        assert start[2] <= end[2]
    doc_id = start[0]
    token_list = list()
    for sentence_id in range(start[1], end[1] + 1):
        if sentence_id == start[1]:
            token_id = start[2]
            while (doc_id, sentence_id, token_id) in token_dict:
                if sentence_id >= end[1] and token_id > end[2]:
                    break
                token_list += [(doc_id, sentence_id, token_id)]
                token_id += 1

    sentence_text = ""
    for token_id, next_token_id in zip(token_list, token_list[1:] + [None]):
        curr_token = token_dict[token_id]
        next_token = token_dict[next_token_id] if next_token_id else None

        start = curr_token['start']
        end = start + curr_token['length']

        sentence_text += curr_token['token']
        if next_token is not None:
            next_start = next_token['start']
            sentence_text += ' ' * (next_start - end)
    return sentence_text


def get_entity_span(entity_dict, token_dict, entity_id):
    """
    :param entity_dict: EntityId -> Entity dict{}
    :param token_dict:  (doc_id, sent_id, token_id) -> Token
    :param entity_id:   EntityId
    :return:
    """
    entity = entity_dict[entity_id]
    start_token = (entity['docid'], entity['tokens'][0][0], entity['tokens'][0][1])
    end_token = (entity['docid'], entity['tokens'][-1][0], entity['tokens'][-1][1])
    return get_span(token_dict, start=start_token, end=end_token)


def get_entity_span_text(entity_dict, token_dict, entity_id):
    """
    :param entity_dict: EntityId -> Entity dict{}
    :param token_dict:  (doc_id, sent_id, token_id) -> Token
    :param entity_id:   EntityId
    :return:
    """
    entity = entity_dict[entity_id]
    if not entity['tokens']:
        return ""
    start_token = (entity['docid'], entity['tokens'][0][0], entity['tokens'][0][1])
    end_token = (entity['docid'], entity['tokens'][-1][0], entity['tokens'][-1][1])
    return get_span_text(token_dict, start=start_token, end=end_token)


def load_sentence_span(token_dict, sent_list_dict):
    sent_span_dict = dict()
    for key in sent_list_dict:
        token_list = sent_list_dict[key]
        start_token = key[0], key[1], token_list[0]
        end_token = key[0], key[1], token_list[-1]
        sentence_span_start = token_dict[start_token]['start']
        sentence_span_end = token_dict[end_token]['start'] + token_dict[end_token]['length'] - 1

        sentence_text = ""
        for token_id, next_token_id in zip(token_list, token_list[1:] + [None]):
            curr_token = token_dict[(key[0], key[1], token_id)]
            next_token = token_dict[(key[0], key[1], next_token_id)] if next_token_id is not None else None

            start = curr_token['start']
            end = start + curr_token['length']

            sentence_text += curr_token['token']
            if next_token is not None:
                next_start = next_token['start']
                sentence_text += ' ' * (next_start - end)
        sent_span_dict[key] = (sentence_span_start, sentence_span_end, sentence_text)
    return sent_span_dict


def load_coref(filename):
    entity_mention_dict = defaultdict(set)
    mention_entity_dict = dict()
    with codecs.open(filename, 'r', 'utf8') as fin:
        for line in fin:
            att = line.strip().split('\t')
            entity_id = att[1]
            for mention_id in att[2:]:
                entity_mention_dict[entity_id].add(mention_id)
                mention_entity_dict[mention_id] = entity_id
    return entity_mention_dict, mention_entity_dict


def load_filler(filename):
    filler_dict = dict()
    with codecs.open(filename, 'r', 'utf8') as fin:
        for line in fin:
            filler = att_to_filler(line.strip().split('\t'))
            filler_dict[filler['filler_id']] = filler
    return filler_dict


def load_event_mention_data(filename):
    evm_dict = dict()
    with codecs.open(filename, 'r', 'utf8') as fin:
        for line in fin:
            att = line.strip().split('\t')
            evm = att_to_event_mention(att)
            evm_dict[evm['evm_id']] = evm
    return evm_dict
