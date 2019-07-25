#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger on 2018/3/14
import codecs
from collections import defaultdict

from .charseq import CharSeq


def read_ace_sgm_file(filename, encoding='utf8'):
    import re
    ml_filter = re.compile('</?\w+[^>]*>')
    doc = codecs.open(filename, 'r', encoding=encoding).read()
    text = ml_filter.sub('', doc)
    return text


def label_entity_token(document, parsed_result, generalized=True):
    """
    :param document:        Document
    :param parsed_result:   Stanford Parsed
    :param generalized:     Contain value time?
    :return:
        (senid, tokenid) -> [(entity_mention, 'B'), (entity_mention, 'I'), ...]
        entity_mention -> [token1, token2, ...]ue
    """
    token_charseq_dict = dict()

    for senid, sentence in enumerate(parsed_result['sentences']):
        for tokenid, token in enumerate(sentence['tokens']):
            charseq = CharSeq.from_stanford_token(token, docid=document.docid)
            token_charseq_dict[(senid, tokenid)] = charseq

    em_token_dict = defaultdict(list)

    if generalized:
        candidate_entities = document.entities + document.values + document.time_expressions
    else:
        candidate_entities = document.entities

    for entity in candidate_entities:
        for entity_mention in entity.mentions:
            for token_iden, charseq in token_charseq_dict.iteritems():
                if charseq.partial_match(entity_mention.head):
                    em_token_dict[entity_mention].append(token_iden)
            em_token_dict[entity_mention].sort()

    token_em_dict = defaultdict(list)

    for entity_mention, tokens in em_token_dict.iteritems():
        if len(tokens) == 0:
            continue
        elif len(tokens) == 1:
            token_em_dict[tokens[0]].append((entity_mention, 'B'))
        else:
            token_em_dict[tokens[0]].append((entity_mention, 'B'))
            for token in tokens[1:]:
                token_em_dict[token].append((entity_mention, 'I'))

    return token_em_dict, em_token_dict
