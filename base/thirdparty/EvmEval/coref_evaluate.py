#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger on 2018/1/30
from collections import defaultdict


def evaluate_evmeval(golden_mention_list, golden_coref_dict, pred_mention_list, predict_coref_dict, verbose=False):
    """
    ['#BeginOfDocument sample',
     'sys     sample  E1      2,8     murder  Conflict_Attack Actual',
     'sys     sample  E2      12,16   kill    Conflict_Attack Actual',
     '@Coreference    R1      E1,E2',
     '#EndOfDocument',]
    :param golden_mention_list: [(docid, event_mention_id, start, length, type), ...]
    :param pred_mention_list: [(docid, event_mention_id, start, length, type), ...]
    :param verbose
    :return:
    """
    import cmath
    from .EvmEvalScorer import event_scorer

    pred_mention_list = list(set(pred_mention_list))

    def to_result(event_mention_list, event_coref_dict):
        doc_dict = defaultdict(list)
        for docid, event_mention_id, start, length, type_name in event_mention_list:
            if type_name == 'other':
                continue
            doc_dict[docid] += ['sys\t%s\t%s\t%s,%s\ttoken\t%s\tActual' % (event_mention_id, docid,
                                                                       start, start + length,
                                                                       type_name)]
        for docid in event_coref_dict:
            event_coref_chain = event_coref_dict[docid]
            if len(event_coref_chain) == 1:
                continue
            for event_id in event_coref_chain:
                doc_dict[docid] += ['@Coreference\t%s\t%s' % (event_id, ','.join(event_coref_chain[event_id]))]

        result_list = list()

        for docid in doc_dict:
            temp_result = list()
            temp_result += ['#BeginOfDocument %s' % docid]
            temp_result += doc_dict[docid]
            temp_result += ['#EndOfDocument']
            result_list += temp_result
            # print('\n'.join(temp_result))

        return result_list

    golden_result = to_result(golden_mention_list, golden_coref_dict)
    pred_result = to_result(pred_mention_list, predict_coref_dict)

    _, _, result = event_scorer.score(golden_result, pred_result)

    print(result)

    # span_result = [0. if cmath.isnan(s) else s for s in span_result['macro']]
    # type_result = [0. if cmath.isnan(s) else s for s in type_result['macro']]

    return 0# span_result + type_result
