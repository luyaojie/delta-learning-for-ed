#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger
import codecs
import os
import time
from collections import defaultdict

import numpy as np

from base.thirdparty.ACE.charseq import CharSeq
from base.thirdparty.ACE.document import Document
from base.thirdparty.ACE.event import Event
from base.thirdparty.ACE.event_mention import EventMention
from base.thirdparty.ACE.utils import read_ace_sgm_file
from base.utils.utils import negative_set


def get_evaluate(evaluate_method):
    if evaluate_method == 'ace' or evaluate_method == 'ace2005':
        return evaluate_ace2005
    elif evaluate_method == 'evmeval':
        return evaluate_evmeval
    else:
        raise NotImplementedError


def evaluate(golden_list, pred_list, trigger_type=True):
    return evaluate_ace2005(golden_list, pred_list, trigger_type)


def table_to_apf(pred_list, apf_folder, doc_id):
    ace_source_folder = os.environ["ACE_SOURCE_FOLDER"]
    source_text = read_ace_sgm_file(ace_source_folder + os.sep + doc_id + ".sgm")
    doc = Document(source_text, source_file=doc_id, source_type='mp', docid=doc_id)
    evid = 0
    for doc_key, start, length, pred_label in pred_list:
        assert doc_key == doc_id

        if pred_label in negative_set:
            continue

        evid += 1
        start = int(start)
        length = int(length)

        ll = list(pred_label)
        for i in range(len(pred_label)):
            if i == 0:
                ll[0] = pred_label[0].upper()
            if i - 1 >= 0:
                if pred_label[i - 1] in [':', '-']:
                    ll[i] = pred_label[i].upper()
        pred_label = ''.join(ll)

        safe_start = max(0, start)
        safe_end = min(start + length, len(source_text))
        trigger = source_text[start:start + length].replace('\n', ' ')
        context = source_text[safe_start:safe_end]
        context = context.replace('\n', ' ')

        trigger = CharSeq(start, end=start + length - 1, text=trigger, docid=doc_id)
        context = CharSeq(safe_start, end=safe_end - 1, text=context, docid=doc_id)

        event = Event(id=doc_id + 'EV' + str(evid), type="", subtype=pred_label.split(':')[1])
        evm = EventMention(id=doc_id + 'EVM' + str(evid), nugget=trigger, extent=context, event=event)

        event.mentions.append(evm)
        doc.events.append(event)
    apf_xml = doc.to_apf_document()
    with codecs.open(apf_folder + os.sep + doc_id + '.apf.xml', 'w', 'utf8') as output:
        output.write(apf_xml)


def evaluate_ace2005(golden_list, pred_list, verbose=False):
    """
    :param golden_list: [(docid, start, length, type), ...]
    :param pred_list: [(docid, start, length, type), ...]
    :param verbose
    :return:
    """

    pred_list = list(set(pred_list))

    pred_dict = defaultdict(list)
    gold_dict = defaultdict(list)

    golden_folder = '/tmp/aceeval_%s_%s_golden' % (os.getpid(), time.time())
    predict_folder = '/tmp/aceeval_%s_%s_predict' % (os.getpid(), time.time())
    file_list_filename = '/tmp/aceeval_%s_%s_filelist' % (os.getpid(), time.time())

    os.makedirs(golden_folder)
    os.makedirs(predict_folder)

    for docid, start, length, typename in pred_list:
        if typename == 'other':
            continue
        pred_dict[docid] += [(docid, start, length, typename)]

    for docid in pred_dict:
        pred_list = pred_dict[docid]
        table_to_apf(pred_list, predict_folder, docid)

    for docid, start, length, typename in golden_list:
        gold_dict[docid] += [(docid, start, length, typename)]

    for docid in gold_dict:
        pred_list = gold_dict[docid]
        table_to_apf(pred_list, golden_folder, docid)

    with open(file_list_filename, 'w') as output:
        for docid in gold_dict:
            output.write("%s\n" % docid)

            if docid not in pred_dict:
                table_to_apf([], predict_folder, docid)

    cmd_prefix = "java -cp base/thirdparty/joint_ere_release/ere.jar edu.rpi.jie.ace.acetypes.EventScorer"
    cp_cmd = "cp -r %s/* %s" % (os.environ["ACE_SOURCE_FOLDER"], golden_folder)
    cmd = "%s %s %s %s" % (cmd_prefix, golden_folder, predict_folder, file_list_filename)
    rm_cmd = "rm -r %s %s %s" % (golden_folder, predict_folder, file_list_filename)

    os.system(cp_cmd)
    result = os.popen(cmd, ).readlines()
    os.system(rm_cmd)

    type_result = result[5].split('\t')
    span_result = result[7].split('\t')
    results = [float(type_result[2]) * 100, float(type_result[4]) * 100, float(type_result[6]) * 100,
               float(span_result[2]) * 100, float(span_result[4]) * 100, float(span_result[6]) * 100]
    results = [0. if np.isnan(r) else r for r in results]
    type_f1, type_p, type_r = results[:3]
    span_f1, span_p, span_r = results[3:]
    return span_p, span_r, span_f1, type_p, type_r, type_f1


def evaluate_evmeval(golden_list, pred_list, verbose=False):
    """
    ['#BeginOfDocument sample',
     'sys     sample  E1      2,8     murder  Conflict_Attack Actual',
     'sys     sample  E2      12,16   kill    Conflict_Attack Actual',
     '@Coreference    R1      E1,E2',
     '#EndOfDocument',]
    :param golden_list: [(docid, start, length, type), ...]
    :param pred_list: [(docid, start, length, type), ...]
    :param verbose
    :return:
    """
    import cmath
    import os
    import time

    pred_list = list(set(pred_list))

    def to_dict(event_list):
        event_dict = defaultdict(list)
        for docid, start, length, type_name in event_list:
            if type_name == 'other':
                continue
            event_dict[(docid, start, length)].append(type_name)
        return event_dict

    golden_result = transform_to_score_list(to_dict(golden_list))
    pred_result = transform_to_score_list(to_dict(pred_list))

    golden_filename = '/tmp/evmeval_%s_%s.golden' % (os.getpid(), time.time())
    predict_filename = '/tmp/evmeval_%s_%s.predict' % (os.getpid(), time.time())
    with open(golden_filename, 'w') as output:
        output.write(''.join(golden_result))
    with open(predict_filename, 'w') as output:
        output.write(''.join(pred_result))

    cmd = "python2 base/thirdparty/EvmEval/scorer_v1.8.py -g %s -s %s" % (golden_filename, predict_filename)
    rm_cmd = "rm %s %s" % (golden_filename, predict_filename)

    result = os.popen(cmd, ).readlines()[-5:]
    os.system(rm_cmd)

    span_result = {'micro': [float(a) for a in result[0].split('\t')[1:4]],
                   'macro': [float(a) for a in result[0].split('\t')[4:7]],
                   }
    type_result = {'micro': [float(a) for a in result[1].split('\t')[1:4]],
                   'macro': [float(a) for a in result[1].split('\t')[4:7]],
                   }

    span_result = [0. if cmath.isnan(s) else s for s in span_result['micro']]
    type_result = [0. if cmath.isnan(s) else s for s in type_result['micro']]

    return span_result + type_result


def transform_to_score_list(golden_dict):
    from collections import defaultdict
    system_name = "KBP2017System"

    def att_to_line(_doc_id, _offset, _length, _label, _e_cnt):
        return '\t'.join([system_name,
                          str(_doc_id),
                          "EV" + str(_e_cnt),
                          str(_offset) + "," + str(_offset + _length),
                          "token",
                          _label,
                          "actual"]) + '\n'

    lines = []
    e_cnt = 0
    doc2data = defaultdict(list)
    for key in golden_dict:
        for label in golden_dict[key]:
            doc_id, offset, length = key
            doc2data[doc_id].append((doc_id, offset, length, label))

    for d in doc2data:
        lines.append("#BeginOfDocument " + d + "\n")
        for doc_id, offset, length, label in doc2data[d]:
            lines.append(att_to_line(doc_id, offset, length, label, e_cnt))
            e_cnt += 1
        lines.append("#EndOfDocument\n")

    return lines
