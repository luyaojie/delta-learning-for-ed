#!/usr/bin/env python
# -*- coding:utf-8 -*-
from argparse import ArgumentParser

import torch

from base.dataloader.data_structure import EventData
from base.dataloader.event_data_utils import FEATURE_INDEX_MAP
from base.dataloader.event_detection_corpus import EventDetectionCorpus
from base.options import add_env_option
from base.pt4nlp import Optimizer, load_model

negative_set = {'NEGATIVE', 'other'}


def mean_list(numbers):
    if len(numbers) == 0:
        return 0.
    return sum(numbers) / len(numbers)


def result2string(result_iter):
    dev_str = "Dev P: %6.2f, R: %6.2f, F1: %6.2f" % (result_iter[0], result_iter[1], result_iter[2])
    tst_str = "Tst P: %6.2f, R: %6.2f, F1: %6.2f" % (result_iter[3], result_iter[4], result_iter[5])
    return dev_str + " | " + tst_str


def get_model_name(args):
    model_name_list = []
    if args.prefix is not None:
        model_name_list += [args.prefix]
    model_name_list += ["%s" % args.model]
    model_name_list += ["neg_%s" % args.neg_ratio]
    model_name_list += ["batch_%s" % args.batch]
    # model_name_list += ["opt_%s" % args.optimizer]
    # model_name_list += ["lr_%s" % args.lr]
    if args.fix_neg:
        model_name_list += ["fixneg"]
    if args.suffix is not None:
        model_name_list += [args.suffix]
    return '_'.join(model_name_list)


def pre_data(args, eval_mode=False):
    train_event_data = EventData(folder_name="%s/train" % args.data,
                                 prefix_name="train")
    dev_event_data = EventData(folder_name="%s/dev" % args.data,
                               prefix_name="dev")
    test_event_data = EventData(folder_name="%s/test" % args.data,
                                prefix_name="test")

    word_d, feature_dict_list = EventDetectionCorpus.load_pre_dict(train_event_data.ids_dat)
    label_d = EventDetectionCorpus.load_label_dictionary("%s/label2id.dat" % args.data)

    word_d, feature_dict_list = EventDetectionCorpus.load_pre_dict(dev_event_data.ids_dat,
                                                                   word_dict=word_d,
                                                                   feature_dict_list=feature_dict_list)
    word_d, feature_dict_list = EventDetectionCorpus.load_pre_dict(test_event_data.ids_dat,
                                                                   word_dict=word_d,
                                                                   feature_dict_list=feature_dict_list)

    use_elmo = True if args.adv_target == 'elmo' else False
    if eval_mode:
        train_data = EventDetectionCorpus(event_data=train_event_data,
                                          word_dictionary=word_d,
                                          label_dictionary=label_d,
                                          feature_dictionary_list=feature_dict_list,
                                          neg_ratio=args.neg_ratio,
                                          device=args.device,
                                          max_length=args.max_length,
                                          fix_neg=args.fix_neg,
                                          trigger_window=args.trigger_window,
                                          word_dropout=args.word_drop,
                                          pos_filter=args.pos_filter,
                                          elmo=False,  # Eval with out Train Dataset's ELMo
                                          label_multi_token=args.label_multi_token,
                                          )
    else:
        train_data = EventDetectionCorpus(event_data=train_event_data,
                                          word_dictionary=word_d,
                                          label_dictionary=label_d,
                                          feature_dictionary_list=feature_dict_list,
                                          neg_ratio=args.neg_ratio,
                                          device=args.device,
                                          max_length=args.max_length,
                                          fix_neg=args.fix_neg,
                                          trigger_window=args.trigger_window,
                                          word_dropout=args.word_drop,
                                          pos_filter=args.pos_filter,
                                          elmo=use_elmo,
                                          label_multi_token=args.label_multi_token,
                                          )
    if eval_mode:
        dev_data = None
    else:
        dev_data = EventDetectionCorpus(event_data=dev_event_data,
                                        word_dictionary=word_d,
                                        label_dictionary=label_d,
                                        feature_dictionary_list=feature_dict_list,
                                        neg_ratio=-1,
                                        device=args.device,
                                        # max_length=args.max_length,
                                        random=False,
                                        neg_sample_seed=0,
                                        trigger_window=args.trigger_window,
                                        word_dropout=0,
                                        pos_filter=args.pos_filter,
                                        elmo=use_elmo,
                                        label_multi_token=args.label_multi_token,
                                        )

    test_data = EventDetectionCorpus(event_data=test_event_data,
                                     word_dictionary=word_d,
                                     label_dictionary=label_d,
                                     feature_dictionary_list=feature_dict_list,
                                     neg_ratio=-1,
                                     device=args.device,
                                     # max_length=args.max_length,
                                     random=False,
                                     neg_sample_seed=0,
                                     trigger_window=args.trigger_window,
                                     word_dropout=0,
                                     pos_filter=args.pos_filter,
                                     elmo=use_elmo,
                                     label_multi_token=args.label_multi_token,
                                     )

    return train_data, dev_data, test_data


def pre_env_option(parser=None):
    if not parser:
        parser = ArgumentParser()
    add_env_option(parser)
    Optimizer.add_optimizer_options(parser, default_optim='Adadelta')
    return parser


def pre_env(parser):
    args = parser.parse_args()

    if args.device >= 0:
        torch.cuda.set_device(args.device)

    return args


def pre_model_path(args):
    if args.save:
        model_path = "%s/%s" % (args.model_folder, get_model_name(args))
    else:
        model_path = None

    return model_path


def init_param(args, event_detector):
    if args.pre_emb is not None and args.pre_emb != 'random':
        event_detector.lexi_spec_embedding_layer.embeddings.load_pretrained_vectors(args.pre_emb)
        event_detector.lexi_free_embedding_layer.embeddings.load_pretrained_vectors(args.pre_emb)
        event_detector.lexi_embedding_layer.embeddings.load_pretrained_vectors(args.pre_emb)

    if args.device >= 0:
        event_detector.cuda(args.device)

    print("Event Detector")
    print(event_detector)

    to_update_param = list()
    for name_param in event_detector.named_parameters():
        name, param = name_param
        if 'bias' in name:
            param.data.zero_()
        to_update_param += [param]

    if args.pre_train_model:
        pre_train_model = load_model(args.pre_train_model, device=args.device)
        event_detector.load_pre_train_model(pre_train_model,
                                            embedding=True, encoder=True, fusion_layer=True, classifier=True)

    if args.load_spec_model:
        pre_train_model = load_model(args.load_spec_model, device=args.device)
        event_detector.load_specific_model(pre_train_model)

    if args.load_free_model:
        pre_train_model = load_model(args.load_free_model, device=args.device)
        event_detector.load_free_model(pre_train_model)

    optimizer = Optimizer.get_optimizer(args, to_update_param, mode='max')
    print("Optimizer: ", optimizer)
    return optimizer


def get_feature_dict_list(train_data, args):
    feature_dict_list = list()
    if args.entity:
        feature_dict_list += [train_data.feature_dictionary_list[FEATURE_INDEX_MAP['entity'] - 1]]
    if args.pos:
        feature_dict_list += [train_data.feature_dictionary_list[FEATURE_INDEX_MAP['pos_tag'] - 1]]
    if args.lemma:
        feature_dict_list += [train_data.feature_dictionary_list[FEATURE_INDEX_MAP['lemma'] - 1]]
    if args.deprole:
        feature_dict_list += [train_data.feature_dictionary_list[FEATURE_INDEX_MAP['deprole'] - 1]]
    return feature_dict_list


def get_feature_flag(args):
    feature_flag = [0]
    if args.entity:
        feature_flag += [FEATURE_INDEX_MAP['entity']]
    if args.pos:
        feature_flag += [FEATURE_INDEX_MAP['pos_tag']]
    if args.lemma:
        feature_flag += [FEATURE_INDEX_MAP['lemma']]
    if args.deprole:
        feature_flag += [FEATURE_INDEX_MAP['deprole']]
    return feature_flag
