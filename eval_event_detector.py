#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger

from base.event_detector import EventDetector
from base.options import *
from base.train_watcher import TrainWatcher, output_predict_to_file
from base.utils.utils import pre_data, pre_env_option, pre_env, get_feature_dict_list, init_param


def main():
    parser = pre_env_option()
    add_cnn_option(parser)
    add_rnn_option(parser)
    parser.add_argument('-predict-prefix', dest='predict_prefix')
    args = pre_env(parser)
    print(args)

    train_data, dev_data, test_data = pre_data(args, eval_mode=True)

    event_detector = EventDetector(args=args,
                                   word_dict=train_data.word_dictionary,
                                   label_dict=train_data.label_dictionary,
                                   feature_dict_list=get_feature_dict_list(train_data, args))
    init_param(args, event_detector)
    watcher = TrainWatcher(model=event_detector, train_data=train_data, dev_data=dev_data,
                           test_data=test_data, args=args)

    (span_p, span_r, span_f1, type_p, type_r, type_f1), output_list = watcher.eval_model(event_detector,
                                                                                         data=test_data,
                                                                                         verbose=True)
    print("Test Span P: %.2f, R: %.2f, F1: %.2f" % (span_p, span_r, span_f1))
    print("Test Type P: %.2f, R: %.2f, F1: %.2f" % (type_p, type_r, type_f1))
    if args.predict_prefix is not None:
        output_name = args.predict_prefix + '.predict'
        output_predict_to_file(output_list, output_name)


if __name__ == "__main__":
    main()
