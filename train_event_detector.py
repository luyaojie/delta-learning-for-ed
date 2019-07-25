#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger
from base.event_detector import EventDetector
from base.options import *
from base.train_watcher import TrainWatcher, train_ed_detector_model
from base.utils.utils import pre_model_path, pre_data, pre_env_option, init_param, get_feature_dict_list, pre_env

parser = pre_env_option()
add_cnn_option(parser)
add_rnn_option(parser)
args = pre_env(parser)
print(args)

model_path = pre_model_path(args)

train_data, dev_data, test_data = pre_data(args)

event_detector = EventDetector(args=args,
                               word_dict=train_data.word_dictionary,
                               label_dict=train_data.label_dictionary,
                               feature_dict_list=get_feature_dict_list(train_data, args))

optimizer = init_param(args, event_detector)

watcher = TrainWatcher(model=event_detector, train_data=train_data, dev_data=dev_data, test_data=test_data,
                       args=args, model_path=model_path)

print("Train Event Detector ...")
train_ed_detector_model(event_detector=event_detector, optimizer=optimizer,
                        train_data=train_data, num_epoch=args.epoch, watcher=watcher)
