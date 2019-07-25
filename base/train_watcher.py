#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger
import codecs
import copy
import os

import numpy as np
import torch
from past.builtins import zip

from .evaluate import get_evaluate
from .utils.infer_utils import model_predict

SPAN_P = 0
SPAN_R = 1
SPAN_F = 2
TYPE_P = 3
TYPE_R = 4
TYPE_F = 5
EPOCH_INDEX = 6
BATCH_INDEX = 7


def mean_list(numbers):
    if len(numbers) == 0:
        return 0.
    return sum(numbers) / len(numbers)


def output_predict_to_file(output_list, output_filename):
    with codecs.open(output_filename, 'w', 'utf8') as output:
        for doc_id, start, length, token, tag in output_list:
            output.write("%s\t%s\t%s\t%s\t%s\n" % (doc_id, start, length, token, tag))


def get_best_log_report_str(epoch_index, batch_index, span_report, type_report, prefix, f1_score):
    if batch_index:
        log_str = "Best Batch %s F1: %s\n" % (prefix, f1_score)
    else:
        log_str = "Best Epoch %s F1: %s\n" % (prefix, f1_score)
    if batch_index:
        log_str += "Epoch %2d, Batch %s, Span %s\n" % (epoch_index, batch_index, span_report)
        log_str += "Epoch %2d, Batch %s, Type %s\n" % (epoch_index, batch_index, type_report)
    else:
        log_str += "Epoch %2d, Span %s\n" % (epoch_index, span_report)
        log_str += "Epoch %2d, Type %s\n" % (epoch_index, type_report)
    return log_str


class TrainWatcher(object):

    def __init__(self, model, train_data, dev_data, test_data, args, model_path=None, ):
        self.model = model
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.batch_index = 0
        self.epoch_index = 0
        self.eval_each_batch = args.eval_each_batch
        self.eval_batch_size = args.eval_batch

        self.evaluate = get_evaluate(args.evaluate)

        # loss, epoch, batch
        self.train_log_loss = list()
        # dev span p, dev span r, dev span f, dev type p, dev type r, dev type f, epoch, batch
        self.dev_batch_result = list()
        # test span p, test span r, test span f, test type p, test type r, test type f, epoch, batch
        self.test_batch_result = list()
        # dev span p, dev span r, dev span f, dev type p, dev type r, dev type f, epoch, batch
        self.dev_epoch_result = list()
        # test span p, test span r, test span f, test type p, test type r, test type f, epoch, batch
        self.test_epoch_result = list()

        self.best_epoch_dev_f = 0.
        self.best_batch_dev_f = 0.
        self.save_threshold = args.save_threshold
        self.curr_score = 0.

        self.model_path = model_path

        self.args = args

        self.save_init_env()

    def save_init_env(self):
        """
        Save Init Config and Data
        :return:
        """
        if self.model_path is not None:
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)

            train_data_path = os.path.join(self.model_path, "train.data")
            model_conf_path = os.path.join(self.model_path, "model.config")

            temp_train_data = copy.deepcopy(self.train_data)
            temp_train_data.clear_data()
            torch.save(temp_train_data, train_data_path)

            with codecs.open(model_conf_path, 'w', 'utf8') as output:
                output.write("%s\n\n%s" % (self.args, self.model))

    @staticmethod
    def save_result(filename, result):

        def to_str(numbers):
            return ["%.4f" % number for number in numbers]

        with open(filename, 'w') as output:
            for line in result:
                output.write("%s,%s,%s\n" % (','.join(to_str(line[:-2])), line[-2], line[-1]))

    def finish_batch(self, log_loss):
        self.batch_index += 1
        self.train_log_loss.append(log_loss + [self.epoch_index, self.batch_index])

        if self.eval_each_batch <= 0:
            return

        if self.batch_index % self.eval_each_batch == 0:

            # span_p, span_r, span_f, type_p, type_r, type_f
            dev_result, dev_pre_log_list = self.eval_model(self.model, self.dev_data)
            test_result, test_pre_log_list = self.eval_model(self.model, self.test_data)

            span_report = self.result2string(dev_result=dev_result[:3], test_result=test_result[:3])
            type_report = self.result2string(dev_result=dev_result[3:], test_result=test_result[3:])

            self.dev_batch_result.append(list(dev_result) + [self.epoch_index, self.batch_index])
            self.test_batch_result.append(list(test_result) + [self.epoch_index, self.batch_index])

            dev_type_f = dev_result[TYPE_F]

            self.curr_score = dev_type_f

            if dev_type_f > self.best_batch_dev_f:
                self.best_batch_dev_f = dev_type_f

                if self.model_path is not None:
                    log_str = get_best_log_report_str(epoch_index=self.epoch_index,
                                                      batch_index=self.batch_index,
                                                      span_report=span_report,
                                                      type_report=type_report,
                                                      prefix='Dev',
                                                      f1_score=dev_type_f)
                    self.save_model(model_path=os.path.join(self.model_path, 'best_dev_f1_batch'),
                                    log_str=log_str,
                                    dev_pred_log_list=dev_pre_log_list,
                                    test_pred_log_list=test_pre_log_list)

    @staticmethod
    def mid_result2string(dev_result, test_result, train_result=None):
        if train_result:
            train_str = 'Train: %2.2f %2.2f %2.2f %2.2f' % (train_result[2], train_result[3],
                                                            train_result[4], train_result[5])
        else:
            train_str = 'Train: --.-- --.-- --.-- --.--'
        dev_str = 'Dev: %2.2f %2.2f %2.2f %2.2f' % (dev_result[2], dev_result[3],
                                                    dev_result[4], dev_result[5])
        test_str = 'Tst: %2.2f %2.2f %2.2f %2.2f' % (test_result[2], test_result[3],
                                                     test_result[4], test_result[5])
        return ' '.join([train_str, dev_str, test_str])

    def finish_epoch(self, time_used=None):

        # span_p, span_r, span_f, type_p, type_r, type_f
        train_result, _ = self.eval_model(self.model, self.train_data, verbose=False)
        dev_result, dev_pre_log_list = self.eval_model(self.model, self.dev_data, verbose=False)
        test_result, test_pre_log_list = self.eval_model(self.model, self.test_data, verbose=False)

        dev_type_f = dev_result[TYPE_F]
        self.curr_score = dev_type_f

        epoch_log_loss = filter(lambda b: b[-2] == self.epoch_index, self.train_log_loss)
        loss_list = zip(*epoch_log_loss)
        log_loss = ' '.join(["%.4f" % mean_list(loss) for loss in loss_list[:-2]])

        span_report = self.result2string(dev_result=dev_result[:3], test_result=test_result[:3])
        type_report = self.result2string(dev_result=dev_result[3:], test_result=test_result[3:])
        result_str = self.mid_result2string(dev_result, test_result, train_result)

        self.dev_epoch_result.append(list(dev_result) + [self.epoch_index, self.batch_index])
        self.test_epoch_result.append(list(test_result) + [self.epoch_index, self.batch_index])

        train_detail = "Epoch %2d, Loss: %s, Time: %.2fs" % (self.epoch_index, log_loss, time_used)

        dev_batch_result = list(filter(lambda b: b[-2] == self.epoch_index, self.dev_batch_result))
        test_batch_result = list(filter(lambda b: b[-2] == self.epoch_index, self.test_batch_result))
        assert len(dev_batch_result) == len(test_batch_result)
        if len(dev_batch_result) > 0:
            dev_batch_result = np.array(dev_batch_result)
            test_batch_result = np.array(test_batch_result)
            best_dev_batch_index = np.argmax(dev_batch_result[:, TYPE_F])

            best_dev_batch_result = list(dev_batch_result[best_dev_batch_index])
            best_test_batch_result = list(test_batch_result[best_dev_batch_index])

            b_result_str = self.mid_result2string(best_dev_batch_result, best_test_batch_result)

            print("%s, %s" % (train_detail, b_result_str))

        print("%s, %s" % (train_detail, result_str))

        if dev_type_f > self.best_epoch_dev_f:
            self.best_epoch_dev_f = dev_type_f

            if self.model_path is not None:
                log_str = get_best_log_report_str(epoch_index=self.epoch_index,
                                                  batch_index=None,
                                                  span_report=span_report,
                                                  type_report=type_report,
                                                  prefix='Dev',
                                                  f1_score=dev_type_f)
                self.save_model(model_path=self.model_path + os.sep + 'best_dev_f1_epoch',
                                log_str=log_str,
                                dev_pred_log_list=dev_pre_log_list,
                                test_pred_log_list=test_pre_log_list)

        self.epoch_index += 1
        return dev_result

    def get_best_iter_log_str(self, iter_type='epoch'):
        if iter_type == 'epoch':
            dev_iter_result = np.array(self.dev_epoch_result)
            test_iter_result = np.array(self.test_epoch_result)
        elif iter_type == 'batch':
            dev_iter_result = np.array(self.dev_batch_result)
            test_iter_result = np.array(self.test_batch_result)
        else:
            raise NotImplementedError

        log_str = ""
        best_dev_iter_span_index = np.argmax(dev_iter_result[:, SPAN_F])
        best_dev_iter_type_index = np.argmax(dev_iter_result[:, TYPE_F])

        temp_str = self.result2string(list(dev_iter_result[best_dev_iter_span_index, :3]),
                                      list(test_iter_result[best_dev_iter_span_index, :3]))
        log_str += "Best Dev  Span Iter %d, Batch %d, %s\n" % (dev_iter_result[best_dev_iter_span_index, -2],
                                                               dev_iter_result[best_dev_iter_span_index, -1],
                                                               temp_str)

        temp_str = self.result2string(list(dev_iter_result[best_dev_iter_type_index, 3:6]),
                                      list(test_iter_result[best_dev_iter_type_index, 3:6]))
        log_str += "Best Dev  Type Iter %d, Batch %d, %s\n" % (dev_iter_result[best_dev_iter_type_index, -2],
                                                               dev_iter_result[best_dev_iter_type_index, -1],
                                                               temp_str)

        return log_str

    def get_best_batch_log_str(self):
        return self.get_best_iter_log_str('batch')

    def get_best_epoch_log_str(self):
        return self.get_best_iter_log_str('epoch')

    def finish_train(self):
        print("Train Done!")

        if self.model_path is not None:
            log_str = ""
            if self.eval_each_batch > 0 and len(self.dev_batch_result) > 0:
                log_str += self.get_best_batch_log_str()
            log_str += self.get_best_epoch_log_str()
            with open(self.model_path + os.sep + 'result.log', 'w') as output:
                output.write(log_str)

            self.save_result(self.model_path + os.sep + "train.loss.csv", self.train_log_loss)
            self.save_result(self.model_path + os.sep + "dev.batch.result.csv", self.dev_batch_result)
            self.save_result(self.model_path + os.sep + "test.batch.result.csv", self.test_batch_result)
            self.save_result(self.model_path + os.sep + "dev.epoch.result.csv", self.dev_epoch_result)
            self.save_result(self.model_path + os.sep + "test.epoch.result.csv", self.test_epoch_result)
            print("Model Path: %s" % self.model_path)

    def eval_model(self, model, data, verbose=False):
        """
        :param model:
        :param data:
        :param verbose:
        :return:
            type_p, type_r, type_f, span_p, span_r, span_f
        """
        predict_list, output_list = model_predict(model, data, self.eval_batch_size)
        span_p, span_r, span_f1, type_p, type_r, type_f1 = self.evaluate(data.gold_list, predict_list, verbose=verbose)
        return (span_p, span_r, span_f1, type_p, type_r, type_f1), output_list

    @staticmethod
    def result2string(dev_result, test_result):
        dev_str = "Dev P: %6.2f, R: %6.2f, F1: %6.2f" % (dev_result[0], dev_result[1], dev_result[2])
        tst_str = "Tst P: %6.2f, R: %6.2f, F1: %6.2f" % (test_result[0], test_result[1], test_result[2])
        return dev_str + " | " + tst_str

    def save_model(self,
                   model_path,
                   log_str,
                   train_pred_log_list=None,
                   dev_pred_log_list=None,
                   test_pred_log_list=None,
                   ):
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        with open(model_path + os.sep + 'event_extractor.log', 'w') as output:
            output.write(log_str)

        torch.save(self.model, model_path + os.sep + 'event_extractor.model')

        if self.curr_score > self.save_threshold:

            if train_pred_log_list is not None:
                output_predict_to_file(train_pred_log_list, model_path + os.sep + 'train_pre_result.log')
            if dev_pred_log_list is not None:
                output_predict_to_file(dev_pred_log_list, model_path + os.sep + 'dev_pre_result.log')
            if test_pred_log_list is not None:
                output_predict_to_file(test_pred_log_list, model_path + os.sep + 'test_pre_result.log')


def eval_enhance_adv(event_detector, data, watcher, prefix=''):
    event_detector.eval()
    torch.set_grad_enabled(False)
    enhance_gold_list = list()
    enhance_pred_list = list()
    adv_gold_list = list()
    adv_pred_list = list()
    for batch in data.next_batch(watcher.args.batch):
        enhance_gold, enhance_pred, adv_gold, adv_pred = event_detector.eval_enhance_adv(feature=batch.feature,
                                                                                         positions=batch.position,
                                                                                         lengths=batch.length)
        enhance_gold_list += enhance_gold
        enhance_pred_list += enhance_pred
        adv_gold_list += adv_gold
        adv_pred_list += adv_pred

    import sklearn.metrics as sm

    def pred_to_label(x):
        return (x >= 0.5).tolist()

    def gold_to_label(x):
        return x.int().tolist()

    enhance_gold_list = gold_to_label(torch.cat(enhance_gold_list, 0))
    enhance_pred_list = pred_to_label(torch.cat(enhance_pred_list, 0).squeeze(1))

    adv_gold_list = gold_to_label(torch.cat(adv_gold_list, 0))
    adv_pred_list = pred_to_label(torch.cat(adv_pred_list, 0).squeeze(1))

    print('%s Enh Acc: %.2f F1:%.2f' % (prefix, sm.accuracy_score(enhance_gold_list, enhance_pred_list) * 100,
                                        sm.f1_score(enhance_gold_list, enhance_pred_list) * 100,
                                        ))
    print('%s Adv Acc: %.2f F1:%.2f' % (prefix, sm.accuracy_score(adv_gold_list, adv_pred_list) * 100,
                                        sm.f1_score(adv_gold_list, adv_pred_list) * 100,
                                        ))


def train_ed_model_epoch(event_detector, optimizer, train_data, watcher):
    from .pt4nlp import clip_weight_norm
    for d in train_data.next_batch(watcher.args.batch):
        event_detector.train()
        torch.set_grad_enabled(True)
        optimizer.zero_grad()

        log_loss, loss_list = event_detector.loss(feature=d.feature,
                                                  positions=d.position,
                                                  lengths=d.length,
                                                  label=d.label)

        log_loss.backward()

        optimizer.step()

        if optimizer.weight_clip > 0:
            clip_weight_norm(event_detector, optimizer.weight_clip,
                             row_norm_params=['event_detector.classifier.encoder.output.weight'])

        watcher.finish_batch(loss_list)


def train_ed_detector_model(event_detector, optimizer, train_data, num_epoch, watcher):
    import time
    if watcher.args.enhance_weight > 0 or watcher.args.adv_weight > 0:
        eval_enhance_adv(event_detector, train_data, watcher, prefix='Prefix')
    for iter_i in range(num_epoch):
        if iter_i < watcher.args.adv_pre_train:
            event_detector.pre_train_adv = True
        else:
            event_detector.pre_train_adv = False
        iter_start_time = time.time()
        train_ed_model_epoch(event_detector, optimizer, train_data, watcher)
        time_used = int(time.time() - iter_start_time)
        dev_result = watcher.finish_epoch(time_used)
        optimizer.step_lr(dev_result[-1])
        if watcher.args.enhance_weight > 0 or watcher.args.adv_weight > 0:
            eval_enhance_adv(event_detector, train_data, watcher, prefix='Epoch %s' % iter_i)
    watcher.finish_train()
