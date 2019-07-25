#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Optimizer(object):

    def __init__(self, optimizer, grad_clip=-1, weight_clip=-1, lr_factor=1, lr_patience=10, mode='min'):
        self.optimizer = optimizer
        self.grad_clip = grad_clip
        self.weight_clip = weight_clip
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        if self.lr_factor == 1:
            self.reduce_optimizer = None
        else:
            self.reduce_optimizer = ReduceLROnPlateau(self.optimizer, mode=mode, factor=self.lr_factor,
                                                      patience=self.lr_patience)
        self.lrs = [float(param_group['lr']) for param_group in self.optimizer.param_groups]

    def step(self):
        self.clip_grad()
        self.optimizer.step()

    def step_lr(self, metrics):
        if self.lr_factor == 1:
            return
        raw_lrs = list()
        for i, param_group in enumerate(self.optimizer.param_groups):
            raw_lrs += [float(param_group['lr'])]
        self.reduce_optimizer.step(metrics=metrics)
        new_lrs = list()
        for i, param_group in enumerate(self.optimizer.param_groups):
            new_lrs += [float(param_group['lr'])]
        self.lrs = list()
        for raw_lr, new_lr in zip(raw_lrs, new_lrs):
            if raw_lr != new_lr:
                print("lr adapt: %s -> %s" % (raw_lr, new_lr))
            self.lrs += [new_lr]

    def zero_grad(self):
        self.optimizer.zero_grad()

    def clip_grad(self):
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], self.grad_clip)

    def __repr__(self):
        str1 = "%s [lr %s][weight-decay %s][grad-clip %s][weight-clip %s]" % (self.optimizer.__class__.__name__,
                                                                              self.optimizer.defaults['lr'],
                                                                              self.optimizer.defaults['weight_decay'],
                                                                              self.grad_clip, self.weight_clip)
        str2 = "[lr-factor %s][lr-patience %s]" % (self.lr_factor, self.lr_patience)
        return str1 + str2

    @staticmethod
    def find_prefix(vars_args, prefix):
        vars_clear_args = dict()
        keys = ['lr', 'optimizer', 'weight_decay', 'grad_clip', 'weight_clip']
        for key in keys:
            vars_clear_args[key] = vars_args["%s_%s" % (prefix, key)]
        return vars_clear_args

    @staticmethod
    def get_optimizer(args, parameters, prefix=None, mode='min'):
        vars_args = vars(args)

        if prefix is not None:
            vars_args = Optimizer.find_prefix(vars_args, prefix=prefix)

        if len(vars_args["optimizer"].split('=')) == 2:
            vars_args["optimizer"], vars_args["lr"] = vars_args["optimizer"].split('=')
            vars_args["lr"] = float(vars_args["lr"])

        weight_decay = 0 if 'weight_decay' not in vars(args) else vars_args["weight_decay"]
        grad_clip = -1 if 'grad_clip' not in vars(args) else vars_args["grad_clip"]
        weight_clip = -1 if 'weight_clip' not in vars(args) else vars_args["weight_clip"]
        lr_factor = 1 if 'lr_factor' not in vars(args) else vars_args["lr_factor"]
        lr_patience = 10 if 'lr_patience' not in vars(args) else vars_args["lr_patience"]

        if vars_args["lr"] is not None:
            optimizer = getattr(torch.optim, vars_args["optimizer"])(parameters, lr=vars_args["lr"],
                                                                     weight_decay=weight_decay)
        else:
            optimizer = getattr(torch.optim, vars_args["optimizer"])(parameters, weight_decay=weight_decay)

        return Optimizer(optimizer, grad_clip=grad_clip, weight_clip=weight_clip,
                         lr_factor=lr_factor, lr_patience=lr_patience, mode=mode)

    @staticmethod
    def add_optimizer_options(parser, default_lr=None, default_optim='Adam', prefix=None,
                              lr_factor=1, lr_patience=10, weight_decay=0, grad_clip=-1, weight_clip=-1):
        if prefix is None:
            parser.add_argument('-lr', type=float, dest="lr", default=default_lr)
            parser.add_argument('-lr-factor', type=float, dest="lr_factor", default=lr_factor)
            parser.add_argument('-lr-patience', type=float, dest="lr_patience", default=lr_patience)
            parser.add_argument('-optim', type=str, dest="optimizer", default=default_optim)
            parser.add_argument('-weight-decay', type=float, dest="weight_decay", default=weight_decay)
            parser.add_argument('-grad-clip', type=float, dest="grad_clip", default=grad_clip)
            parser.add_argument('-weight-clip', type=float, dest="weight_clip", default=weight_clip)
        else:
            parser.add_argument('-%s-lr' % prefix, type=float, dest="%s_lr" % prefix, default=default_lr)
            parser.add_argument('-%s-lr-factor', type=float, dest="%s_lr_factor", default=lr_factor)
            parser.add_argument('-%s-lr-patience', type=float, dest="%s_lr_patience", default=lr_patience)
            parser.add_argument('-%s-optim' % prefix, type=str, dest="%s_optimizer" % prefix, default=default_optim)
            parser.add_argument('-%s-weight-decay' % prefix, type=float, dest="%s_weight_decay" % prefix,
                                default=weight_decay)
            parser.add_argument('-%s-grad-clip' % prefix, type=float, dest="%s_grad_clip" % prefix, default=grad_clip)
            parser.add_argument('-%s-weight-clip' % prefix, type=float, dest="%s_weight_clip" % prefix,
                                default=weight_clip)
