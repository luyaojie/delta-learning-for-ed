#!/usr/bin/env bash
# -*- coding:utf-8 -*- 
python -u train_event_detector.py \
      -data data/kbp2017_english \
      -pre-emb data/embedding/default.embedding \
      -device 0 \
      -neg-ratio 5 \
      -prefix kbpbase \
      -weight-clip 9 \
      -rp \
      -eval evmeval \
      -eval-each-batch -1 \
      -feat-vec-size 50 \
      -rnn-cat \
      -rnn-context mlp \
      -lexi-wind -1 \
      -emb-dropout 0.3 \
      -rnn-type GRU \
      -full-model-update \
      $*
