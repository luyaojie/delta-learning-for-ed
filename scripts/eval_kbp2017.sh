#!/usr/bin/env bash
# -*- coding:utf-8 -*- 
python -u eval_event_detector.py \
      -data data/kbp2017_english \
      -pre-emb kbp2007.word.embedding.bin \
      -device 0 \
      -neg-ratio 5 \
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
      -prefix kbpbase \
      $*
