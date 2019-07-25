#!/usr/bin/env bash
# -*- coding:utf-8 -*-
export ACE_SOURCE_FOLDER=./data/ace2005_english/ace2005_all_source
python -u eval_event_detector.py \
      -data data/ace2005_english \
      -pre-emb ace2005.word.embedding.bin \
      -device 0 \
      -neg-ratio -1 \
      -cnn-filter 3 \
      -prefix acebase \
      -weight-clip 9 \
      -max-length 80 \
      -rp \
      -entity \
      -feat-vec-size 50 \
      -rnn-type GRU \
      -rnn-cat \
      -rnn-context mlp \
      -lexi-wind -1 \
      -emb-dropout 0.3 \
      $*
