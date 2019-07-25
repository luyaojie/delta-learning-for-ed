#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger
from .pt4nlp.attention import ATTENTION_TYPE


def add_data_option(parser):
    # Data Path
    parser.add_argument('-data', type=str, dest="data", default="data/default_dataset")
    # Pre-trained Word Embedding
    parser.add_argument('-pre-emb', type=str, dest="pre_emb",
                        default="random")
    # Label each token in Multi-Token Nugget, Default is False
    parser.add_argument('-label-multi-token', dest='label_multi_token', action='store_true')

    # Use Relative Position
    parser.add_argument('-rp', dest='rp', action='store_true')
    # Use Entity Tag
    parser.add_argument('-entity', dest='entity', action='store_true')
    # Use Part-of-Speech, False in this paper
    parser.add_argument('-pos', dest='pos', action='store_true')
    # Use Lemma, False in this paper
    parser.add_argument('-lemma', dest='lemma', action='store_true')
    # Use Dep Role, False in this paper
    parser.add_argument('-deprole', dest='deprole', action='store_true')
    # Use Char from CharCNN, False in this paper
    parser.add_argument('-char', dest='char', action='store_true')

    # Dropout on Word Embedding
    parser.add_argument('-emb-dropout', type=float, dest='emb_dropout', default=0., help='Dropout Rate in Embedding')

    # Fix Neg Sample in each Epoch, False means sample neg sample in each Epoch
    parser.add_argument('-fix-neg', action='store_true', dest='fix_neg')
    # Sample Ratio for NIL Instances
    parser.add_argument('-neg-ratio', type=int, dest="neg_ratio", default=50)
    # Scale Weight for NIL Instances
    parser.add_argument('-neg-scale', type=float, dest="neg_scale", default=1.)
    # Max-Sentence Length
    parser.add_argument('-max-length', dest="max_length", type=int, default=300)
    # Context Window Size for Trigger Candidate
    parser.add_argument('-tri-wind', dest="trigger_window", type=int, default=18)
    # Drop Word in Context Window
    parser.add_argument('-word-drop', dest="word_drop", type=float, default=0)
    # Filter according to POS, False in this paper
    parser.add_argument('-pos-filter', dest="pos_filter", action='store_true')
    # Remove Stop Word in Context Window, False in this paper
    parser.add_argument('-remove-stop-word', dest="remove_stop_word", action='store_true')

    # Eval Batch Size
    parser.add_argument('-eval-batch', type=int, dest="eval_batch", default=500,
                        help='Bigger is faster, Require More Memory')
    # Eval Each Batch
    parser.add_argument('-eval-each-batch', type=int, dest="eval_each_batch", default=500,
                        help='Eval Model Each N Batch')

    # Pre-Train Lexi Classifier Epoch
    parser.add_argument('-adv-pre-train', type=int, dest='adv_pre_train', default=0, )
    # Weight for Lexical Adversarial
    parser.add_argument('-adv-weight', type=float, dest='adv_weight', default=0., )
    # Neg. Num for Lexical Adversarial/Enhance
    parser.add_argument('-adv-num', type=int, dest='adv_num', default=1, )
    # Weight for Lexical Enhance
    parser.add_argument('-enhance-weight', type=float, dest='enhance_weight', default=0., )
    # Type Lexi Representation
    parser.add_argument('-rw', type=str, dest='adv_target', default='word', choices=['word', 'elmo'])
    # Load Pre-train model
    parser.add_argument('-pre-train-model', type=str, dest='pre_train_model')
    # Load Pre-train Lexi-Spec model
    parser.add_argument('-load-spec-model', type=str, dest='load_spec_model')
    # Load Pre-train Lexi-Spec model
    parser.add_argument('-load-free-model', type=str, dest='load_free_model')
    # False is only Update Gate (Fix Lexi-Spec and Lexi-free), True is Update Global Model
    parser.add_argument('-full-model-update', dest='full_model_update', action='store_true')


def add_env_option(parser):
    # ENV
    # GPU Device Number, -1 is CPU
    parser.add_argument('-device', type=int, dest="device", default=-1)
    # Epoch Number
    parser.add_argument('-epoch', type=int, dest="epoch", default=50)
    # Batch size
    parser.add_argument('-batch', type=int, dest="batch", default=128, help='Bigger is faster, Require More Memory')
    # Word Embedding Dimension
    parser.add_argument('-word-vec-size', type=int, dest="word_vec_size", default=300, help='Word Vector Size')
    # Feature Vector Dimension
    parser.add_argument('-feat-vec-size', type=int, dest="feat_vec_size", default=50, help='Feature Vector Size')
    # Eval Type， evmeval for KBP2017， ace for ACE 2005
    parser.add_argument('-eval', type=str, dest="evaluate", default='ace', choices=['ace', 'evmeval'],
                        help='Evaluate Method')

    # DATA
    add_data_option(parser)

    # MODEL
    # No Save Model
    parser.add_argument('-no-save', dest="save", action='store_false',
                        help='No Save Model')
    # Model Name
    parser.add_argument('-model', dest='model', type=None, default='delta_learning')
    # Model Folder for Saving
    parser.add_argument('-model-folder', dest='model_folder', type=str, default='model', help='save model to folder')
    # Prefix Name
    parser.add_argument('-prefix', dest='prefix', type=str, default=None, help='prefix name for model filename')
    # Suffix Name
    parser.add_argument('-suffix', dest='suffix', type=str, default=None, help='suffix name for model filename')
    # Min F threshold for Save
    parser.add_argument('-save-threshold', dest='save_threshold', type=float, default=40.0)

    # Loss Function，This Paper use nll
    parser.add_argument('-loss', dest='loss', type=str, default='nll')

    # Lexi Window Size
    parser.add_argument('-lexi-wind', type=int, dest="lexi_wind", default=-1,
                        help='Lexi Window Size, 0 is trigger, -1 is None')

    # Trigger Mask, False in this paper
    parser.add_argument('-trigger-mask', action='store_true', help='Trigger Mask')

    # MLP Classifier
    # MLP Layer Num and Size, Default is Single Layer
    parser.add_argument('-mlp-hidden', nargs='+', dest='mlp_hiddens', default=[], type=int,
                        help='Hidden in MLP, as list')
    # Act in MLP
    parser.add_argument('-mlp-act', dest='mlp_act', default='ReLU', type=str)
    # Batch Normalization in MLP, False in this paper
    parser.add_argument('-mlp-bn', dest='mlp_bn', action='store_true')
    # Dropout in MLP
    parser.add_argument('-mlp-dropout', type=float, dest='mlp_dropout', default=0.5, help='Dropout Rate in MLP')


def add_cnn_option(parser):
    # CNN
    parser.add_argument('-cnn-filter', nargs='+', dest='cnn_filter', default=[3])
    parser.add_argument('-cnn-size', type=int, default=300, dest='cnn_hidden_size')
    parser.add_argument('-cnn-act', type=str, default='Tanh', dest='cnn_act')
    parser.add_argument('-cnn-bn', action='store_true', dest='cnn_bn')
    parser.add_argument('-cnn-dropout', type=float, dest="cnn_dropout", default=0.,
                        help='Dropout Rate on CNN Encoder')


def add_rnn_option(parser):
    # RNN
    parser.add_argument('-rnn-type', type=str, dest="rnn_type", default="LSTM", choices=['LSTM', 'GRU', 'RNN'],
                        help='RNN Type: LSTM, GRU, RNN')
    parser.add_argument('-rnn-dropout', type=float, dest="rnn_dropout", default=0.2,
                        help='Dropout Rate on RNN Encoder')
    parser.add_argument('-rnn-layer', type=int, dest="rnn_layer", default=1,
                        help='RNN Layer Number')
    parser.add_argument('-rnn-size', type=int, dest="rnn_size", default=300,
                        help='RNN Layer Number')
    parser.add_argument('-rnn-context', type=str, dest="rnn_context", choices=ATTENTION_TYPE + ['none'], default='none')
    # Cat RNN Hidden State with Attention Context (Ignore Trigger Candidate)
    parser.add_argument('-rnn-cat', action='store_true', dest="rnn_cat")
