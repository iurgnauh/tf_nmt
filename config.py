#!/usr/bin/env python
# -*- coding:utf8 -*-

# ================================================================================
# Copyright 2018 Alibaba Inc. All Rights Reserved.
#
# History:
# 2018.04.26. Be created by jiangshi.lxq. Forked and adatped from tensor2tensor. 
# For internal use only. DON'T DISTRIBUTE.
# ================================================================================


params = {}

params['train_src'] = '../data/train.all_beam.src.shuffle'
params['train_trg'] = '../data/train.all_beam.tgt.shuffle'
params['dev_src'] = '../data/dev.src'
params['dev_trg'] = '../data/dev.tgt'
params['vocab_src'] = 'vocab/vocab.big.src'
params['vocab_trg'] = 'vocab/vocab.big.tgt'

#params["num_gpus"] = 4
params["save_checkpoints_steps"]=5000
params["keep_checkpoint_max"]=20
params["max_len"] = 70
params["train_batch_size_words"]=4096
params["optimizer"] = 'adam'  # adam or sgd
params["learning_rate_decay"] = 'sqrt'  # sqrt: 0->peak->sqrt, exp: peak->exp
params["learning_rate_peak"] = 0.0003
params["warmup_steps"] = 8000  # only for sqrt decay
params["decay_steps"] = 100  # only for exp decay, decay every n steps
params["decay_rate"] = 0.9  # only for exp decay
params["src_vocab_size"] = 50000
params["trg_vocab_size"] = 50000
params["hidden_size"] = 512
params["filter_size"] = 2048
params["num_hidden_layers"] = 6
params["num_heads"] = 8
params['gradient_clip_value'] = 5.0
params["confidence"] = 0.9
params["prepost_dropout"] = 0.1
params["relu_dropout"] = 0.1
params["attention_dropout"] = 0.1
params["preproc_actions"] = 'n'
params["postproc_actions"] = 'da'

params["beam_size"] = 10
params["alpha"] = 0.6
params["max_decoded_trg_len"] = 70

params['model_dir'] = 'models/src2tgt_big'
params['output_dir'] = 'output/src2tgt_big/dev'
params['copy_dir'] = 'ema'

import os
if not os.path.exists(params['model_dir']):
    os.mkdir(params['model_dir'])
if not os.path.exists(params['output_dir']):
    os.mkdir(params['output_dir'])

params['output_path'] = params['output_dir'] + '/output'
params['output_all_beam'] = False


