#!/usr/bin/env python
# -*- coding:utf8 -*-

# ================================================================================
# Copyright 2018 Alibaba Inc. All Rights Reserved.
#
# History:
# 2018.04.27. Be created by jiangshi.lxq. Forked and adatped from tensor2tensor. 
# For internal use only. DON'T DISTRIBUTE.
# ================================================================================

from config import *
import tensorflow as tf
import sys
import numpy as np
import common_attention
import common_layers
import subprocess
import re
from random import shuffle
from tensorflow.python.client import device_lib
from importlib import import_module
#from dp import GraphDispatcher
tf.logging.set_verbosity(tf.logging.INFO)

assert len(sys.argv) == 2, 'python xx.py config_file'
_, config_file = sys.argv
if config_file.endswith('.py'):
    config_file = config_file.rsplit('.', 1)[0]
tf.logging.info('Using config from %s'%config_file)
in_config = import_module(config_file)
params = getattr(in_config, 'params')


def prepare_encoder_input(src_wids, src_masks, params):
    src_vocab_size = params["src_vocab_size"]
    hidden_size = params["hidden_size"]
    with tf.variable_scope('Source_Side'):
        src_emb = common_layers.embedding(src_wids, src_vocab_size, hidden_size)
    src_emb *= hidden_size**0.5
    encoder_self_attention_bias = common_attention.attention_bias_ignore_padding(1-src_masks)
    encoder_input = common_attention.add_timing_signal_1d(src_emb)
    encoder_input = tf.multiply(encoder_input,tf.expand_dims(src_masks,2))
    return encoder_input,encoder_self_attention_bias

def prepare_decoder_input(trg_wids, params):
    trg_vocab_size = params["trg_vocab_size"]
    hidden_size = params["hidden_size"]
    with tf.variable_scope('Target_Side'):
        trg_emb = common_layers.embedding(trg_wids, trg_vocab_size, hidden_size)
    trg_emb *= hidden_size**0.5
    decoder_self_attention_bias = common_attention.attention_bias_lower_triangle(\
            tf.shape(trg_emb)[1])
    decoder_input = common_layers.shift_left_3d(trg_emb)
    decoder_input = common_attention.add_timing_signal_1d(decoder_input)
    return decoder_input,decoder_self_attention_bias

def layer_process(x, y, flag, dropout):
    if flag == None:
        return y
    for c in flag:
        if c == 'a':
            y = x+y
        elif c == 'n':
            y = common_layers.layer_norm(y)
        elif c == 'd':
            y = tf.nn.dropout(y, 1.0 - dropout)
    return y

def transformer_ffn_layer(x, params):
    filter_size = params["filter_size"]
    hidden_size = params["hidden_size"]
    relu_dropout = params["relu_dropout"]
    return common_layers.conv_hidden_relu(
            x,
            filter_size,
            hidden_size,
            dropout=relu_dropout)

def transformer_encoder(encoder_input,
                        encoder_self_attention_bias,
                        mask,
                        params={},
                        name="encoder"):
    num_hidden_layers = params["num_hidden_layers"]
    hidden_size = params["hidden_size"]
    num_heads = params["num_heads"]
    prepost_dropout = params["prepost_dropout"]
    attention_dropout = params["attention_dropout"]
    preproc_actions = params['preproc_actions']
    postproc_actions = params['postproc_actions']
    x = encoder_input
    mask = tf.expand_dims(mask,2)
    with tf.variable_scope(name):
        for layer in xrange(num_hidden_layers):
            with tf.variable_scope("layer_%d" % layer):
                o,w = common_attention.multihead_attention(
                        layer_process(None,x,preproc_actions,prepost_dropout),
                        None,
                        encoder_self_attention_bias,
                        hidden_size,
                        hidden_size,
                        hidden_size,
                        num_heads,
                        attention_dropout,
                        summaries=False,
                        name="encoder_self_attention")
                x = layer_process(x,o,postproc_actions,prepost_dropout)
                o = transformer_ffn_layer(layer_process(None,x,preproc_actions,prepost_dropout), params)
                x = layer_process(x,o,postproc_actions,prepost_dropout)
                x = tf.multiply(x,mask)
        return layer_process(None,x,preproc_actions,prepost_dropout)

def transformer_decoder(decoder_input,
                        encoder_output,
                        decoder_self_attention_bias,
                        encoder_decoder_attention_bias,
                        states_key=None,
                        states_val=None,
                        params={},
                        name="decoder"):
    num_hidden_layers = params["num_hidden_layers"]
    hidden_size = params["hidden_size"]
    num_heads = params["num_heads"]
    prepost_dropout = params["prepost_dropout"]
    attention_dropout = params["attention_dropout"]
    preproc_actions = params['preproc_actions']
    postproc_actions = params['postproc_actions']
    x = decoder_input
    with tf.variable_scope(name):
        for layer in xrange(num_hidden_layers):
            with tf.variable_scope("layer_%d" % layer):
                o,w = common_attention.multihead_attention(
                        layer_process(None,x,preproc_actions,prepost_dropout),
                        None,
                        decoder_self_attention_bias,
                        hidden_size,
                        hidden_size,
                        hidden_size,
                        num_heads,
                        attention_dropout,
                        states_key=states_key,
                        states_val=states_val,
                        layer=layer,
                        summaries=False,
                        name="decoder_self_attention")
                x = layer_process(x,o,postproc_actions,prepost_dropout)
                o,w = common_attention.multihead_attention(
                        layer_process(None,x,preproc_actions,prepost_dropout),
                        encoder_output,
                        encoder_decoder_attention_bias,
                        hidden_size,
                        hidden_size,
                        hidden_size,
                        num_heads,
                        attention_dropout,
                        summaries=False,
                        name="encdec_attention")
                x = layer_process(x,o,postproc_actions,prepost_dropout)
                o = transformer_ffn_layer(layer_process(None,x,preproc_actions,prepost_dropout), params)
                x = layer_process(x,o,postproc_actions,prepost_dropout)
        return layer_process(None,x,preproc_actions,prepost_dropout), w

def transformer_body(body_input, params):
    encoder_input, encoder_self_attention_bias, src_masks, \
                     decoder_input, decoder_self_attention_bias = body_input
    # encode
    encoder_output = transformer_encoder(encoder_input, encoder_self_attention_bias,\
                src_masks, params)
    #decode
    decoder_output, attention_weights = transformer_decoder(decoder_input, encoder_output, \
            decoder_self_attention_bias, encoder_self_attention_bias, \
            states_key=None, states_val=None, params=params)
    return decoder_output, attention_weights

def output_layer(decoder_output, params):
    hidden_size = params["hidden_size"]
    trg_vocab_size = params["trg_vocab_size"]
    with tf.variable_scope('Target_Side'):
        with tf.variable_scope('WordEmbedding',reuse=True):
            trg_emb = tf.get_variable('C')
    shape = tf.shape(decoder_output)[:-1]
    decoder_output = tf.reshape(decoder_output, [-1, hidden_size])
    logits = tf.matmul(decoder_output, trg_emb, transpose_b=True)
    logits = tf.reshape(logits, tf.concat([shape, [trg_vocab_size]], 0))
    return logits

def _expand_to_beam_size(tensor, beam_size):
    tensor = tf.expand_dims(tensor, axis=1)
    tile_dims = [1] * tensor.shape.ndims
    tile_dims[1] = beam_size
    return tf.tile(tensor, tile_dims)

def _get_shape(tensor):
    static = tensor.shape.as_list()
    dynamic = tf.unstack(tf.shape(tensor))
    return [s[1] if s[0] is None else s[0] for s in zip(static, dynamic)]

def _merge_beam_dim(tensor):
    shape = _get_shape(tensor)
    shape[0] *= shape[1]  # batch -> batch * beam_size
    shape.pop(1)  # Remove beam dim
    return tf.reshape(tensor, shape)
    
def _unmerge_beam_dim(tensor, batch_size, beam_size):
    shape = _get_shape(tensor)
    new_shape = [batch_size] + [beam_size] + shape[1:]
    return tf.reshape(tensor, new_shape)

def compute_batch_indices(batch_size, beam_size):
    batch_pos = tf.range(batch_size * beam_size) // beam_size
    batch_pos = tf.reshape(batch_pos, [batch_size, beam_size])
    return batch_pos

def predict_next(encoder_output, encoder_self_attention_bias, trg_wids,\
        states_key, states_val, params={}):
    decoder_input,decoder_self_attention_bias = prepare_decoder_input(trg_wids, params)
    decoder_input = decoder_input[:,-1:,:]
    decoder_self_attention_bias = decoder_self_attention_bias[:,:,-1:,:]
    decoder_output, attention_weights = transformer_decoder(decoder_input, encoder_output,
            decoder_self_attention_bias, encoder_self_attention_bias, states_key, states_val,\
                    params=params)
    decoder_output_last = decoder_output[:,-1,:]
    attention_weights_last = attention_weights[:,-1,:]
    with tf.variable_scope('Target_Side'):
        with tf.variable_scope('WordEmbedding',reuse=True):
            trg_emb = tf.get_variable('C')
    logits = tf.matmul(decoder_output_last,trg_emb,transpose_b=True)
    logits = tf.nn.softmax(logits)
    logits = tf.log(logits)
    return logits, attention_weights_last, states_key, states_val


def beam_search(src_wids, src_masks, params, var_scope):
    var_scope.reuse_variables()
    INF = 1.0 * 1e7
    beam_size = params["beam_size"]
    vocab_size = params["trg_vocab_size"]
    hidden_size = params["hidden_size"]
    num_layers = params["num_hidden_layers"]
    alpha = params["alpha"]
    max_decoded_trg_len = params["max_decoded_trg_len"]
    source_length = tf.shape(src_wids)[1]
    batch_size = tf.shape(src_wids)[0]

    shift_src_mask = src_masks[:,1:] # NOTE should not be needed
    shift_src_mask = tf.pad(shift_src_mask,[[0,0],[0,1]])
    shift_src_mask = _expand_to_beam_size(shift_src_mask, beam_size)

    src_wids = _expand_to_beam_size(src_wids, beam_size)
    src_wids = _merge_beam_dim(src_wids)
    src_masks = _expand_to_beam_size(src_masks, beam_size)
    src_masks = _merge_beam_dim(src_masks)

    encoder_input, encoder_self_attention_bias = prepare_encoder_input(src_wids, src_masks, params)
    encoder_output = transformer_encoder(encoder_input, encoder_self_attention_bias,\
                src_masks, params)

    alive_seq = tf.zeros([batch_size, beam_size, 1], dtype=tf.int32)
    #alive_scores = tf.zeros([batch_size, beam_size])
    initial_scores = tf.constant([[0.] + [-INF] * (beam_size - 1)])
    alive_scores = tf.tile(initial_scores, [batch_size, 1])
    alive_att = tf.zeros([batch_size, beam_size, 1, source_length])
    
    finish_seq = tf.zeros([batch_size, beam_size, 1], dtype=tf.int32)
    finish_scores = tf.ones([batch_size, beam_size]) * -INF
    finish_flags = tf.zeros([batch_size, beam_size], dtype=tf.bool)
    finish_att = tf.zeros([batch_size, beam_size, 1, source_length])

    states_key = [tf.zeros([batch_size, 0, hidden_size]) for layer in range(num_layers)]
    states_val = [tf.zeros([batch_size, 0, hidden_size]) for layer in range(num_layers)]

    # Set 2nd dim to None since it's not invariant in the tf.while_loop
    for layer in range(num_layers):
      states_key[layer].set_shape(tf.TensorShape([None, None, hidden_size]))
      states_val[layer].set_shape(tf.TensorShape([None, None, hidden_size]))

    states_key = [_expand_to_beam_size(states_key[layer], beam_size) for layer in range(num_layers)]
    states_val = [_expand_to_beam_size(states_val[layer], beam_size) for layer in range(num_layers)]

    def _step(i, alive_seq, alive_scores, alive_att, finish_seq, finish_scores, \
            finish_att, finish_flags, states_key, states_val):
        reuse = tf.greater(i,0)
        flat_ids = tf.reshape(alive_seq, [batch_size * beam_size, -1]) # batch_size*beam_size x T
        flat_states_key = [_merge_beam_dim(states_key[layer]) for layer in range(num_layers)]
        flat_states_val = [_merge_beam_dim(states_val[layer]) for layer in range(num_layers)]

        flat_logits, flat_weights, flat_states_key, flat_states_val = predict_next(encoder_output, \
                encoder_self_attention_bias, flat_ids, flat_states_key, flat_states_val, params)

        states_key = [_unmerge_beam_dim(flat_states_key[layer], batch_size, beam_size) \
                for layer in range(num_layers)]
        states_val = [_unmerge_beam_dim(flat_states_val[layer], batch_size, beam_size) \
                for layer in range(num_layers)]

        step_scores = tf.reshape(flat_logits, (batch_size, beam_size, -1))
        step_weights = tf.reshape(flat_weights, (batch_size, beam_size, -1))
        scores = tf.expand_dims(alive_scores, axis=2) + step_scores
        flat_scores = tf.reshape(scores, [batch_size, beam_size * vocab_size])
        topk_scores, topk_ids = tf.nn.top_k(flat_scores, k=2*beam_size)
        topk_beam_index = topk_ids // vocab_size
        topk_ids %= vocab_size

        topk_flags = tf.equal(topk_ids,0)
        batch_pos = compute_batch_indices(batch_size, 2*beam_size)
        topk_coordinates = tf.stack([batch_pos, topk_beam_index], axis=2)
        topk_seq = tf.gather_nd(alive_seq, topk_coordinates)
        topk_seq = tf.concat([topk_seq[:,:,:-1], tf.expand_dims(topk_ids, axis=2)], axis=2)
        topk_seq = tf.concat([topk_seq, tf.zeros([batch_size, 2*beam_size,1],dtype=tf.int32)], axis=2)
        topk_att = tf.gather_nd(alive_att, topk_coordinates)
        curr_att = tf.gather_nd(step_weights, topk_coordinates)
        topk_att = tf.concat([topk_att, tf.expand_dims(curr_att, axis=2)], axis=2)
        states_key = [tf.gather_nd(states_key[layer], topk_coordinates) for layer in range(num_layers)]
        states_val = [tf.gather_nd(states_val[layer], topk_coordinates) for layer in range(num_layers)]


        # pick alive hypo
        virtual_scores = topk_scores + tf.to_float(topk_flags) * -INF
        _, topk_indexes = tf.nn.top_k(virtual_scores, k=beam_size)
        batch_pos = compute_batch_indices(batch_size, beam_size)
        top_coordinates = tf.stack([batch_pos, topk_indexes], axis=2)
        alive_seq = tf.gather_nd(topk_seq, top_coordinates)
        alive_scores = tf.gather_nd(virtual_scores, top_coordinates)
        alive_att = tf.gather_nd(topk_att, top_coordinates)
        states_key = [tf.gather_nd(states_key[layer], top_coordinates) for layer in range(num_layers)]
        states_val = [tf.gather_nd(states_val[layer], top_coordinates) for layer in range(num_layers)]

        # pick dead hypo
        finish_seq = tf.concat([finish_seq, tf.zeros([batch_size, beam_size, 1], tf.int32)], axis=2)
        finish_att = tf.concat([finish_att, tf.zeros([batch_size, beam_size, 1, source_length])], axis=2)
        # nomalize topk_scores with length penalty
        length_penalty = tf.pow(((5. + tf.to_float(i + 1)) / 6.), alpha)
        topk_norm_scores = topk_scores / length_penalty

        virtual_scores = topk_norm_scores + (1 - tf.to_float(topk_flags)) * -INF
        mix_seq = tf.concat([finish_seq, topk_seq], axis=1)
        mix_scores = tf.concat([finish_scores, virtual_scores], axis=1)
        mix_att = tf.concat([finish_att, topk_att], axis=1)
        mix_flags = tf.concat([finish_flags, topk_flags], axis=1)
        _, topk_indexes = tf.nn.top_k(mix_scores, k=beam_size)
        top_coordinates = tf.stack([batch_pos, topk_indexes], axis=2)
        finish_seq = tf.gather_nd(mix_seq, top_coordinates)
        finish_flags = tf.gather_nd(mix_flags, top_coordinates)
        finish_scores = tf.gather_nd(mix_scores, top_coordinates)
        finish_scores = finish_scores + (1 - tf.to_float(finish_flags)) * -INF
        finish_att = tf.gather_nd(mix_att, top_coordinates)

        return i+1, alive_seq, alive_scores, alive_att, \
                finish_seq, finish_scores, finish_att, finish_flags, states_key, states_val

                
    def is_finished(i,alive_seq,alive_score,alive_att,\
            finish_seq,finish_score,finish_att,finish_flag,states_key, states_val):
      max_length_penalty = tf.pow(((5. + tf.to_float(max_decoded_trg_len)) / 6.),alpha)
      # The best possible score of the most likley alive sequence
      lower_bound_alive_scores = alive_score[:, 0] / max_length_penalty

      lowest_score_of_fininshed = tf.reduce_min(finish_score * tf.to_float(finish_flag), axis=1)
      lowest_score_of_fininshed += ((1. - tf.to_float(tf.reduce_any(finish_flag, 1))) * -INF)

      bound_is_met = tf.reduce_all(tf.greater(lowest_score_of_fininshed,
                     lower_bound_alive_scores))
      return tf.logical_and( tf.less(i, max_decoded_trg_len),tf.logical_not(bound_is_met) )


    _, alive_seq, alive_scores, alive_att,\
            finish_seq, finish_scores, finish_att, finish_flags, states_key, states_val = \
            tf.while_loop( is_finished, _step, \
            [tf.constant(0), alive_seq, alive_scores, alive_att, finish_seq, \
                finish_scores, finish_att, finish_flags, states_key, states_val \
            ],
         shape_invariants=[
             tf.TensorShape([]),
             tf.TensorShape([None, None, None]),
             tf.TensorShape([None, None]),
             tf.TensorShape([None, None, None, None]),
             tf.TensorShape([None, None, None]),
             tf.TensorShape([None, None]),
             tf.TensorShape([None, None, None, None]),
             tf.TensorShape([None, None]), 
             [tf.TensorShape([None, None, None, hidden_size]) for layer in range(num_layers)],
             [tf.TensorShape([None, None, None, hidden_size]) for layer in range(num_layers)]
             ],
         parallel_iterations=1,
         back_prop=False)
    finish_seq_len = tf.shape(finish_seq)[2]
    shift_src_mask = tf.expand_dims(shift_src_mask,2)
    shift_src_mask = tf.tile(shift_src_mask, [1,1,finish_seq_len,1])
    finish_att = tf.multiply(finish_att, tf.to_float(shift_src_mask))  # no attention for eos
    #final_trans = tf.to_int64(tf.transpose(finish_seq[:,:,:-1], [2,0,1]))
    final_score = finish_scores
    beam_idx = tf.argmax(finish_scores,1,output_type=tf.int32)
    batch_idx = tf.range(batch_size)
    best_idx = tf.stack([batch_idx, beam_idx], axis=1)
    final_trans = tf.gather_nd(finish_seq[:,:,:-1], best_idx)
    final_att = tf.to_int32(tf.transpose(tf.argmax(finish_att[:,:,1:,:-1],3), [2,0,1]))
    return [final_trans, final_score, final_att]

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)
      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(grads, 0)
    grad = tf.reduce_sum(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def transformer_model_fn(features, labels, mode, params):
    with tf.variable_scope('NmtModel') as var_scope:
        if mode == tf.estimator.ModeKeys.PREDICT:
            src_wids = features
            src_masks = tf.to_float(tf.not_equal(src_wids,0))    #NOTE no padding for eos
            output_wids, scores, atts = beam_search(src_wids, src_masks, params)
            predictions = {"translation":output_wids}  #TODO
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        def get_loss(features, labels, params):
            #features = tf.Print(features,[tf.shape(features)])
            last_padding = tf.zeros([tf.shape(features)[0],1],tf.int64)
            src_wids = tf.concat([features,last_padding],1)
            src_masks = tf.to_float(tf.not_equal(src_wids,0))
            shift_src_masks = src_masks[:,:-1]
            shift_src_masks = tf.pad(shift_src_masks,[[0,0],[1,0]],constant_values=1)
            trg_wids = tf.concat([labels,last_padding],1)
            trg_masks = tf.to_float(tf.not_equal(trg_wids,0))
            shift_trg_masks = trg_masks[:,:-1]
            shift_trg_masks = tf.pad(shift_trg_masks,[[0,0],[1,0]],constant_values=1)

            encoder_input, encoder_self_attention_bias = prepare_encoder_input(src_wids, shift_src_masks, params)
            encoder_input = tf.nn.dropout(encoder_input,\
                                              1.0 - params['prepost_dropout'])
            decoder_input, decoder_self_attention_bias = prepare_decoder_input(trg_wids, params)
            decoder_input = tf.nn.dropout(decoder_input,\
                                              1.0 - params['prepost_dropout'])
            body_input = encoder_input, encoder_self_attention_bias, shift_src_masks, \
                         decoder_input, decoder_self_attention_bias
            body_output = transformer_body(body_input, params)
            decoder_output, attention_weights = body_output
            logits = output_layer(decoder_output, params)

            confidence = params["confidence"]
            trg_vocab_size = params["trg_vocab_size"]
            low_confidence = (1.0 - confidence) / tf.to_float(trg_vocab_size - 1)
            soft_targets = tf.one_hot(tf.cast(trg_wids, tf.int32), depth=trg_vocab_size,
                    on_value=confidence, off_value=low_confidence)
            mask = tf.cast(shift_trg_masks,logits.dtype)
            xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=soft_targets) * mask
            loss = tf.reduce_sum(xentropy) / tf.reduce_sum(mask)
            return loss

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_gpus = params['num_gpus']
            gradient_clip_value = params['gradient_clip_value']
            step = tf.to_float(tf.train.get_global_step())
            warmup_steps = params['warmup_steps']
            if params['learning_rate_decay'] == 'sqrt':
                lr_warmup = params['learning_rate_peak'] * tf.minimum(1.0,step/warmup_steps)
                lr_decay = params['learning_rate_peak'] * tf.minimum(1.0,tf.sqrt(warmup_steps/step))
                lr = tf.where(step < warmup_steps, lr_warmup, lr_decay)
            elif params['learning_rate_decay'] == 'exp':
                lr = tf.train.exponential_decay(params['learning_rate_peak'], 
                        global_step=step, 
                        decay_steps=params['decay_steps'],
                        decay_rate=params['decay_rate'])
            else:
                tf.logging.info("learning rate decay strategy not supported")
                sys.exit()
            if params['optimizer'] == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr)
            elif params['optimizer'] == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.997, epsilon=1e-09)
            else:
                tf.logging.info("optimizer not supported")
                sys.exit()
            #optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.98, epsilon=1e-09)
            optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, gradient_clip_value)

            #dispatcher = GraphDispatcher(num_gpus, daisy_chain_variables=True)
            n = (tf.shape(features)[0]//num_gpus ) * num_gpus
            features = features[:n]
            labels = labels[:n]
            feature_shards = tf.split(features, num_gpus)
            label_shards = tf.split(labels, num_gpus)
            #loss_shards = dispatcher(get_loss, feature_shards, label_shards, params)
            devices = [x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"]
            loss_shards = []
            grad_shards = []
            for i, device in enumerate(devices):
                #if i > 0:
                    #var_scope.reuse_variables()
                with tf.variable_scope( tf.get_variable_scope(), reuse=True if i > 0 else None):
                    with tf.device(device):
                        loss = get_loss(feature_shards[i], label_shards[i], params)
                        grads = optimizer.compute_gradients(loss)
                        #tf.get_variable_scope().reuse_variables()
                        loss_shards.append(loss)
                        grad_shards.append(grads)
            #loss_shards = tf.Print(loss_shards,[loss_shards])
            loss = tf.reduce_mean(loss_shards)
            grad = average_gradients(grad_shards)
            train_op = optimizer.apply_gradients(grad, global_step=tf.train.get_global_step())
            ema = tf.train.ExponentialMovingAverage(decay=0.9999)
            with tf.control_dependencies([train_op]):
                train_op = ema.apply(tf.trainable_variables())
            #train_op = optimizer.minimize(loss, tf.train.get_global_step())

            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        if mode == tf.estimator.ModeKeys.EVAL: 
            loss = get_loss(features, labels, params)
            src_wids = tf.concat([features,last_padding],1)
            src_masks = tf.to_float(tf.not_equal(src_wids,0))
            shift_src_masks = src_masks[:,:-1]
            shift_src_masks = tf.pad(shift_src_masks,[[0,0],[1,0]],constant_values=1)
            output_wids,_,_ = beam_search(src_wids, shift_src_masks, params, var_scope)
            predictions = {"wids":output_wids}
            add_dict_to_collection("predictions", predictions)
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss)

def input_fn(src_file,
             trg_file,
             src_vocab_file,
             trg_vocab_file,
             num_buckets=20,
             max_len=100,
             is_shuffle=True,
             batch_size=200,
             batch_size_words=4096,
             num_gpus=1,
             is_train=True):
    src_vocab = tf.contrib.lookup.index_table_from_file(src_vocab_file,default_value=1) # NOTE unk -> 1
    trg_vocab = tf.contrib.lookup.index_table_from_file(trg_vocab_file,default_value=1)
    src_dataset = tf.data.TextLineDataset(src_file)
    trg_dataset = tf.data.TextLineDataset(trg_file)
    src_trg_dataset = tf.data.Dataset.zip((src_dataset, trg_dataset))
    #if is_train == True:
        #src_trg_dataset = src_trg_dataset.shard(4, int(sys.argv[1]))
    src_trg_dataset = src_trg_dataset.map(
            lambda src, trg: ( tf.string_split([src]).values, tf.string_split([trg]).values),
            num_parallel_calls=10).prefetch(1000000)
    src_trg_dataset = src_trg_dataset.map(
            lambda src, trg: (src_vocab.lookup(src), trg_vocab.lookup(trg)),
            num_parallel_calls=10).prefetch(1000000)
    #if is_shuffle:
    #    src_trg_dataset = src_trg_dataset.shuffle(buffer_size=1000000)

    #if False:
    if is_train == True:
      def key_func(src_data, trg_data):
        bucket_width = (max_len + num_buckets - 1) // num_buckets
        bucket_id = tf.maximum(tf.size(src_data)  // bucket_width, tf.size(trg_data) // bucket_width)
        return tf.to_int64(tf.minimum(num_buckets, bucket_id))

      def reduce_func(unused_key, windowed_data):
        return windowed_data.padded_batch(batch_size_words, padded_shapes=([None],[None]))

      def window_size_func(key):
        bucket_width = (max_len + num_buckets - 1) // num_buckets
        key += 1  # For bucket_width == 1, key 0 is unassigned.
        size = (num_gpus * batch_size_words // (key * bucket_width))
        return tf.to_int64(size)

      src_trg_dataset = src_trg_dataset.filter(
                  lambda src, trg: tf.logical_and(tf.size(src)<=max_len, tf.size(trg)<=max_len))
      src_trg_dataset = src_trg_dataset.apply(
          tf.contrib.data.group_by_window(
              key_func=key_func, reduce_func=reduce_func, window_size_func=window_size_func))

    else:
        src_trg_dataset = src_trg_dataset.padded_batch(batch_size*num_gpus, padded_shapes=([None],[None]))
    #src_trg_dataset = src_trg_dataset.repeat(num_epochs)
    iterator = src_trg_dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    features,labels = iterator.get_next()
    return features,labels

def cal_bleu(output_file, ref_file):
    bleu_out = subprocess.check_output(['tools/multi-bleu.perl', ref_file], \
            stdin=open(output_file), stderr=subprocess.STDOUT)
    bleu_out = bleu_out.decode("utf-8")
    bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
    return bleu_score

def add_dict_to_collection(collection_name, dict_):
  key_collection = collection_name + "_keys"
  value_collection = collection_name + "_values"
  for key, value in dict_.items():
    tf.add_to_collection(key_collection, key)
    tf.add_to_collection(value_collection, value)

def get_dict_from_collection(collection_name):
  key_collection = collection_name + "_keys"
  value_collection = collection_name + "_values"
  keys = tf.get_collection(key_collection)
  values = tf.get_collection(value_collection)
  return dict(zip(keys, values))

def extract_batches(tensors):
  if not isinstance(tensors, dict):
    for tensor in tensors:
      yield tensor
  else:
    batch_size = None
    for value in tensors.values():
      batch_size = batch_size or value.shape[0]
    for b in range(batch_size):
      yield {
          key: value[b] for key, value in tensors.items()
      }

class SaveEvaluationPredictionHook(tf.train.SessionRunHook):
  def __init__(self, output_file, ref_file, trg_vocab_file):
    self._output_file = output_file
    self._ref_file = ref_file
    self._trg_vocab_file = trg_vocab_file

  def begin(self):
    self._predictions = get_dict_from_collection("predictions")
    self._global_step = tf.train.get_global_step()

  def before_run(self, run_context):
    return tf.train.SessionRunArgs([self._predictions, self._global_step])

  def after_run(self, run_context, run_values):
    predictions, self._current_step = run_values.results
    self._output_path = "{}.{}".format(self._output_file, self._current_step)
    trg_rvocab = dict([(i,w.strip()) for i,w in enumerate(open(self._trg_vocab_file))])
    with open(self._output_path, "a") as output_file:
      for prediction in extract_batches(predictions):
        wids = list(prediction['wids']) + [0]
        wids = wids[:wids.index(0)]
        o = ' '.join([trg_rvocab[wid] for wid in wids]).replace('@@ ','').replace('@@','')
        print >>output_file, o

  def end(self, session):
    tf.logging.info("Evaluation predictions saved to %s", self._output_path)
    score = cal_bleu(self._output_path, self._ref_file)
    tf.logging.info("BLEU: %s", score)

def shuffle_train(train_src, train_trg):
    line_pairs = []
    for src_line,trg_line in zip(open(train_src),open(train_trg)):
        line_pairs.append((src_line,trg_line))
    shuffle(line_pairs)
    fsrc = open(train_src,'w')
    ftrg = open(train_trg,'w')
    for src_line,trg_line in line_pairs:
        print >>fsrc, src_line.strip()
        print >>ftrg, trg_line.strip()

def main(_):

    gpu_devices = [x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"]
    if len(gpu_devices) == 0:
        params['num_gpus'] = 1
    else:
        params['num_gpus'] = len(gpu_devices)
    print(params)

    transformer = tf.estimator.Estimator(model_fn=transformer_model_fn,\
            config=tf.estimator.RunConfig(\
            save_checkpoints_steps=params['save_checkpoints_steps'],\
            keep_checkpoint_max = params['keep_checkpoint_max']),\
            model_dir=params['model_dir'],params=params)

    train_src = params['train_src']
    train_trg = params['train_trg']
    vocab_src = params['vocab_src']
    vocab_trg = params['vocab_trg']
    
    epoch = 0
    while True:
        epoch += 1
        tf.logging.info("Epoch %i", epoch)
        #shuffle_train(train_src,train_trg)
        train_input_fn = lambda: input_fn(
            train_src,
            train_trg,
            vocab_src,
            vocab_trg,
            batch_size_words=params['train_batch_size_words'],
            max_len=params['max_len'],
            num_gpus=params['num_gpus'],
            is_shuffle=False,
            is_train=True
        )
        transformer.train(train_input_fn)

if __name__ == '__main__':
    tf.app.run()

