# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to build and run the tensorflow graph for the sequence-to-sequence model"""

import os
import time
import numpy as np
import tensorflow as tf
from attention_decoder import attention_decoder
from tensorflow.contrib.tensorboard.plugins import projector
import data

FLAGS = tf.app.flags.FLAGS

class SummarizationModel(object):
  """A class to represent a sequence-to-sequence model for text summarization. Supports both baseline mode, pointer-generator mode, and coverage"""

  def __init__(self, hps, vocab):
    self._hps = hps
    self._vocab = vocab

  def _add_placeholders(self):
    """Add placeholders to the graph. These are entry points for any input data."""
    hps = self._hps

    # encoder part
    self._enc_batch = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch')
    self._enc_lens = tf.placeholder(tf.int32, [hps.batch_size], name='enc_lens')
    self._enc_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, None], name='enc_padding_mask')
    if hps.title_engaged or hps.title_guided:
      self._title_batch = tf.placeholder(tf.int32, [hps.batch_size, None], name='title_batch')
      self._title_lens = tf.placeholder(tf.int32, [hps.batch_size], name='title_lens')
      self._title_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, None], name='title_padding_mask')
      
    if hps.pointer_gen:
      self._enc_batch_extend_vocab = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch_extend_vocab')
      self._max_art_oovs = tf.placeholder(tf.int32, [], name='max_art_oovs')
      if hps.co_occurrence or hps.prev_relation or hps.co_occurrence_h or hps.markov_attention or hps.markov_attention_contribution:
        self._cooccurrence_matrix = tf.placeholder(tf.float32, [hps.batch_size, None, None], name='cooccurrence_matrix')
      if hps.co_occurrence_i or (hps.coverage and hps.coverage_weighted) or hps.attention_weighted or hps.markov_attention or hps.markov_attention_contribution:
        self._cooccurrence_weight = tf.placeholder(tf.float32, [hps.batch_size, None], name='cooccurrence_weight')
      if hps.mode == "decode" and (hps.markov_attention or hps.markov_attention_contribution):
        self._prev_attention_dist = tf.placeholder(tf.float32, [hps.batch_size, None], name='prev_attention_dist')
      if hps.tagger_attention or hps.tagger_encoding:
        self._tagger_matrix = tf.placeholder(tf.float32, [hps.batch_size, None, len(data.TAGS_SET) + 1], name='tagger_matrix')
    # decoder part
    self._dec_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='dec_batch')
    self._target_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='target_batch')
    self._dec_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, hps.max_dec_steps], name='dec_padding_mask')

    if hps.mode=="decode" and hps.coverage:
      self.prev_coverage = tf.placeholder(tf.float32, [hps.batch_size, None], name='prev_coverage')

  def _make_feed_dict(self, batch, just_enc=False):
    """Make a feed dictionary mapping parts of the batch to the appropriate placeholders.

    Args:
      batch: Batch object
      just_enc: Boolean. If True, only feed the parts needed for the encoder.
    """
    feed_dict = {}
    feed_dict[self._enc_batch] = batch.enc_batch
    feed_dict[self._enc_lens] = batch.enc_lens
    feed_dict[self._enc_padding_mask] = batch.enc_padding_mask
    if FLAGS.title_engaged or FLAGS.title_guided:
      feed_dict[self._title_batch] = batch.title_batch
      feed_dict[self._title_lens] = batch.title_lens
      feed_dict[self._title_padding_mask] = batch.title_padding_mask
    if FLAGS.pointer_gen:
      feed_dict[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
      feed_dict[self._max_art_oovs] = batch.max_art_oovs
      if FLAGS.co_occurrence or FLAGS.prev_relation or FLAGS.co_occurrence_h or FLAGS.markov_attention or FLAGS.markov_attention_contribution:
        feed_dict[self._cooccurrence_matrix] = batch.cooccurrence_matrix
      if FLAGS.co_occurrence_i or (FLAGS.coverage and FLAGS.coverage_weighted) or FLAGS.attention_weighted or FLAGS.markov_attention or FLAGS.markov_attention_contribution:
        feed_dict[self._cooccurrence_weight] = batch.cooccurrence_weight
      if FLAGS.tagger_attention or FLAGS.tagger_encoding:
        feed_dict[self._tagger_matrix] = batch.tagger_matrix
    if not just_enc:
      feed_dict[self._dec_batch] = batch.dec_batch
      feed_dict[self._target_batch] = batch.target_batch
      feed_dict[self._dec_padding_mask] = batch.dec_padding_mask
    return feed_dict

  def _add_encoder(self, encoder_inputs, seq_len, variable_scope_name="encoder", enc_outputs=None):
    """Add a single-layer bidirectional LSTM encoder to the graph.

    Args:
      encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
      seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].

    Returns:
      encoder_outputs:
        A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
      fw_state, bw_state:
        Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
    """
    with tf.variable_scope(variable_scope_name):
      if self._hps.cell_type == 'GRU':
        cell_fw = tf.contrib.rnn.GRUCell(self._hps.hidden_dim, kernel_initializer=self.rand_unif_init)
        cell_bw = tf.contrib.rnn.GRUCell(self._hps.hidden_dim, kernel_initializer=self.rand_unif_init)
      else:
        cell_fw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
        cell_bw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)

        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=(1.0 - self._hps.dropout))
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=(1.0 - self._hps.dropout))

      (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs, dtype=tf.float32, sequence_length=seq_len, swap_memory=True)
      encoder_outputs = tf.concat(axis=2, values=encoder_outputs) # concatenate the forwards and backwards states
      if enc_outputs is not None:
        encoder_outputs = 0.5 * enc_outputs + 0.5 * encoder_outputs

      if self._hps.source_siding_bridge:
        encoder_outputs = tf.concat(axis=2, values=[encoder_outputs, encoder_inputs])
      return encoder_outputs, fw_st, bw_st

  def _reduce_states(self, fw_st, bw_st):
    """Add to the graph a linear layer to reduce the encoder's final FW and BW state into a single initial state for the decoder. This is needed because the encoder is bidirectional but the decoder is not.

    Args:
      fw_st: LSTMStateTuple with hidden_dim units.
      bw_st: LSTMStateTuple with hidden_dim units.

    Returns:
      state: LSTMStateTuple with hidden_dim units.
    """
    hidden_dim = self._hps.hidden_dim
    with tf.variable_scope('reduce_final_st'):
      if self._hps.cell_type == "GRU":
        w_reduce_s = tf.get_variable('w_reduce_s', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
        bias_reduce_s = tf.get_variable('bias_reduce_s', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
        old_s = tf.concat(axis=1, values=[fw_st, bw_st])
        new_s = tf.nn.relu(tf.matmul(old_s, w_reduce_s) + bias_reduce_s)
        return new_s
      else:
        # Define weights and biases to reduce the cell and reduce the state
        w_reduce_c = tf.get_variable('w_reduce_c', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
        w_reduce_h = tf.get_variable('w_reduce_h', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
        bias_reduce_c = tf.get_variable('bias_reduce_c', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
        bias_reduce_h = tf.get_variable('bias_reduce_h', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
        # Apply linear layer
        old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c]) # Concatenation of fw and bw cell
        old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h]) # Concatenation of fw and bw state
        new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c) # Get new cell from old cell
        new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h) # Get new state from old state
        return tf.contrib.rnn.LSTMStateTuple(new_c, new_h) # Return new cell and state

  def _add_decoder(self, inputs, decoder_input_ids, emb_enc_inputs):
    """Add attention decoder to the graph. In train or eval mode, you call this once to get output on ALL steps. In decode (beam search) mode, you call this once for EACH decoder step.

    Args:
      inputs: inputs to the decoder (word embeddings). A list of tensors shape (batch_size, emb_dim)

    Returns:
      outputs: List of tensors; the outputs of the decoder
      out_state: The final state of the decoder
      attn_dists: A list of tensors; the attention distributions
      p_gens: A list of scalar tensors; the generation probabilities
      coverage: A tensor, the current coverage vector
    """
    hps = self._hps
    if hps.cell_type == 'GRU':
      cell = tf.contrib.rnn.GRUCell(hps.hidden_dim, kernel_initializer=self.rand_unif_init)
    else:
      cell = tf.contrib.rnn.LSTMCell(hps.hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)
    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=(1.0 - hps.dropout))

    prev_coverage = self.prev_coverage if hps.mode == "decode" and hps.coverage else None # In decode mode, we run attention_decoder one step at a time and so need to pass in the previous step's coverage vector each time
    prev_attn_dist = self._prev_attention_dist if hps.mode == "decode" and hps.markov_attention_contribution else None
    co_matrix = self._cooccurrence_matrix if hps.co_occurrence or hps.co_occurrence_h or hps.markov_attention_contribution else None
    enc_batch_extend_vocab = self._enc_batch_extend_vocab if hps.co_occurrence_h else None
    attn_weight = self._cooccurrence_weight if hps.attention_weighted or (hps.coverage and hps.coverage_weighted) or hps.markov_attention_contribution else None
    tagger_matrix = self._tagger_matrix if hps.tagger_attention else None
    title_encoder_states = self._title_states if hps.title_engaged or hps.title_guided else None
    title_padding_mask = self._title_padding_mask if hps.title_engaged or hps.title_guided else None

    outputs, out_state, attn_dists, p_gens, coverage = attention_decoder(inputs, self._dec_in_state, self._enc_states, self._enc_padding_mask, cell, initial_state_attention=(hps.mode=="decode"), pointer_gen=hps.pointer_gen, use_coverage=hps.coverage, prev_coverage=prev_coverage, matrix=co_matrix, enc_batch_extend_vocab=enc_batch_extend_vocab, decoder_input_ids=decoder_input_ids, attention_weight=attn_weight, emb_enc_inputs=emb_enc_inputs, prev_attention_dist=prev_attn_dist, tagger_matrix=tagger_matrix, title_encoder_states=title_encoder_states, title_padding_mask=title_padding_mask)

    return outputs, out_state, attn_dists, p_gens, coverage

  def _calc_final_dist(self, vocab_dists, attn_dists, init_attn=None):
    """Calculate the final distribution, for the pointer-generator model

    Args:
      vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.
      attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays
      init_attn: The textrank distributions. shape (batch_size, attn_len)

    Returns:
      final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
    """
    with tf.variable_scope('final_distribution'):
      
      attn_len = tf.shape(self._enc_batch_extend_vocab)[1] # number of states we attend over
      if self._hps.prev_relation or self._hps.markov_attention:
        co_matrix = tf.slice(self._cooccurrence_matrix, [0, 0, 0], [-1, attn_len, attn_len])

      # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
      if not self._hps.generation_only:
        vocab_dists = [p_gen * dist for (p_gen, dist) in zip(self.p_gens, vocab_dists)]

      if self._hps.markov_attention:
        mark_dists = []
        for i in range(self._hps.max_dec_steps):
          prev_attn = init_attn if i == 0 else attn_dists[i - 1]
          prev_attn_dist = tf.tile(tf.expand_dims(prev_attn, 1), [1, attn_len, 1])
          mark_dist = tf.reduce_sum(tf.multiply(tf.matrix_transpose(co_matrix), prev_attn_dist), 2)
          mark_dists.append(mark_dist)
        attn_dists = [dist1 + dist2 for (dist1, dist2) in zip(mark_dists, attn_dists)]

      if self._hps.prev_relation:
        p_r = 0.2
        attn_dists = [(1 - p_gen) * (1 - p_r) * dist for (p_gen, dist) in zip(self.p_gens, attn_dists)]
      else:
        if not self._hps.copy_only:
          attn_dists = [(1 - p_gen) * dist for (p_gen, dist) in zip(self.p_gens, attn_dists)]

      # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
      extended_vsize = self._vocab.size() + self._max_art_oovs # the maximum (over the batch) size of the extended vocabulary
      if not self._hps.copy_only:
        extra_zeros = tf.zeros((self._hps.batch_size, self._max_art_oovs))
        vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists] # list length max_dec_steps of shape (batch_size, extended_vsize)

      if not self._hps.generation_only:
        # Project the values in the attention distributions onto the appropriate entries in the final distributions
        # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary, then we add 0.1 onto the 500th entry of the final distribution
        # This is done for each decoder timestep.
        # This is fiddly; we use tf.scatter_nd to do the projection
        batch_nums = tf.range(0, limit=self._hps.batch_size) # shape (batch_size)
        batch_nums = tf.expand_dims(batch_nums, 1) # shape (batch_size, 1)
        batch_nums = tf.tile(batch_nums, [1, attn_len]) # shape (batch_size, attn_len)
        indices = tf.stack( (batch_nums, self._enc_batch_extend_vocab), axis=2) # shape (batch_size, enc_t, 2)
        shape = [self._hps.batch_size, extended_vsize]
        attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists] # list length max_dec_steps (batch_size, extended_vsize)

      if self._hps.prev_relation:
        # p_r = tf.get_variable("p_r", [1], initializer=tf.constant_initializer(0.2))
        # self._p_r = p_r[0]        
        relation_dists = []
        for i in range(self._hps.max_dec_steps):
          single_relation_dists = []
          for j in range(self._hps.batch_size):
            m = co_matrix[j]
            v = self._enc_batch_extend_vocab[j]
            x = self._dec_batch[j][i]
            t = tf.where(tf.equal(v, x))
            d = tf.cond(tf.shape(t)[0] > 0, lambda: m[t[0][0]], lambda: tf.zeros([attn_len]))
            single_relation_dists.append(d)
          relation_dists.append(single_relation_dists)

        relation_dists_projected = [tf.scatter_nd(indices, (1 - p_gen) * p_r * relation_dist, shape) for (relation_dist, p_gen) in zip(relation_dists, self.p_gens)]
        final_dists = [vocab_dist + copy_dist + relation_dist for (vocab_dist, copy_dist, relation_dist) in zip(vocab_dists_extended, attn_dists_projected, relation_dists_projected)]
      else:
        # Add the vocab distributions and the copy distributions together to get the final distributions
        # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving the final distribution for that decoder timestep
        # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
        if self._hps.generation_only:
          final_dists = vocab_dists_extended
        elif self._hps.copy_only:
          final_dists = attn_dists_projected
        else:
          final_dists = [vocab_dist + copy_dist for (vocab_dist, copy_dist) in zip(vocab_dists_extended, attn_dists_projected)]

      return final_dists

  def _add_emb_vis(self, embedding_var):
    """Do setup so that we can view word embedding visualization in Tensorboard, as described here:
    https://www.tensorflow.org/get_started/embedding_viz
    Make the vocab metadata file, then make the projector config file pointing to it."""
    train_dir = os.path.join(FLAGS.log_root, "train")
    vocab_metadata_path = os.path.join(train_dir, "vocab_metadata.tsv")
    self._vocab.write_metadata(vocab_metadata_path) # write metadata file
    summary_writer = tf.summary.FileWriter(train_dir)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = vocab_metadata_path
    projector.visualize_embeddings(summary_writer, config)

  def _add_matching_layer(self, enc_states):
    context_title_states = attention_decoder(None, None, enc_states, title_encoder_states=self._title_states, title_padding_mask=self._title_padding_mask, match_layer=True)
    return context_title_states    

  def _add_seq2seq(self):
    """Add the whole sequence-to-sequence model to the graph."""
    hps = self._hps
    vsize = self._vocab.size() # size of the vocabulary

    with tf.variable_scope('seq2seq'):
      # Some initializers
      self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123)
      self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)

      # Add embedding matrix (shared by the encoder and decoder inputs)
      with tf.variable_scope('embedding'):
        embedding = tf.get_variable('embedding', [vsize, hps.emb_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
        if hps.mode=="train": self._add_emb_vis(embedding) # add to tensorboard
        emb_enc_inputs = tf.nn.embedding_lookup(embedding, self._enc_batch) # tensor with shape (batch_size, max_enc_steps, emb_size)
        emb_dec_inputs = [tf.nn.embedding_lookup(embedding, x) for x in tf.unstack(self._dec_batch, axis=1)] # list length max_dec_steps containing shape (batch_size, emb_size)
        decoder_input_ids = [x for x in tf.unstack(self._dec_batch, axis=1)] if hps.co_occurrence_h else None

        emb_dim = hps.emb_dim
        if hps.co_occurrence_i:
          emb_dim += 1
          emb_enc_inputs = tf.reshape(tf.concat([emb_enc_inputs, tf.expand_dims(self._cooccurrence_weight, 2)], 2), [hps.batch_size, -1, emb_dim])
        if hps.tagger_encoding:
          emb_dim += (len(data.TAGS_SET) + 1)
          emb_enc_inputs = tf.reshape(tf.concat([emb_enc_inputs, self._tagger_matrix], 2), [hps.batch_size, -1, emb_dim])

      # Add the encoder.
        enc_outputs, fw_st, bw_st = self._add_encoder(emb_enc_inputs, self._enc_lens)

      if hps.title_engaged or hps.title_guided:
        emb_title_inputs = tf.nn.embedding_lookup(embedding, self._title_batch)
        title_enc_outputs, _, _ = self._add_encoder(emb_title_inputs, self._title_lens, "title_encoder")
        self._title_states = title_enc_outputs

      if hps.title_guided:
        with tf.variable_scope('merging_layer'):
          context_title_states = self._add_matching_layer(enc_outputs)
          emb_mrg_inputs = tf.concat([enc_outputs, context_title_states], 2)
          enc_outputs, fw_st, bw_st = self._add_encoder(emb_mrg_inputs, self._enc_lens, "merging_layer_encoder", enc_outputs)

      self._enc_states = enc_outputs
      # Our encoder is bidirectional and our decoder is unidirectional so we need to reduce the final encoder hidden state to the right size to be the initial decoder hidden state
      self._dec_in_state = self._reduce_states(fw_st, bw_st)
      # Add the decoder.
      with tf.variable_scope('decoder'):
        decoder_outputs, self._dec_out_state, self.attn_dists, self.p_gens, self.coverage = self._add_decoder(emb_dec_inputs, decoder_input_ids, emb_enc_inputs=(emb_enc_inputs if hps.target_siding_bridge else None))

      # Add the output projection to obtain the vocabulary distribution
      with tf.variable_scope('output_projection'):
        w = tf.get_variable('w', [hps.hidden_dim, vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
        w_t = tf.transpose(w)
        v = tf.get_variable('v', [vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
        vocab_scores = [] # vocab_scores is the vocabulary distribution before applying softmax. Each entry on the list corresponds to one decoder step
        for i,output in enumerate(decoder_outputs):
          if i > 0:
            tf.get_variable_scope().reuse_variables()
          vocab_scores.append(tf.nn.xw_plus_b(output, w, v)) # apply the linear layer

        vocab_dists = [tf.nn.softmax(s) for s in vocab_scores] # The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.


      # For pointer-generator model, calc final distribution from copy distribution and vocabulary distribution
      if hps.pointer_gen:
        init_attn = None
        if hps.markov_attention:
          if hps.mode in ['train', 'eval']:
            init_attn = self._cooccurrence_weight
          else:
            init_attn = self._prev_attention_dist
        final_dists = self._calc_final_dist(vocab_dists, self.attn_dists, init_attn)
      else: # final distribution is just vocabulary distribution
        final_dists = vocab_dists


      if hps.mode in ['train', 'eval']:
        # Calculate the loss
        with tf.variable_scope('loss'):
          if hps.pointer_gen:
            # Calculate the loss per step
            # This is fiddly; we use tf.gather_nd to pick out the probabilities of the gold target words
            loss_per_step = [] # will be list length max_dec_steps containing shape (batch_size)
            batch_nums = tf.range(0, limit=hps.batch_size) # shape (batch_size)
            for dec_step, dist in enumerate(final_dists):
              targets = self._target_batch[:, dec_step] # The indices of the target words. shape (batch_size)
              indices = tf.stack((batch_nums, targets), axis=1) # shape (batch_size, 2)
              gold_probs = tf.gather_nd(dist, indices) # shape (batch_size). prob of correct words on this step
              losses = -tf.log(gold_probs)
              loss_per_step.append(losses)

            # Apply dec_padding_mask and get loss
            self._loss = _mask_and_avg(loss_per_step, self._dec_padding_mask)

          else: # baseline model
            self._loss = tf.contrib.seq2seq.sequence_loss(tf.stack(vocab_scores, axis=1), self._target_batch, self._dec_padding_mask) # this applies softmax internally

          tf.summary.scalar('loss', self._loss)

          # Calculate coverage loss from the attention distributions
          if hps.coverage:
            with tf.variable_scope('coverage_loss'):
              self._coverage_loss = _coverage_loss(self.attn_dists, self._dec_padding_mask)
              tf.summary.scalar('coverage_loss', self._coverage_loss)
            self._total_loss = self._loss + hps.cov_loss_wt * self._coverage_loss
            tf.summary.scalar('total_loss', self._total_loss)

          # if hps.prev_relation:
          #   tf.summary.scalar('p_r', self._p_r)

    if hps.mode == "decode":
      # We run decode beam search mode one decoder step at a time
      assert len(final_dists)==1 # final_dists is a singleton list containing shape (batch_size, extended_vsize)
      final_dists = final_dists[0]
      topk_probs, self._topk_ids = tf.nn.top_k(final_dists, hps.batch_size * hps.beam_depth) # take the k largest probs. note batch_size=beam_size in decode mode
      self._topk_log_probs = tf.log(topk_probs)

  def _add_train_op(self):
    """Sets self._train_op, the op to run for training."""
    # Take gradients of the trainable variables w.r.t. the loss function to minimize
    loss_to_minimize = self._total_loss if self._hps.coverage else self._loss
    tvars = tf.trainable_variables()
    gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

    # Clip the gradients
    with tf.device("/gpu:0"):
      grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)

    # Add a summary
    tf.summary.scalar('global_norm', global_norm)

    # Apply adagrad optimizer
    if self._hps.optimizer == 'Adam':
      optimizer = tf.train.AdamOptimizer(self._hps.lr)
    else:
      optimizer = tf.train.AdagradOptimizer(self._hps.lr, initial_accumulator_value=self._hps.adagrad_init_acc)
    with tf.device("/gpu:0"):
      self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')

  def build_graph(self):
    """Add the placeholders, model, global step, train_op and summaries to the graph"""
    tf.logging.info('Building graph...')
    t0 = time.time()
    self._add_placeholders()
    with tf.device("/gpu:0"):
      self._add_seq2seq()
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    if self._hps.mode == 'train':
      self._add_train_op()
    self._summaries = tf.summary.merge_all()
    t1 = time.time()
    tf.logging.info('Time to build graph: %i seconds', t1 - t0)

  def run_train_step(self, sess, batch):
    """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""
    feed_dict = self._make_feed_dict(batch)
    to_return = {
        'train_op': self._train_op,
        'summaries': self._summaries,
        'loss': self._loss,
        'global_step': self.global_step,
    }
    if self._hps.coverage:
      to_return['coverage_loss'] = self._coverage_loss
    return sess.run(to_return, feed_dict)

  def run_eval_step(self, sess, batch):
    """Runs one evaluation iteration. Returns a dictionary containing summaries, loss, global_step and (optionally) coverage loss."""
    feed_dict = self._make_feed_dict(batch)
    to_return = {
        'summaries': self._summaries,
        'loss': self._loss,
        'global_step': self.global_step,
    }
    if self._hps.coverage:
      to_return['coverage_loss'] = self._coverage_loss
    return sess.run(to_return, feed_dict)

  def run_encoder(self, sess, batch):
    """For beam search decoding. Run the encoder on the batch and return the encoder states and decoder initial state.

    Args:
      sess: Tensorflow session.
      batch: Batch object that is the same example repeated across the batch (for beam search)

    Returns:
      enc_states: The encoder states. A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim].
      dec_in_state: A LSTMStateTuple of shape ([1,hidden_dim],[1,hidden_dim])
    """
    feed_dict = self._make_feed_dict(batch, just_enc=True) # feed the batch into the placeholders
    if self._hps.title_engaged or self._hps.title_guided:
      (enc_states, dec_in_state, global_step, title_states) = sess.run([self._enc_states, self._dec_in_state, self.global_step, self._title_states], feed_dict) # run the encoder
    else:
      (enc_states, dec_in_state, global_step) = sess.run([self._enc_states, self._dec_in_state, self.global_step], feed_dict) # run the encoder
      title_states = None
    # dec_in_state is LSTMStateTuple shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
    # Given that the batch is a single example repeated, dec_in_state is identical across the batch so we just take the top row.
    if self._hps.cell_type == "GRU":
      dec_in_state = dec_in_state[0]
    else:
      dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_in_state.c[0], dec_in_state.h[0])
    return enc_states, dec_in_state, title_states

  def decode_onestep(self, sess, batch, latest_tokens, enc_states, dec_init_states, prev_coverage, prev_attn_dist=None, title_states=None):
    """For beam search decoding. Run the decoder for one step.

    Args:
      sess: Tensorflow session.
      batch: Batch object containing single example repeated across the batch
      latest_tokens: Tokens to be fed as input into the decoder for this timestep
      enc_states: The encoder states.
      dec_init_states: List of beam_size LSTMStateTuples; the decoder states from the previous timestep
      prev_coverage: List of np arrays. The coverage vectors from the previous timestep. List of None if not using coverage.

    Returns:
      ids: top 2k ids. shape [beam_size, 2*beam_size]
      probs: top 2k log probabilities. shape [beam_size, 2*beam_size]
      new_states: new states of the decoder. a list length beam_size containing
        LSTMStateTuples each of shape ([hidden_dim,],[hidden_dim,])
      attn_dists: List length beam_size containing lists length attn_length.
      p_gens: Generation probabilities for this step. A list length beam_size. List of None if in baseline mode.
      new_coverage: Coverage vectors for this step. A list of arrays. List of None if coverage is not turned on.
    """

    hps = self._hps
    beam_size = len(dec_init_states)

    if hps.cell_type == "GRU":
      new_dec_in_state = np.array(dec_init_states)
    else:
      # Turn dec_init_states (a list of LSTMStateTuples) into a single LSTMStateTuple for the batch
      cells = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
      hiddens = [np.expand_dims(state.h, axis=0) for state in dec_init_states]
      new_c = np.concatenate(cells, axis=0)  # shape [batch_size,hidden_dim]
      new_h = np.concatenate(hiddens, axis=0)  # shape [batch_size,hidden_dim]
      new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

    feed = {
        self._enc_states: enc_states,
        self._enc_padding_mask: batch.enc_padding_mask,
        self._dec_in_state: new_dec_in_state,
        self._dec_batch: np.transpose(np.array([latest_tokens]))
    }

    to_return = {
      "ids": self._topk_ids,
      "probs": self._topk_log_probs,
      "states": self._dec_out_state,
      "attn_dists": self.attn_dists
    }

    if hps.title_engaged or hps.title_guided:
      feed[self._title_padding_mask] = batch.title_padding_mask
      feed[self._title_states] = title_states

    if hps.pointer_gen:
      feed[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
      feed[self._max_art_oovs] = batch.max_art_oovs
      to_return['p_gens'] = self.p_gens
      if hps.co_occurrence or hps.prev_relation or hps.co_occurrence_h or hps.markov_attention or hps.markov_attention_contribution:
        feed[self._cooccurrence_matrix] = batch.cooccurrence_matrix
      if hps.co_occurrence_i or (hps.coverage and hps.coverage_weighted) or hps.attention_weighted or hps.markov_attention or hps.markov_attention_contribution:
        feed[self._cooccurrence_weight] = batch.cooccurrence_weight
      if hps.markov_attention or hps.markov_attention_contribution:
        feed[self._prev_attention_dist] = prev_attn_dist
      if hps.tagger_attention or hps.tagger_encoding:
        feed[self._tagger_matrix] = batch.tagger_matrix
    if self._hps.coverage:
      feed[self.prev_coverage] = np.stack(prev_coverage, axis=0)
      to_return['coverage'] = self.coverage
    if hps.target_siding_bridge:
      feed[self._enc_batch] = batch.enc_batch

    results = sess.run(to_return, feed_dict=feed) # run the decoder step

    if hps.cell_type == "GRU":
      new_states = [results['states'][i] for i in range(beam_size)]
    else:
      # Convert results['states'] (a single LSTMStateTuple) into a list of LSTMStateTuple -- one for each hypothesis
      new_states = [tf.contrib.rnn.LSTMStateTuple(results['states'].c[i, :], results['states'].h[i, :]) for i in range(beam_size)]

    # Convert singleton list containing a tensor to a list of k arrays
    assert len(results['attn_dists'])==1
    attn_dists = results['attn_dists'][0].tolist()

    if FLAGS.pointer_gen:
      # Convert singleton list containing a tensor to a list of k arrays
      assert len(results['p_gens'])==1
      p_gens = results['p_gens'][0].tolist()
    else:
      p_gens = [None for _ in range(beam_size)]

    # Convert the coverage tensor to a list length k containing the coverage vector for each hypothesis
    if FLAGS.coverage:
      new_coverage = results['coverage'].tolist()
      assert len(new_coverage) == beam_size
    else:
      new_coverage = [None for _ in range(beam_size)]

    return results['ids'], results['probs'], new_states, attn_dists, p_gens, new_coverage


def _mask_and_avg(values, padding_mask):
  """Applies mask to values then returns overall average (a scalar)

  Args:
    values: a list length max_dec_steps containing arrays shape (batch_size).
    padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.

  Returns:
    a scalar
  """

  dec_lens = tf.reduce_sum(padding_mask, axis=1) # shape batch_size. float32
  dec_lens = [tf.maximum(dec_lens[i], 1.0) for i in range(FLAGS.batch_size)]
  values_per_step = [v * padding_mask[:, dec_step] for dec_step, v in enumerate(values)]
  values_per_ex = sum(values_per_step) / dec_lens # shape (batch_size); normalized value for each batch member
  return tf.reduce_mean(values_per_ex) # overall average


def _coverage_loss(attn_dists, padding_mask):
  """Calculates the coverage loss from the attention distributions.

  Args:
    attn_dists: The attention distributions for each decoder timestep. A list length max_dec_steps containing shape (batch_size, attn_length)
    padding_mask: shape (batch_size, max_dec_steps).

  Returns:
    coverage_loss: scalar
  """
  coverage = tf.zeros_like(attn_dists[0]) # shape (batch_size, attn_length). Initial coverage is zero.
  covlosses = [] # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
  for a in attn_dists:
    covloss = tf.reduce_sum(tf.minimum(a, coverage), [1]) # calculate the coverage loss for this step
    covlosses.append(covloss)
    coverage += a # update the coverage vector
  coverage_loss = _mask_and_avg(covlosses, padding_mask)
  return coverage_loss
