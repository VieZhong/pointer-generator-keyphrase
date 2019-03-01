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

"""This file defines the decoder"""

import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
import data

FLAGS = tf.app.flags.FLAGS

# Note: this function is based on tf.contrib.legacy_seq2seq_attention_decoder, which is now outdated.
# In the future, it would make more sense to write variants on the attention mechanism using the new seq2seq library for tensorflow 1.0: https://www.tensorflow.org/api_guides/python/contrib.seq2seq#Attention
def attention_decoder(decoder_inputs, initial_state, encoder_states, enc_padding_mask=None, cell=None, initial_state_attention=False, pointer_gen=True, use_coverage=False, prev_coverage=None, matrix=None, enc_batch_extend_vocab=None, decoder_input_ids=None, attention_weight=None, emb_enc_inputs=None, prev_attention_dist=None, tagger_matrix=None, title_encoder_states=None, title_padding_mask=None, match_layer=False):
  """
  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor [batch_size x cell.state_size].
    encoder_states: 3D Tensor [batch_size x attn_length x attn_size].
    enc_padding_mask: 2D Tensor [batch_size x attn_length] containing 1s and 0s; indicates which of the encoder locations are padding (0) or a real token (1).
    cell: rnn_cell.RNNCell defining the cell function and size.
    initial_state_attention:
      Note that this attention decoder passes each decoder input through a linear layer with the previous step's context vector to get a modified version of the input. If initial_state_attention is False, on the first decoder step the "previous context vector" is just a zero vector. If initial_state_attention is True, we use initial_state to (re)calculate the previous step's context vector. We set this to False for train/eval mode (because we call attention_decoder once for all decoder steps) and True for decode mode (because we call attention_decoder once for each decoder step).
    pointer_gen: boolean. If True, calculate the generation probability p_gen for each decoder step.
    use_coverage: boolean. If True, use coverage mechanism.
    prev_coverage:
      If not None, a tensor with shape (batch_size, attn_length). The previous step's coverage vector. This is only not None in decode mode when using coverage.
    title_encoder_states: batch_size x title_attn_length x title_attn_size
  Returns:
    outputs: A list of the same length as decoder_inputs of 2D Tensors of
      shape [batch_size x cell.output_size]. The output vectors.
    state: The final state of the decoder. A tensor shape [batch_size x cell.state_size].
    attn_dists: A list containing tensors of shape (batch_size,attn_length).
      The attention distributions for each decoder step.
    p_gens: List of scalars. The values of p_gen for each decoder step. Empty list if pointer_gen=False.
    coverage: Coverage vector on the last step computed. None if use_coverage=False.
  """
  with variable_scope.variable_scope("attention_decoder", reuse=tf.AUTO_REUSE) as scope:
    batch_size = encoder_states.get_shape()[0].value # if this line fails, it's because the batch size isn't defined
    attn_size = encoder_states.get_shape()[2].value # if this line fails, it's because the attention length isn't defined

    # To calculate attention, we calculate
    #   v^T tanh(W_h h_i + W_s s_t + b_attn)
    # where h_i is an encoder state, and s_t a decoder state.
    # attn_vec_size is the length of the vectors v, b_attn, (W_h h_i) and (W_s s_t).
    # We set it to be equal to the size of the encoder states.
    attention_vec_size = attn_size

    # Get the weight matrix W_h and apply it to each encoder state to get (W_h h_i), the encoder features
    W_h = variable_scope.get_variable("W_h", [1, 1, attn_size, attention_vec_size])
    # Reshape encoder_states (need to insert a dim)
    encoder_states = tf.expand_dims(encoder_states, axis=2) # now is shape (batch_size, attn_len, 1, attn_size)

    if FLAGS.co_occurrence_h or FLAGS.markov_attention_contribution or FLAGS.coverage_weighted_expansion or FLAGS.title_engaged or FLAGS.title_guided:
      attn_len = tf.shape(encoder_states)[1]

    if FLAGS.title_engaged or (FLAGS.title_guided and match_layer):
      title_attn_size = title_encoder_states.get_shape()[2].value
      title_attn_len = tf.shape(title_encoder_states)[1]
      W_t_c = variable_scope.get_variable("W_t_c", [1, 1, attn_size, title_attn_size])
      
      score = [] # batch_size x attn_length x title_attn_length
      score_matrix = nn_ops.conv2d(encoder_states, W_t_c, [1, 1, 1, 1], "SAME") # batch_size x attn_length x 1 x title_attn_size
      score_matrix = tf.reshape(score_matrix, [-1, attn_len, title_attn_size]) # batch_size x attn_length x title_attn_size
      for batch_index in range(batch_size):
        score.append(math_ops.reduce_sum(tf.multiply(tf.tile(tf.expand_dims(score_matrix[batch_index], 1), [1, title_attn_len, 1]), title_encoder_states[batch_index]), -1))      

      title_attn_dist = nn_ops.softmax(score) # take softmax. shape (batch_size, attn_length, title_attn_length)
      title_attn_dist *= tf.tile(tf.expand_dims(title_padding_mask, 1), [1, attn_len, 1]) # apply mask
      title_masked_sums = tf.reduce_sum(title_attn_dist, axis=2)
      title_attn_dist =  title_attn_dist / tf.reshape(title_masked_sums, [-1, attn_len, 1]) # re-normalize

      context_title_states =[] # batch_size x attn_length x title_attn_size
      for batch_index in range(batch_size):
        # transpose_title_encoder_state = tf.transpose(title_encoder_states[batch_index])
        # context_title_state = tf.map_fn(lambda x: x * transpose_title_encoder_state, title_attn_dist[batch_index]) # attn_length x title_attn_size x title_attn_length
        # context_title_states.append(math_ops.reduce_sum(context_title_state, -1)) # attn_length x title_attn_size
        context_title_state = tf.tile(tf.expand_dims(title_attn_dist[batch_index], -1), [1, 1, title_attn_size]) * title_encoder_states[batch_index] # attn_length x title_attn_length x title_attn_size
        context_title_states.append(math_ops.reduce_sum(context_title_state, 1)) # attn_length x title_attn_size
      context_title_states = tf.expand_dims(context_title_states, axis=2) # batch_size x attn_length x 1 x title_attn_size
    
    if FLAGS.title_guided and match_layer:
      return tf.reshape(context_title_states, [batch_size, -1, title_attn_size])
    if FLAGS.title_engaged:
      W_e = variable_scope.get_variable("W_e", [1, 1, title_attn_size, attention_vec_size])
      title_features = nn_ops.conv2d(context_title_states, W_e, [1, 1, 1, 1], "SAME")

    if FLAGS.co_occurrence_h or FLAGS.markov_attention_contribution:
      co_matrix = tf.slice(matrix, [0, 0, 0], [-1, attn_len, attn_len]) # shape (batch_size, attn_length, attn_length).

    if FLAGS.attention_weighted and attention_weight is not None:
      attn_weight = tf.tile(tf.expand_dims(tf.expand_dims(attention_weight, 2), 3), [1, 1, 1, attn_size])
      weighted_encoder_states = attn_weight * encoder_states
    else:
      weighted_encoder_states = encoder_states
    encoder_features = nn_ops.conv2d(weighted_encoder_states, W_h, [1, 1, 1, 1], "SAME") # shape (batch_size,attn_length,1,attention_vec_size)

    if FLAGS.co_occurrence:
      c_matrix = tf.expand_dims(matrix, axis=2) # now is shape (batch_size, max_enc_steps, 1, max_enc_steps)
      W_p = variable_scope.get_variable("W_p", [1, 1, FLAGS.max_enc_steps, attention_vec_size])
      matrix_features = nn_ops.conv2d(c_matrix, W_p, [1, 1, 1, 1], "SAME")
      matrix_features = tf.slice(matrix_features, [0, 0, 0, 0], [-1, tf.shape(enc_padding_mask)[1], -1, -1])

    if FLAGS.tagger_attention:
      tag_matrix = tf.expand_dims(tagger_matrix, axis=2) # now is shape (batch_size, max_enc_steps, 1, tag_size)
      W_t = variable_scope.get_variable("W_t", [1, 1, len(data.TAGS_SET) + 1, attention_vec_size])
      tag_features = nn_ops.conv2d(tag_matrix, W_t, [1, 1, 1, 1], "SAME")

    # Get the weight vectors v and w_c (w_c is for coverage)
    v = variable_scope.get_variable("v", [attention_vec_size])
    if use_coverage:
      with variable_scope.variable_scope("coverage"):
        w_c = variable_scope.get_variable("w_c", [1, 1, 1, attention_vec_size])

    if prev_coverage is not None: # for beam search mode with coverage
      # reshape from (batch_size, attn_length) to (batch_size, attn_len, 1, 1)
      prev_coverage = tf.expand_dims(tf.expand_dims(prev_coverage, 2), 3)


    def attention(decoder_state, coverage=None, input_ids=None):
      """Calculate the context vector and attention distribution from the decoder state.

      Args:
        decoder_state: state of the decoder
        coverage: Optional. Previous timestep's coverage vector, shape (batch_size, attn_len, 1, 1).

      Returns:
        context_vector: weighted sum of encoder_states
        attn_dist: attention distribution
        coverage: new coverage vector. shape (batch_size, attn_len, 1, 1)
      """
      with variable_scope.variable_scope("Attention"):
        # Pass the decoder state through a linear layer (this is W_s s_t + b_attn in the paper)
        decoder_features = linear(decoder_state, attention_vec_size, True) # shape (batch_size, attention_vec_size)
        decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1) # reshape to (batch_size, 1, 1, attention_vec_size)

        def masked_attention(e):
          """Take softmax of e then apply enc_padding_mask and re-normalize"""
          attn_dist = nn_ops.softmax(e) # take softmax. shape (batch_size, attn_length)
          attn_dist *= enc_padding_mask # apply mask
          masked_sums = tf.reduce_sum(attn_dist, axis=1) # shape (batch_size)
          return attn_dist / tf.reshape(masked_sums, [-1, 1]) # re-normalize

        if use_coverage and coverage is not None: # non-first step of coverage

          # Multiply coverage vector by w_c to get coverage_features.
          if FLAGS.coverage_weighted and attention_weight is not None:
            co_weight = tf.expand_dims(tf.expand_dims(attention_weight, 2), 3)
            weighted_coverage = co_weight * coverage
            if FLAGS.coverage_weighted_expansion:
              weighted_coverage *= tf.to_float(attn_len)
          else:
            weighted_coverage = coverage
          coverage_features = nn_ops.conv2d(weighted_coverage, w_c, [1, 1, 1, 1], "SAME") # c has shape (batch_size, attn_length, 1, attention_vec_size)

          sum_features = encoder_features + decoder_features + coverage_features
          if FLAGS.co_occurrence:
            sum_features += matrix_features
          if FLAGS.tagger_attention:
            sum_features += tag_features
          if FLAGS.title_engaged:
            sum_features += title_features

          # Calculate v^T tanh(W_h h_i + W_s s_t + w_c c_i^t + b_attn)
          e = math_ops.reduce_sum(v * math_ops.tanh(sum_features), [2, 3])  # shape (batch_size,attn_length)

          # Calculate attention distribution
          attn_dist = masked_attention(e)

          # Update coverage vector
          coverage += array_ops.reshape(attn_dist, [batch_size, -1, 1, 1])
        else:
          # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
          sum_features = encoder_features + decoder_features
          if FLAGS.co_occurrence:
            sum_features += matrix_features
          if FLAGS.tagger_attention:
            sum_features += tag_features
          if FLAGS.title_engaged:
            sum_features += title_features
          e = math_ops.reduce_sum(v * math_ops.tanh(sum_features), [2, 3]) # calculate e

          # Calculate attention distribution
          attn_dist = masked_attention(e)

          if use_coverage: # first step of training
            coverage = tf.expand_dims(tf.expand_dims(attn_dist, 2), 2) # initialize coverage

        # Calculate the context vector from attn_dist and encoder_states
        if FLAGS.co_occurrence_h and input_ids is not None: 
          p_oc = tf.get_variable("p_oc", [1], initializer=tf.constant_initializer(0.2))
          p_dist = []
          for j in range(FLAGS.batch_size):
            m = co_matrix[j]
            vb = enc_batch_extend_vocab[j]
            xj = input_ids[j]
            t = tf.where(tf.equal(vb, xj))
            d = tf.cond(tf.shape(t)[0] > 0, lambda: m[tf.to_int32(t[0][0])], lambda: tf.zeros([attn_len]))
            p_dist.append(d)
          context_vector = math_ops.reduce_sum(array_ops.reshape((p_oc *  p_dist + (1 - p_oc) * attn_dist), [batch_size, -1, 1, 1]) * encoder_states, [1, 2]) # shape (batch_size, attn_size).
        else:
          context_vector = math_ops.reduce_sum(array_ops.reshape(attn_dist, [batch_size, -1, 1, 1]) * encoder_states, [1, 2]) # shape (batch_size, attn_size).
        context_vector = array_ops.reshape(context_vector, [-1, attn_size])

      return context_vector, attn_dist, coverage

    outputs = []
    attn_dists = []
    p_gens = []
    state = initial_state
    coverage = prev_coverage # initialize coverage to None or whatever was passed in
    context_vector = array_ops.zeros([batch_size, attn_size])
    context_vector.set_shape([None, attn_size])  # Ensure the second shape of attention vectors is set.
    if initial_state_attention: # true in decode mode
      # Re-calculate the context vector from the previous step so that we can pass it through a linear layer with this step's input to get a modified version of the input
      context_vector, _, coverage = attention(initial_state, coverage, decoder_input_ids[0] if FLAGS.co_occurrence_h else None) # in decode mode, this is what updates the coverage vector
    for i, inp in enumerate(decoder_inputs):
      tf.logging.info("Adding attention_decoder timestep %i of %i", i, len(decoder_inputs))
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()

      # Merge input and previous attentions into one vector x of the same size as inp
      input_size = inp.get_shape().with_rank(2)[1]
      if input_size.value is None:
        raise ValueError("Could not infer input size from input: %s" % inp.name)
      x = linear([inp] + [context_vector], input_size, True)

      # Run the decoder RNN cell. cell_output = decoder state
      cell_output, state = cell(x, state)

      # Run the attention mechanism.
      if i == 0 and initial_state_attention:  # always true in decode mode
        with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=True): # you need this because you've already run the initial attention(...) call
          context_vector, attn_dist, _ = attention(state, coverage, decoder_input_ids[i] if FLAGS.co_occurrence_h else None) # don't allow coverage to update
      else:
        if not FLAGS.co_occurrence_h:
          context_vector, attn_dist, coverage = attention(state, coverage)
        else:
          context_vector, attn_dist, coverage = attention(state, coverage, decoder_input_ids[i])
      if FLAGS.markov_attention_contribution:
        if prev_attention_dist is not None:
          prev_attn = prev_attention_dist
        else:
          prev_attn = attention_weight if i == 0 else attn_dists[i - 1]
        prev_attn_dist = tf.tile(tf.expand_dims(prev_attn, 1), [1, attn_len, 1])
        mark_dist = tf.reduce_sum(tf.multiply(tf.matrix_transpose(co_matrix), prev_attn_dist), 2)
        if FLAGS.markov_attention_contribution_used_x: 
          p_attn = tf.sigmoid(linear([state.c, state.h, x], 1, True, scope="markov_attn"))
        else:
          p_attn = tf.sigmoid(linear([state.c, state.h], 1, True, scope="markov_attn"))
        attn_dist = p_attn * attn_dist + (1 - p_attn) * mark_dist
      attn_dists.append(attn_dist)

      # Calculate p_gen
      if pointer_gen:
        with tf.variable_scope('calculate_pgen'):
          if FLAGS.cell_type == "GRU":
            p_gen = linear([context_vector, state, x], 1, True) # a scalar
          else:
            p_gen = linear([context_vector, state.c, state.h, x], 1, True) # a scalar
          p_gen = tf.sigmoid(p_gen)
          p_gens.append(p_gen)

      # Concatenate the cell_output (= decoder state) and the context vector, and pass them through a linear layer
      # This is V[s_t, h*_t] + b in the paper
      with variable_scope.variable_scope("AttnOutputProjection"):
        if FLAGS.target_siding_bridge:
          max_attn_index = tf.expand_dims(tf.argmax(attn_dist, 1, output_type=tf.int32), 1)
          batch_nums = tf.expand_dims(tf.range(0, limit=batch_size), 1) # shape (batch_size, 1)
          max_attn_index = tf.stack((batch_nums, max_attn_index), axis=2)
          max_prop_input = tf.reshape(tf.gather_nd(emb_enc_inputs, max_attn_index), [-1, FLAGS.emb_dim if not FLAGS.co_occurrence_i else FLAGS.emb_dim + 1])
          output = linear([cell_output] + [context_vector] + [max_prop_input], cell.output_size, True)
        else:
          output = linear([cell_output] + [context_vector], cell.output_size, True)
      outputs.append(output)

    # If using coverage, reshape it
    if coverage is not None:
      coverage = array_ops.reshape(coverage, [batch_size, -1])

    return outputs, state, attn_dists, p_gens, coverage



def linear(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (isinstance(args, (list, tuple)) and not args):
    raise ValueError("`args` must be specified")
  if not isinstance(args, (list, tuple)):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  # Now the computation.
  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
    if len(args) == 1:
      res = tf.matmul(args[0], matrix)
    else:
      res = tf.matmul(tf.concat(axis=1, values=args), matrix)
    if not bias:
      return res
    bias_term = tf.get_variable(
        "Bias", [output_size], initializer=tf.constant_initializer(bias_start))
  return res + bias_term
