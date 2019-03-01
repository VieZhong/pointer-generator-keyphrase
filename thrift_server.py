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

"""This is the top-level file to train, evaluate or test your summarization model"""
# coding=utf-8
import sys
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import numpy as np
from collections import namedtuple
from data import Vocab, get_stop_word_ids
from batcher import Batcher
from model import SummarizationModel
from decode import BeamSearchDecoder
import util
from tensorflow.python import debug as tf_debug

FLAGS = tf.app.flags.FLAGS

# Where to find data
# tf.app.flags.DEFINE_string('data_path', '/tmp/test/example.txt', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('vocab_path', '/project/data/test_for_generator_keyphrase/finished_files/vocab', 'Path expression to text vocabulary file.')
tf.app.flags.DEFINE_string('stop_words_path', '/project/data/stopword/stopword_en.txt', 'Path expression to stop words file')
tf.app.flags.DEFINE_string('ref_dir', '/data/__DATASET__/val_reference/', 'Path to reference words')

# Important settings
tf.app.flags.DEFINE_string('mode', 'decode', 'must be one of train/eval/decode')
tf.app.flags.DEFINE_boolean('single_pass', True, 'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')
tf.app.flags.DEFINE_boolean('decode_only', True, 'If True, only decode, do not calculate f1 score. only for chinese, only for special format data_path')

# Where to save output
tf.app.flags.DEFINE_string('log_root', '/tmp/test-pointer-generater/log/', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('exp_name', 'test_test_thrift_experiment', 'Name for experiment. Logs will be saved in a directory with this name, under log_root.')
tf.app.flags.DEFINE_string('language', 'english', 'language')

# Encoder and decoder settings
tf.app.flags.DEFINE_string('cell_type', 'LSTM', 'LSTM or GRU')
tf.app.flags.DEFINE_float('dropout', 0.0, 'for dropout')

# Hyperparameters
tf.app.flags.DEFINE_integer('hidden_dim', 256, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')
tf.app.flags.DEFINE_integer('batch_size', 16, 'minibatch size')
tf.app.flags.DEFINE_integer('max_enc_steps', 400, 'max timesteps of encoder (max source text tokens)')
tf.app.flags.DEFINE_integer('max_dec_steps', 8, 'max timesteps of decoder (max summary tokens)')
tf.app.flags.DEFINE_integer('beam_size', 200, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('beam_depth', 6, 'beam depth for beam search decoding')
tf.app.flags.DEFINE_integer('min_dec_steps', 1, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
tf.app.flags.DEFINE_integer('vocab_size', 50000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
tf.app.flags.DEFINE_integer('max_keyphrase_num', 10, 'max keyphrase number')
tf.app.flags.DEFINE_integer('max_title_len', 20, 'max title length')
tf.app.flags.DEFINE_integer('occurrence_window_size', 3, 'window size while calculating occurrence matrix')
tf.app.flags.DEFINE_float('lr', 0.15, 'learning rate')
tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
tf.app.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')
tf.app.flags.DEFINE_string('optimizer', 'Adagrad', 'Adagrad or Adam')

# Pointer-generator or baseline model
tf.app.flags.DEFINE_boolean('pointer_gen', True, 'If True, use pointer-generator model. If False, use baseline model.')

# Coverage hyperparameters
tf.app.flags.DEFINE_boolean('coverage', True, 'Use coverage mechanism. Note, the experiments reported in the ACL paper train WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards. i.e. to reproduce the results in the ACL paper, turn this off for most of training then turn on for a short phase at the end.')
tf.app.flags.DEFINE_float('cov_loss_wt', 1.0, 'Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.')

# VieZhong Improvement Hyperparameters
tf.app.flags.DEFINE_boolean('attention_weighted', False, 'Whether attention mechanism is weighted.')
tf.app.flags.DEFINE_boolean('coverage_weighted', False, 'Whether coverage mechanism is weighted.')
tf.app.flags.DEFINE_boolean('coverage_weighted_expansion', False, 'Whether coverage mechanism is weighted.')
tf.app.flags.DEFINE_boolean('co_occurrence', False, 'Whether to use co_occurrence factor.')
tf.app.flags.DEFINE_boolean('co_occurrence_h', True, 'Whether to use co_occurrence_h factor.')
tf.app.flags.DEFINE_boolean('co_occurrence_i', False, 'Whether to concat co_occurrence matrix to encoder embeddings.')
tf.app.flags.DEFINE_boolean('prev_relation', False, 'Whether to use the previous output word to predict the next output word.')
tf.app.flags.DEFINE_boolean('source_siding_bridge', False, 'Whether to use source siding bridging model.')
tf.app.flags.DEFINE_boolean('target_siding_bridge', False, 'Whether to use target siding bridging model.')
tf.app.flags.DEFINE_boolean('markov_attention', False, 'Whether to use markov attention mechanism.')
tf.app.flags.DEFINE_boolean('markov_attention_contribution', False, 'Whether to use markov attention contribution mechanism.')
tf.app.flags.DEFINE_boolean('markov_attention_contribution_used_x', False, 'Whether to use markov attention contribution mechanism.')
tf.app.flags.DEFINE_boolean('tagger_attention', False, 'Whether to use tagger_attention factor')
tf.app.flags.DEFINE_boolean('tagger_encoding', False, 'Whether to use tagger_attention factor')
tf.app.flags.DEFINE_boolean('title_engaged', False, 'Whether to use title_engaged factor')
tf.app.flags.DEFINE_boolean('title_guided', False, 'Whether to use title_guided factor')

tf.app.flags.DEFINE_boolean('generation_only', False, 'Whether in generation mode only')
tf.app.flags.DEFINE_boolean('copy_only', False, 'Whether in copy mode only')

# Utility flags, for restoring and changing checkpoints
tf.app.flags.DEFINE_boolean('convert_to_coverage_model', False, 'Convert a non-coverage model to a coverage model. Turn this on and run in train mode. Your current training model will be copied to a new version (same name with _cov_init appended) that will be ready to run with coverage flag turned on, for the coverage training stage.')
tf.app.flags.DEFINE_boolean('restore_best_model', False, 'Restore the best model in the eval/ dir and save it in the train/ dir, ready to be used for further training. Useful for early stopping, or if your training checkpoint has become corrupted with e.g. NaN values.')

# Debugging. See https://www.tensorflow.org/programmers_guide/debugger
tf.app.flags.DEFINE_boolean('debug', False, "Run in tensorflow's debug mode (watches for NaN/inf values)")



def main(articles):

  tf.logging.set_verbosity(tf.logging.INFO) # choose what level of logging you want
  tf.logging.info('Starting seq2seq_attention in %s mode...', (FLAGS.mode))

  # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
  FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)
  if not os.path.exists(FLAGS.log_root):
    if FLAGS.mode=="train":
      os.makedirs(FLAGS.log_root)
    else:
      raise Exception("Logdir %s doesn't exist. Run in train mode to create it." % (FLAGS.log_root))

  vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size) # create a vocabulary

  stop_word_ids = get_stop_word_ids(FLAGS.stop_words_path, vocab) if FLAGS.pointer_gen and (FLAGS.co_occurrence or FLAGS.prev_relation or FLAGS.co_occurrence_h or FLAGS.co_occurrence_i or (FLAGS.coverage and FLAGS.coverage_weighted)) or FLAGS.attention_weighted or FLAGS.markov_attention or FLAGS.markov_attention_contribution else None

  # If in decode mode, set batch_size = beam_size
  # Reason: in decode mode, we decode one example at a time.
  # On each step, we have beam_size-many hypotheses in the beam, so we need to make a batch of these hypotheses.
  if FLAGS.mode == 'decode':
    FLAGS.batch_size = FLAGS.beam_size

  # If single_pass=True, check we're in decode mode
  if FLAGS.single_pass and FLAGS.mode!='decode':
    raise Exception("The single_pass flag should only be True in decode mode")

  # if FLAGS.prev_relation and not FLAGS.co_occurrence:
  #   raise Exception("The co_occurrence flag should be True when the prev_relation flag is True")

  # Make a namedtuple hps, containing the values of the hyperparameters that the model needs
  hparam_list = ['decode_only', 'generation_only', 'copy_only', 'occurrence_window_size', 'max_title_len', 'title_engaged', 'title_guided', 'ref_dir', 'tagger_encoding', 'tagger_attention', 'source_siding_bridge', 'target_siding_bridge', 'language', 'dropout', 'optimizer', 'mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm', 'hidden_dim', 'emb_dim', 'batch_size', 'beam_depth', 'max_dec_steps', 'max_enc_steps', 'max_keyphrase_num', 'attention_weighted', 'coverage', 'coverage_weighted', 'coverage_weighted_expansion', 'co_occurrence', 'prev_relation', 'co_occurrence_h', 'co_occurrence_i', 'cov_loss_wt', 'pointer_gen', 'cell_type', 'markov_attention', 'markov_attention_contribution', 'markov_attention_contribution_used_x']
  hps_dict = {}
  for key,val in FLAGS.__flags.items(): # for each flag
    if key in hparam_list: # if it's in the list
      hps_dict[key] = val # add it to the dict
  hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)

  sess = tf.Session(config=util.get_config())
  sess.run(tf.global_variables_initializer())

  # Create a batcher object that will create minibatches of data
  # batcher = Batcher(FLAGS.data_path, vocab, hps, single_pass=FLAGS.single_pass, stop_words=stop_word_ids)
  batcher = Batcher(articles, vocab, hps, single_pass=FLAGS.single_pass, stop_words=stop_word_ids)

  tf.set_random_seed(111) # a seed value for randomness

  if hps.mode == 'decode':
    decode_model_hps = hps  # This will be the hyperparameters for the decoder model
    decode_model_hps = hps._replace(max_dec_steps=1) # The model is configured with max_dec_steps=1 because we only ever run one step of the decoder at a time (to do beam search). Note that the batcher is initialized with max_dec_steps equal to e.g. 100 because the batches need to contain the full summaries
    model = SummarizationModel(decode_model_hps, vocab)
    decoder = BeamSearchDecoder(model, batcher, vocab)
    return decoder.decode() # decode indefinitely (unless single_pass=True, in which case deocde the dataset exactly once)
  else:
    raise ValueError("The 'mode' flag must be one of train/eval/decode")



from keyphrase import KeyphraseModel
from keyphrase import ttypes
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

from tf_app_run import run


__HOST = 'localhost'
__PORT = 8080

class KeyphrasesHandler(object):
  def predict(self, articles):
    article_list = [{"id": a.id, "title": a.title, "text": a.text} for a in articles]
    decode_results = run(argv=[article_list])
    return [ttypes.Keyphrase(r["id"], r["keyphrases"]) for r in decode_results]


if __name__ == '__main__':
  handler = KeyphrasesHandler()

  processor = KeyphraseModel.Processor(handler)
  transport = TSocket.TServerSocket(__HOST, __PORT)
  tfactory = TTransport.TBufferedTransportFactory()
  pfactory = TBinaryProtocol.TBinaryProtocolFactory()

  rpcServer = TServer.TSimpleServer(processor,transport, tfactory, pfactory)

  print('Starting the rpc server at', __HOST,':', __PORT)
  rpcServer.serve()