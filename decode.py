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

"""This file contains code to run beam search decoding, including running F1-SCORE evaluation and producing JSON datafiles for the in-browser attention visualizer, which can be found here https://github.com/abisee/attn_vis"""

import os
import time
import tensorflow as tf
import beam_search
import data
import json
import util
import logging
import numpy as np
import Stemmer
import hashlib

FLAGS = tf.app.flags.FLAGS

SECS_UNTIL_NEW_CKPT = 60  # max number of seconds before loading new checkpoint
DECODE_MAX_NUM = 120


class BeamSearchDecoder(object):
  """Beam search decoder."""

  def __init__(self, model, batcher, vocab):
    """Initialize decoder.

    Args:
      model: a Seq2SeqAttentionModel object.
      batcher: a Batcher object.
      vocab: Vocabulary object
    """
    self._model = model
    self._model.build_graph()
    self._batcher = batcher
    self._vocab = vocab
    self._saver = tf.train.Saver() # we use this to load checkpoints for decoding
    self._sess = tf.Session(config=util.get_config())

    # Load an initial checkpoint to use for decoding
    ckpt_path = util.load_ckpt(self._saver, self._sess)

    if not FLAGS.decode_only:
      if FLAGS.single_pass:
        # Make a descriptive decode directory name
        ckpt_name = "ckpt-" + ckpt_path.split('-')[-1] # this is something of the form "ckpt-123456"
        self._decode_dir = os.path.join(FLAGS.log_root, get_decode_dir_name(ckpt_name))
        if os.path.exists(self._decode_dir):
          raise Exception("single_pass decode directory %s should not already exist" % self._decode_dir)

      else: # Generic decode dir name
        self._decode_dir = os.path.join(FLAGS.log_root, "decode")

      # Make the decode dir if necessary
      if not os.path.exists(self._decode_dir): os.mkdir(self._decode_dir)

      if FLAGS.single_pass:
        # Make the dirs to contain output written in the correct format for pyrouge
        self._rouge_ref_dir = FLAGS.ref_dir.replace('__DATASET__', 'kp20k' if FLAGS.language == 'english' else 'nssd_data')
        if not os.path.exists(self._rouge_ref_dir): os.mkdir(self._rouge_ref_dir)
        self._rouge_dec_dir = os.path.join(self._decode_dir, "decoded")
        if not os.path.exists(self._rouge_dec_dir): os.mkdir(self._rouge_dec_dir)

    else:
      self._decode_dir = os.path.dirname(FLAGS.data_path)


  def decode(self):
    """Decode examples until data is exhausted (if FLAGS.single_pass) and return, or decode indefinitely, loading latest checkpoint at regular intervals"""
    t0 = time.time()
    counter = 0
    hashList = []
    decode_result = []
    while True:
      batch = self._batcher.next_batch()  # 1 example repeated across batch
      if batch is None: # finished decoding dataset in single_pass mode
        assert FLAGS.single_pass, "Dataset exhausted, but we are not in single_pass mode"
        tf.logging.info("Decoder has finished reading dataset for single_pass.")
        if not FLAGS.decode_only:
          tf.logging.info("Output has been saved in %s and %s. Now starting F1_SCORE eval...", self._rouge_ref_dir, self._rouge_dec_dir)
          f1_score = f1_score_eval(self._rouge_ref_dir, self._rouge_dec_dir)
          f1_score_log(f1_score, self._decode_dir)
        else:
          write_decode_result_in_file(decode_result, self._decode_dir)
        return

      original_article = batch.original_articles[0]  # string
      original_abstract = batch.original_abstracts[0]  # string
      original_abstract_sents = batch.original_abstracts_sents_all[0]  # list of strings

      art_hash = hashhex(original_article)
      if art_hash in hashList:
        continue
      hashList.append(art_hash)

      article_withunks = data.show_art_oovs(original_article, self._vocab) # string
      abstract_withunks = data.show_abs_oovs(original_abstract, self._vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None)) # string

      # Run beam search to get best Hypothesis
      all_hyp = beam_search.run_beam_search(self._sess, self._model, self._vocab, batch)
      best_hyp = all_hyp[0]

      decoded_words = []
      for hyp in all_hyp:
        if len(decoded_words) < DECODE_MAX_NUM:
          # Extract the output ids from the hypothesis and convert back to words
          output_ids = [int(t) for t in hyp.tokens[1:]]
          decoded_words_1 = data.outputids2words(output_ids, self._vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None))

          # Remove the [STOP] token from decoded_words, if necessary
          while len(decoded_words_1) and decoded_words_1[0] in [',', '.', '-lrb-', data.STOP_DECODING]:
            decoded_words_1 = decoded_words_1[1:]
          for symbol in [',', '.', '-lrb-', data.STOP_DECODING]:
            try:
              stop_idx = decoded_words_1.index(symbol) # index of the (first) [STOP] symbol
              decoded_words_1 = decoded_words_1[:stop_idx]
            except ValueError:
              continue
          if not len(decoded_words_1) or (len(decoded_words) and decoded_words_1[0] in [words[0] for words in decoded_words]) or '[UNK]' in decoded_words_1 or '<digit>' in decoded_words_1:
            continue
          if len(decoded_words_1) == 1 and ('“' == decoded_words_1[0] or '”' == decoded_words_1[0]):
            continue
          decoded_words.append(decoded_words_1)
      decoded_output = ' '.join(flat(decoded_words)) # single string          

      # # Extract the output ids from the hypothesis and convert back to words
      # output_ids = [int(t) for t in best_hyp.tokens[1:]]
      # decoded_words = data.outputids2words(output_ids, self._vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None))

      # # Remove the [STOP] token from decoded_words, if necessary
      # try:
      #   fst_stop_idx = decoded_words.index(data.STOP_DECODING) # index of the (first) [STOP] symbol
      #   decoded_words = decoded_words[:fst_stop_idx]
      # except ValueError:
      #   decoded_words = decoded_words
      # decoded_output = ' '.join(decoded_words) # single string
      if FLAGS.decode_only:
        result = []
        for words in decoded_words[:10]:
          result.append(''.join(words) if FLAGS.language == 'chinese' else ' '.join(words))
        decode_result.append({"id": original_abstract_sents[0], "keyphrases": ';'.join(result)})
        counter += 1
        tf.logging.info("We\'ve been decoded %i articles.", counter)
      elif FLAGS.single_pass:
        self.write_for_f1_eval(original_abstract_sents, decoded_words, counter) # write ref summary and decoded summary to file, to eval with pyrouge later
        counter += 1 # this is how many examples we've decoded
      else:
        print_results(article_withunks, abstract_withunks, decoded_output) # log output to screen
        self.write_for_attnvis(article_withunks, abstract_withunks, decoded_words, best_hyp.attn_dists, best_hyp.p_gens) # write info to .json file for visualization tool

        # Check if SECS_UNTIL_NEW_CKPT has elapsed; if so return so we can load a new checkpoint
        t1 = time.time()
        if t1-t0 > SECS_UNTIL_NEW_CKPT:
          tf.logging.info('We\'ve been decoding with same checkpoint for %i seconds. Time to load new checkpoint', t1-t0)
          _ = util.load_ckpt(self._saver, self._sess)
          t0 = time.time()


  def write_for_f1_eval(self, reference_sents, decoded_words_list, ex_index):
    """Write output to file in correct format for eval with pyrouge. This is called in single_pass mode.

    Args:
      reference_sents: list of strings
      decoded_words: list of strings
      ex_index: int, the index with which to label the files
    """
    # First, divide decoded output into sentences
    decoded_sents = []
    for decoded_words in decoded_words_list:
      while len(decoded_words) > 0:
        try:
          fst_period_idx = decoded_words.index(".")
        except ValueError: # there is text remaining that doesn't end in "."
          fst_period_idx = len(decoded_words)
        sent = decoded_words[:fst_period_idx+1] # sentence up to and including the period
        decoded_words = decoded_words[fst_period_idx + 1:] # everything else
        decoded_sents.append(' '.join(sent))

    # pyrouge calls a perl script that puts the data into HTML files.
    # Therefore we need to make our output HTML safe.
    decoded_sents = [make_html_safe(w) for w in decoded_sents]
    reference_sents = [make_html_safe(w) for w in reference_sents]

    # Write to file
    ref_file = os.path.join(self._rouge_ref_dir, "%06d_reference.txt" % ex_index)
    decoded_file = os.path.join(self._rouge_dec_dir, "%06d_decoded.txt" % ex_index)

    if not os.path.exists(ref_file):
      with open(ref_file, "w") as f:
        for idx,sent in enumerate(reference_sents):
          f.write(sent) if idx==len(reference_sents)-1 else f.write(sent+"\n")
    with open(decoded_file, "w") as f:
      for idx,sent in enumerate(decoded_sents):
        f.write(sent) if idx==len(decoded_sents)-1 else f.write(sent+"\n")

    tf.logging.info("Wrote example %i to file" % ex_index)


  def write_for_attnvis(self, article, abstract, decoded_words, attn_dists, p_gens):
    """Write some data to json file, which can be read into the in-browser attention visualizer tool:
      https://github.com/abisee/attn_vis

    Args:
      article: The original article string.
      abstract: The human (correct) abstract string.
      attn_dists: List of arrays; the attention distributions.
      decoded_words: List of strings; the words of the generated summary.
      p_gens: List of scalars; the p_gen values. If not running in pointer-generator mode, list of None.
    """
    article_lst = article.split() # list of words
    decoded_lst = flat(decoded_words) # list of decoded words
    to_write = {
        'article_lst': [make_html_safe(t) for t in article_lst],
        'decoded_lst': [make_html_safe(t) for t in decoded_lst],
        'abstract_str': make_html_safe(abstract),
        'attn_dists': attn_dists
    }
    if FLAGS.pointer_gen:
      to_write['p_gens'] = p_gens
    output_fname = os.path.join(self._decode_dir, 'attn_vis_data.json')
    with open(output_fname, 'w') as output_file:
      json.dump(to_write, output_file)
    tf.logging.info('Wrote visualization data to %s', output_fname)


def flat(l):
  for k in l:
    if not isinstance(k, (list, tuple)):
      yield k
    else:
      yield from flat(k)


def print_results(article, abstract, decoded_output):
  """Prints the article, the reference summmary and the decoded summary to screen"""
  print("---------------------------------------------------------------------------")
  tf.logging.info('ARTICLE:  %s', article)
  tf.logging.info('REFERENCE SUMMARY: %s', abstract)
  tf.logging.info('GENERATED SUMMARY: %s', decoded_output)
  print("---------------------------------------------------------------------------")


def make_html_safe(s):
  """Replace any angled brackets in string s to avoid interfering with HTML attention visualizer."""
  s.replace("<", "&lt;")
  s.replace(">", "&gt;")
  return s


def read_text_file(text_file):
  lines = []
  with open(text_file, "r", encoding='utf-8') as f:
    for line in f:
      line = line.strip()
      if line:
        lines.append(line)
  return lines


def get_f1_score(ref_words, dec_words, stemmer):
  total_ref = len(ref_words)
  total_dec = len(dec_words)
  
  if total_ref < 1 or total_dec < 1:
    return 0

  num = min(FLAGS.max_keyphrase_num, total_ref, total_dec)
  num_overlap = 0
  if FLAGS.language == 'english':
    dec_stem_words = [' '.join(stemmer.stemWords(w.split())) for w in dec_words[:num]]
    ref_stem_words = [' '.join(stemmer.stemWords(w.split())) for w in ref_words[:num]]
    for d_words in dec_stem_words:
      d_words = d_words.split()
      is_overlap = False
      for r_words in ref_stem_words:
        r_words = r_words.split()
        if len(r_words) == len(d_words):
          is_in = True
          for i, d_w in enumerate(d_words):
            # if d_w not in r_words:
            if d_w != r_words[i]:
              is_in = False
              break
          if is_in:
            is_overlap = True
            break
      if is_overlap:
        num_overlap = num_overlap + 1
    if num_overlap < 1:
      return 0
  else:
    dec_stem_words = [''.join(w.split()) for w in dec_words[:num]]
    ref_stem_words = [''.join(w.split()) for w in ref_words[:num]]
    for d_words in dec_stem_words:
      is_overlap = False
      for r_words in ref_stem_words:
        if r_words == d_words:
          is_overlap = True
          break
      if is_overlap:
        num_overlap = num_overlap + 1
    if num_overlap < 1:
      return 0
  recall = num_overlap / len(ref_stem_words)
  precision = num_overlap / len(dec_stem_words)
  return 2.0 * precision * recall / (precision + recall)


def f1_score_eval(ref_dir, dec_dir):
  # "%06d_reference.txt", "%06d_decoded.txt"
  ref_files = os.listdir(ref_dir)
  dec_files = os.listdir(dec_dir)

  stemmer = Stemmer.Stemmer('english') if FLAGS.language == 'english' else None

  f1_score_result = []
  for ref_file in ref_files:
    name = ref_file.split('_')[0]
    dec_file = ("%s_decoded.txt" % name)
    if dec_file in dec_files:
      ref_words = read_text_file(os.path.join(ref_dir, ref_file))
      dec_words = read_text_file(os.path.join(dec_dir, dec_file))
      f1_score_result.append(get_f1_score(ref_words, dec_words, stemmer))
  if len(f1_score_result) < 1:
    return 0.0
  return sum(f1_score_result) / len(f1_score_result)


def f1_score_log(result, dir_to_write):
  """Log ROUGE results to screen and write to file.

  Args:
    results_dict: the dictionary returned by pyrouge
    dir_to_write: the directory where we will write the results to"""
  log_str = ("f1 score: %s" % result)
  tf.logging.info(log_str) # log to screen
  results_file = os.path.join(dir_to_write, "F1_results.txt")
  tf.logging.info("Writing final F1_SCORE results to %s...", results_file)
  with open(results_file, "w") as f:
    f.write(log_str)


def get_decode_dir_name(ckpt_name):
  """Make a descriptive name for the decode dir, including the name of the checkpoint we use to decode. This is called in single_pass mode."""

  if "train" in FLAGS.data_path: dataset = "train"
  elif "val" in FLAGS.data_path: dataset = "val"
  elif "test" in FLAGS.data_path: dataset = "test"
  else: raise ValueError("FLAGS.data_path %s should contain one of train, val or test" % (FLAGS.data_path))
  dirname = "decode_%s_%imaxenc_%ibeam_%imindec_%imaxdec_%imaxnum" % (dataset, FLAGS.max_enc_steps, FLAGS.beam_size, FLAGS.min_dec_steps, FLAGS.max_dec_steps, DECODE_MAX_NUM)
  if FLAGS.generation_only:
    dirname += '_generation'
  elif FLAGS.copy_only:
    dirname += '_copy'
  elif FLAGS.copy_only_after_generation:
    dirname += '_copy_after_generation'
  if ckpt_name is not None:
    dirname += "_%s" % ckpt_name
  return dirname


def hashhex(s):
  """Returns a heximal formated SHA1 hash of the input string."""
  h = hashlib.sha1()
  h.update(s.encode('utf-8'))
  return h.hexdigest()


def write_decode_result_in_file(result, dir_to_write):
  output_file_name = os.path.basename(FLAGS.data_path).replace("input", "output")
  with open(os.path.join(dir_to_write, output_file_name), "w", encoding="utf-8") as f:
    for r in result:
      f.write("%s\n" % json.dumps(r, ensure_ascii=False))
    