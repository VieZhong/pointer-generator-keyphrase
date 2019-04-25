# encoding=utf-8
import sys
import os
import hashlib
import struct
import subprocess
import collections
import json
import tensorflow as tf
import shutil
import jieba
from tensorflow.core.example import example_pb2


dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

# all_train_urls = "url_lists/all_train.txt"
# all_val_urls = "url_lists/all_val.txt"
# all_test_urls = "url_lists/all_test.txt"

tokenized_dir = "/data/nssd_data/with_title/article_tokenized"
finished_files_dir = "/data/nssd_data/with_title/finished_files"
chunks_dir = os.path.join(finished_files_dir, "chunked")

# These are the number of .story files we expect there to be in cnn_stories_dir and dm_stories_dir
num_expected_cnn_stories = 92579
num_expected_dm_stories = 219506

VOCAB_SIZE = 250000
CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data


def chunk_file(set_name, finished_files_dir):
  in_file = '%s/%s.bin' % (finished_files_dir, set_name)
  reader = open(in_file, "rb")
  chunk = 0
  finished = False
  while not finished:
    chunk_fname = os.path.join(chunks_dir, '%s_%05d.bin' % (set_name, chunk)) # new chunk
    with open(chunk_fname, 'wb') as writer:
      for _ in range(CHUNK_SIZE):
        len_bytes = reader.read(8)
        if not len_bytes:
          finished = True
          break
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, example_str))
      chunk += 1


def chunk_all(finished_files_dir):
  # Make a dir to hold the chunks
  if not os.path.isdir(chunks_dir):
    os.mkdir(chunks_dir)
  # Chunk the data
  for set_name in ['train', 'val', 'test']:
    print("Splitting %s data into chunks..." % set_name)
    chunk_file(set_name, finished_files_dir)
  print("Saved chunked data in %s" % chunks_dir)


def tokenize_stories(file_dir, tokenized_dir):
  """Maps a whole directory of .story files to a tokenized version using Stanford CoreNLP Tokenizer"""
  print("Preparing to tokenize %s to %s..." % (file_dir, tokenized_dir))

  if not os.path.exists("tmp"): os.makedirs("tmp")

  json_files = os.listdir(file_dir)
  stories = dict()

  for j_file in json_files:
    lines = read_text_file(os.path.join(file_dir, j_file))
    names = []
    for line in lines:
      try:
        result = json.loads(line)
      except json.decoder.JSONDecodeError:
        print ("读取错误，直接跳过本行")
        continue
      if "title" not in result or "abstract" not in result or "keyword" not in result:
        continue
      if not len(result['title']) or not len(result['abstract']) or not len(result['keyword']):
        continue
      file_name = ("%s.txt" % hashhex(result['title']))
      split_word_title = ' '.join(jieba.cut(result['title']))
      split_word_abstract = ' '.join(jieba.cut(result['abstract']))
      split_word_keyword = ';'.join([' '.join(jieba.cut(w)) for w in result['keyword'].split(';')])
      names.append(file_name)
      with open(("tmp/%s" % file_name), "w") as wf:
        wf.write("@title\n %s\n" % split_word_title)
        wf.write("@abstract\n %s\n" % split_word_abstract)
        wf.write("@keyphrases\n %s" % split_word_keyword)
        wf.close()
    stories[j_file.split('.')[0]] = names
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping.txt", "w") as f:
      for s in names:
        f.write("%s \t %s\n" % (os.path.join("tmp", s), os.path.join(tokenized_dir, s)))
    command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-lowerCase', '-ioFileList', '-preserveLines', 'mapping.txt']
    print("Tokenizing %i files in %s/%s and saving in %s..." % (len(names), file_dir, j_file, tokenized_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping.txt")
  
  shutil.rmtree("tmp")

  return stories

  # Check that the tokenized stories directory contains the same number of files as the original directory
  # num_orig = len(os.listdir(stories_dir))
  # num_tokenized = len(os.listdir(tokenized_stories_dir))
  # if num_orig != num_tokenized:
  #   raise Exception("The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
  # print "Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir)


def read_text_file(text_file):
  lines = []
  with open(text_file, "r", encoding='utf-8') as f:
    for line in f:
      lines.append(line.strip())
  return lines


def hashhex(s):
  """Returns a heximal formated SHA1 hash of the input string."""
  h = hashlib.sha1()
  h.update(s.encode('utf-8'))
  return h.hexdigest()


def get_url_hashes(url_list):
  return [hashhex(url) for url in url_list]


def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if "@title" in line: return line
  if "@abstract" in line: return line
  if "@keyphrases" in line: return line
  if line=="": return line
  if line[-1] in END_TOKENS: return line
  # print line[-1]
  return line + " 。"


def get_art_abs(story_file):
  lines = read_text_file(story_file)

  # Lowercase everything
  lines = [line.lower() for line in lines]

  # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
  # lines = [fix_missing_period(line) for line in lines]

  # Separate out article and abstract sentences
  article_lines = []
  keyphrases = None
  next_is = {
    "title": False,
    "abstract": False,
    "keyphrase": False
  }
  for idx,line in enumerate(lines):
    if line == "":
      continue # empty line
    elif line.startswith("@keyphrases"):
      next_is["keyphrase"] = True
    elif next_is["keyphrase"]:
      keyphrases = line.split(';')
      next_is["keyphrase"] = False
    elif line.startswith("@title"):
      next_is["title"] = True
    elif next_is["title"]:
      title = line
      next_is["title"] = False
    elif line.startswith("@abstract"):
      next_is["abstract"] = True
    elif next_is["abstract"]:
      next_is["abstract"] = False
      article_lines.append(fix_missing_period(line))

  # Make article into a single string
  article = ' '.join(article_lines)

  # Make abstract into a signle string, putting <s> and </s> tags around the sentences
  keyword = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in keyphrases])

  return title, article, keyword


def write_to_bin(stories, out_file, makevocab=False):
  """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""
  # print "Making bin file for URLs listed in %s..." % url_file
  # url_list = read_text_file(url_file)
  # url_hashes = get_url_hashes(url_list)
  # story_fnames = [s+".txt" for s in stories]
  num_stories = len(stories)

  if makevocab:
    vocab_counter = collections.Counter()

  with open(out_file, 'wb') as writer:
    for idx,s in enumerate(stories):
      if idx % 1000 == 0:
        print("Writing story %i of %i; %.2f percent done" % (idx, num_stories, float(idx)*100.0/float(num_stories)))

      # Look in the tokenized story dirs to find the .story file corresponding to this url
      if os.path.isfile(os.path.join(tokenized_dir, s)):
        story_file = os.path.join(tokenized_dir, s)
      else:
        print("Error: Couldn't find tokenized story file %s in tokenized directories %s. Was there an error during tokenization?" % (s, tokenized_dir))
        # Check again if tokenized stories directories contain correct number of files
        # print "Checking that the tokenized stories directories %s and %s contain correct number of files..." % (cnn_tokenized_stories_dir, dm_tokenized_stories_dir)
        # check_num_stories(cnn_tokenized_stories_dir, num_expected_cnn_stories)
        # check_num_stories(dm_tokenized_stories_dir, num_expected_dm_stories)
        raise Exception("Tokenized stories directories %s contain correct number of files but story file %s found in neither." % (tokenized_dir, s))

      # Get the strings to write to .bin file
      title, article, keyword = get_art_abs(story_file)

      # Write to tf.Example
      tf_example = example_pb2.Example()
      tf_example.features.feature['title'].bytes_list.value.extend([bytes(title, encoding="utf8")])
      tf_example.features.feature['article'].bytes_list.value.extend([bytes(article, encoding="utf8")])
      tf_example.features.feature['keyword'].bytes_list.value.extend([bytes(keyword, encoding="utf8")])
      tf_example_str = tf_example.SerializeToString()
      str_len = len(tf_example_str)
      writer.write(struct.pack('q', str_len))
      writer.write(struct.pack('%ds' % str_len, tf_example_str))

      # Write the vocab to file, if applicable
      if makevocab:
        tit_tokens = title.split(' ')
        art_tokens = article.split(' ')
        abs_tokens = keyword.split(' ')
        abs_tokens = [t for t in abs_tokens if t not in [SENTENCE_START, SENTENCE_END]] # remove these tags from vocab
        tokens = art_tokens + abs_tokens + tit_tokens
        tokens = [t.strip() for t in tokens] # strip
        tokens = [t for t in tokens if t!=""] # remove empty
        vocab_counter.update(tokens)

  print("Finished writing file %s\n" % out_file)

  # write vocab to file
  if makevocab:
    print("Writing vocab file...")
    with open(os.path.join(finished_files_dir, "vocab"), 'w', encoding="utf-8") as writer:
      for word, count in vocab_counter.most_common(VOCAB_SIZE):
        if not word.isdigit():
          writer.write((word + ' ' + str(count) + '\n'))
    print("Finished writing vocab file")


def check_num_stories(stories_dir, num_expected):
  num_stories = len(os.listdir(stories_dir))
  if num_stories != num_expected:
    raise Exception("stories directory %s contains %i files but should contain %i" % (stories_dir, num_stories, num_expected))


if __name__ == '__main__':
  if len(sys.argv) != 2:
    print("USAGE: python make_datafiles.py <jsons_dir>")
    sys.exit()
  jsons_dir = sys.argv[1]

  # Create some new directories
  if not os.path.exists(tokenized_dir): os.makedirs(tokenized_dir)
  # if not os.path.exists(dm_tokenized_stories_dir): os.makedirs(dm_tokenized_stories_dir)
  if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

  # Run stanford tokenizer on both stories dirs, outputting to tokenized stories directories
  stories = tokenize_stories(jsons_dir, tokenized_dir)
  # tokenize_stories(dm_stories_dir, dm_tokenized_stories_dir)

  # Read the tokenized stories, do a little postprocessing then write to bin files
  write_to_bin(stories["testing"], os.path.join(finished_files_dir, "test.bin"))
  write_to_bin(stories["validation"], os.path.join(finished_files_dir, "val.bin"))
  write_to_bin(stories["training"], os.path.join(finished_files_dir, "train.bin"), makevocab=True)

  # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunks
  chunk_all(finished_files_dir)