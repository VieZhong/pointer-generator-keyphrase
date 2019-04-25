import os
import json
import pickle
import Stemmer

dataset = 'kp20k'
path = "/tmp/test-pointer-generater/log"
experiment_name = "coverage_weighted_experiment"
model_name = "decode_val_400maxenc_40beam_1mindec_10maxdec_10maxnum_ckpt-1884567"
validation_json_text_path = "/data/%s/validation_json_token/validation.json" % dataset
validation_json_pickle_path = "/data/%s/validation_json_token/validation.pkl" % dataset
val_reference_pickle_path = "/data/%s/val_reference.pkl" % dataset
Max_Article_Length = 400

def read_text_file(text_file):
  lines = []
  with open(text_file, "r", encoding='utf-8') as f:
    for line in f:
      line = line.strip()
      if line:
        lines.append(line)
  return lines

def get_f1_score(ref_words, dec_words, stemmer, max_keyphrase_num):
  total_ref = len(ref_words)
  total_dec = len(dec_words)
  
  if total_ref < 1 or total_dec < 1:
    return 0

  num_overlap = 0
  dec_stem_words = [' '.join(stemmer.stemWords(w.split())) for w in dec_words[:min(max_keyphrase_num, total_ref)]]
  ref_stem_words = [' '.join(stemmer.stemWords(w.split())) for w in ref_words[:min(max_keyphrase_num, total_ref)]]
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
  recall = num_overlap / len(ref_stem_words)
  precision = num_overlap / len(dec_stem_words)
  return 2.0 * precision * recall / (precision + recall)

def score_eval(ref_dir, dec_dir, validation_data):
  # "%06d_reference.txt", "%06d_decoded.txt"
  
  dec_files = os.listdir(dec_dir)
  stemmer = Stemmer.Stemmer('english')

  f1_score_result_5 = []
  f1_score_result_10 = []
  absent_recall_result_10 = []
  absent_recall_result_50 = []
  if os.path.exists(val_reference_pickle_path):
    val_reference = pickle.load(open(val_reference_pickle_path, 'rb'))
    for name in val_reference:
      dec_file = ("%s_decoded.txt" % name)
      if dec_file in dec_files:
        present_ref_words = val_reference[name]["present_ref"]
        absent_ref_words = val_reference[name]["absent_ref"]
        
        article = val_reference[name]["article"]
        dec_words = read_text_file(os.path.join(dec_dir, dec_file))
        present_dec_words, _ = choose_present_and_absent_dec_words(dec_words, article)

        if len(present_ref_words) > 0 and len(present_dec_words) > 0:
          f1_score_result_5.append(get_f1_score(present_ref_words, present_dec_words, stemmer, 5))
          f1_score_result_10.append(get_f1_score(present_ref_words, present_dec_words, stemmer, 10))

        if len(absent_ref_words) > 0 and len(dec_words) > 0:
          absent_recall_result_10.append(get_absent_recall(absent_ref_words, dec_words, stemmer, 10))
          absent_recall_result_50.append(get_absent_recall(absent_ref_words, dec_words, stemmer, 50))
  else:
    ref_files = os.listdir(ref_dir)
    val_reference = dict()
    for ref_file in ref_files:
      name = ref_file.split('_')[0]
      dec_file = ("%s_decoded.txt" % name)
      if dec_file in dec_files:
        ref_words = read_text_file(os.path.join(ref_dir, ref_file))
        present_ref_words, absent_ref_words, article = choose_present_and_absent_ref_words(ref_words, validation_data)
        
        val_reference[name] = {
          "ref": ref_words,
          "present_ref": present_ref_words,
          "absent_ref": absent_ref_words,
          "article": article
        }

        dec_words = read_text_file(os.path.join(dec_dir, dec_file))
        present_dec_words, _ = choose_present_and_absent_dec_words(dec_words, article)

        if len(present_ref_words) > 0 and len(present_dec_words) > 0:
          f1_score_result_5.append(get_f1_score(present_ref_words, present_dec_words, stemmer, 5))
          f1_score_result_10.append(get_f1_score(present_ref_words, present_dec_words, stemmer, 10))

        if len(absent_ref_words) > 0 and len(dec_words) > 0:
          absent_recall_result_10.append(get_absent_recall(absent_ref_words, dec_words, stemmer, 10))
          absent_recall_result_50.append(get_absent_recall(absent_ref_words, dec_words, stemmer, 50))
    pickle.dump(val_reference, open(val_reference_pickle_path, 'wb'))

  return sum(f1_score_result_5) / len(f1_score_result_5), sum(f1_score_result_10) / len(f1_score_result_10), sum(absent_recall_result_10) / len(absent_recall_result_10), sum(absent_recall_result_50) / len(absent_recall_result_50)

def choose_present_and_absent_ref_words(keywords, data):
  present = []
  absent = []
  text = None
  for article in data:
    title = article["title"]
    abstract = article["abstract"]
    keyword = article["keyword"]
    is_in = True
    for x in keywords:
      if x not in keyword:
        is_in = False
        break
    if is_in:
      text = article
      for x in keywords:
        if x in title or x in abstract:
          present.append(x)
        else:
          absent.append(x)
      break
  return present, absent, text

def choose_present_and_absent_dec_words(keywords, article):
  present = []
  absent = []
  
  if article is not None:
    title = article["title"]
    abstract = article["abstract"]

    for x in keywords:
      if x in title or x in abstract:
        present.append(x)
      else:
        absent.append(x)

  return present, absent

def f1_score_log(result, dir_to_write, name):
  log_str = ("f1@%s score: %s" % (name, result))
  print(log_str) # log to screen
  results_file = os.path.join(dir_to_write, "F1_results_@%s.txt" % name)
  print("Writing final F1_SCORE results to %s..." % results_file)
  with open(results_file, "w") as f:
    f.write(log_str)

def absent_recall_log(result, dir_to_write, name):
  log_str = ("absent_keywords recall@%s score: %s" % (name, result))
  print(log_str) # log to screen
  results_file = os.path.join(dir_to_write, "Absent_recall_@%s.txt" % name)
  print("Writing final Absent_Recall results to %s..." % results_file)
  with open(results_file, "w") as f:
    f.write(log_str)

def get_validation_data(file_path):
  lines = read_text_file(file_path)
  articles = []
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
    title = result['title'].split()
    abstract = result['abstract'].split()
    if len(title) + len(abstract) > Max_Article_Length:
      result['abstract'] = ' '.join(abstract[:Max_Article_Length - len(title)])
    articles.append(result)
  return articles

def get_absent_recall(ref_words, dec_words, stemmer, max_keyphrase_num):
  total_ref = len(ref_words)
  total_dec = len(dec_words)
  
  if total_ref < 1 or total_dec < 1:
    return 0

  num_overlap = 0
  dec_stem_words = [' '.join(stemmer.stemWords(w.split())) for w in dec_words[:max_keyphrase_num]]
  ref_stem_words = [' '.join(stemmer.stemWords(w.split())) for w in ref_words[:min(max_keyphrase_num, total_ref)]]
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
  return num_overlap / len(ref_stem_words)


if __name__ == '__main__':

  if os.path.exists(validation_json_pickle_path):
    validation_data = pickle.load(open(validation_json_pickle_path, 'rb'))
  else:
    validation_data = get_validation_data(validation_json_text_path)
    pickle.dump(validation_data, open(validation_json_pickle_path, 'wb'))

  f1_score_5, f1_score_10, absent_recall_10, absent_recall_50 = score_eval("/data/%s/val_reference" % dataset, "%s/%s/%s/decoded" % (path, experiment_name, model_name), validation_data)
  f1_score_log(f1_score_5, "%s/%s/%s" % (path, experiment_name, model_name), 5)
  f1_score_log(f1_score_10, "%s/%s/%s" % (path, experiment_name, model_name), 10)
  absent_recall_log(absent_recall_10, "%s/%s/%s" % (path, experiment_name, model_name), 10)
  absent_recall_log(absent_recall_50, "%s/%s/%s" % (path, experiment_name, model_name), 50)

