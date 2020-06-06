'''
17.04.20
tollef jørgensen

this file converts the format of the PreCo dataset to a CoNLL-compatible jsonline format, as is the output by:
https://github.com/kentonl/e2e-coref/blob/master/minimize.py
'''
import json

class PrecoFormatter():
  def __init__(self, json, coreference_only=True, sent_key="sentences", cluster_key="mention_clusters"):
    self.sentences = json[sent_key]
    self.clusters = json[cluster_key]
    self.coreference_only = coreference_only

    # some sentences in the dataset are simply whitespace. ignore them.
    self.invalid = None
    self.set_invalid_sents()

    # a 1-to-1 map of sentence index of a token
    self.sentence_map = None  
    self.build_sentence_map()

    self.conll_format = None
    self.to_conll()

  def get(self):
    return self.conll_format

  def set_invalid_sents(self):
    self.invalid = [i for i, sent in enumerate(self.sentences) if len(sent) <= 1]

  def build_sentence_map(self):
    idx = 0
    mapping = []
    for sent in self.sentences:
        # add sentence index for the number of tokens in a sentence
        mapping.extend([idx]*len(sent))
        idx += 1
    self.sentence_map = mapping

  def to_conll(self):
    if self.coreference_only:
      # remove non-group clusters (i.e. singular mentions)
      clusters = [c for c in self.clusters if len(c) > 1]
    else:
      clusters = self.clusters

    def length_of_prev_sentences(sent_index):
      tokencount = len([idx for idx in self.sentence_map if idx < sent_index and idx not in self.invalid])
      return tokencount | 0

    conll = []
    for cluster in clusters:
      conll_cluster = []
      for mention in cluster:
        idx, m1, m2 = mention
        prev_token_offset = length_of_prev_sentences(idx)
        x1 = m1 + prev_token_offset
        x2 = m2 + prev_token_offset - 1 # PreCo adds 1 to the end index, remove it.
        conll_cluster.append([x1, x2])
      conll.append(conll_cluster)

    self.conll_format = conll

class SpanBERTIndexUpdater:
  def __init__(self, json):
    self.clusters = json["predicted_clusters"]
    self.sentences = json["sentences"]
    self.subtoken_map =  json["subtoken_map"]

    self.preco_map = None
    self.build_map()

    self.preco_indexes = None
    self.to_preco_index()

  def get(self):
    return self.preco_indexes

  # build a mapping from subtoken index to original index,
  # as the original subtoken mapping is offset by previous tokens
  def build_map(self):
    # get initial offset by the first non-zero subtoken
    sorted_unique_map = sorted(list(set(self.subtoken_map)))
    initial_offset = sorted_unique_map[1]
    
    mapping = {}
    previdx = 0
    token_count = 0

    for subtok_index in self.subtoken_map:
      previdx = subtok_index

      if initial_offset > 1:
        pointer = previdx - initial_offset
      else:
        pointer = previdx
      # update the map with old_index => new_index
      mapping[token_count] = pointer
      token_count += 1

    self.preco_map = mapping

  def to_preco_index(self):
    updated_indexes = []
    for cluster in self.clusters:
      updated_cluster = []
      for mention in cluster:
        m1, m2 = mention
        start = self.preco_map[m1]
        end = self.preco_map[m2]
        updated_cluster.append([start, end])
      updated_indexes.append(updated_cluster)

    self.preco_indexes = updated_indexes
