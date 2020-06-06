'''
tollef jørgensen, 13.04.20


functionality:

parse a cluster-object from a line of jsonline formatted conll files

{
  id: ...,
  sentences: ...,
  clusters: ...,
}

consider clusters as truth and predict a new set.
compare the two clusters.

'''
from utils import flatten, tuplify_clusters
import json

class JsonEval(object):
  def __init__(self):
    self.model = None
    self.sent_key = "sentences"
    self.cluster_key = "clusters"

  def load_model(self, model):
    self.model = model

  # define new keys for the json file to load
  def set_document_type(self, sent_key, cluster_key):
    self.sent_key = sent_key
    self.cluster_key = cluster_key

  def new_document(self, document, jsonify=False):
    # load if the passed document is simply a string read from file
    if jsonify:
      document = json.loads(document)

    self.sents = document[self.sent_key]
    self.clusters = document[self.cluster_key]

    # correct the format from [[]] to [()]
    self.clusters = tuplify_clusters(self.clusters)

    self.parsed_sents = []
    self.parse_sentences()
    self.tokens = []
    self.tokenize()

    print("Document has {} sentences and {} clusters, with a total of {} tokens".format(
      len(self.sents), len(self.clusters), len(self.tokens))
    )

  def tokenize(self):
    self.tokens = flatten(self.sents)
  
  def parse_sentences(self):
    for sent in self.sents:
      self.parsed_sents.append(' '.join(sent).strip())

  #  TODO detect whether adjust_index should be true or false, based on the dataset
  #  PreCo: false
  #  Ontonotes: true
  # ...
  def predict(self, sentence, adjust_index=False):
    if isinstance(sentence, list):
      sentence = ' '.join(sentence).strip()
    clusters = self.resolve(sentence)

    pred = []
    for mentions in clusters.values():
        parsed_mentions = []
        for _, mention in mentions:
            start, end = mention
            # alter the start/end index for Ontonotes dataset:
            if adjust_index:
              end -= 1
            parsed_mentions.append([start, end])
        pred.append(parsed_mentions)

    tuplified_predictions = tuplify_clusters(pred)
    return tuplified_predictions

  # coref resolving with neuralcoref          
  def resolve(self, sentence):
      doc = self.model(sentence)
      clusters = doc._.coref_clusters
      
      corefs = {}
      for c in clusters:
          # e.g. TOM HANKS. find all mentions of him:
          references = []
          for m in c.mentions:
              offset = (m.start, m.end)
              references.append([m, offset])
          corefs[str(c.main)] = references
      return corefs
