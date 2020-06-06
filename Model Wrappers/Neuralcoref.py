'''
A wrapper around NeuralCoref (https://github.com/huggingface/neuralcoref)
and spacy. Returning the nlp-object with neuralcoref in its pipeline

Dismissed from the modules/tollef_coref module
'''
import spacy
import neuralcoref

def get_cluster(doc, idx=None):
  if idx:
    return doc._.coref_clusters[idx]
  return doc._.coref_clusters

# params:
# blacklist: ["i", "me", "my", "you", "your"]
blacklist = True  # DO consider the above words when parsing
conv_dict = {
    "SomeRareWordOrName": ["woman", "waitress"]
}
max_dist = 50
max_dist_match = 500  # not considering texts above length 250
greed = 0.53

COREF = "neuralcoref"

def get_param(obj, key, default):
  if key in obj:
    return obj[key]
  else:
    return default

class Coref(object):
  def __init__(self, params, spacy_size="md", gpu=True, viz=False, verbose=False):
    self.greed = get_param(params, "greed", greed)
    self.max_dist = get_param(params, "max_dist", max_dist)
    self.max_dist_match = get_param(params, "max_dist_match", max_dist_match)
    self.blacklist = get_param(params, "blacklist", blacklist)

    spacy_model = "en_core_web_{}".format(spacy_size)
    # DISABLED = ["ner"]  # disable the ner module
    self.verbose = verbose
    if self.verbose:
      print("Loading spacy model...")
    self.nlp = spacy.load(spacy_model, disable=["ner"])

    if gpu:
      spacy.prefer_gpu()

    if viz:
      self.viz = viz

    self.doc = None

    self.init_coref()

  def init_coref(self):
    # if already instantiated, remove it.
    if COREF in self.nlp.pipe_names:
        self.nlp.remove_pipe(COREF)
    coref = neuralcoref.NeuralCoref(self.nlp.vocab,
                                    blacklist = self.blacklist,
                                    #conv_dict = conv_dict,
                                    max_dist = self.max_dist,
                                    max_dist_match = self.max_dist_match,
                                    greedyness = self.greed)



    if self.verbose:
      print("Added neuralcoref to pipeline!")
    self.nlp.add_pipe(coref, name='neuralcoref')

  # a function to show the outputs of neuralcoref
  def verbose(self, doc):
      print(doc.text, end="\n-----\n")
      clusters = get_cluster(doc)
      print("clusters: {}".format(clusters))
      for cluster in clusters:
          print("Found mentions: ", cluster.mentions)
      #    print("Cluster: ", cluster.main, "-- mentions: ", cluster.mentions)
          for mention in cluster.mentions:
              print("Cluster: ", cluster.main)
              print("MENTION: ", mention)
              print("Offset: ", mention.start, mention.end)
              print("Subtree: ", [t for t in mention.subtree])
              print("Sentence context:", mention.sent)
              print()
  
  def add_doc(self, doc):
    self.doc = self.nlp(doc)

  def cluster_resolved(self):
    return self.doc._.cluster_resolved

  def clusters(self):
    clusters = []
    if self.doc._.coref_clusters is not None:
      for cluster in self.doc._.coref_clusters:
        mention_clusters = []
        for mention in cluster.mentions:
          mention_clusters.append([mention.start, mention.end-1])
        clusters.append(mention_clusters)
    return clusters

  def tokens(self):
    return [tok.text for tok in self.doc]

  def get_tokens(self):
    return self.tokens()

  def predict(self, text):
    self.add_doc(text)
    return self.clusters()

  def show(self):
    if self.viz:
      self.viz.render(self.doc, huggingface=True)