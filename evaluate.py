import spacy
import neuralcoref
import os
import argparse

from metrics import CorefEvaluator  # scorer by Lee et. al, 2018
from document import Document  # a structure to hold documents to be scored
from evaluator import JsonEval  # custom .jsonline evaluator


def main(args):
  # lazy match filename:
  dataset = args.dataset
  if "." not in dataset:  # i.e. does not include extension
    for filename in os.listdir(args.path):
      if dataset in filename:
        dataset = filename
        break
    print("Identified dataset: {}".format(dataset))

  path = os.path.join(args.path, dataset)
  print("File to predict: {}".format(path))

  if args.gpu:
    spacy.prefer_gpu()
  spacy_model = "en_core_web_{}".format(args.modelsize)
  print("Loading spacy model: {}".format(spacy_model))
  nlp = spacy.load(spacy_model)

  NEURALCOREF = "neuralcoref"
  if NEURALCOREF in nlp.pipe_names:
      nlp.remove_pipe(NEURALCOREF)

  # params
  # blacklist: ["i", "me", "my", "you", "your"]
  blacklist = True  # resolve words in the blacklist, False: otherwise
  store_scores = False  # for less memory usage, True if the scores should be tweaked
  # TODO: update scores with knowledge dataS
  # conv_dict = {"SomeName": ["woman", "girl"]}
  # TODO: update with knowledge data
  max_dist = 50  # max mention distance
  max_dist_match = 250  # not considering texts above length 250
  greedyness = 0.50
  coref = neuralcoref.NeuralCoref(nlp.vocab,
                                  blacklist=blacklist,
                                  store_scores=store_scores,
                                  #conv_dict=conv_dict,
                                  max_dist=max_dist,
                                  max_dist_match=max_dist_match,
                                  greedyness=greedyness)

  print("Adding neuralcoref to spacy pipeline")
  nlp.add_pipe(coref, name=NEURALCOREF)

  # load the .jsonline evaluator with nlp pipeline
  evaluator = JsonEval()
  evaluator.set_document_type(sent_key=args.sentkey, cluster_key=args.clusterkey)
  evaluator.load_model(nlp)

  # initialize an empty CorefEvaluator, to be updated with data from documents and score them
  scorer = CorefEvaluator()
  print("Initializing scorer on the following evalautors: {}".format([str(e) for e in scorer.evaluators]))

  def convert_local_to_global_index(local_predictions, prev_tokens):
    global_preds = []
    for cluster in local_predictions:
      c = []
      for mention in cluster:
        c.append([i + prev_tokens for i in mention])
      global_preds.append(c)
    return global_preds

  def preco_strip_idx(preds):
    stripped = []
    for cluster in preds:
      c = []
      for mention in cluster:
        m = []
        for idx, m1, m2 in mention:
          m.append((m1, m2))
        c.append(tuple(m))
      stripped.append(tuple(c))
    return stripped


  # open the gold file:
  with open(path, "r") as gold_file:
    n = 0
    for doc in gold_file:
      if n == 0:
        n += 1
        continue
      # load the evaluator with a new document
      evaluator.new_document(doc, jsonify=True)
      _gold = evaluator.clusters
      print("gold: ", _gold)
      sents = evaluator.sents
      print("gold: ", preco_strip_idx(_gold))

      # store num of computed tokens in a doc (group of sentences)
      # this handles conversion of local to global indexes (per sentence)
      # TODO: find out whether a dataset has global or local indexes
      # prev_tokens = 0

      # _pred = evaluator.predict(evaluator.tokens)
      # print("pred: ", _pred)
      # scorer = CorefEvaluator()
      # d = Document(_pred, _gold)
      # scorer.update(d)
      # print(scorer)

      doc_preds = []
      for sent in sents:
        print(sent)
        # score the document
        _pred = evaluator.predict(sent)
        doc_preds.append(_pred)
        # prev_tokens += len(sent)
      print(doc_preds)

      break




if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--path", required=True, help="Path in which the datasets reside")
  parser.add_argument("--dataset", required=True, help="Dataset to run predictions on")
  parser.add_argument("--gpu", help="Use spaCy with GPU", default=True)
  parser.add_argument("--modelsize", help="spaCy model size [sm, md, lg]", default="sm")
  parser.add_argument("--clusterkey", help="the cluster identifier in the target dataset", default="clusters")
  parser.add_argument("--sentkey", help="the sentence identifier in the target dataset", default="sentences")
  
  main(parser.parse_args())