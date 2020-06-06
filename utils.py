import os

def get_cluster(doc, idx=None):
  if idx:
    return doc._.coref_clusters[idx]
  return doc._.coref_clusters

def get_score_dict(doc):
  return doc._.coref_scores.items()

def is_coref(doc):
  return doc._.in_coref

def get_mentions(cluster_piece, idx):
  return cluster_piece[idx].mentions

def get_main_cluster(cluster):
  return cluster.main

def get_highest_score(mention):
  scores = get_score_dict(mention)
  highscore  = []
  for name, score in scores:
    highscore.append((name, score))
  best = sorted(highscore)[-1]
  mention, score = best
  return (mention, score)

def get_context(doc, mention):
  start, end = mention.start, mention.end
  surround = 8
  start = start - surround
  end = end + surround
  
  span = doc[start:end]
  return span

def flatten(_list):
  return [item for sublist in _list for item in sublist]

def tuplify_clusters(clusters):
  tuplified_clusters = []
  for cluster in clusters:
    new_cluster = []
    for mention in cluster:
      new_cluster.append(tuple(mention))
    tuplified_clusters.append(tuple(new_cluster))
  return tuplified_clusters

def file_finder(path, target):
  if '.' not in target:
    print("File type not specified, looking for it...")
    for filename in os.listdir(path):
        if target in filename:
            target = filename
  path = os.path.join(path, target)
  print("Identified dataset: {}".format(path))
  return path