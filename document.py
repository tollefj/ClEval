from utils import tuplify_clusters

class Document(object):
  def __init__(self, predicted, truth):
    self.pred = tuplify_clusters(predicted)
    self.gold = tuplify_clusters(truth)
    self.pred_mentions = self.mentionize(self.pred)
    self.gold_mentions = self.mentionize(self.gold)
    
  def mentionize(self, clusters):
    mentions = {}
    for mention_group in clusters:
        for part_mention in mention_group:
            mentions[part_mention] = mention_group
    return mentions

  def __str__(self):
    return "Predicted:\n{}\n\nGold:\n{}\n".format(self.pred, self.gold)