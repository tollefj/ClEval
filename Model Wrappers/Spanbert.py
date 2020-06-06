from allennlp.predictors.predictor import Predictor
import allennlp_models.coref

class SpanBert:
  def __init__(self, viz=False):
    pretrained = "../../pretrained/allen-spanbert-large/"
    self.predictor = Predictor.from_path(pretrained)
    self.viz = viz
    self.doc = None
  
  def get_tokens(self):
    return self.doc["document"]

  def predict(self, text):
    self.doc = self.predictor.predict(text)
    return self.doc["clusters"]
      
  def predict_tokens(self, text):
    self.doc = self.predictor.predict_tokenized(text)
    return self.doc["clusters"]

  # print out all mentions corresponding to predicted clusters
  def describe(self):
    for cl in self.doc["clusters"]:
      for m in cl:
        print(self.doc["document"][m[0]:m[1]+1])
      print()
      
  def show(self):
    if self.viz:
      self.viz.render(self.doc, allen=True)