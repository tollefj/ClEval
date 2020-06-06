from stanza.server import CoreNLPClient

class CoreNLP:
  def __init__(self, ram="5G", viz=False):
    self.client = None

    deterministic = ["tokenize", "ssplit", "pos", "lemma", "ner", "parse", "dcoref"]
    statistical = ["tokenize", "ssplit", "pos", "lemma", "ner", "parse", "coref"]
    with CoreNLPClient(annotators=statistical,
                      timeout=600000,
                      memory=ram) as client:
      self.client = client
        
    self.annotation = None  # object holding all data
    self.verbose = False
    
    self.dummy_init = self.predict("")
    
    self.viz = viz

  def get_tokens(self):
    tokens = []
    for sent in self.annotation.sentence:
      tokens.extend([t.value for t in sent.token])
    return tokens

  def get_clusters(self):
    corefs = self.annotation.corefChain
    clusters = []

    for coref in corefs:
      mentions = list(coref.mention)
      mentionclusters = []
      for m in mentions:
        start = m.beginIndex
        end = m.endIndex
        head = m.headIndex
        sentidx = m.sentenceIndex
        sent_by_idx = self.annotation.sentence[sentidx]
        
        # get the sentence token offset
        offset = sent_by_idx.tokenOffsetBegin
        mentionclusters.append([offset + start, offset + end - 1])
        
        if self.verbose:
          sent_toks = [tok.value for tok in sent_by_idx.token]
          print(sent_toks[start:end], "head:", sent_toks[head])
          print("mention:", [offset + start, offset + end - 1])
      clusters.append(mentionclusters)
    return clusters

  def predict(self, text):
    self.annotation = self.client.annotate(text)
    return self.get_clusters()
  
  def show(self):
    if self.viz:
      self.viz.render(self.annotation, corenlp=True)