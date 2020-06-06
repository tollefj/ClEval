from nltk.data import load
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.tokenize import WhitespaceTokenizer
import re

# a basic wrapper to avoid multiple loads of tokenizers
# https://www.nltk.org/_modules/nltk/tokenize.html

class Tokenizer:
  def __init__(self, lang="english", preserve_lines=True):
    self.lang = lang
    self.preserve_lines = preserve_lines
    self.punkt = load('tokenizers/punkt/{0}.pickle'.format(self.lang))
    self.treebank = self.init_treebank()
    self.whitespacer = self.init_whitespace()

  def init_treebank(self):
    treebank = TreebankWordTokenizer()

    improved_open_quote_regex = re.compile(u'([«“‘„]|[`]+)', re.U)
    improved_open_single_quote_regex = re.compile(r"(?i)(\')(?!re|ve|ll|m|t|s|d)(\w)\b", re.U)
    improved_close_quote_regex = re.compile(u'([»”’])', re.U)
    improved_punct_regex = re.compile(r'([^\.])(\.)([\]\)}>"\'' u'»”’ ' r']*)\s*$', re.U)

    treebank.STARTING_QUOTES.insert(0, (improved_open_quote_regex, r' \1 '))
    treebank.STARTING_QUOTES.append((improved_open_single_quote_regex, r'\1 \2'))
    treebank.ENDING_QUOTES.insert(0, (improved_close_quote_regex, r' \1 '))
    treebank.PUNCTUATION.insert(0, (improved_punct_regex, r'\1 \2 \3 '))
    
    return treebank

  def init_whitespace(self):
    whitespace = WhitespaceTokenizer()
    # custom rules here eventually
    return whitespace

  def tokenize(self, text):
    return self.treebank.tokenize(text)

  def sentences(self, text):
    return self.punkt.tokenize(text)

  def words(self, text):
    sentences = None
    if self.preserve_lines:
        sentences = [text]
    else:
        sentences = self.sentences(text)

    return [term for sent in sentences for term in self.tokenize(sent)]

  def whitespace(self, text):
    return self.whitespacer.tokenize(text)
