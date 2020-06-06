import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.getcwd()))

from document import Document
from metrics import CorefEvaluator
from preco_spanbert_formatter import PrecoFormatter, SpanBERTIndexUpdater
from utils import file_finder


scorer = CorefEvaluator()

def main(args):
  print(args)
  dataset = file_finder(args.path, args.eval)

  with open(dataset, 'r') as preco, open(args.target, 'r') as spanbert:
    for json_preco, json_spanbert in zip(preco, spanbert):
      preco_data = json.loads(json_preco)
      spanbert_data = json.loads(json_spanbert)
      
      preco_formatter = PrecoFormatter(preco_data)
      gold = preco_formatter.get()

      spanbert_indexer = SpanBERTIndexUpdater(spanbert_data)
      pred = spanbert_indexer.get()

      doc = Document(pred, gold)  # a document parsing the data into mention objects
      scorer.update(doc)

  scorer.detailed_score()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--path", required=True, help="Path of the PreCo dataset")
  parser.add_argument("--eval", required=True, help="PreCo dataset to use for evaluation")
  parser.add_argument("--target", required=True, help="The SpanBERT/BERT annotated dataset in jsonline format")
  main(parser.parse_args())
