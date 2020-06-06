import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.getcwd()))
from spanbert.bert import tokenization

genre = "nw"
# The Ontonotes data for training the model contains text from several sources
# of very different styles. You need to specify the most suitable one out of:
# "bc": broadcast conversation
# "bn": broadcast news
# "mz": magazine
# "nw": newswire
# "pt": Bible text
# "tc": telephone conversation
# "wb": web data

model_name = "spanbert_base"
# The fine-tuned model to use. Options are:
# bert_base
# spanbert_base
# bert_large
# spanbert_large

def main(args):
  dataset = args.dataset
  if "." not in dataset:  # i.e. does not include extension
    for filename in os.listdir(args.path):
      if dataset in filename:
        dataset = filename
        break
    print("Identified dataset: {}".format(dataset))

  path = os.path.join(args.path, dataset)
  print("Path: {}".format(path))

  # Determine Max Segment
  max_segment = None
  for line in open('spanbert/experiments.conf'):
      if line.startswith(model_name):
          max_segment = True
      elif line.strip().startswith("max_segment_len"):
          if max_segment:
              max_segment = int(line.strip().split()[-1])
              break
  print("Max segment size: {}".format(max_segment))

  tokenizer = tokenization.FullTokenizer(vocab_file="spanbert/cased_config_vocab/vocab.txt", do_lower_case=False)

  subtoken_num = 0

  # iterate the input dataset
  out_name = "{}_{}_spanbert_{}.jsonl".format(args.name, args.dataset, args.genre)
  out_path = os.path.join("spanbert_tagged_data", out_name)
  print("Writing to file: {}".format(out_path))
  with open(out_path, 'w') as out_file:
    with open(path, 'r') as input_file:
      for jsonline in input_file.readlines():
        data = {
            'doc_key': args.genre,
            'sentences': [["[CLS]"]],
            'speakers': [["[SPL]"]],
            'clusters': [],
            'sentence_map': [0],
            'subtoken_map': [0],
        }

        as_json = json.loads(jsonline)
        text = as_json[args.sentkey]
        for sent_num, line in enumerate(text):
            raw_tokens = line
            to_sent = ' '.join(line)
            tokens = tokenizer.tokenize(to_sent)
            if len(tokens) + len(data['sentences'][-1]) >= max_segment:
                data['sentences'][-1].append("[SEP]")
                data['sentences'].append(["[CLS]"])
                data['speakers'][-1].append("[SPL]")
                data['speakers'].append(["[SPL]"])
                data['sentence_map'].append(sent_num - 1)
                data['subtoken_map'].append(subtoken_num - 1)
                data['sentence_map'].append(sent_num)
                data['subtoken_map'].append(subtoken_num)

            ctoken = raw_tokens[0]
            cpos = 0
            for token in tokens:
                data['sentences'][-1].append(token)
                data['speakers'][-1].append("-")
                data['sentence_map'].append(sent_num)
                data['subtoken_map'].append(subtoken_num)
                
                if token.startswith("##"):
                    token = token[2:]
                if len(ctoken) == len(token):
                    subtoken_num += 1
                    cpos += 1
                    if cpos < len(raw_tokens):
                        ctoken = raw_tokens[cpos]
                else:
                    ctoken = ctoken[len(token):]

        data['sentences'][-1].append("[SEP]")
        data['speakers'][-1].append("[SPL]")
        data['sentence_map'].append(sent_num - 1)
        data['subtoken_map'].append(subtoken_num - 1)
        
        json.dump(data, out_file, sort_keys=True)
        out_file.write('\n')


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--path", required=True, help="Path in which the datasets reside")
  parser.add_argument("--dataset", required=True, help="Dataset to read")
  parser.add_argument("--name", required=True, help="Out dataset main name")
  parser.add_argument("--genre", required=True, default="nz", help="Genre in the Ontonotes set")
  parser.add_argument("--sentkey", required=False, default="sentences", help="The sentence identifier of the jsonline object")
  main(parser.parse_args())
