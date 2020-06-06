import json
import sys

import tensorflow as tf

import util


def pred(infile, out):
  config = util.initialize_from_env()
  model = util.get_model(config)

  with tf.Session() as session:
    model.restore(session)

    with open(out, "w") as output_file:
      with open(infile, "r") as input_file:
        for data in input_file.readlines():
          jsondata = json.loads(data)
          tensorized_example = model.tensorize_example(jsondata, is_training=False)
          feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
          _, _, _, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = session.run(model.predictions, feed_dict=feed_dict)
          predicted_antecedents = model.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
          jsondata["predicted_clusters"], _ = model.get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)
          jsondata["top_spans"] = list(zip((int(i) for i in top_span_starts), (int(i) for i in top_span_ends)))
          jsondata['head_scores'] = []

          json.dump(jsondata, output_file, sort_keys=True)
          output_file.write("\n")

if __name__ == "__main__":
  in_file = sys.argv[1]
  out_file = sys.argv[2]
  pred(in_file, out_file)
