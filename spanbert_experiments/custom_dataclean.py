from bert import tokenization
import json
import sys
import os
import json

import tensorflow as tf
import util

filename = "none"

text = [
"Firefly is an American space Western drama television series which ran from 2002-2003, created by writer and director Joss Whedon, under his Mutant Enemy Productions label.",
"Whedon served as an executive producer, along with Tim Minear.",
"The series is set in the year 2517, after the arrival of humans in a new star system and follows the adventures of the renegade crew of Serenity, a 'Firefly-class' spaceship.",
"The ensemble cast portrays the nine characters who live on Serenity.",
"Whedon pitched the show as 'nine people looking into the blackness of space and seeing nine different things.'",
"The show explores the lives of a group of people, some of whom fought on the losing side of a civil war, who make a living on the fringes of society as part of the pioneer culture of their star system.",
"In this future, the only two surviving superpowers, the United States and China, fused to form the central federal government, called the Alliance, resulting in the fusion of the two cultures.",
"According to Whedon's vision, 'nothing will change in the future: technology will advance, but we will still have the same political, moral, and ethical problems as today.'",
"Firefly premiered in the U.S. on the Fox network on September 20, 2002.",
"By mid-December, Firefly had averaged 4.7 million viewers per episode and was 98th in Nielsen ratings.",
"It was canceled after 11 of the 14 produced episodes were aired.",
"Despite the relatively short life span of the series, it received strong sales when it was released on DVD and has large fan support campaigns.",
"It won a Primetime Emmy Award in 2003 for Outstanding Special Visual Effects for a Series.",
"TV Guide ranked the series at No. 5 on their 2013 list of 60 shows that were 'Cancelled Too Soon.'",
"The post-airing success of the show led Whedon and Universal Pictures to produce Serenity, a 2005 film which continues from the story of the series, and the Firefly franchise expanded to other media, including comics and a role-playing game.",
]

preco_2 = [["At", "the", "moment", ",", "it", "may", "be", "difficult", "to", "imagine", ",", "but", "many", "people", "believe", "that", ",", "by", "the", "year", "2100", ",", "we", "will", "live", "on", "the", "planet", "Mars", "."], ["Our", "own", "planet", ",", "Earth", ",", "is", "becoming", "more", "and", "more", "crowed", "and", "polluted", "."], ["Luckily", ",", "we", "can", "start", "again", "and", "build", "a", "better", "world", "on", "Mars", "."], ["Here", "is", "what", "life", "could", "be", "like", "."], [" "], ["First", "of", "all", ",", "transport", "should", "be", "much", "better", "."], ["At", "present", ",", "our", "spaceships", "are", "too", "slow", "to", "carry", "large", "numbers", "of", "people", "to", "Mars", "--", "it", "takes", "months", "."], ["However", ",", "by", "2100", ",", "spaceship", "can", "travel", "at", "half", "the", "speed", "of", "light", "."], ["It", "might", "take", "us", "two", "or", "three", "days", "to", "get", "to", "Mars", "!"], [" "], ["Secondly", ",", "humans", "need", "food", ",", "water", "and", "air", "to", "live", "."], ["Scientists", "should", "be", "able", "to", "develop", "plants", "that", "can", "be", "grown", "on", "Mars", "."], ["These", "plants", "will", "produce", "the", "food", "and", "air", "that", "we", "need", "."], ["However", ",", "can", "these", "plants", "produce", "water", "for", "us", "?"], ["There", "is", "no", "answer", "now", "."], [" "], ["There", "is", "a", "problem", "for", "us", "to", "live", "on", "Mars", "."], ["Mars", "pulls", "us", "much", "less", "than", "the", "Earth", "does", "."], ["This", "will", "be", "dangerous", "because", "we", "could", "easily", "jump", "too", "high", "and", "fly", "slowly", "away", "into", "space", "."], ["To", "prevent", "this", ",", "humans", "on", "Mars", "have", "to", "wear", "special", "shoes", "to", "make", "themselves", "heavier", "."], [" "], ["Life", "on", "Mars", "will", "be", "better", "than", "that", "on", "Earth", "in", "many", "ways", ",", "People", "will", "have", "more", "space", "."], ["Living", "in", "a", "large", "building", "with", "only", "10", "bedrooms", "is", "highly", "possible", "."], ["Many", "people", "believe", "that", "robot", "will", "do", "most", "of", "our", "work", ",", "so", "we", "have", "more", "time", "for", "our", "hobbies", "."], [" "], ["There", "will", "probably", "be", "no", "school", "on", "Mars", "."], ["Every", "student", "will", "have", "a", "computer", "at", "home", "which", "is", "connected", "to", "the", "internet", "."], ["They", "can", "study", ",", "do", "their", "homework", "and", "take", "exams", "in", "online", "schools", "."], ["Each", "student", "will", "also", "have", "their", "own", "online", "teacher", "called", "``", "e-teacher", "''", "."], [" "], ["However", ",", "in", "some", "ways", ",", "life", "on", "Mars", "may", "not", "be", "better", "than", "that", "on", "the", "earth", "today", "."], ["Food", "will", "not", "be", "the", "same", "--", "meals", "will", "probably", "be", "in", "the", "form", "of", "pills", "and", "will", "not", "be", "as", "delicious", "as", "they", "are", "today", ",", "Also", ",", "space", "travel", "will", "make", "many", "people", "feel", "ill", "."], ["The", "spaceship", "will", "travel", "fast", "but", "the", "journey", "to", "Mars", "will", "probably", "be", "very", "uncomfortable", "."]]

if filename != "none":
    text = [l.strip() for l in open(filename).readlines()]

text = preco_2

genre = "nz"
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

data = {
    'doc_key': genre,
    'sentences': [["[CLS]"]],
    'speakers': [["[SPL]"]],
    'clusters': [],
    'sentence_map': [0],
    'subtoken_map': [0],
}

# Determine Max Segment
max_segment = None
for line in open('experiments.conf'):
    if line.startswith(model_name):
        max_segment = True
    elif line.strip().startswith("max_segment_len"):
        if max_segment:
            max_segment = int(line.strip().split()[-1])
            break

tokenizer = tokenization.FullTokenizer(vocab_file="cased_config_vocab/vocab.txt", do_lower_case=False)
subtoken_num = 0
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


#with open("sample.in.json", 'w') as out:
#    json.dump(data, out, sort_keys=True)

def pred(indata, out):
  config = util.initialize_from_env()
  model = util.get_model(config)

  with tf.Session() as session:
    model.restore(session)

    with open(out, "w") as output_file:
        tensorized_example = model.tensorize_example(indata, is_training=False)
        feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
        _, _, _, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = session.run(model.predictions, feed_dict=feed_dict)
        predicted_antecedents = model.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
        indata["predicted_clusters"], _ = model.get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)
        indata["top_spans"] = list(zip((int(i) for i in top_span_starts), (int(i) for i in top_span_ends)))
        indata['head_scores'] = []

        output_file.write(json.dumps(indata))
        output_file.write("\n")

pred(data, "precoout.txt")