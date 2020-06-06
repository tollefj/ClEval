'''
Accessed 13.04.20 by Tollef Jørgensen
Source: https://github.com/kentonl/e2e-coref/blob/master/metrics.py
Heavily based on the work by Clark Manning: https://github.com/clarkkev/deep-coref/blob/master/evaluation.py 

No changes have been made.

the update function has been updated according to the original implementation by Manning.
as well as the function "print_scores"

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from collections import Counter
from sklearn.utils.linear_assignment_ import linear_assignment

def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)

class CorefEvaluator(object):
    def __init__(self):
        self.doc_count = 0
        self.conll_evals = [muc, b_cubed, ceafe]
        self.evaluators = [Evaluator(m) for m in self.conll_evals]
        self.all = [Evaluator(m) for m in [muc, b_cubed, ceafe, lea]]
        
    def eval_documents(self, docs):
        for d in docs:
            for e in self.all:
                e.update(d)
        print("evaluated {} documents with {} metrics".format(len(docs), len(self.all)))
        self.detailed_score("", "")

    def update(self, doc):
        self.doc_count += 1
        for e in self.all:
            e.update(doc)

    def update_conll(self, doc):
        for e in self.evaluators:
            e.update(doc)


    def detailed_score(self, modelname, dataset, verbose=True):
        conll_avg_f1 = 0
        leascore = 0
        if verbose:
            print("Evaluating {} on dataset: {}".format(modelname, dataset))
        for e in self.all:
            if verbose: 
              print("Running metric: {}".format(e.metric.__name__))
            p, r, f = e.get_prf()
            if e.metric in self.conll_evals:
                conll_avg_f1 += f
            else:
                leascore += f
            if verbose:
                print("Precision:\t{p}\nRecall:\t\t{r}\nF1 score:\t{f1}".format(p=p, r=r, f1=f))
                print("-----------------------------------")
        if verbose:
            print("\nCoNLL-2012 F1 score: {}".format(conll_avg_f1/3))
            print("Evaluated {} documents total".format(self.doc_count))
        return conll_avg_f1/3, leascore

    
    def get_conll(self):
      f1 = 0
      for e in self.all:
        if e.metric in self.conll_evals:
          f1 += e.get_f1()
      return f1 / 3

    def get_f1(self, evals):
        return sum(e.get_f1() for e in evals) / len(evals)

    def get_recall(self, evals):
        return sum(e.get_recall() for e in evals) / len(evals)

    def get_precision(self, evals):
        return sum(e.get_precision() for e in evals) / len(evals)

    def get_prf(self):
        p = self.get_precision(self.all)
        r = self.get_recall(self.all)
        f = self.get_f1(self.all)
        return p, r, f
        # return self.get_precision(), self.get_recall(), self.get_f1()

    def get_prf_conll(self):
        p = self.get_precision(self.evaluators)
        r = self.get_recall(self.evaluators)
        f = self.get_f1(self.evaluators)
        return p, r, f

class Evaluator(object):
    def __init__(self, metric, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = metric
        self.beta = beta

    def __str__(self):
        return self.metric.__name__

    def update(self, doc):
        predicted = doc.pred
        gold = doc.gold
        predicted_mentions = doc.pred_mentions
        gold_mentions = doc.gold_mentions

        if self.metric == ceafe:
            pn, pd, rn, rd = self.metric(predicted, gold)
        else:
            pn, pd = self.metric(predicted, gold_mentions)
            rn, rd = self.metric(gold, predicted_mentions)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    def get_f1(self):
        return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def get_counts(self):
        return self.p_num, self.p_den, self.r_num, self.r_den


def evaluate_documents(documents, metric, beta=1):
    evaluator = Evaluator(metric, beta=beta)
    for document in documents:
        evaluator.update(document)
    return evaluator.get_precision(), evaluator.get_recall(), evaluator.get_f1()


def b_cubed(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        gold_counts = Counter()
        correct = 0
        for m in c:
            if m in mention_to_gold:
                gold_counts[tuple(mention_to_gold[m])] += 1
        for c2, count in gold_counts.items():
            if len(c2) != 1:
                correct += count * count

        num += correct / float(len(c))
        dem += len(c)

    return num, dem


def muc(clusters, mention_to_gold):
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = set()
        for m in c:
            if m in mention_to_gold:
                linked.add(mention_to_gold[m])
            else:
                tp -= 1
        tp -= len(linked)
    return tp, p


def phi4(c1, c2):
    return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))


def ceafe(clusters, gold_clusters):
    clusters = [c for c in clusters if len(c) != 1]
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi4(gold_clusters[i], clusters[j])
    matching = linear_assignment(-scores)
    similarity = sum(scores[matching[:, 0], matching[:, 1]])
    return similarity, len(clusters), similarity, len(gold_clusters)


def importance(entity):
    return len(entity)

def lea(clusters, mention_to_gold):
    num, dem = 0, 0


    for c in clusters:
        if len(c) == 1:
            continue

        common_links = 0
        all_links = len(c) * (len(c) - 1) / 2.0
        for i, m in enumerate(c):
            if m in mention_to_gold:
                for m2 in c[i + 1:]:
                    if m2 in mention_to_gold and mention_to_gold[m] == mention_to_gold[m2]:
                        common_links += 1

        resolution_score = common_links / float(all_links)
        num += importance(c) * resolution_score
        dem += importance(c)

    return num, dem