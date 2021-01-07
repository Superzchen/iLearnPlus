#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import numpy as np
import pandas as pd
import math
from sklearn.metrics import roc_curve, auc, precision_recall_curve

class Metrics(object):
    def __init__(self, scores, labels, pos_label=None, threshold=0.5):
        super(Metrics, self).__init__()
        self.scores = scores
        label_sets = sorted(set(labels))
        if pos_label is None:
            self.pos_label = label_sets[-1]
        else:
            self.pos_label = pos_label
        self.labels = labels
        self.threshold = threshold
        # evaluation metrics
        self.sensitivity = 0
        self.specificity = 0
        self.accuracy = 0
        self.mcc = 0
        self.precision = 0
        self.f1 = 0
        self.auc = 0
        self.prc = 0
        self.aucDot = None
        self.prcDot = None

        if len(label_sets) == 2:
            self.calculateBinaryTaskMetrics()
        elif len(label_sets) > 2:
            self.calculateMultiTaskMetrics()
        else:
            pass

    def calculateBinaryTaskMetrics(self):
        tp, tn, fp, fn = 0, 0, 0, 0
        for i in range(len(self.scores)):
            if self.labels[i] == self.pos_label:
                if self.scores[i] >= self.threshold:
                    tp += 1
                else:
                    fn += 1
            else:
                if self.scores[i] < self.threshold:
                    tn += 1
                else:
                    fp += 1

        self.sensitivity = round(tp / (tp + fn) * 100, 2) if (tp + fn) != 0 else 0
        self.specificity = round(tn / (fp + tn) * 100, 2) if (fp + tn) != 0 else 0
        self.accuracy = round((tp + tn) / (tp + fn + tn + fp) * 100, 2)
        self.mcc = round((tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)), 4) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) != 0 else 0
        self.precision = round(tp / (tp + fp) * 100, 2) if (tp + fp) != 0 else 0
        self.f1 = round(2 * tp / (2 * tp + fp + fn), 4) if (2 * tp + fp + fn) != 0 else 0
        # roc        
        fpr, tpr, _ = roc_curve(self.labels, self.scores, pos_label=self.pos_label)
        self.auc = round(auc(fpr, tpr), 4)
        self.aucDot = pd.DataFrame(np.hstack((fpr.reshape((-1, 1)), tpr.reshape((-1, 1)))), columns=['fpr', 'tpr'])
        # prc
        precision, recall, _ = precision_recall_curve(self.labels, self.scores, pos_label=self.pos_label)
        self.prc = round(auc(recall, precision), 4)
        self.prcDot = pd.DataFrame(np.hstack((recall.reshape((-1, 1)), precision.reshape((-1, 1)))),
                                   columns=['recall', 'precision'])

    def calculateMultiTaskMetrics(self):
        result = []
        for item in self.scores:
            result.append(max(enumerate(item), key=lambda x: x[1])[0])
        result = np.array(result)
        self.accuracy = round(len(result[result == self.labels]) / len(self.labels) * 100, 2)

    @staticmethod
    def getBinaryTaskMetrics(scores, labels, pos_label=None, threshold=0.5):
        evaluationMetrics = Metrics(scores, labels, pos_label, threshold)
        return (evaluationMetrics.sensitivity, evaluationMetrics.specificity, evaluationMetrics.precision, evaluationMetrics.accuracy, evaluationMetrics.mcc,
                evaluationMetrics.f1, evaluationMetrics.auc, evaluationMetrics.prc), True

    @staticmethod
    def getMutiTaskMetrics(scores, labels):
        evaluationMetricss = Metrics(scores, labels)
        return evaluationMetricss.accuracy, True


if __name__ == '__main__':
    data = np.loadtxt('SVM_IND.txt', delimiter='\t')
    metrics = Metrics(data[:, 1:], data[:, 0])
    print(metrics.aucDot)

    # acc = Metrics.getMutiTaskMetrics(data[:, 1:], data[:, 0])
    # print(acc)
