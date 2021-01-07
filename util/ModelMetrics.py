#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import numpy as np
import pandas as pd
import copy

class ModelMetrics(object):
    def __init__(self):
        super(ModelMetrics, self).__init__()
        self.metrics = pd.DataFrame(data=None, columns=['Id', 'StartTime', 'EndTime', 'Sn', 'Sp', 'Pre', 'Acc', 'MCC', 'F1', 'AUROC', 'AUPRC'])
        self.classification_task = None
        self.aucData = {}                                               # dict key=Id, value=Pandas DataFrame
        self.prcData = {}                                               # dict key=Id, value=Pandas DataFrame
        self.prediction_scores = {}                                     # dict key=Id, value=Pandas DataFrame
        self.models = {}                                                # dict key=Id, value=model_list

    def insert_data(self, metric_ndarray, id, auc_df, prc_df, prediction_score, model=None):
        df = pd.DataFrame(data=metric_ndarray)
        self.metrics = self.metrics.append(df, ignore_index=True)
        self.aucData[id] = auc_df
        self.prcData[id] = prc_df
        self.prediction_scores[id] = prediction_score
        if not model is None:
            self.models[id] = copy.deepcopy(model)


if __name__ == '__main__':
    model = ModelMetrics()
    df = pd.DataFrame([['id', '000', '111', 98, 99, 98, 99, 0.9, 0.8, 0.90, 0.91]], columns=['Id', 'StartTime', 'EndTime', 'Sn', 'Sp', 'Pre', 'Acc', 'MCC', 'F1', 'AUROC', 'AUPRC'])
    df1 = pd.DataFrame([['id', '111', '111', 98, 99, 98, 99, 0.9, 0.8, 0.90, 0.91]], columns=['Id', 'StartTime', 'EndTime', 'Sn', 'Sp', 'Pre', 'Acc', 'MCC', 'F1', 'AUROC', 'AUPRC'])

    model.insert_data(df, None, None)
    model.insert_data(df1, None, None)


