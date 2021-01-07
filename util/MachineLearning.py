#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import os, sys, re, platform

pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)

import warnings
import pandas as pd
import numpy as np
import torch
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import lightgbm as lgb
import xgboost as xgb
from scipy import interp
from EvaluationMetrics import Metrics
from Nets import (DealDataset, Net_CNN_1, Net_CNN_11, Net_RNN_2, Net_ABCNN_4, Net_ResNet_5, Net_AutoEncoder_6)
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class ILearnMachineLearning(object):
    def __init__(self, kw):
        self.kw = kw                                     # parameters dict
        self.training_dataframe = None                   # Pandas DataFrame
        self.training_datalabel = None                   # ndarray
        self.testing_dataframe = None                    # Pandas DataFrame
        self.testing_datalabel = None                    # ndarray
        self.training_score = None                       # 2D ndarray [fold, label, scores...]
        self.testing_score = None                        # 2D ndarray [0, label, scores...]
        self.best_model = None                           # list
        self.best_n_trees = 0                            # int
        self.metrics = None                              # Pandas DataFrame
        self.aucData = None                              # list
        self.prcData = None                              # list
        self.meanAucData = None                          # Pandas DataFrame
        self.meanPrcData = None                          # Pandas DataFrame
        self.indepAucData = None                         # Pandas DataFrame
        self.indepPrcData = None                         # Pandas DataFrame
        self.algorithm = None                            # string
        self.error_msg = None                            # string
        self.boxplot_data = None                         # dict key -> metrics.columns  value -> BoxPlotData object
        self.message = None                              # string
        self.task = None                                 # string 'binary' or 'muti-task'

    def load_data(self, file, target='Training'):
        if not os.path.exists(file):
            self.error_msg = 'data file does not exist.'
            return False
        dataframe, datalabel = None, None
        try:
            if file.endswith('.tsv'):
                df = pd.read_csv(file, sep='\t', header=None)
                dataframe = df.iloc[:, 1:]
                dataframe.index=['Sample_%s'%i for i in range(dataframe.values.shape[0])]
                dataframe.columns = ['F_%s'%i for i in range(dataframe.values.shape[1])]
                datalabel = np.array(df.iloc[:, 0]).astype(int)
            elif file.endswith('.csv'):
                df = pd.read_csv(file, sep=',', header=None)
                dataframe = df.iloc[:, 1:]
                dataframe.index=['Sample_%s'%i for i in range(dataframe.values.shape[0])]
                dataframe.columns = ['F_%s'%i for i in range(dataframe.values.shape[1])]
                datalabel = np.array(df.iloc[:, 0]).astype(int)
            elif file.endswith('.svm'):
                with open(file) as f:
                    record = f.read().strip()
                record = re.sub('\d+:', '', record)
                array = np.array([[i for i in item.split()] for item in record.split('\n')])
                dataframe = pd.DataFrame(array[:, 1:], dtype=float)
                dataframe.index=['Sample_%s'%i for i in range(dataframe.values.shape[0])]
                dataframe.columns = ['F_%s'%i for i in range(dataframe.values.shape[1])]
                datalabel = array[:, 0].astype(int)
            else:
                with open(file) as f:
                    record = f.read().strip().split('@')[-1].split('\n')[1:]
                array = np.array([item.split(',') for item in record])
                dataframe = pd.DataFrame(array[:, 0:-1], dtype=float)
                dataframe.index=['Sample_%s'%i for i in range(dataframe.values.shape[0])]
                dataframe.columns = ['F_%s'%i for i in range(dataframe.values.shape[1])]
                label = []
                for i in array[:, -1]:
                    if i == 'yes':
                        label.append(1)
                    else:
                        label.append(0)
                datalabel = np.array(label)
            if target == 'Training':
                self.training_dataframe = dataframe
                self.training_datalabel = datalabel
                if len(set(datalabel)) == 2:
                    self.task = 'binary'
                if len(set(datalabel)) > 2:
                    self.task = 'muti-task'
            if target == 'Testing':
                self.testing_dataframe = dataframe
                self.testing_datalabel = datalabel
        except Exception as e:
            self.error_msg = str(e)
            return False
        return True

    def import_training_data(self, dataframe, label):
        self.training_dataframe = dataframe
        self.training_datalabel = label
        if len(set(label)) == 2:
            self.task = 'binary'
        if len(set(label)) > 2:
            self.task = 'muti-task'

    def import_testing_data(self, dataframe, label):
        self.testing_dataframe = dataframe
        self.testing_datalabel = label

    def RandomForest(self):
        try:
            self.algorithm = 'RF'
            auto = self.kw['auto']
            fold = self.kw['FOLD']
            n_trees = self.kw['n_trees']
            n_jobs = self.kw['cpu']
            if auto:
                tree_range = self.kw['tree_range']
            else:
                tree_range = (n_trees, n_trees + 1, 100)
        
            if not self.training_dataframe is None:
                categories = sorted(set(self.training_datalabel))
                X, y = self.training_dataframe.values, self.training_datalabel
                # best model selection
                best_n_trees = tree_range[0]
                best_auc = 0
                best_accuracy = 0
                best_model = []
                best_training_score = None
                for tree in range(tree_range[0], tree_range[1] + 1, tree_range[2]):
                    training_score = np.zeros((X.shape[0], len(categories) + 2))
                    training_score[:, 1] = y
                    model = []
                    folds = StratifiedKFold(fold).split(X, y) # python iterator can be used only once
                    for i, (train, valid) in enumerate(folds):
                        train_X, train_y = X[train], y[train]
                        valid_X, valid_y = X[valid], y[valid]
                        rfc_model = RandomForestClassifier(n_estimators=tree, bootstrap=False, n_jobs=n_jobs)
                        rfc = rfc_model.fit(train_X, train_y)
                        model.append(rfc)
                        training_score[valid, 0] = i
                        training_score[valid, 2:] = rfc.predict_proba(valid_X)

                    if len(categories) == 2:
                        metrics, ok = Metrics.getBinaryTaskMetrics(training_score[:, 3], training_score[:, 1], pos_label=categories[-1])
                        if metrics[6] > best_auc:
                            best_auc = metrics[6]
                            best_n_trees = tree
                            best_model = model
                            best_training_score = training_score
                    if len(categories) > 2:
                        metrics = Metrics.getMutiTaskMetrics(training_score[:, 2:], training_score[:, 1])
                        if metrics[0] > best_accuracy:
                            best_accuracy = metrics[0]
                            best_n_trees = tree
                            best_model = model
                            best_training_score = training_score

                self.training_score = pd.DataFrame(best_training_score, columns=['Fold', 'Label'] + ['Score_%s' %i for i in categories])
                self.best_model = best_model
                self.best_n_trees = best_n_trees

                # independent dataset
                if not self.testing_dataframe is None:
                    indep = self.testing_dataframe.values
                    testing_score = np.zeros((indep.shape[0], len(categories) + 2))
                    testing_score[:, 1] = self.testing_datalabel
                    for rfc in self.best_model:
                        testing_score[:, 2:] += rfc.predict_proba(indep)
                    testing_score[:, 2:] /= fold
                    self.testing_score = pd.DataFrame(testing_score, columns=['Fold', 'Label'] + ['Score_%s' %i for i in categories])
                    self.testing_score['Fold'] = 'NA'
                # calculate metrics
                self.calculateEvaluationMetrics()
                self.message = 'Best n_trees is %d' %self.best_n_trees
                return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def SupportVectorMachine(self):
        try:
            self.algorithm = 'SVM'
            kernel = self.kw['kernel']
            auto = self.kw['auto']
            fold= self.kw['FOLD']
            penality = self.kw['penality']
            gamma = 1/self.training_dataframe.values.shape[1] if self.kw['gamma'] == 'auto' else self.kw['gamma']
            penalityRange = self.kw['penalityRange']
            gammaRange = self.kw['gammaRange']

            parameters = {'kernel': ['linear'], 'C': penalityRange} if kernel == 'linear' else {'kernel': [kernel], 'C': penalityRange, 'gamma': 2.0 ** np.arange(gammaRange[0], gammaRange[1])}
        
            if auto:
                optimizer = GridSearchCV(svm.SVC(probability=True), parameters).fit(self.training_dataframe.values, self.training_datalabel)
                params = optimizer.best_params_
                penality = params['C']
                if kernel != 'linear':
                    gamma = params['gamma']
            categories = sorted(set(self.training_datalabel))
            X, y = self.training_dataframe.values, self.training_datalabel
            training_score = np.zeros((X.shape[0], len(categories) + 2))
            training_score[:, 1] = y
            model = []
            folds = StratifiedKFold(fold).split(X, y)
            for i, (train, valid) in enumerate(folds):
                train_X, train_y = X[train], y[train]
                valid_X, valid_y = X[valid], y[valid]
                svm_model = svm.SVC(C=penality, kernel=kernel, degree=3, gamma=gamma, coef0=0.0, shrinking=True, probability=True, random_state=1)
                svc = svm_model.fit(train_X, train_y)
                model.append(svc)
                training_score[valid, 0] = i
                training_score[valid, 2:] = svc.predict_proba(valid_X)
            self.training_score = pd.DataFrame(training_score, columns=['Fold', 'Label'] + ['Score_%s' %i for i in categories])
            self.best_model = model

            # independent dataset
            if not self.testing_dataframe is None:
                indep = self.testing_dataframe.values
                testing_score = np.zeros((indep.shape[0], len(categories) + 2))
                testing_score[:, 1] = self.testing_datalabel
                for svc in self.best_model:
                    testing_score[:, 2:] += svc.predict_proba(indep)
                testing_score[:, 2:] /= fold
                self.testing_score = pd.DataFrame(testing_score, columns=['Fold', 'Label'] + ['Score_%s' %i for i in categories])
                self.testing_score['Fold'] = 'NA'
            # calculate metrics
            self.calculateEvaluationMetrics()
            self.message = 'penality: %s' %penality
            self.message += ' gamma: %s' %gamma if kernel != 'linear' else ''
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def MultiLayerPerceptron(self):
        try:
            self.algorithm = 'MLP'
            hidden_layer_size = tuple([int(i) for i in self.kw['layer'].split(';')])
            for item in hidden_layer_size:
                if item > 256:
                    self.error_msg = 'The layer size is out of range (1~256).'
                    return False

            activation = self.kw['activation']
            optimizer = self.kw['optimizer']
            epochs = self.kw['epochs']
            fold = self.kw['FOLD']
            lr=0.001
            categories = sorted(set(self.training_datalabel))
            X, y = self.training_dataframe.values, self.training_datalabel
            training_score = np.zeros((X.shape[0], len(categories) + 2))
            training_score[:, 1] = y
            model = []
            folds = StratifiedKFold(fold).split(X, y)
            for i, (train, valid) in enumerate(folds):
                train_X, train_y = X[train], y[train]
                valid_X, valid_y = X[valid], y[valid]
                mlp_model = MLPClassifier(activation=activation, alpha=1e-05, batch_size='auto', beta_1=0.9, beta_2=0.999,
                                      early_stopping=False, epsilon=1e-08, hidden_layer_sizes=hidden_layer_size,
                                      learning_rate='constant', learning_rate_init=lr, max_iter=epochs, momentum=0.9,
                                      nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True, solver=optimizer,
                                      tol=0.0001, validation_fraction=0.1, verbose=False, warm_start=False)
                mlp = mlp_model.fit(train_X, train_y)
                model.append(mlp)
                training_score[valid, 0] = i
                training_score[valid, 2:] = mlp.predict_proba(valid_X)
            self.training_score = pd.DataFrame(training_score, columns=['Fold', 'Label'] + ['Score_%s' %i for i in categories])
            self.best_model = model

            # independent dataset
            if not self.testing_dataframe is None:
                indep = self.testing_dataframe.values
                testing_score = np.zeros((indep.shape[0], len(categories) + 2))
                testing_score[:, 1] = self.testing_datalabel
                for mlp in self.best_model:
                    testing_score[:, 2:] += mlp.predict_proba(indep)
                testing_score[:, 2:] /= fold
                self.testing_score = pd.DataFrame(testing_score, columns=['Fold', 'Label'] + ['Score_%s' %i for i in categories])
                self.testing_score['Fold'] = 'NA'
            # calculate metrics
            self.calculateEvaluationMetrics()
            self.message = 'Complete.'
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def LogisticRegressionClassifier(self):
        try:
            self.algorithm = 'LR'
            fold = self.kw['FOLD']        
            categories = sorted(set(self.training_datalabel))
            X, y = self.training_dataframe.values, self.training_datalabel
            training_score = np.zeros((X.shape[0], len(categories) + 2))
            training_score[:, 1] = y
            model = []
            folds = StratifiedKFold(fold).split(X, y)
            for i, (train, valid) in enumerate(folds):
                train_X, train_y = X[train], y[train]
                valid_X, valid_y = X[valid], y[valid]
                lr_model = LogisticRegression(C=1.0, random_state=0).fit(train_X, train_y)
                model.append(lr_model)
                training_score[valid, 0] = i
                training_score[valid, 2:] = lr_model.predict_proba(valid_X)
            self.training_score = pd.DataFrame(training_score, columns=['Fold', 'Label'] + ['Score_%s' %i for i in categories])
            self.best_model = model

            # independent dataset
            if not self.testing_dataframe is None:
                indep = self.testing_dataframe.values
                testing_score = np.zeros((indep.shape[0], len(categories) + 2))
                testing_score[:, 1] = self.testing_datalabel
                for lrc in self.best_model:
                    testing_score[:, 2:] += lrc.predict_proba(indep)
                testing_score[:, 2:] /= fold
                self.testing_score = pd.DataFrame(testing_score, columns=['Fold', 'Label'] + ['Score_%s' %i for i in categories])
                self.testing_score['Fold'] = 'NA'
            # calculate metrics
            self.calculateEvaluationMetrics()
            self.message = 'Complete.'
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def LDAClassifier(self):
        try:
            self.algorithm = 'LDA'
            fold = self.kw['FOLD']        
            categories = sorted(set(self.training_datalabel))
            X, y = self.training_dataframe.values, self.training_datalabel
            training_score = np.zeros((X.shape[0], len(categories) + 2))
            training_score[:, 1] = y
            model = []
            folds = StratifiedKFold(fold).split(X, y)
            for i, (train, valid) in enumerate(folds):
                train_X, train_y = X[train], y[train]
                valid_X, valid_y = X[valid], y[valid]
                lda_model = LinearDiscriminantAnalysis(solver='svd', store_covariance=True).fit(train_X, train_y)
                model.append(lda_model)
                training_score[valid, 0] = i
                training_score[valid, 2:] = lda_model.predict_proba(valid_X)
            self.training_score = pd.DataFrame(training_score,
                                               columns=['Fold', 'Label'] + ['Score_%s' % i for i in categories])
            self.best_model = model

            # independent dataset
            if not self.testing_dataframe is None:
                indep = self.testing_dataframe.values
                testing_score = np.zeros((indep.shape[0], len(categories) + 2))
                testing_score[:, 1] = self.testing_datalabel
                for lda in self.best_model:
                    testing_score[:, 2:] += lda.predict_proba(indep)
                testing_score[:, 2:] /= fold
                self.testing_score = pd.DataFrame(testing_score,
                                                  columns=['Fold', 'Label'] + ['Score_%s' % i for i in categories])
                self.testing_score['Fold'] = 'NA'
            # calculate metrics
            self.calculateEvaluationMetrics()
            self.message = 'Complete.'
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def QDAClassifier(self):
        try:
            self.algorithm = 'QDA'
            fold = self.kw['FOLD']        
            categories = sorted(set(self.training_datalabel))
            X, y = self.training_dataframe.values, self.training_datalabel
            training_score = np.zeros((X.shape[0], len(categories) + 2))
            training_score[:, 1] = y
            model = []
            folds = StratifiedKFold(fold).split(X, y)
            for i, (train, valid) in enumerate(folds):
                train_X, train_y = X[train], y[train]
                valid_X, valid_y = X[valid], y[valid]
                qda_model = QuadraticDiscriminantAnalysis(store_covariance=True).fit(train_X, train_y)
                model.append(qda_model)
                training_score[valid, 0] = i
                training_score[valid, 2:] = qda_model.predict_proba(valid_X)            
            self.training_score = pd.DataFrame(training_score,
                                                columns=['Fold', 'Label'] + ['Score_%s' % i for i in categories])
            self.best_model = model

            # independent dataset
            if not self.testing_dataframe is None:
                indep = self.testing_dataframe.values
                testing_score = np.zeros((indep.shape[0], len(categories) + 2))
                testing_score[:, 1] = self.testing_datalabel
                for qda in self.best_model:
                    testing_score[:, 2:] += qda.predict_proba(indep)
                testing_score[:, 2:] /= fold
                self.testing_score = pd.DataFrame(testing_score,
                                                    columns=['Fold', 'Label'] + ['Score_%s' % i for i in categories])
                self.testing_score['Fold'] = 'NA'
            # calculate metrics
            self.calculateEvaluationMetrics()
            self.message = 'Complete.'
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def KNeighbors(self):
        try:
            self.algorithm = 'KNN'
            fold = self.kw['FOLD']
            n_neighbors = self.kw['topKValue']        
            categories = sorted(set(self.training_datalabel))
            X, y = self.training_dataframe.values, self.training_datalabel
            training_score = np.zeros((X.shape[0], len(categories) + 2))
            training_score[:, 1] = y
            model = []
            folds = StratifiedKFold(fold).split(X, y)
            for i, (train, valid) in enumerate(folds):
                train_X, train_y = X[train], y[train]
                valid_X, valid_y = X[valid], y[valid]
                knn_model = KNeighborsClassifier(n_neighbors=n_neighbors).fit(train_X, train_y)
                model.append(knn_model)
                training_score[valid, 0] = i
                training_score[valid, 2:] = knn_model.predict_proba(valid_X)
            self.training_score = pd.DataFrame(training_score, columns=['Fold', 'Label'] + ['Score_%s' %i for i in categories])
            self.best_model = model

            # independent dataset
            if not self.testing_dataframe is None:
                indep = self.testing_dataframe.values
                testing_score = np.zeros((indep.shape[0], len(categories) + 2))
                testing_score[:, 1] = self.testing_datalabel
                for lrc in self.best_model:
                    testing_score[:, 2:] += lrc.predict_proba(indep)
                testing_score[:, 2:] /= fold
                self.testing_score = pd.DataFrame(testing_score, columns=['Fold', 'Label'] + ['Score_%s' %i for i in categories])
                self.testing_score['Fold'] = 'NA'
            # calculate metrics
            self.calculateEvaluationMetrics()
            self.message = 'Complete.'
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def LightGBMClassifier(self):
        try:
            self.algorithm = 'LightGBM'
            fold= self.kw['FOLD']
            boosting_type = self.kw['boosting_type']
            num_leaves = self.kw['num_leaves']
            max_depth = self.kw['max_depth']
            learning_rate = self.kw['learning_rate']
            leaves_range = self.kw['leaves_range']
            depth_range = self.kw['depth_range']
            rate_range = self.kw['rate_range']
            auto = self.kw['auto']
            n_jobs = self.kw['cpu']
            
            parameters = {
                'num_leaves': list(range(leaves_range[0], leaves_range[1], leaves_range[2])),
                'max_depth': list(range(depth_range[0], depth_range[1], depth_range[2])),
                'learning_rate': list(np.arange(rate_range[0], rate_range[1], rate_range[2]))
            }
            if auto:
                gbm = lgb.LGBMClassifier(boosting_type=boosting_type)
                gsearch = GridSearchCV(gbm, param_grid=parameters, n_jobs=n_jobs).fit(self.training_dataframe.values, self.training_datalabel)
                best_parameters = gsearch.best_params_
                num_leaves = best_parameters['num_leaves']
                max_depth = best_parameters['max_depth']
                learning_rate = best_parameters['learning_rate']

            categories = sorted(set(self.training_datalabel))
            X, y = self.training_dataframe.values, self.training_datalabel
            training_score = np.zeros((X.shape[0], len(categories) + 2))
            training_score[:, 1] = y
            model = []
            folds = StratifiedKFold(fold).split(X, y)
            for i, (train, valid) in enumerate(folds):
                train_X, train_y = X[train], y[train]
                valid_X, valid_y = X[valid], y[valid]
                gbm_model = lgb.LGBMClassifier(boosting_type=boosting_type, num_leaves=num_leaves, max_depth=max_depth, learning_rate=learning_rate).fit(train_X, train_y)
                model.append(gbm_model)
                training_score[valid, 0] = i
                training_score[valid, 2:] = gbm_model.predict_proba(valid_X)
            self.training_score = pd.DataFrame(training_score, columns=['Fold', 'Label'] + ['Score_%s' %i for i in categories])
            self.best_model = model

            # independent dataset
            if not self.testing_dataframe is None:
                indep = self.testing_dataframe.values
                testing_score = np.zeros((indep.shape[0], len(categories) + 2))
                testing_score[:, 1] = self.testing_datalabel
                for gbm in self.best_model:
                    testing_score[:, 2:] += gbm.predict_proba(indep)
                testing_score[:, 2:] /= fold
                self.testing_score = pd.DataFrame(testing_score, columns=['Fold', 'Label'] + ['Score_%s' %i for i in categories])
                self.testing_score['Fold'] = 'NA'
            # calculate metrics
            self.calculateEvaluationMetrics()
            if auto:
                self.message = 'Best parameters: ' + str(best_parameters)
            self.message = 'Complete.'
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def XGBoostClassifier(self):
        try:
            self.algorithm = 'XGBoost'
            fold= self.kw['FOLD']
            booster = self.kw['booster']
            max_depth = self.kw['max_depth']
            learning_rate = self.kw['learning_rate']
            depth_range = self.kw['depth_range']
            rate_range = self.kw['rate_range']
            auto = self.kw['auto']
            n_jobs = self.kw['cpu']
            n_estimator = self.kw['n_estimator']
            colsample_bytree = self.kw['colsample_bytree']
            parameters = {
                'max_depth': list(range(depth_range[0], depth_range[1], depth_range[2])),
                'learning_rate': list(np.arange(rate_range[0], rate_range[1], rate_range[2]))
            }
        
            if auto:
                bst = xgb.XGBClassifier(booster=booster, n_estimators=n_estimator, n_jobs=n_jobs, colsample_bytree=colsample_bytree)
                gsearch = GridSearchCV(bst, param_grid=parameters, n_jobs=n_jobs).fit(self.training_dataframe.values, self.training_datalabel)
                best_parameters = gsearch.best_params_
                max_depth = best_parameters['max_depth']
                learning_rate = best_parameters['learning_rate']
            categories = sorted(set(self.training_datalabel))
            X, y = self.training_dataframe.values, self.training_datalabel
            training_score = np.zeros((X.shape[0], len(categories) + 2))
            training_score[:, 1] = y
            model = []
            folds = StratifiedKFold(fold).split(X, y)
            for i, (train, valid) in enumerate(folds):
                train_X, train_y = X[train], y[train]
                valid_X, valid_y = X[valid], y[valid]
                bst_model = xgb.XGBClassifier(booster=booster, max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimator, n_jobs=n_jobs, colsample_bytree=colsample_bytree).fit(train_X, train_y)
                model.append(bst_model)
                training_score[valid, 0] = i
                training_score[valid, 2:] = bst_model.predict_proba(valid_X)
            self.training_score = pd.DataFrame(training_score, columns=['Fold', 'Label'] + ['Score_%s' %i for i in categories])
            self.best_model = model

            # independent dataset
            if not self.testing_dataframe is None:
                indep = self.testing_dataframe.values
                testing_score = np.zeros((indep.shape[0], len(categories) + 2))
                testing_score[:, 1] = self.testing_datalabel
                for bst in self.best_model:
                    testing_score[:, 2:] += bst.predict_proba(indep)
                testing_score[:, 2:] /= fold
                self.testing_score = pd.DataFrame(testing_score, columns=['Fold', 'Label'] + ['Score_%s' %i for i in categories])
                self.testing_score['Fold'] = 'NA'
            # calculate metrics
            self.calculateEvaluationMetrics()
            if auto:
                self.message = 'Best parameters: ' + str(best_parameters)
            self.message = 'Complete.'
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def StochasticGradientDescentClassifier(self):
        try:
            self.algorithm = 'SGD'
            fold = self.kw['FOLD']
        
            categories = sorted(set(self.training_datalabel))
            X, y = self.training_dataframe.values, self.training_datalabel
            training_score = np.zeros((X.shape[0], len(categories) + 2))
            training_score[:, 1] = y
            model = []
            folds = StratifiedKFold(fold).split(X, y)
            for i, (train, valid) in enumerate(folds):
                train_X, train_y = X[train], y[train]
                valid_X, valid_y = X[valid], y[valid]
                sgd = SGDClassifier(loss='log').fit(train_X, train_y)
                model.append(sgd)
                training_score[valid, 0] = i
                training_score[valid, 2:] = sgd.predict_proba(valid_X)
            self.training_score = pd.DataFrame(training_score, columns=['Fold', 'Label'] + ['Score_%s' %i for i in categories])
            self.best_model = model

            # independent dataset
            if not self.testing_dataframe is None:
                indep = self.testing_dataframe.values
                testing_score = np.zeros((indep.shape[0], len(categories) + 2))
                testing_score[:, 1] = self.testing_datalabel
                for sgd in self.best_model:
                    testing_score[:, 2:] += sgd.predict_proba(indep)
                testing_score[:, 2:] /= fold
                self.testing_score = pd.DataFrame(testing_score, columns=['Fold', 'Label'] + ['Score_%s' %i for i in categories])
                self.testing_score['Fold'] = 'NA'
            # calculate metrics
            self.calculateEvaluationMetrics()
            self.message = 'Complete.'
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def DecisionTree(self):
        try:
            self.algorithm = 'DecisionTree'
            fold = self.kw['FOLD']
        
            categories = sorted(set(self.training_datalabel))
            X, y = self.training_dataframe.values, self.training_datalabel
            training_score = np.zeros((X.shape[0], len(categories) + 2))
            training_score[:, 1] = y
            model = []
            folds = StratifiedKFold(fold).split(X, y)
            for i, (train, valid) in enumerate(folds):
                train_X, train_y = X[train], y[train]
                valid_X, valid_y = X[valid], y[valid]
                dtc = DecisionTreeClassifier().fit(train_X, train_y)
                model.append(dtc)
                training_score[valid, 0] = i
                training_score[valid, 2:] = dtc.predict_proba(valid_X)
            self.training_score = pd.DataFrame(training_score, columns=['Fold', 'Label'] + ['Score_%s' %i for i in categories])
            self.best_model = model

            # independent dataset
            if not self.testing_dataframe is None:
                indep = self.testing_dataframe.values
                testing_score = np.zeros((indep.shape[0], len(categories) + 2))
                testing_score[:, 1] = self.testing_datalabel
                for dtc in self.best_model:
                    testing_score[:, 2:] += dtc.predict_proba(indep)
                testing_score[:, 2:] /= fold
                self.testing_score = pd.DataFrame(testing_score, columns=['Fold', 'Label'] + ['Score_%s' %i for i in categories])
                self.testing_score['Fold'] = 'NA'
            # calculate metrics
            self.calculateEvaluationMetrics()
            self.message = 'Complete.'
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def GaussianNBClassifier(self):
        try:
            self.algorithm = 'Bayes'
            fold = self.kw['FOLD']
        
            categories = sorted(set(self.training_datalabel))
            X, y = self.training_dataframe.values, self.training_datalabel
            training_score = np.zeros((X.shape[0], len(categories) + 2))
            training_score[:, 1] = y
            model = []
            folds = StratifiedKFold(fold).split(X, y)
            for i, (train, valid) in enumerate(folds):
                train_X, train_y = X[train], y[train]
                valid_X, valid_y = X[valid], y[valid]
                gsn = GaussianNB().fit(train_X, train_y)
                model.append(gsn)
                training_score[valid, 0] = i
                training_score[valid, 2:] = gsn.predict_proba(valid_X)
            self.training_score = pd.DataFrame(training_score, columns=['Fold', 'Label'] + ['Score_%s' %i for i in categories])
            self.best_model = model

            # independent dataset
            if not self.testing_dataframe is None:
                indep = self.testing_dataframe.values
                testing_score = np.zeros((indep.shape[0], len(categories) + 2))
                testing_score[:, 1] = self.testing_datalabel
                for gsn in self.best_model:
                    testing_score[:, 2:] += gsn.predict_proba(indep)
                testing_score[:, 2:] /= fold
                self.testing_score = pd.DataFrame(testing_score, columns=['Fold', 'Label'] + ['Score_%s' %i for i in categories])
                self.testing_score['Fold'] = 'NA'
            # calculate metrics
            self.calculateEvaluationMetrics()
            self.message = 'Complete.'
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def AdaBoost(self):
        try:
            self.algorithm = 'AdaBoost'
            fold = self.kw['FOLD']
        
            categories = sorted(set(self.training_datalabel))
            X, y = self.training_dataframe.values, self.training_datalabel
            training_score = np.zeros((X.shape[0], len(categories) + 2))
            training_score[:, 1] = y
            model = []
            folds = StratifiedKFold(fold).split(X, y)
            for i, (train, valid) in enumerate(folds):
                train_X, train_y = X[train], y[train]
                valid_X, valid_y = X[valid], y[valid]
                abc = AdaBoostClassifier(n_estimators=200, random_state=0).fit(train_X, train_y)
                model.append(abc)
                training_score[valid, 0] = i
                training_score[valid, 2:] = abc.predict_proba(valid_X)
            self.training_score = pd.DataFrame(training_score, columns=['Fold', 'Label'] + ['Score_%s' %i for i in categories])
            self.best_model = model

            # independent dataset
            if not self.testing_dataframe is None:
                indep = self.testing_dataframe.values
                testing_score = np.zeros((indep.shape[0], len(categories) + 2))
                testing_score[:, 1] = self.testing_datalabel
                for abc in self.best_model:
                    testing_score[:, 2:] += abc.predict_proba(indep)
                testing_score[:, 2:] /= fold
                self.testing_score = pd.DataFrame(testing_score, columns=['Fold', 'Label'] + ['Score_%s' %i for i in categories])
                self.testing_score['Fold'] = 'NA'
            # calculate metrics
            self.calculateEvaluationMetrics()
            self.message = 'Complete.'
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Bagging(self):
        try:
            self.algorithm = 'Bagging'
            fold = self.kw['FOLD']
            n_estimators = self.kw['n_estimator']
            n_jobs = self.kw['cpu']


            
            categories = sorted(set(self.training_datalabel))
            X, y = self.training_dataframe.values, self.training_datalabel
            training_score = np.zeros((X.shape[0], len(categories) + 2))
            training_score[:, 1] = y
            model = []
            folds = StratifiedKFold(fold).split(X, y)
            for i, (train, valid) in enumerate(folds):
                train_X, train_y = X[train], y[train]
                valid_X, valid_y = X[valid], y[valid]
                bgc = BaggingClassifier(n_estimators=n_estimators, n_jobs=n_jobs).fit(train_X, train_y)
                model.append(bgc)
                training_score[valid, 0] = i
                training_score[valid, 2:] = bgc.predict_proba(valid_X)
            self.training_score = pd.DataFrame(training_score, columns=['Fold', 'Label'] + ['Score_%s' %i for i in categories])
            self.best_model = model

            # independent dataset
            if not self.testing_dataframe is None:
                indep = self.testing_dataframe.values
                testing_score = np.zeros((indep.shape[0], len(categories) + 2))
                testing_score[:, 1] = self.testing_datalabel
                for bgc in self.best_model:
                    testing_score[:, 2:] += bgc.predict_proba(indep)
                testing_score[:, 2:] /= fold
                self.testing_score = pd.DataFrame(testing_score, columns=['Fold', 'Label'] + ['Score_%s' %i for i in categories])
                self.testing_score['Fold'] = 'NA'
            # calculate metrics
            self.calculateEvaluationMetrics()
            self.message = 'Complete.'
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def GBDTClassifier(self):
        try:
            self.algorithm = 'GBDT'
            fold = self.kw['FOLD']
        
            categories = sorted(set(self.training_datalabel))
            X, y = self.training_dataframe.values, self.training_datalabel
            training_score = np.zeros((X.shape[0], len(categories) + 2))
            training_score[:, 1] = y
            model = []
            folds = StratifiedKFold(fold).split(X, y)
            for i, (train, valid) in enumerate(folds):
                train_X, train_y = X[train], y[train]
                valid_X, valid_y = X[valid], y[valid]
                gbc = GradientBoostingClassifier().fit(train_X, train_y)
                model.append(gbc)
                training_score[valid, 0] = i
                training_score[valid, 2:] = gbc.predict_proba(valid_X)
            self.training_score = pd.DataFrame(training_score, columns=['Fold', 'Label'] + ['Score_%s' %i for i in categories])
            self.best_model = model

            # independent dataset
            if not self.testing_dataframe is None:
                indep = self.testing_dataframe.values
                testing_score = np.zeros((indep.shape[0], len(categories) + 2))
                testing_score[:, 1] = self.testing_datalabel
                for gbc in self.best_model:
                    testing_score[:, 2:] += gbc.predict_proba(indep)
                testing_score[:, 2:] /= fold
                self.testing_score = pd.DataFrame(testing_score, columns=['Fold', 'Label'] + ['Score_%s' %i for i in categories])
                self.testing_score['Fold'] = 'NA'
            # calculate metrics
            self.calculateEvaluationMetrics()
            self.message = 'Complete.'
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def calculateEvaluationMetrics(self):
        if not self.training_score is None:
            fold = sorted(set(self.training_score['Fold']))
            data = self.training_score.values
            categories = sorted(set(self.training_datalabel))
            index_name = []
            column_name = ['Sn', 'Sp', 'Pre', 'Acc', 'MCC', 'F1', 'AUROC', 'AUPRC']
            if len(categories) == 2:
                dataMetrics = np.zeros((len(fold), 8))
                aucData = []
                prcData = []
                # data for plot mean ROC in cross-validation
                mean_fpr = np.linspace(0, 1, 100)
                tprs = []
                # data for plot mean PRC in cross-validation
                mean_recall = np.linspace(0, 1, 100)
                precisions = []

                for f in fold:
                    tmp_data = data[data[:, 0] == f]
                    metrics = Metrics(tmp_data[:, -1], tmp_data[:, 1])
                    aucData.append(['Fold %d AUROC = %s' %(f, metrics.auc), metrics.aucDot])
                    prcData.append(['Fold %d AUPRC = %s' %(f, metrics.prc), metrics.prcDot])
                    tprs.append(interp(mean_fpr, metrics.aucDot['fpr'], metrics.aucDot['tpr']))
                    tprs[-1][0] = 0
                    precisions.append(interp(mean_recall, metrics.prcDot['recall'][::-1], metrics.prcDot['precision'][::-1]))
                    dataMetrics[int(f)] = np.array([metrics.sensitivity, metrics.specificity, metrics.precision, metrics.accuracy, metrics.mcc, metrics.f1, metrics.auc, metrics.prc]).reshape((1, -1))
                    index_name.append('Fold %d' %f)
                    del metrics
                meanValue = np.around(np.mean(dataMetrics, axis=0), decimals=4)
                dataMetrics = np.vstack((dataMetrics, meanValue))
                index_name.append('Mean')
                self.aucData = aucData
                self.prcData = prcData
                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[-1] = 1.0
                mean_precision = np.mean(precisions, axis=0)
                self.meanAucData = ['Mean AUROC = %s' %dataMetrics[-1][-2], pd.DataFrame(np.hstack((mean_fpr.reshape((-1, 1)), mean_tpr.reshape((-1, 1)))), columns=['fpr', 'tpr'])]
                self.meanPrcData = ['Mean AUPRC = %s' %dataMetrics[-1][-1], pd.DataFrame(np.hstack((mean_recall.reshape((-1, 1)), mean_precision.reshape((-1, 1)))), columns=['recall', 'precision'])]

                if not self.testing_score is None:
                    data = self.testing_score.values[:, 1:].astype(float)
                    metrics = Metrics(data[:, -1], data[:, 0])
                    metrics_ind = np.array([metrics.sensitivity, metrics.specificity, metrics.precision, metrics.accuracy, metrics.mcc, metrics.f1, metrics.auc, metrics.prc]).reshape((1, -1))
                    dataMetrics = np.vstack((dataMetrics, metrics_ind))
                    index_name.append('Indep')
                    self.indepAucData = ['Indep AUROC = %s' %metrics.auc, metrics.aucDot]
                    self.indepPrcData = ['Indep AUPRC = %s' %metrics.prc, metrics.prcDot]
                    del metrics
                self.metrics = pd.DataFrame(dataMetrics, index=index_name, columns=column_name)
            elif len(categories) > 2:
                dataMetrics = np.zeros((len(fold), 8)).astype(str)
                for f in fold:
                    index_name.append('Fold %d' %f)
                    tmp_data = data[data[:, 0] == f]
                    metrics = Metrics(tmp_data[:, 2:], tmp_data[:, 1])
                    dataMetrics[int(f)] = np.array(['NA', 'NA', 'NA', '%s' %metrics.accuracy, 'NA', 'NA', 'NA', 'NA'])
                    del metrics
                self.metrics = pd.DataFrame(dataMetrics, index=index_name, columns=column_name)
            else:
                pass

    def save_prediction_score(self, file, type='training'):
        try:
            if type == 'training':
                df = pd.DataFrame(self.training_score, columns=['Fold', 'Label'] + ['Score_%s' %i for i in sorted(set(self.training_datalabel))])
                df.to_csv(file, sep='\t', header=True, index=False)
            else:
                df = pd.DataFrame(self.testing_score, columns=['Fold', 'Label'] + ['Score_%s' %i for i in sorted(set(self.training_datalabel))])
                df.to_csv(file, sep='\t', header=True, index=False)
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def save_metrics(self, file):
        try:
            self.metrics.to_csv(file, sep='\t', header=True, index=True)
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def save_coder(self, file, type='training'):
        try:
            if type == 'training':
                data = np.hstack((self.training_datalabel.reshape((-1, 1)), self.training_dataframe.values))
            else:
                data = np.hstack((self.testing_datalabel.reshape((-1, 1)), self.testing_dataframe.values))
            if file.endswith('.csv'):
                np.savetxt(file, data, fmt="%s", delimiter=',')
            if file.endswith('.tsv'):
                np.savetxt(file, data, fmt="%s", delimiter=',')
            if file.endswith('.svm'):
                with open(file, 'w') as f:
                    for line in data:
                        f.write('%s' % line[0])
                        for i in range(1, len(line)):
                            f.write('  %d:%s' % (i, line[i]))
                        f.write('\n')
            if file.endswith('.arff'):
                with open(file, 'w') as f:
                    f.write('@relation descriptor\n\n')
                    for i in range(1, len(data[0])):
                        f.write('@attribute f.%d numeric\n' % i)
                    f.write('@attribute play {yes, no}\n\n')
                    f.write('@data\n')
                    for line in data:
                        for fea in line[1:]:
                            f.write('%s,' % fea)
                        if int(line[0]) == 1:
                            f.write('yes\n')
                        else:
                            f.write('no\n')
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def calculate_boxplot_data(self):
        df = self.metrics.iloc[0:self.kw['FOLD']]
        task = len(set(self.training_datalabel))

        boxplot_dict = {}
        if task == 2:
            for item in df.columns:
                boxplot_dict[item] = BoxPlotData(df.loc[:, item])
        if task >= 3:
            boxplot_dict['Acc'] = BoxPlotData(df.loc[:, 'Acc'].values.astype(float))
        self.boxplot_data = boxplot_dict

    """ run deep learning frames """
    def to_categorical(self, y, num_classes=None):
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, num_classes))
        categorical[np.arange(n), y] = 1
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)
        return categorical

    def run_networks(self, network=1):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            fold = self.kw['FOLD']
            batch_size = self.kw['batch_size']
            categories = sorted(set(self.training_datalabel))
            categories_onehot = self.to_categorical(categories)         
            
            X, y = self.training_dataframe.values, self.training_datalabel
            training_score = np.zeros((X.shape[0], categories_onehot.shape[1] + 2))
            training_score[:, 1] = y
            model = []
            folds = StratifiedKFold(fold).split(X, y)
            for i, (train, valid) in enumerate(folds):
                train_set = DealDataset(X[train], y[train])
                valid_set = DealDataset(X[valid], y[valid])
                train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
                valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
                # if network == 1 and len(categories) > 2:
                #     self.algorithm = 'CNN'
                #     net = Net_CNN_1(device=device, category=categories_onehot.shape[1], input_size=self.kw['input_channel'], sequence_length=self.kw['input_length'], out_channel=self.kw['output_channel'], padding=self.kw['padding'], conv_kernel_size=self.kw['kernel_size'], dense_size=self.kw['fc_size'], dropout=self.kw['dropout']).to(device)
                #     net.fit(train_loader, valid_loader, epochs=self.kw['epochs'], early_stopping=self.kw['early_stopping'], lr=self.kw['learning_rate'])
                # elif network == 1 and len(categories) == 2:
                #     self.algorithm = 'CNN'                
                #     net = Net_CNN_11(device=device, input_size=self.kw['input_channel'], sequence_length=self.kw['input_length'], out_channel=self.kw['output_channel'], padding=self.kw['padding'], conv_kernel_size=self.kw['kernel_size'], dense_size=self.kw['fc_size'], dropout=self.kw['dropout']).to(device)
                #     net.fit(train_loader, valid_loader, epochs=self.kw['epochs'], early_stopping=self.kw['early_stopping'], lr=self.kw['learning_rate'])
                if network == 1:
                    self.algorithm = 'CNN'
                    net = Net_CNN_1(device=device, category=categories_onehot.shape[1], input_size=self.kw['input_channel'], sequence_length=self.kw['input_length'], out_channel=self.kw['output_channel'], padding=self.kw['padding'], conv_kernel_size=self.kw['kernel_size'], dense_size=self.kw['fc_size'], dropout=self.kw['dropout']).to(device)
                    net.fit(train_loader, valid_loader, epochs=self.kw['epochs'], early_stopping=self.kw['early_stopping'], lr=self.kw['learning_rate'])
                elif network == 2:
                    self.algorithm = 'RNN'
                    net = Net_RNN_2(device=device, category=categories_onehot.shape[1], input_size=self.kw['input_channel'], sequence_length=self.kw['input_length'], hidden_size=self.kw['rnn_hidden_size'], num_layers=self.kw['rnn_hidden_layers'], dense_size=self.kw['fc_size'], dropout=self.kw['dropout']).to(device)
                    net.fit(train_loader, valid_loader, epochs=self.kw['epochs'], early_stopping=self.kw['early_stopping'], lr=self.kw['learning_rate'])
                elif network == 3:
                    self.algorithm = 'BCNN'
                    net = Net_RNN_2(device=device, category=categories_onehot.shape[1], input_size=self.kw['input_channel'], sequence_length=self.kw['input_length'], hidden_size=self.kw['rnn_hidden_size'], num_layers=self.kw['rnn_hidden_layers'], dense_size=self.kw['fc_size'], dropout=self.kw['dropout'], bidirectional=True).to(device)
                    net.fit(train_loader, valid_loader, epochs=self.kw['epochs'], early_stopping=self.kw['early_stopping'], lr=self.kw['learning_rate'])
                elif network == 4:
                    self.algorithm = 'ABCNN'
                    net = Net_ABCNN_4(device=device, category=categories_onehot.shape[1], input_size=self.kw['input_channel'], sequence_length=self.kw['input_length'], dropout=self.kw['dropout']).to(device)
                    net.fit(train_loader, valid_loader, epochs=self.kw['epochs'], early_stopping=self.kw['early_stopping'], lr=self.kw['learning_rate'])
                elif network == 5:
                    self.algorithm = 'ResNet'
                    net = Net_ResNet_5(device=device, category=categories_onehot.shape[1], input_size=self.kw['input_channel'], sequence_length=self.kw['input_length']).to(device)
                    net.fit(train_loader, valid_loader)
                elif network == 6:
                    self.algorithm = 'AE'
                    net = Net_AutoEncoder_6(device=device, category=categories_onehot.shape[1], input_dim=self.training_dataframe.values.shape[1]).to(device)
                    net.fit(train_loader, valid_loader, epochs=self.kw['epochs'], early_stopping=self.kw['early_stopping'], lr=self.kw['learning_rate'])
                    net.re_build_net()
                    net.re_fit(train_loader, valid_loader, epochs=self.kw['epochs'], early_stopping=self.kw['early_stopping'], lr=self.kw['learning_rate'])
            
                model.append(net)
                training_score[valid, 0] = i
                training_score[valid, 2:] = net.predict(valid_loader)
                self.training_score = pd.DataFrame(training_score, columns=['Fold', 'Label'] + ['Score_%s' %i for i in categories])
                self.best_model = model

            # independent dataset
            if not self.testing_dataframe is None:
                indep = self.testing_dataframe.values
                testing_score = np.zeros((indep.shape[0], categories_onehot.shape[1] + 2))
                testing_score[:, 1] = self.testing_datalabel
                indep_set = DealDataset(indep, self.testing_datalabel)
                indep_loader = DataLoader(indep_set, batch_size, shuffle=False)
                for net in self.best_model:
                    testing_score[:, 2:] += net.predict(indep_loader)
                testing_score[:, 2:] /= fold
                self.testing_score = pd.DataFrame(testing_score, columns=['Fold', 'Label'] + ['Score_%s' %i for i in categories])
                self.testing_score['Fold'] = 'NA'

            # calculate metrics
            self.calculateEvaluationMetrics()
            self.message = 'Complete.'
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False


class BoxPlotData(object):
    def __init__(self, data, whis=1.5):
        super(BoxPlotData, self).__init__()
        self.data = data
        self.whis = whis
        self.boxplot_dict = self.calculate()
    def calculate(self):
        try:
            median_value = np.median(self.data)
            Q1 = None
            Q3 = None
            i = len(self.data) // 4
            a = sorted(self.data)
            if len(self.data) % 4 == 0:
                Q1 = a[i-1] * 0.25 + a[i] * 0.75
                Q3 = a[3*i-1] * 0.75 + a[3*i] * 0.25
            elif len(self.data) % 4 == 1:
                Q1 = a[i]
                Q3 = a[3*i]
            elif len(self.data) % 4 == 2:
                Q1 = a[i] * 0.75 + a[i+1] * 0.25
                Q3 = a[3*i] * 0.25 + a[3*i+1] * 0.75
            elif len(self.data) % 4 == 3:
                Q1 = a[i] * 0.5 + a[i+1] * 0.5
                Q3 = a[3*i+1] * 0.5 + a[3*i+2] * 0.5
            IQR = (Q3 - Q1) * self.whis
            outlier = []
            Max_value = Q3 + IQR
            Min_value = Q1 - IQR
            for item in a:
                if item >= (Q1 - IQR):
                    Min_value = item
                    break
                else:
                    outlier.append(item)
            for item in a[::-1]:
                if item <= (Q3 + IQR):
                    Max_value = item
                    break
                else:
                    outlier.append(item)
            return {'median': median_value, 'Q1': Q1, 'Q3': Q3, 'maximum': Max_value, 'minimum': Min_value, 'outlier': outlier}
        except Exception as e:
            return {}


if __name__ == '__main__':
    ml_defatult_para = {
        'FOLD': 5,
        'cpu': 1,
        'auto': False,
        'n_trees': 100,
        'tree_range': (100, 1000, 100),
        'kernel': 'rbf',
        'penality': 1.0,
        'gamma': 'auto',
        'penalityRange': (1.0, 15.0),
        'gammaRange': (-10.0, 5.0),
        'layer': '32;32',
        'activation': 'relu',
        'optimizer': 'adam',
        'topKValue': 3,
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': -1,
        'learning_rate': 0.01,
        'leaves_range': (20, 100, 10),
        'depth_range': (15, 55, 10),
        'rate_range': (0.01, 0.15, 0.02),
        'booster': 'gbtree',
        'n_estimator': 100,
        'colsample_bytree': 0.8,
        'input_channel': 4,
        'input_length': 101,
        'output_channel': 64,
        'padding': 2,
        'kernel_size': 5,
        'dropout': 0.5,
        'epochs': 1000,
        'early_stopping': 100,
        'batch_size': 64,
        'rnn_hidden_size': 32,
        'rnn_num_layers': 1,
        'fc_size': 64,
    }
    Data = ILearnMachineLearning(ml_defatult_para)
    Data.load_data('../data/binary.csv')
    Data.load_data('../data/binary_ind.csv', target='Testing')
    Data.run_networks(5)
    print(Data.metrics)







