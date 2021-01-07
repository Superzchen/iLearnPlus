#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import sys, os
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QFileDialog, QLabel, QHBoxLayout, QGroupBox, QTextEdit,
                             QVBoxLayout, QTreeWidget, QTreeWidgetItem, QSplitter, QTableWidget, QTabWidget,
                             QTableWidgetItem, QInputDialog, QMessageBox, QFormLayout, QHeaderView, QAbstractItemView)
from PyQt5.QtGui import QIcon, QFont, QMovie
from PyQt5.QtCore import Qt, pyqtSignal
from util import (InputDialog, MachineLearning, ModelMetrics, PlotWidgets)
import qdarkstyle
import numpy as np
import pandas as pd
from itertools import combinations
import copy
import threading
import datetime
import joblib
import sip
import torch

class ILearnPlusAutoML(QWidget):
    # signal
    display_signal = pyqtSignal(list)
    display_curves_signal = pyqtSignal()
    append_msg_signal = pyqtSignal(str)
    close_signal = pyqtSignal(str)

    def __init__(self):
        super(ILearnPlusAutoML, self).__init__()
        # signal
        self.display_signal.connect(self.display_metrics)
        self.display_curves_signal.connect(self.display_curves)
        self.append_msg_signal.connect(self.append_message)

        self.MLData = None                                                     # Machine learning object
        self.algorithms_selected = set([])                                     # selected algorithms
        self.metrics = ModelMetrics.ModelMetrics()                             # data for display result
        self.current_data_index = 0
        self.boxplot_data = {}                                                 # dict
        self.pen_color = [
            (50, 116, 161),
            (225, 129, 44),
            (58, 146, 58),
            (192, 61, 62),
            (147, 114, 178),
            (132, 91, 83),
            (214, 132, 189),
            (127, 127, 127)
        ]
        self.lineStyle = {
            0: Qt.SolidLine,
            1: Qt.SolidLine,
            2: Qt.SolidLine,
            3: Qt.SolidLine,
            4: Qt.SolidLine,
            5: Qt.SolidLine,
            6: Qt.SolidLine,
            7: Qt.SolidLine,
            8: Qt.DashLine,
            9: Qt.DashLine,
            10: Qt.DashLine,
            11: Qt.DashLine,
            12: Qt.DashLine,
            13: Qt.DashLine,
            14: Qt.DashLine,
            15: Qt.DashLine,
        }
        self.ml_defatult_para = {
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
            'input_channel': 1,
            'input_length': 100,
            'output_channel': 64,
            'padding': 2,
            'kernel_size': 5,
            'dropout': 0.5,
            'epochs': 1000,
            'early_stopping': 100,
            'batch_size': 64,
            'fc_size': 64,
            'rnn_hidden_size': 32,
            'rnn_hidden_layers': 1,
            'rnn_bidirection': False,
            'mlp_input_dim': None,
        }                                          # default machine learning parameters
        self.para_rf = {
            'cpu': 1,
            'auto': False,
        }
        self.para_svm = {
            'auto': False,
        }
        self.para_lightgbm = {
            'max_depth': -1,
            'learning_rate': 0.01,
            'cpu': 1,
            'auto': False,
            'depth_range': (15, 55, 10),
            'rate_range': (0.01, 0.15, 0.02),
        }
        self.para_xgboost = {
            'max_depth': -1,
            'learning_rate': 0.01,
            'n_estimator': 100,
            'auto': False,
            'depth_range': (15, 55, 10),
            'rate_range': (0.01, 0.15, 0.02),
        }
        self.para_bagging = {
            'n_estimator': 100,
            'cpu': 1,
        }
        self.para_net_1 = {
            'input_channel': 1,
            'input_length': 100,
            'output_channel': 64,
            'padding': 2,
            'kernel_size': 5,
            'dropout': 0.5,
            'epochs': 1000,
            'early_stopping': 100,
            'batch_size': 64,
            'fc_size': 64,
            'rnn_hidden_size': 32,
            'rnn_hidden_layers': 1,
            'rnn_bidirection': False,
            'mlp_input_dim': None,
        }
        self.para_net_11 = {
            'input_channel': 1,
            'input_length': 100,
            'output_channel': 64,
            'padding': 2,
            'kernel_size': 5,
            'dropout': 0.5,
            'epochs': 1000,
            'early_stopping': 100,
            'batch_size': 64,
            'fc_size': 64,
            'rnn_hidden_size': 32,
            'rnn_hidden_layers': 1,
            'rnn_bidirection': False,
            'mlp_input_dim': None,
        }
        self.para_net_2 = {
            'input_channel': 1,
            'input_length': 100,
            'output_channel': 64,
            'padding': 2,
            'kernel_size': 5,
            'dropout': 0.5,
            'epochs': 1000,
            'early_stopping': 100,
            'batch_size': 64,
            'fc_size': 64,
            'rnn_hidden_size': 32,
            'rnn_hidden_layers': 1,
            'rnn_bidirection': False,
            'mlp_input_dim': None,
        }
        self.para_net_3 = {
            'input_channel': 1,
            'input_length': 100,
            'output_channel': 64,
            'padding': 2,
            'kernel_size': 5,
            'dropout': 0.5,
            'epochs': 1000,
            'early_stopping': 100,
            'batch_size': 64,
            'fc_size': 64,
            'rnn_hidden_size': 32,
            'rnn_hidden_layers': 1,
            'rnn_bidirection': False,
            'mlp_input_dim': None,
        }
        self.para_net_4 = {
            'input_channel': 1,
            'input_length': 100,
            'output_channel': 64,
            'padding': 2,
            'kernel_size': 5,
            'dropout': 0.5,
            'epochs': 1000,
            'early_stopping': 100,
            'batch_size': 64,
            'fc_size': 64,
            'rnn_hidden_size': 32,
            'rnn_hidden_layers': 1,
            'rnn_bidirection': False,
            'mlp_input_dim': None,
        }
        self.para_net_5 = {
            'input_channel': 1,
            'input_length': 100,
            'output_channel': 64,
            'padding': 2,
            'kernel_size': 5,
            'dropout': 0.5,
            'epochs': 1000,
            'early_stopping': 100,
            'batch_size': 64,
            'fc_size': 64,
            'rnn_hidden_size': 32,
            'rnn_hidden_layers': 1,
            'rnn_bidirection': False,
            'mlp_input_dim': None,
        }
        self.para_net_6 = {
            'input_channel': 1,
            'input_length': 100,
            'output_channel': 64,
            'padding': 2,
            'kernel_size': 5,
            'dropout': 0.5,
            'epochs': 1000,
            'early_stopping': 100,
            'batch_size': 64,
            'fc_size': 64,
            'rnn_hidden_size': 32,
            'rnn_hidden_layers': 1,
            'rnn_bidirection': False,
            'mlp_input_dim': None,
        }
        self.ml_combination_para = {
            'FOLD': 5,
            'cpu': 1,
            'auto': True,
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
            'input_channel': 1,
            'input_length': 100,
            'output_channel': 64,
            'padding': 2,
            'kernel_size': 5,
            'dropout': 0.5,
            'epochs': 1000,
            'early_stopping': 100,
            'batch_size': 64,
            'fc_size': 64,
            'rnn_hidden_size': 32,
            'rnn_hidden_layers': 1,
            'rnn_bidirection': False,
            'mlp_input_dim': None,
        } 

        # for combination of models
        self.MLData_cb = None     
        self.bestPerformance = 0
        self.bestCombinations = ()
        self.bestMetrics = None
        self.bestAUC = None
        self.bestPRC = None
        self.bestModels = None
        self.bestTrainingScore = None

        # status bar
        self.gif = QMovie('images/progress_bar.gif')
        self.initUI()

    def initUI(self):
        self.setWindowTitle('iLearnPlus AutoML')
        self.resize(800, 600)
        self.setWindowState(Qt.WindowMaximized)
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.setup_UI()

    def setup_UI(self):
        # file
        topGroupBox = QGroupBox('Load data', self)
        topGroupBox.setFont(QFont('Arial', 10))
        # topGroupBox.setMinimumHeight(100)
        topGroupBoxLayout = QFormLayout()
        trainFileButton = QPushButton('Open')
        trainFileButton.clicked.connect(lambda:self.data_from_file_ml('Training'))
        # testFileButton = QPushButton('Open')
        # testFileButton.clicked.connect(lambda:self.data_from_file_ml('Testing'))
        topGroupBoxLayout.addRow('Open training file:', trainFileButton)
        # topGroupBoxLayout.addRow('Open testing file:', testFileButton)
        topGroupBox.setLayout(topGroupBoxLayout)

        # tree
        treeGroupBox = QGroupBox('Machine learning algorithms', self)
        treeGroupBox.setFont(QFont('Arial', 10))
        treeLayout = QHBoxLayout()
        self.ml_treeWidget = QTreeWidget()
        self.ml_treeWidget.setColumnCount(2)
        self.ml_treeWidget.setMinimumWidth(300)
        self.ml_treeWidget.setColumnWidth(0, 150)
        self.ml_treeWidget.setFont(QFont('Arial', 8))
        self.ml_treeWidget.setHeaderLabels(['Methods', 'Definition'])
        self.machineLearningAlgorighms = QTreeWidgetItem(self.ml_treeWidget)
        self.machineLearningAlgorighms.setExpanded(True)  # set node expanded
        self.machineLearningAlgorighms.setText(0, 'Machine learning algorithms')
        rf = QTreeWidgetItem(self.machineLearningAlgorighms)
        rf.setText(0, 'RF')
        rf.setText(1, 'Random Forest')
        rf.setCheckState(0, Qt.Unchecked)
        dtree = QTreeWidgetItem(self.machineLearningAlgorighms)
        dtree.setText(0, 'DecisionTree')
        dtree.setText(1, 'Decision Tree')
        dtree.setCheckState(0, Qt.Unchecked)
        lightgbm = QTreeWidgetItem(self.machineLearningAlgorighms)
        lightgbm.setText(0, 'LightGBM')
        lightgbm.setText(1, 'LightGBM')
        lightgbm.setCheckState(0, Qt.Unchecked)
        svm = QTreeWidgetItem(self.machineLearningAlgorighms)
        svm.setText(0, 'SVM')
        svm.setText(1, 'Support Verctor Machine')
        svm.setCheckState(0, Qt.Unchecked)
        mlp = QTreeWidgetItem(self.machineLearningAlgorighms)
        mlp.setText(0, 'MLP')
        mlp.setText(1, 'Multi-layer Perceptron')
        mlp.setCheckState(0, Qt.Unchecked)
        xgboost = QTreeWidgetItem(self.machineLearningAlgorighms)
        xgboost.setText(0, 'XGBoost')
        xgboost.setText(1, 'XGBoost')
        xgboost.setCheckState(0, Qt.Unchecked)
        knn = QTreeWidgetItem(self.machineLearningAlgorighms)
        knn.setText(0, 'KNN')
        knn.setText(1, 'K-Nearest Neighbour')
        knn.setCheckState(0, Qt.Unchecked)
        lr = QTreeWidgetItem(self.machineLearningAlgorighms)
        lr.setText(0, 'LR')
        lr.setText(1, 'Logistic Regression')
        lr.setCheckState(0, Qt.Unchecked)
        lda = QTreeWidgetItem(self.machineLearningAlgorighms)
        lda.setText(0, 'LDA')
        lda.setText(1, 'Linear Discriminant Analysis')
        lda.setCheckState(0, Qt.Unchecked)
        qda = QTreeWidgetItem(self.machineLearningAlgorighms)
        qda.setText(0, 'QDA')
        qda.setText(1, 'Quadratic Discriminant Analysis')
        qda.setCheckState(0, Qt.Unchecked)
        sgd = QTreeWidgetItem(self.machineLearningAlgorighms)
        sgd.setText(0, 'SGD')
        sgd.setText(1, 'Stochastic Gradient Descent')
        sgd.setCheckState(0, Qt.Unchecked)
        bayes = QTreeWidgetItem(self.machineLearningAlgorighms)
        bayes.setText(0, 'NaiveBayes')
        bayes.setText(1, 'NaiveBayes')
        bayes.setCheckState(0, Qt.Unchecked)
        bagging = QTreeWidgetItem(self.machineLearningAlgorighms)
        bagging.setText(0, 'Bagging')
        bagging.setText(1, 'Bagging')
        bagging.setCheckState(0, Qt.Unchecked)
        adaboost = QTreeWidgetItem(self.machineLearningAlgorighms)
        adaboost.setText(0, 'AdaBoost')
        adaboost.setText(1, 'AdaBoost')
        adaboost.setCheckState(0, Qt.Unchecked)
        gbdt = QTreeWidgetItem(self.machineLearningAlgorighms)
        gbdt.setText(0, 'GBDT')
        gbdt.setText(1, 'Gradient Tree Boosting')
        gbdt.setCheckState(0, Qt.Unchecked)
        # deep learning algorighms
        net1 = QTreeWidgetItem(self.machineLearningAlgorighms)
        net1.setText(0, 'Net_1_CNN')
        net1.setText(1, 'Convolutional Neural Network')
        net1.setCheckState(0, Qt.Unchecked)
        net11 = QTreeWidgetItem(self.machineLearningAlgorighms)
        net11.setText(0, 'Net_1_CNN_binary')
        net11.setText(1, 'Convolutional Neural Network')
        net11.setCheckState(0, Qt.Unchecked)
        net2 = QTreeWidgetItem(self.machineLearningAlgorighms)
        net2.setText(0, 'Net_2_RNN')
        net2.setText(1, 'Recurrent Neural Network')
        net2.setCheckState(0, Qt.Unchecked)
        net3 = QTreeWidgetItem(self.machineLearningAlgorighms)
        net3.setText(0, 'Net_3_BRNN')
        net3.setText(1, 'Bidirectional Recurrent Neural Network')
        net3.setCheckState(0, Qt.Unchecked)
        net4 = QTreeWidgetItem(self.machineLearningAlgorighms)
        net4.setText(0, 'Net_4_ABCNN')
        net4.setText(1, 'Attention Based Convolutional Neural Network')
        net4.setCheckState(0, Qt.Unchecked)
        net5 = QTreeWidgetItem(self.machineLearningAlgorighms)
        net5.setText(0, 'Net_5_ResNet')
        net5.setText(1, 'Deep Residual Network')
        net5.setCheckState(0, Qt.Unchecked)
        net6 = QTreeWidgetItem(self.machineLearningAlgorighms)
        net6.setText(0, 'Net_6_AE')
        net6.setText(1, 'AutoEncoder')
        net6.setCheckState(0, Qt.Unchecked)
        treeLayout.addWidget(self.ml_treeWidget)
        treeGroupBox.setLayout(treeLayout)
        self.ml_treeWidget.clicked.connect(self.ml_tree_clicked)
        self.ml_treeWidget.itemChanged.connect(self.ml_tree_checkState)

        ## parameter
        paraGroupBox = QGroupBox('Set K-fold Cross-Validation', self)
        paraGroupBox.setFont(QFont('Arial', 10))
        paraLayout = QFormLayout(paraGroupBox)
        self.fold_lineEdit = InputDialog.MyLineEdit('5')
        self.fold_lineEdit.setFont(QFont('Arial', 8))
        self.fold_lineEdit.clicked.connect(self.setFold)
        paraLayout.addRow('Cross-Validation:', self.fold_lineEdit)

        # operation
        startGroupBox = QGroupBox('Operator', self)
        startGroupBox.setFont(QFont('Arial', 10))
        startLayout = QHBoxLayout(startGroupBox)
        self.start_button = QPushButton('Start')
        self.start_button.clicked.connect(self.run_ml)
        self.start_button.setFont(QFont('Arial', 10))
        startLayout.addWidget(self.start_button)

        ### layout
        left_vertical_layout = QVBoxLayout()
        left_vertical_layout.addWidget(topGroupBox)
        left_vertical_layout.addWidget(treeGroupBox)
        left_vertical_layout.addWidget(paraGroupBox)
        left_vertical_layout.addWidget(startGroupBox)

        #### widget
        leftWidget = QWidget()
        leftWidget.setLayout(left_vertical_layout)

        resultTabWidget = QWidget()
        resultTabLayout = QVBoxLayout(resultTabWidget)
        resultTabControlLayout = QHBoxLayout()
        self.resultSaveBtn = QPushButton('Save metrics')
        self.resultSaveBtn.clicked.connect(self.save_result)
        self.modelSaveBtn = QPushButton('Save model')
        self.modelSaveBtn.clicked.connect(self.save_model)
        self.displayCorrBtn = QPushButton(' Display Correlation ')
        self.displayCorrBtn.clicked.connect(self.display_correlation_heatmap)
        self.combineModelBtn = QPushButton(' Combine models ')
        self.combineModelBtn.clicked.connect(self.combineModels)

        resultTabControlLayout.addStretch(1)
        resultTabControlLayout.addWidget(self.resultSaveBtn)
        resultTabControlLayout.addWidget(self.modelSaveBtn)
        resultTabControlLayout.addWidget(self.displayCorrBtn)
        resultTabControlLayout.addWidget(self.combineModelBtn)
        resultTabControlLayout.addStretch(1)
        self.metricsTableWidget = QTableWidget()
        self.metricsTableWidget.setFont(QFont('Arial', 8))
        self.metricsTableWidget.setColumnCount(11)
        self.metricsTableWidget.setHorizontalHeaderLabels(['Id', 'StartTime', 'EndTime', 'Sn (%)', 'Sp (%)', 'Pre (%)', 'Acc (%)', 'MCC', 'F1', 'AUROC', 'AUPRC'])
        self.metricsTableWidget.verticalHeader().setHidden(True)
        self.metricsTableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.metricsTableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.metricsTableWidget.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.metricsTableWidget.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.metricsTableWidget.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.metricsTableWidget.resizeRowsToContents()
        resultTabLayout.addWidget(self.metricsTableWidget)
        resultTabLayout.addLayout(resultTabControlLayout)

        """ boxplot using matplotlib """
        boxplotWidget = QWidget()
        self.boxplotLayout = QVBoxLayout(boxplotWidget)
        self.boxplotGraph = PlotWidgets.BoxplotWidget()
        self.boxplotLayout.addWidget(self.boxplotGraph)

        """ ROC curve using matplotlib """
        rocWidget = QWidget()
        self.rocLayout = QVBoxLayout(rocWidget)
        self.rocCurveGraph = PlotWidgets.CurvesWidget()
        self.rocLayout.addWidget(self.rocCurveGraph)

        logWidget = QWidget()
        logLayout = QHBoxLayout(logWidget)
        self.logTextEdit = QTextEdit()
        self.logTextEdit.setFont(QFont('Arial', 8))
        logLayout.addWidget(self.logTextEdit)

        plotTabWidget = QTabWidget()
        plotTabWidget.addTab(resultTabWidget, '  Result  ')
        plotTabWidget.addTab(boxplotWidget, '  Box plot  ')
        plotTabWidget.addTab(rocWidget, '  ROC and PRC curve  ')
        plotTabWidget.addTab(logWidget, '  Log  ')

        ##### splitter
        splitter_1 = QSplitter(Qt.Horizontal)
        splitter_1.addWidget(leftWidget)
        splitter_1.addWidget(plotTabWidget)
        splitter_1.setSizes([100, 1200])

        ###### vertical layout
        vLayout = QVBoxLayout()

        ## status bar
        statusGroupBox = QGroupBox('Status', self)
        statusGroupBox.setFont(QFont('Arial', 10))
        statusLayout = QHBoxLayout(statusGroupBox)
        self.status_label = QLabel('Welcome to iLearnPlus Analysis')
        self.progress_bar = QLabel()
        self.progress_bar.setMaximumWidth(230)
        statusLayout.addWidget(self.status_label)
        statusLayout.addWidget(self.progress_bar)

        splitter_2 = QSplitter(Qt.Vertical)
        splitter_2.addWidget(splitter_1)
        splitter_2.addWidget(statusGroupBox)
        splitter_2.setSizes([1000, 100])
        vLayout.addWidget(splitter_2)
        self.setLayout(vLayout)

    def data_from_file_ml(self, target='Training'):
        file_name, ok = QFileDialog.getOpenFileName(self, 'Open', './data', 'CSV Files (*.csv);;TSV Files (*.tsv);;SVM Files(*.svm);;Weka Files (*.arff)')
        if ok:
            if self.MLData is None:
                self.MLData = MachineLearning.ILearnMachineLearning(self.ml_defatult_para)
            ok1 = self.MLData.load_data(file_name, target)
            if ok1:
                if target == 'Training':
                    shape = self.MLData.training_dataframe.values.shape
                else:
                    shape = self.MLData.testing_dataframe.values.shape
                self.logTextEdit.append('%s\tLoad %s file; Datashape: %s' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), target, shape))
            else:
                self.logTextEdit.append('%s\tLoad %s file failed.' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), target))

    def ml_tree_clicked(self, index):
        item = self.ml_treeWidget.currentItem()
        if item.text(0) in ['RF']:
            num, range, cpu, auto, ok = InputDialog.QRandomForestInput.getValues()
            if ok:
                self.ml_defatult_para['n_trees'] = num
                self.ml_defatult_para['tree_range'] = range
                # self.ml_defatult_para['auto'] = auto
                # self.ml_defatult_para['cpu'] = cpu
                self.para_rf['auto'] = auto
                self.para_rf['cpu'] = cpu
                if auto:
                    self.logTextEdit.append('%s\tAlgorithm: %s\t[Auto-optimization\ttree_range=%s]' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), str(range)))
                else:
                    self.logTextEdit.append('%s\tAlgorithm: %s\t[tree number=%s]' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), num))
        elif item.text(0) in ['SVM']:
            kernel, penality, gamma, auto, penalityRange, gammaRange, ok = InputDialog.QSupportVectorMachineInput.getValues()
            if ok:
                self.ml_defatult_para['kernel'] = kernel
                self.ml_defatult_para['penality'] = penality
                self.ml_defatult_para['gamma'] = gamma
                # self.ml_defatult_para['auto'] = auto
                self.para_svm['auto'] = auto
                self.ml_defatult_para['penalityRange'] = penalityRange
                self.ml_defatult_para['gammaRange'] = gammaRange
                if auto:
                    self.logTextEdit.append('%s\tAlgorithm: %s\t[kernel=%s\tAuto-optimization\tpenality range=%s\tgamma range=%s]' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), kernel, str(penalityRange), str(gammaRange)))
                else:
                    self.logTextEdit.append('%s\tAlgorithm: %s\t[kernel=%s\tpenality=%s\tgamma=%s]' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), kernel, penality, gamma))
        elif item.text(0) in ['MLP']:
            layer, epochs, activation, optimizer, ok = InputDialog.QMultiLayerPerceptronInput.getValues()
            if ok:
                self.ml_defatult_para['layer'] = layer
                self.ml_defatult_para['epochs'] = epochs
                self.ml_defatult_para['activation'] = activation
                self.ml_defatult_para['optimizer'] = optimizer
                self.logTextEdit.append('%s\tAlgorithm: %s\t[layer=%s\tepochs=%s\tactivation function=%s\toptimizer=%s]' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), layer, epochs, activation, optimizer))
        elif item.text(0) in ['LR', 'SGD', 'DecisionTree', 'NaiveBayes', 'AdaBoost', 'GBDT', 'LDA', 'QDA']:
            self.logTextEdit.append('%s\tAlgorithm: %s' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0)))
        elif item.text(0) in ['KNN']:
            topKValue, ok = InputDialog.QKNeighborsInput.getValues()
            if ok:
                self.ml_defatult_para['topKValue'] = topKValue
                self.logTextEdit.append('%s\tAlgorithm: %s\t[top K value=%s]' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), topKValue))
        elif item.text(0) in ['LightGBM']:
            type, leaves, depth, rate, leavesRange, depthRange, rateRange, threads, auto, ok = InputDialog.QLightGBMInput.getValues()
            if ok:
                self.ml_defatult_para['boosting_type'] = type
                self.ml_defatult_para['num_leaves'] = leaves
                # self.ml_defatult_para['max_depth'] = depth
                # self.ml_defatult_para['learning_rate'] = rate
                # self.ml_defatult_para['auto'] = auto
                self.ml_defatult_para['leaves_range'] = leavesRange
                # self.ml_defatult_para['depth_range'] = depthRange
                # self.ml_defatult_para['rate_range'] = rateRange
                # self.ml_defatult_para['cpu'] = threads
                self.para_lightgbm['max_depth'] = depth
                self.para_lightgbm['learning_rate'] = rate
                self.para_lightgbm['auto'] = auto
                self.para_lightgbm['depth_range'] = depthRange
                self.para_lightgbm['rate_range'] = rateRange
                self.para_lightgbm['cpu'] = threads
                if auto:
                    self.logTextEdit.append('%s\tAlgorithm: %s\t[Auto optimization\tLeaves range=%s\tDepth range=%s\tLearning rate range=%s]' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), leavesRange, depthRange, rateRange))
                else:
                    self.logTextEdit.append('%s\tAlgorithm: %s\t[boosting type=%s\tnumber of leaves=%s\tmax depth=%s\tlearning rate=%s\tnumber of threads=%s]' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), type, leaves, depth, rate, threads))
        elif item.text(0) in ['XGBoost']:
            booster, maxdepth, rate, estimator, colsample, depthRange, rateRange, threads, auto, ok = InputDialog.QXGBoostInput.getValues()
            self.ml_defatult_para['booster'] = booster
            # self.ml_defatult_para['max_depth'] = maxdepth
            # self.ml_defatult_para['learning_rate'] = rate
            # self.ml_defatult_para['n_estimator'] = estimator
            self.ml_defatult_para['colsample_bytree'] = colsample
            # self.ml_defatult_para['depth_range'] = depthRange
            # self.ml_defatult_para['rate_range'] = rateRange
            # self.ml_defatult_para['cpu'] = threads
            # self.ml_defatult_para['auto'] = auto
            self.para_xgboost['max_depth'] = maxdepth
            self.para_xgboost['learning_rate'] = rate
            self.para_xgboost['n_estimator'] = estimator
            self.para_xgboost['depth_range'] = depthRange
            self.para_xgboost['rate_range'] = rateRange
            self.para_xgboost['cpu'] = threads
            self.para_xgboost['auto'] = auto
            if auto:
                self.logTextEdit.append('%s\tAlgorithm: %s\t[Auto optimization\tDepth range=%s\tLearning rate range=%s]' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), depthRange, rateRange))
            else:
                self.logTextEdit.append('%s\tAlgorithm: %s\t[Booster=%s\tMax depth=%s\tLearning rate=%s\tn_estimator=%s\tcolsample_bytree=%s]' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), booster, maxdepth, rate, estimator, colsample))
        elif item.text(0) in ['Bagging']:
            n_estimators, threads, ok = InputDialog.QBaggingInput.getValues()
            if ok:
                self.para_bagging['n_estimator'] = n_estimators
                self.para_bagging['cpu'] = threads
                self.logTextEdit.append('%s\tAlgorithm: %s\t[n_estimators=%s\tThreads=%s]' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), n_estimators, threads))
        elif item.text(0) in ['Net_1_CNN']:
            if not self.MLData is None:
                input_channel, input_length, output_channel, padding, kernel_size, dropout, learning_rate, epochs, early_stopping, batch_size, fc_size, ok = InputDialog.QNetInput_1.getValues(self.MLData.training_dataframe.values.shape[1])
                if ok:
                    self.para_net_1['input_channel'] = input_channel
                    self.para_net_1['input_length'] = input_length
                    self.para_net_1['output_channel'] = output_channel
                    self.para_net_1['padding'] = padding
                    self.para_net_1['kernel_size'] = kernel_size
                    self.para_net_1['dropout'] = dropout
                    self.para_net_1['learning_rate'] = learning_rate
                    self.para_net_1['epochs'] = epochs
                    self.para_net_1['early_stopping'] = early_stopping
                    self.para_net_1['batch_size'] = batch_size
                    self.para_net_1['fc_size'] = fc_size
                    self.logTextEdit.append('%s\tAlgorithm: %s; Input channel=%s; Input_length=%s; Output_channel=%s; Padding=%s; Kernel_size=%s; Dropout=%s; lr=%s; Epochs=%s; Early_stopping=%s; Batch_size=%s' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), input_channel, input_length, output_channel, padding, kernel_size, dropout, learning_rate, epochs, early_stopping, batch_size))
            else:
                QMessageBox.warning(self, 'Warning', 'Please input training data at first!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        elif item.text(0) in ['Net_1_CNN_binary']:
            if not self.MLData is None:
                input_channel, input_length, output_channel, padding, kernel_size, dropout, learning_rate, epochs, early_stopping, batch_size, fc_size, ok = InputDialog.QNetInput_1.getValues(self.MLData.training_dataframe.values.shape[1])
                if ok:
                    self.para_net_11['input_channel'] = input_channel
                    self.para_net_11['input_length'] = input_length
                    self.para_net_11['output_channel'] = output_channel
                    self.para_net_11['padding'] = padding
                    self.para_net_11['kernel_size'] = kernel_size
                    self.para_net_11['dropout'] = dropout
                    self.para_net_11['learning_rate'] = learning_rate
                    self.para_net_11['epochs'] = epochs
                    self.para_net_11['early_stopping'] = early_stopping
                    self.para_net_11['batch_size'] = batch_size
                    self.para_net_11['fc_size'] = fc_size
                    self.logTextEdit.append('%s\tAlgorithm: %s; Input channel=%s; Input_length=%s; Output_channel=%s; Padding=%s; Kernel_size=%s; Dropout=%s; lr=%s; Epochs=%s; Early_stopping=%s; Batch_size=%s' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), input_channel, input_length, output_channel, padding, kernel_size, dropout, learning_rate, epochs, early_stopping, batch_size))
            else:
                QMessageBox.warning(self, 'Warning', 'Please input training data at first!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        elif item.text(0) in ['Net_2_RNN']:
            if not self.MLData is None:
                input_channel, input_length, hidden_size, num_layers, dropout, learning_rate, epochs, early_stopping, batch_size, fc_size, ok = InputDialog.QNetInput_2.getValues(self.MLData.training_dataframe.values.shape[1])
                if ok:
                    self.para_net_2['input_channel'] = input_channel
                    self.para_net_2['input_length'] = input_length
                    self.para_net_2['rnn_hidden_size'] = hidden_size
                    self.para_net_2['rnn_hidden_layers'] = num_layers
                    self.para_net_2['rnn_bidirection'] = False
                    self.para_net_2['dropout'] = dropout
                    self.para_net_2['learning_rate'] = learning_rate
                    self.para_net_2['epochs'] = epochs
                    self.para_net_2['early_stopping'] = early_stopping
                    self.para_net_2['batch_size'] = batch_size
                    self.para_net_2['rnn_bidirectional'] = False
                    self.logTextEdit.append('%s\tAlgorithm: %s; Input size=%s; Input_length=%s; Hidden_size=%s; Num_hidden_layers=%s; Dropout=%s; lr=%s; Epochs=%s; Early_stopping=%s; Batch_size=%s' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), input_channel, input_length, hidden_size, num_layers, dropout, learning_rate, epochs, early_stopping, batch_size))
            else:
                QMessageBox.warning(self, 'Warning', 'Please input training data at first!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        elif item.text(0) in ['Net_3_BRNN']:
            if not self.MLData is None:
                input_channel, input_length, hidden_size, num_layers, dropout, learning_rate, epochs, early_stopping, batch_size, fc_size, ok = InputDialog.QNetInput_2.getValues(self.MLData.training_dataframe.values.shape[1])
                if ok:
                    self.para_net_3['input_channel'] = input_channel
                    self.para_net_3['input_length'] = input_length
                    self.para_net_3['rnn_hidden_size'] = hidden_size
                    self.para_net_3['rnn_hidden_layers'] = num_layers
                    self.para_net_3['rnn_bidirection'] = False
                    self.para_net_3['dropout'] = dropout
                    self.para_net_3['learning_rate'] = learning_rate
                    self.para_net_3['epochs'] = epochs
                    self.para_net_3['early_stopping'] = early_stopping
                    self.para_net_3['batch_size'] = batch_size
                    self.para_net_3['rnn_bidirectional'] = True
                    self.logTextEdit.append('%s\tAlgorithm: %s; Input size=%s; Input_length=%s; Hidden_size=%s; Num_hidden_layers=%s; Dropout=%s; lr=%s; Epochs=%s; Early_stopping=%s; Batch_size=%s' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), input_channel, input_length, hidden_size, num_layers, dropout, learning_rate, epochs, early_stopping, batch_size))
            else:
                QMessageBox.warning(self, 'Warning', 'Please input training data at first!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        elif item.text(0) in ['Net_4_ABCNN']:
            if not self.MLData is None:
                input_channel, input_length, dropout, learning_rate, epochs, early_stopping, batch_size, ok = InputDialog.QNetInput_4.getValues(self.MLData.training_dataframe.values.shape[1])
                if ok:
                    self.para_net_4['input_channel'] = input_channel
                    self.para_net_4['input_length'] = input_length
                    self.para_net_4['dropout'] = dropout
                    self.para_net_4['learning_rate'] = learning_rate
                    self.para_net_4['epochs'] = epochs
                    self.para_net_4['early_stopping'] = early_stopping
                    self.para_net_4['batch_size'] = batch_size
                    self.logTextEdit.append('%s\tAlgorithm: %s; Input size=%s; Input_length=%s; Dropout=%s; lr=%s; Epochs=%s; Early_stopping=%s; Batch_size=%s' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), input_channel, input_length, dropout, learning_rate, epochs, early_stopping, batch_size))
            else:
                QMessageBox.warning(self, 'Warning', 'Please input training data at first!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        elif item.text(0) in ['Net_5_ResNet']:
            if not self.MLData is None:
                input_channel, input_length, learning_rate, epochs, early_stopping, batch_size, ok = InputDialog.QNetInput_5.getValues(self.MLData.training_dataframe.values.shape[1])
                if ok:
                    self.para_net_5['input_channel'] = input_channel
                    self.para_net_5['input_length'] = input_length
                    self.para_net_5['learning_rate'] = learning_rate
                    self.para_net_5['epochs'] = epochs
                    self.para_net_5['early_stopping'] = early_stopping
                    self.para_net_5['batch_size'] = batch_size
                    self.logTextEdit.append('%s\tAlgorithm: %s; Input size=%s; Input_length=%s; lr=%s; Epochs=%s; Early_stopping=%s; Batch_size=%s'  %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), input_channel, input_length, learning_rate, epochs, early_stopping, batch_size))
            else:
                QMessageBox.warning(self, 'Warning', 'Please input training data at first!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        elif item.text(0) in ['Net_6_AE']:
            if not self.MLData is None:
                input_dim, dropout, learning_rate, epochs, early_stopping, batch_size, ok = InputDialog.QNetInput_6.getValues(self.MLData.training_dataframe.values.shape[1])
                if ok:
                    self.para_net_6['mlp_input_dim'] = input_dim
                    self.para_net_6['dropout'] = dropout
                    self.para_net_6['learning_rate'] = learning_rate
                    self.para_net_6['epochs'] = epochs
                    self.para_net_6['early_stopping'] = early_stopping
                    self.para_net_6['batch_size'] = batch_size
                    self.logTextEdit.append('%s\tAlgorithm: %s; Input dimension=%s; Dropout=%s; lr=%s; Epochs=%s; Early_stopping=%s; Batch_size=%s' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), input_dim, dropout, learning_rate, epochs, early_stopping, batch_size))
            else:
                QMessageBox.warning(self, 'Warning', 'Please input training data at first!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

    def ml_tree_checkState(self, item, column):
        if item and item.text(0) not in ['Machine learning algorithms']:
            self.ml_treeWidget.setCurrentItem(item)
            if item.checkState(column) == Qt.Checked:
                self.algorithms_selected.add(item.text(0))
            if item.checkState(column) == Qt.Unchecked:
                self.algorithms_selected.discard(item.text(0))

    def setFold(self):
        fold, ok = QInputDialog.getInt(self, 'Fold number', 'Setting K-fold cross-validation', 5, 2, 100, 1)
        if ok:
            self.fold_lineEdit.setText(str(fold))
            self.ml_defatult_para['FOLD'] = fold
            self.logTextEdit.append('%s\tSet %s-fold Cross-validation' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), fold))

    def panel_clear(self):
        # metricsTableWidget init
        self.metricsTableWidget.clear()
        self.metricsTableWidget.setColumnCount(11)
        self.metricsTableWidget.setRowCount(0)
        self.current_data_index = 0
        self.metricsTableWidget.setHorizontalHeaderLabels(['Id', 'StartTime', 'EndTime', 'Sn (%)', 'Sp (%)', 'Pre (%)', 'Acc (%)', 'MCC', 'F1', 'AUROC', 'AUPRC'])
        self.descriptor = None
        self.fasta_file = None
        #
        self.metrics = ModelMetrics.ModelMetrics()
        self.boxplot_data = {}

    def restart_init(self):
        # metricsTableWidget init
        self.metricsTableWidget.clear()
        self.metricsTableWidget.setColumnCount(11)
        self.metricsTableWidget.setRowCount(0)
        self.current_data_index = 0
        self.metricsTableWidget.setHorizontalHeaderLabels(['Id', 'StartTime', 'EndTime', 'Sn (%)', 'Sp (%)', 'Pre (%)', 'Acc (%)', 'MCC', 'F1', 'AUROC', 'AUPRC'])
        self.metrics = ModelMetrics.ModelMetrics()
        self.boxplot_data = {}
        self.combineModelBtn.setDisabled(False)

    def run_ml(self):
        self.setDisabled(True)
        self.restart_init()
        if self.MLData is None:
            QMessageBox.critical(self, 'Error', 'Please load training file.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        elif len(self.algorithms_selected) == 0:
            QMessageBox.critical(self, 'Error', 'Please select at least one machine learning algorithm.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        elif self.ml_defatult_para['FOLD'] <= 1:
            QMessageBox.critical(self, 'Error', 'FOLD need > 1', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        else:
            self.logTextEdit.append('%s\tJobs start ...' %datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            self.progress_bar.setMovie(self.gif)
            self.gif.start()
            t = threading.Thread(target=self.train_model)
            t.start()

    def train_model(self):
        for algo in self.algorithms_selected:
            self.status_label.setText('Start training %s model  ...' % algo)
            start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if algo == 'RF':
                for key in self.para_rf:
                    self.ml_defatult_para[key] = self.para_rf[key]                    
                ok = self.MLData.RandomForest()
            elif algo == 'SVM':
                for key in self.para_svm:
                    self.ml_defatult_para[key] = self.para_svm[key]
                ok = self.MLData.SupportVectorMachine()
            elif algo == 'MLP':
                ok = self.MLData.MultiLayerPerceptron()
            elif algo == 'LR':
                ok = self.MLData.LogisticRegressionClassifier()
            elif algo == 'LDA':
                ok = self.MLData.LDAClassifier()
            elif algo == 'QDA':
                ok = self.MLData.QDAClassifier()
            elif algo == 'KNN':
                ok = self.MLData.KNeighbors()
            elif algo == 'LightGBM':
                for key in self.para_lightgbm:
                    self.ml_defatult_para[key] = self.para_lightgbm[key]
                ok = self.MLData.LightGBMClassifier()
            elif algo == 'XGBoost':
                for key in self.para_xgboost:
                    self.ml_defatult_para[key] = self.para_xgboost[key]
                ok = self.MLData.XGBoostClassifier()
            elif algo == 'SGD':
                ok = self.MLData.StochasticGradientDescentClassifier()
            elif algo == 'DecisionTree':
                ok = self.MLData.DecisionTree()
            elif algo == 'NaiveBayes':
                ok = self.MLData.GaussianNBClassifier()
            elif algo == 'AdaBoost':
                ok = self.MLData.AdaBoost()
            elif algo == 'Bagging':
                for key in self.para_bagging:
                    self.ml_defatult_para[key] = self.para_bagging[key]
                ok = self.MLData.Bagging()
            elif algo == 'GBDT':
                ok = self.MLData.GBDTClassifier()
            elif algo == 'Net_1_CNN':
                for key in self.para_net_1:
                    self.ml_defatult_para[key] = self.para_net_1[key]
                ok = self.MLData.run_networks(1)
            elif algo == 'Net_1_CNN_binary':
                for key in self.para_net_11:
                    self.ml_defatult_para[key] = self.para_net_11[key]
                ok = self.MLData.run_networks(11)
            elif algo == 'Net_2_RNN':
                for key in self.para_net_2:
                    self.ml_defatult_para[key] = self.para_net_2[key]
                ok = self.MLData.run_networks(2)
            elif algo == 'Net_3_BRNN':
                for key in self.para_net_3:
                    self.ml_defatult_para[key] = self.para_net_3[key]
                ok = self.MLData.run_networks(3)
            elif algo == 'Net_4_ABCNN':
                for key in self.para_net_4:
                    self.ml_defatult_para[key] = self.para_net_4[key]
                ok = self.MLData.run_networks(4)
            elif algo == 'Net_5_ResNet':
                for key in self.para_net_5:
                    self.ml_defatult_para[key] = self.para_net_5[key]
                ok = self.MLData.run_networks(5)
            elif algo == 'Net_6_AE':
                for key in self.para_net_6:
                    self.ml_defatult_para[key] = self.para_net_6[key]
                ok = self.MLData.run_networks(6)

            self.MLData.calculate_boxplot_data()
            end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if ok:
                self.status_label.setText('%s model training complete.' %algo)
                data_list = ['%s_model' %algo, start_time, end_time] + list(self.MLData.metrics.values[-1])
                item = pd.DataFrame([data_list], columns=['Id', 'StartTime', 'EndTime', 'Sn', 'Sp', 'Pre', 'Acc', 'MCC', 'F1', 'AUROC', 'AUPRC'])
                self.metrics.insert_data(item, '%s_model' %algo, self.MLData.meanAucData, self.MLData.meanPrcData, self.MLData.training_score, self.MLData.best_model)
                self.boxplot_data[algo] = self.MLData.metrics.iloc[0:self.ml_defatult_para['FOLD'], :]
                self.display_signal.emit(data_list)
            else:
                self.status_label.setText('%s model training failed.' %algo)
        self.display_curves_signal.emit()

    def display_metrics(self, data_list):
        index = self.current_data_index
        self.metricsTableWidget.insertRow(index)
        self.current_data_index += 1
        for i in range(len(data_list)):
            self.metricsTableWidget.setItem(index, i, QTableWidgetItem(str(data_list[i])))
        self.metricsTableWidget.resizeRowsToContents()

    def display_curves(self):
        self.status_label.setText('Plotting ... ')
        self.rocLayout.removeWidget(self.rocCurveGraph)
        sip.delete(self.rocCurveGraph)
        self.rocCurveGraph = PlotWidgets.CurvesWidget()
        self.rocLayout.addWidget(self.rocCurveGraph)
        try:
            if self.MLData.task == 'binary':
                self.rocCurveGraph.init_roc_data(0, 'ROC curve', self.metrics.aucData)
            if self.MLData.task == 'binary':
                self.rocCurveGraph.init_prc_data(1, 'PRC curve', self.metrics.prcData)
        except Exception as e:
            QMessageBox.warning(self, 'Warning', str(e), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

        """ boxplot using matplotlib """
        self.boxplotLayout.removeWidget(self.boxplotGraph)
        sip.delete(self.boxplotGraph)
        self.boxplotGraph = PlotWidgets.BoxplotWidget()
        self.boxplotLayout.addWidget(self.boxplotGraph)
        try:
            if not self.MLData is None and self.MLData.task == 'binary':
                self.boxplotGraph.init_data(['Sn', 'Sp', 'Pre', 'Acc', 'MCC', 'F1', 'AUROC', 'AUPRC'], self.boxplot_data)
            if not self.MLData is None and self.MLData.task == 'muti-task':
                self.boxplotGraph.init_data(['Acc'], self.boxplot_data)
        except Exception as e:
            QMessageBox.warning(self, 'Warning', str(e), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

        # init data for p_value calculation (bootstrap test)
        self.rocCurveGraph.init_prediction_scores(self.MLData.task, self.metrics.prediction_scores)

        self.gif.stop()
        self.progress_bar.clear()
        self.start_button.setDisabled(False)
        self.setDisabled(False)
        self.status_label.setText('Operation complete.')

    def save_result(self):
        try:
            saved_file, ok = QFileDialog.getSaveFileName(self, 'Save', './data', 'TSV Files (*.tsv)')
            if ok:
                self.metrics.metrics.to_csv(saved_file, sep='\t', header=True, index=False)
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

    def save_model(self):
        try:
            model_list = self.metrics.metrics.loc[:, 'Id'].values.tolist()
            if len(model_list) > 0:
                model_id, ok = InputDialog.QSelectModel.getValues(model_list)
                if ok:
                    if model_id in self.metrics.models:
                        save_directory = QFileDialog.getExistingDirectory(self, 'Save', './data')
                        if not os.path.exists(save_directory):
                            pass
                        else:
                            for i, model in enumerate(self.metrics.models[model_id]):
                                model_name = '%s/%s_%s.pkl' %(save_directory, model_id, i+1)
                                if 'predict_proba' in dir(model):
                                    joblib.dump(model, model_name)
                                else:
                                    torch.save(model, model_name)
                            QMessageBox.information(self, 'Model saved',
                                                    'The models have been saved to directory %s' % save_directory,
                                                    QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
                    else:
                        QMessageBox.critical(self, 'Error', 'An error has been encountered in saving the model.',
                                            QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
            else:
                QMessageBox.critical(self, 'Error', 'No model can be saved.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

    def combineModels(self):
        model_list = self.metrics.metrics.loc[:, 'Id'].values.tolist()
        if len(model_list) >= 2:
            combineAlgorithm, ok = InputDialog.QSCombineModelDialog.getValues(model_list)
            combinations_array = []
            if ok:
                for model_num in range(2, len(model_list) + 1):
                    combinations_array += list(combinations(model_list, model_num))
                self.logTextEdit.append('%s\tStarting train combined models ...' %datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                self.progress_bar.setMovie(self.gif)
                self.gif.start()
                t = threading.Thread(target=lambda : self.calculateCombinations(combineAlgorithm, combinations_array))
                t.start()
        else:
            QMessageBox.critical(self, 'Error', 'At least two models needed.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

    def calculateCombinations(self, combineAlgorithm, combinations_array):
        self.setDisabled(True)
        start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.ml_combination_para['FOLD'] = self.ml_defatult_para['FOLD']        
        for c in combinations_array:
            self.status_label.setText('Training combined model for combination %s with %s method.' %(str(c), combineAlgorithm))
            msg = '%s\tTraining model for combinations %s with %s method ...' %(start_time, str(c), combineAlgorithm)
            self.append_msg_signal.emit(msg)
            # if self.metrics.classification_task == 'binary':
            #     ensemble_data_X = np.concatenate(tuple(self.metrics.prediction_scores[m].values[:, 3].reshape((-1, 1)) for m in c), axis=1)
            # else:
            #     ensemble_data_X = np.concatenate(tuple(self.metrics.prediction_scores[m].values[:, 2:] for m in c), axis=1)
            ensemble_data_X = np.concatenate(tuple(self.metrics.prediction_scores[m].values[:, 2:] for m in c), axis=1)
            ensemble_data_y = self.metrics.prediction_scores[c[0]].values[:, 1]            
            df = pd.DataFrame(ensemble_data_X.astype(float))
            label = ensemble_data_y.astype(int)
            self.MLData_cb = MachineLearning.ILearnMachineLearning(self.ml_combination_para)
            self.MLData_cb.import_training_data(df, label)
            
            if combineAlgorithm == 'RF':
                ok = self.MLData_cb.RandomForest()
            elif combineAlgorithm == 'SVM':
                ok = self.MLData_cb.SupportVectorMachine()
            elif combineAlgorithm == 'MLP':
                ok = self.MLData_cb.MultiLayerPerceptron()
            elif combineAlgorithm == 'LR':
                ok = self.MLData_cb.LogisticRegressionClassifier()
            elif combineAlgorithm == 'LDA':
                ok = self.MLData_cb.LDAClassifier()
            elif combineAlgorithm == 'QDA':
                ok = self.MLData_cb.QDAClassifier()
            elif combineAlgorithm == 'KNN':
                ok = self.MLData_cb.KNeighbors()
            elif combineAlgorithm == 'LightGBM':
                ok = self.MLData_cb.LightGBMClassifier()
            elif combineAlgorithm == 'XGBoost':
                ok = self.MLData_cb.XGBoostClassifier()
            elif combineAlgorithm == 'SGD':
                ok = self.MLData_cb.StochasticGradientDescentClassifier()
            elif combineAlgorithm == 'DecisionTree':
                ok = self.MLData_cb.DecisionTree()
            elif combineAlgorithm == 'NaiveBayes':
                ok = self.MLData_cb.GaussianNBClassifier()
            elif combineAlgorithm == 'AdaBoost':
                ok = self.MLData_cb.AdaBoost()
            elif combineAlgorithm == 'Bagging':
                ok = self.MLData_cb.Bagging()
            elif combineAlgorithm == 'GBDT':
                ok = self.MLData_cb.GBDTClassifier()

            end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')            
            if ok:
                self.status_label.setText('Training combined model for combination %s complete.' %str(c))
                msg = '%s\tTraining combined model for combination %s complete.' %(end_time, str(c))
                self.append_msg_signal.emit(msg)
                if self.metrics.classification_task == 'binary':
                    if self.MLData_cb.metrics.loc['Mean', 'AUROC'] > self.bestPerformance:
                        self.bestPerformance = self.MLData_cb.metrics.loc['Mean', 'AUROC']                           
                        self.bestMetrics = self.MLData_cb.metrics
                        self.bestAUC = self.MLData_cb.meanAucData
                        self.bestPRC = self.MLData_cb.meanPrcData
                        self.bestModels = self.MLData_cb.best_model
                        self.bestTrainingScore = self.MLData_cb.training_score
                        self.boxplot_data['Combined'] = self.MLData_cb.metrics.iloc[0:self.ml_defatult_para['FOLD'], :]

                else:
                    if self.MLData_cb.metrics.loc['Mean', 'AUROC'] > self.bestPerformance:
                        self.bestPerformance = self.MLData_cb.metrics.loc['Mean', 'Acc']
                        self.bestMetrics = self.MLData_cb.metrics
                        self.bestAUC = self.MLData_cb.meanAucData
                        self.bestPRC = self.MLData_cb.meanPrcData
                        self.bestModels = self.MLData_cb.best_model
                        self.bestTrainingScore = self.MLData_cb.training_score
                        self.boxplot_data['Combined'] = self.MLData_cb.metrics.iloc[0:self.ml_defatult_para['FOLD'], :]

        data_list = ['Combined_model', start_time, end_time] + list(self.bestMetrics.values[-1])
        item = pd.DataFrame([data_list], columns=['Id', 'StartTime', 'EndTime', 'Sn', 'Sp', 'Pre', 'Acc', 'MCC', 'F1', 'AUROC', 'AUPRC'])
        self.metrics.insert_data(item, 'Combined_model', self.bestAUC, self.bestPRC, self.bestTrainingScore, self.bestModels)

        msg = '%s\tThe combination with best performance is %s' % (end_time, str(c))
        self.append_msg_signal.emit(msg)

        self.display_signal.emit(data_list)
        self.display_curves_signal.emit()
        self.setDisabled(False)
        self.combineModelBtn.setDisabled(True)

    def append_message(self, message):
        self.logTextEdit.append(message)

    def display_correlation_heatmap(self):
        dataframe = self.metrics.metrics.iloc[:, 3:]
        dataframe.index = self.metrics.metrics.iloc[:, 0].values
        try:
            if self.MLData.task == 'binary':
                self.corrWindow = PlotWidgets.HeatmapWidget(dataframe)
                self.corrWindow.show()
            else:
                QMessageBox.critical(self, 'Error', 'Correlation can not be calculated for muti-classification task.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Confirm Exit', 'Are you sure want to quit iLearnPlus?',
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close_signal.emit('AutoML')
            self.close()
        else:
            if event:
                event.ignore()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setFont(QFont('Arial', 10))
    win = ILearnPlusAutoML()
    win.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    win.show()
    sys.exit(app.exec_())