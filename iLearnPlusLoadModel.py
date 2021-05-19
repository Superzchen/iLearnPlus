#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import sys, os, re
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QFileDialog, QLabel, QHBoxLayout, QGroupBox, QTextEdit,
                             QVBoxLayout, QSplitter, QTableWidget, QTabWidget,
                             QTableWidgetItem, QMessageBox, QFormLayout, QRadioButton,
                             QHeaderView,
                             QAbstractItemView)
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt, pyqtSignal
from util import PlotWidgets
import numpy as np
import pandas as pd
import torch
from util.EvaluationMetrics import Metrics
from torch.utils.data import DataLoader
from util.Nets import (DealDataset, Net_CNN_1, Net_CNN_11, Net_RNN_2, Net_ABCNN_4, Net_ResNet_5, Net_AutoEncoder_6)
import qdarkstyle
import sip
import joblib

class iLearnPlusLoadModel(QWidget):
    close_signal = pyqtSignal(str)
    def __init__(self):
        super(iLearnPlusLoadModel, self).__init__()

        """ Machine Learning Variable """
        self.data_index = {
            'Training_data': None,
            'Testing_data': None,
            'Training_score': None,
            'Testing_score': None,
            'Metrics': None,
            'ROC': None,
            'PRC': None,
            'Model': None,
        }
        self.current_data_index = 0
        self.ml_running_status = False

        self.model_list = []
        self.dataframe = None
        self.datalabel = None
        self.score = None
        self.metrics = None
        self.aucData = None
        self.prcData = None

        # initialize UI
        self.initUI()

    def initUI(self):
        self.setWindowTitle('iLearnPlus LoadModel')
        self.resize(800, 600)
        self.setWindowState(Qt.WindowMaximized)
        self.setWindowIcon(QIcon(os.path.join(pPath, 'images', 'logo.ico')))

        # file
        topGroupBox = QGroupBox('Load data', self)
        topGroupBox.setFont(QFont('Arial', 10))
        topGroupBox.setMinimumHeight(100)
        topGroupBoxLayout = QFormLayout()
        modelFileButton = QPushButton('Load')
        modelFileButton.setToolTip('One or more models could be loaded.')
        modelFileButton.clicked.connect(self.loadModel)
        testFileButton = QPushButton('Open')
        testFileButton.clicked.connect(self.loadDataFile)
        topGroupBoxLayout.addRow('Open model file(s):', modelFileButton)
        topGroupBoxLayout.addRow('Open testing file:', testFileButton)
        topGroupBox.setLayout(topGroupBoxLayout)

        # start button
        startGroupBox = QGroupBox('Operator', self)
        startGroupBox.setFont(QFont('Arial', 10))
        startLayout = QHBoxLayout(startGroupBox)
        self.ml_start_button = QPushButton('Start')
        self.ml_start_button.clicked.connect(self.run_model)
        self.ml_start_button.setFont(QFont('Arial', 10))
        self.ml_save_button = QPushButton('Save')
        self.ml_save_button.setFont(QFont('Arial', 10))
        self.ml_save_button.clicked.connect(self.save_ml_files)
        startLayout.addWidget(self.ml_start_button)
        startLayout.addWidget(self.ml_save_button)

        # log
        logGroupBox = QGroupBox('Log', self)
        logGroupBox.setFont(QFont('Arial', 10))
        logLayout = QHBoxLayout(logGroupBox)
        self.logTextEdit = QTextEdit()
        self.logTextEdit.setFont(QFont('Arial', 8))
        logLayout.addWidget(self.logTextEdit)


        ### layout
        left_vertical_layout = QVBoxLayout()
        left_vertical_layout.addWidget(topGroupBox)
        left_vertical_layout.addWidget(logGroupBox)
        left_vertical_layout.addWidget(startGroupBox)

        #### widget
        leftWidget = QWidget()
        leftWidget.setLayout(left_vertical_layout)

        #### view region
        scoreTabWidget = QTabWidget()
        trainScoreWidget = QWidget()
        scoreTabWidget.setFont(QFont('Arial', 8))
        scoreTabWidget.addTab(trainScoreWidget, 'Prediction score and evaluation metrics')
        train_score_layout = QVBoxLayout(trainScoreWidget)
        self.train_score_tableWidget = QTableWidget()
        self.train_score_tableWidget.setFont(QFont('Arial', 8))
        self.train_score_tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.train_score_tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        train_score_layout.addWidget(self.train_score_tableWidget)

        self.metricsTableWidget = QTableWidget()
        self.metricsTableWidget.setFont(QFont('Arial', 8))
        self.metricsTableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.metricsTableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.metricsTableWidget.resizeRowsToContents()
        splitter_middle = QSplitter(Qt.Vertical)
        splitter_middle.addWidget(scoreTabWidget)
        splitter_middle.addWidget(self.metricsTableWidget)

        self.dataTableWidget = QTableWidget(2, 4)
        self.dataTableWidget.setFont(QFont('Arial', 8))
        self.dataTableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.dataTableWidget.setShowGrid(False)
        self.dataTableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.dataTableWidget.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.dataTableWidget.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.dataTableWidget.setHorizontalHeaderLabels(['Select', 'Data', 'Shape', 'Source'])
        self.dataTableWidget.verticalHeader().setVisible(False)

        self.roc_curve_widget = PlotWidgets.CurveWidget()
        self.prc_curve_widget = PlotWidgets.CurveWidget()
        plotTabWidget = QTabWidget()
        plotTabWidget.setFont(QFont('Arial', 8))
        rocWidget = QWidget()
        self.rocLayout = QVBoxLayout(rocWidget)
        self.rocLayout.addWidget(self.roc_curve_widget)
        prcWidget = QWidget()
        self.prcLayout = QHBoxLayout(prcWidget)
        self.prcLayout.addWidget(self.prc_curve_widget)
        plotTabWidget.addTab(rocWidget, 'ROC curve')
        plotTabWidget.addTab(prcWidget, 'PRC curve')
        splitter_right = QSplitter(Qt.Vertical)
        splitter_right.addWidget(self.dataTableWidget)
        splitter_right.addWidget(plotTabWidget)
        splitter_right.setSizes([100, 300])

        splitter_view = QSplitter(Qt.Horizontal)
        splitter_view.addWidget(splitter_middle)
        splitter_view.addWidget(splitter_right)
        splitter_view.setSizes([100, 200])

        ##### splitter
        splitter_1 = QSplitter(Qt.Horizontal)
        splitter_1.addWidget(leftWidget)
        splitter_1.addWidget(splitter_view)
        splitter_1.setSizes([100, 1200])

        ###### vertical layout
        vLayout = QVBoxLayout()

        ## status bar
        statusGroupBox = QGroupBox('Status', self)
        statusGroupBox.setFont(QFont('Arial', 10))
        statusLayout = QHBoxLayout(statusGroupBox)
        self.ml_status_label = QLabel('Welcome to iLearnPlus LoadModel')
        self.ml_progress_bar = QLabel()
        self.ml_progress_bar.setMaximumWidth(230)
        statusLayout.addWidget(self.ml_status_label)
        statusLayout.addWidget(self.ml_progress_bar)

        splitter_2 = QSplitter(Qt.Vertical)
        splitter_2.addWidget(splitter_1)
        splitter_2.addWidget(statusGroupBox)
        splitter_2.setSizes([1000, 100])
        vLayout.addWidget(splitter_2)
        self.setLayout(vLayout)

    def loadModel(self):
        model_files, ok = QFileDialog.getOpenFileNames(self, 'Open', os.path.join(pPath, 'data'), 'PKL Files (*.pkl)')
        if len(model_files) > 0:
            self.model_list = []
            for file in model_files:
                error_tag_ml = False
                error_tag_dl = False
                model = None
                try:
                    model = joblib.load(file)
                except Exception as e:
                    error_tag_ml = True
                
                if error_tag_ml:
                    try:
                        model = torch.load(file)
                    except Exception as e:
                        error_tag_dl = True
                else:
                    if 'predict_proba' not in dir(model):
                        try:
                            model = torch.load(file)
                        except Exception as e:
                            error_tag_dl = True

                if not error_tag_dl or not error_tag_ml:
                    self.model_list.append(model)
            if len(self.model_list) > 0:                
                self.logTextEdit.append('Load model successfully.')
                self.logTextEdit.append('Model number: %s' %len(model_files))
                return True
            else:
                QMessageBox.critical(self, 'Error', 'Load model failed.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
                self.model_list = []
                return False
        else:
            return False

    def loadDataFile(self):
        file, ok = QFileDialog.getOpenFileName(self, 'Open', os.path.join(pPath, 'data'), 'CSV Files (*.csv);;TSV Files (*.tsv);;SVM Files(*.svm);;Weka Files (*.arff)')
        if ok:
            if not os.path.exists(file):
                QMessageBox.critical(self, 'Error', 'Data file does not exist.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
                return False
            self.dataframe, self.datalabel = None, None
            try:
                if file.endswith('.tsv'):
                    df = pd.read_csv(file, sep='\t', header=None)
                    self.dataframe = df.iloc[:, 1:]
                    self.dataframe.index = ['Sample_%s' % i for i in range(dataframe.values.shape[0])]
                    self.dataframe.columns = ['F_%s' % i for i in range(dataframe.values.shape[1])]
                    self.datalabel = np.array(df.iloc[:, 0]).astype(int)
                elif file.endswith('.csv'):
                    df = pd.read_csv(file, sep=',', header=None)
                    self.dataframe = df.iloc[:, 1:]
                    self.dataframe.index = ['Sample_%s' % i for i in range(self.dataframe.values.shape[0])]
                    self.dataframe.columns = ['F_%s' % i for i in range(self.dataframe.values.shape[1])]
                    self.datalabel = np.array(df.iloc[:, 0]).astype(int)
                elif file.endswith('.svm'):
                    with open(file) as f:
                        record = f.read().strip()
                    record = re.sub('\d+:', '', record)
                    array = np.array([[i for i in item.split()] for item in record.split('\n')])
                    self.dataframe = pd.DataFrame(array[:, 1:], dtype=float)
                    self.dataframe.index = ['Sample_%s' % i for i in range(self.dataframe.values.shape[0])]
                    self.dataframe.columns = ['F_%s' % i for i in range(self.dataframe.values.shape[1])]
                    self.datalabel = array[:, 0].astype(int)
                else:
                    with open(file) as f:
                        record = f.read().strip().split('@')[-1].split('\n')[1:]
                    array = np.array([item.split(',') for item in record])
                    self.dataframe = pd.DataFrame(array[:, 0:-1], dtype=float)
                    self.dataframe.index = ['Sample_%s' % i for i in range(self.dataframe.values.shape[0])]
                    self.dataframe.columns = ['F_%s' % i for i in range(self.dataframe.values.shape[1])]
                    label = []
                    for i in array[:, -1]:
                        if i == 'yes':
                            label.append(1)
                        else:
                            label.append(0)
                    self.datalabel = np.array(label)

            except Exception as e:
                QMessageBox.critical(self, 'Error', 'Open data file failed.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
                return False
            self.logTextEdit.append('Load data file successfully.')
            self.logTextEdit.append('Data shape: %s' %(str(self.dataframe.values.shape)))
            return True
        else:
            return False

    def run_model(self):
        # reset
        self.score = None
        self.metrics = None

        if len(self.model_list) > 0 and not self.dataframe is None:
            try:
                prediction_score = None
                for model in self.model_list:
                    if 'predict_proba' not in dir(model):
                        valid_set = DealDataset(self.dataframe.values, self.datalabel.reshape((-1, 1)))
                        valid_loader = DataLoader(valid_set, batch_size=512, shuffle=False)
                        tmp_prediction_score = model.predict(valid_loader)
                    else:
                        tmp_prediction_score = model.predict_proba(self.dataframe.values)

                    if prediction_score is None:
                        prediction_score = tmp_prediction_score
                    else:
                        prediction_score += tmp_prediction_score
                prediction_score /= len(self.model_list)
                self.score = prediction_score

                # display prediction score
                if not self.score is None:
                    data = self.score
                    self.train_score_tableWidget.setRowCount(data.shape[0])
                    self.train_score_tableWidget.setColumnCount(data.shape[1])
                    self.train_score_tableWidget.setHorizontalHeaderLabels(['Score for category %s' %i for i in range(data.shape[1])])
                    for i in range(data.shape[0]):
                        for j in range(data.shape[1]):
                            cell = QTableWidgetItem(str(round(data[i][j], 4)))
                            self.train_score_tableWidget.setItem(i, j, cell)
                    if self.data_index['Training_score'] is None:
                        # index = self.current_data_index
                        index = 0
                        self.data_index['Training_score'] = index
                        self.dataTableWidget.insertRow(index)
                        self.current_data_index += 1
                    else:
                        # index = self.data_index['Training_score']
                        index = 0
                    self.training_score_radio = QRadioButton()
                    self.dataTableWidget.setCellWidget(index, 0, self.training_score_radio)
                    self.dataTableWidget.setItem(index, 1, QTableWidgetItem('Training score'))
                    self.dataTableWidget.setItem(index, 2, QTableWidgetItem(str(data.shape)))
                    self.dataTableWidget.setItem(index, 3, QTableWidgetItem('NA'))

                # calculate and display evaluation metrics
                column_name = ['Sn', 'Sp', 'Pre', 'Acc', 'MCC', 'F1', 'AUROC', 'AUPRC']
                if not self.score is None and self.score.shape[1] == 2:
                    # calculate metrics
                    data = self.score
                    metrics = Metrics(data[:, -1], self.datalabel)
                    metrics_ind = np.array(
                        [metrics.sensitivity, metrics.specificity, metrics.precision, metrics.accuracy, metrics.mcc,
                         metrics.f1, metrics.auc, metrics.prc]).reshape((1, -1))
                    index_name = ['Metrics value']
                    self.aucData = ['AUROC = %s' % metrics.auc, metrics.aucDot]
                    self.prcData = ['AUPRC = %s' % metrics.prc, metrics.prcDot]
                    del metrics
                    self.metrics = pd.DataFrame(metrics_ind, index=index_name, columns=column_name)

                    # display metrics
                    data = self.metrics.values
                    self.metricsTableWidget.setRowCount(data.shape[0])
                    self.metricsTableWidget.setColumnCount(data.shape[1])
                    self.metricsTableWidget.setHorizontalHeaderLabels(
                        ['Sn (%)', 'Sp (%)', 'Pre (%)', 'Acc (%)', 'MCC', 'F1', 'AUROC', 'AUPRC'])
                    self.metricsTableWidget.setVerticalHeaderLabels(self.metrics.index)
                    for i in range(data.shape[0]):
                        for j in range(data.shape[1]):
                            cell = QTableWidgetItem(str(data[i][j]))
                            self.metricsTableWidget.setItem(i, j, cell)
                    if self.data_index['Metrics'] is None:
                        # index = self.current_data_index
                        index = 1
                        self.data_index['Metrics'] = index
                        self.dataTableWidget.insertRow(index)
                        self.current_data_index += 1
                    else:
                        # index = self.data_index['Metrics']
                        index = 1
                    self.metrics_radio = QRadioButton()
                    self.dataTableWidget.setCellWidget(index, 0, self.metrics_radio)
                    self.dataTableWidget.setItem(index, 1, QTableWidgetItem('Evaluation metrics'))
                    self.dataTableWidget.setItem(index, 2, QTableWidgetItem(str(data.shape)))
                    self.dataTableWidget.setItem(index, 3, QTableWidgetItem('NA'))

                #plot ROC
                if not self.aucData is None:
                    self.rocLayout.removeWidget(self.roc_curve_widget)
                    sip.delete(self.roc_curve_widget)
                    self.roc_curve_widget = PlotWidgets.CurveWidget()
                    self.roc_curve_widget.init_data(0, 'ROC curve', ind_data=self.aucData)
                    self.rocLayout.addWidget(self.roc_curve_widget)

                # plot PRC
                if not self.prcData is None:
                    self.prcLayout.removeWidget(self.prc_curve_widget)
                    sip.delete(self.prc_curve_widget)
                    self.prc_curve_widget = PlotWidgets.CurveWidget()
                    self.prc_curve_widget.init_data(1, 'PRC curve', ind_data=self.prcData)
                    self.prcLayout.addWidget(self.prc_curve_widget)
            except Exception as e:
                QMessageBox.critical(self, 'Error', str(e), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        else:
            QMessageBox.critical(self, 'Error', 'Please load the model file(s) or data file.', QMessageBox.Ok | QMessageBox.No,
                                 QMessageBox.Ok)

    def save_ml_files(self):
        tag = 0
        try:
            if self.training_score_radio.isChecked():
                tag = 1
                save_file, ok = QFileDialog.getSaveFileName(self, 'Save', './data', 'TSV Files (*.tsv)')
                if ok:
                    ok1 = self.save_prediction_score(save_file)                    
                    if not ok1:
                        QMessageBox.critical(self, 'Error', 'Save file failed.', QMessageBox.Ok | QMessageBox.No,
                                            QMessageBox.Ok)
        except Exception as e:
            pass

        try:
            if self.metrics_radio.isChecked():
                tag = 1
                save_file, ok = QFileDialog.getSaveFileName(self, 'Save', './data', 'TSV Files (*.tsv)')
                if ok:
                    ok1 = self.save_metrics(save_file)                    
                    if not ok1:
                        QMessageBox.critical(self, 'Error', 'Save file failed.', QMessageBox.Ok | QMessageBox.No,
                                            QMessageBox.Ok)
        except Exception as e:
            pass

        if tag == 0:
            QMessageBox.critical(self, 'Error', 'Please select which data to save.', QMessageBox.Ok | QMessageBox.No,
                                 QMessageBox.Ok)

    def save_prediction_score(self, file):
        try:
            df = pd.DataFrame(self.score, columns=['Score_%s' %i for i in range(self.score.shape[1])])
            df.to_csv(file, sep='\t', header=True, index=False)
            return True
        except Exception as e:
            print(e)
            return False

    def save_metrics(self, file):
        try:
            self.metrics.to_csv(file, sep='\t', header=True, index=True)
            return True
        except Exception as e:
            return False

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Confirm Exit', 'Are you sure want to quit iLearnPlus?', QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close_signal.emit('LoadModel')
            self.close()
        else:
            if event:
                event.ignore()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = iLearnPlusLoadModel()
    window.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window.show()
    sys.exit(app.exec_())