#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import os, sys
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QWidget, QApplication, QHBoxLayout, QVBoxLayout, QSizePolicy, QSpacerItem, QComboBox,
                             QPushButton, QMessageBox)
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import pyqtSignal
import seaborn as sns
import pandas as pd
# from sklearn.neighbors.kde import KernelDensity
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy import stats
import numpy as np
import qdarkstyle
import InputDialog
import threading
import random

plt.style.use('ggplot')

""" Seaborn """
class ClusteringDiagram(QWidget):
    def __init__(self):
        super(ClusteringDiagram, self).__init__()
        self.initUI()

    def initUI(self):
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.layout = QVBoxLayout(self)

    def seabornPlot(self):
        g = sns.FacetGrid(self.df, hue="time", palette="Set1",
                          hue_order=["Dinner", "Lunch", "Dinner1", "Dinner2", "Dinner3", "Dinner4", "Dinner5",
                                     "Dinner6", "Dinner7", "Dinner8", "Dinner9"])
        g.map(plt.scatter, "total_bill", "tip", s=50, edgecolor="w", norm=0)
        g.add_legend()
        return g.fig

    def initPlot(self, df):
        self.df = df
        fig = self.seabornPlot()
        figureCanvas = FigureCanvas(fig)
        figureCanvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        figureCanvas.updateGeometry()
        navigationToolbar = NavigationToolbar2QT(figureCanvas, self)
        control_layout = QHBoxLayout()
        control_layout.addStretch(1)
        control_layout.addWidget(navigationToolbar)
        control_layout.addStretch(1)
        self.layout.addWidget(figureCanvas)
        self.layout.addLayout(control_layout)
        self.show()

""" Matplotlib """
class MyFigureCanvas(FigureCanvas):
    """ Canvas """

    def __init__(self):
        self.figure = Figure()
        super().__init__(self.figure)

# Cluster diagram
class ClusteringDiagramMatplotlib(QWidget):
    def __init__(self):
        super(ClusteringDiagramMatplotlib, self).__init__()
        self.fontdict = {
            'family': 'Arial',
            'size': 16,
            'color': '#282828',
        }
        self.colorlist = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#FFFF33', '#A65628', '#F781BF',
                          '#999999']
        self.marklist = ['o'] * 9 + ['v'] * 9 + ['^'] * 9 + ['+'] * 9
        self.initUI()

    def initUI(self):
        self.setWindowTitle('iLearnPlus Scatter plot')
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.resize(800, 600)
        layout = QVBoxLayout(self)
        self.figureCanvas = MyFigureCanvas()
        layout.addWidget(self.figureCanvas)
        self.navigationToolbar = NavigationToolbar2QT(self.figureCanvas, self)
        hLayout = QHBoxLayout()
        hLayout.addStretch(1)
        hLayout.addWidget(self.navigationToolbar)
        hLayout.addStretch(1)
        layout.addLayout(hLayout)

    def init_data(self, method, data, xlabel=None, ylabel=None):
        self.xlabel = xlabel if not xlabel is None else 'PC 1'
        self.ylabel = ylabel if not ylabel is None else 'PC 2'
        self.method = method
        self.prefix = 'Cluster:' if method == 'Clustering' else 'Sample category:'
        self.data = data
        self.fig = self.figureCanvas.figure.add_subplot(111)
        self.__draw_figure__()

    def __draw_figure__(self):
        self.fig.cla()
        self.fig.set_facecolor('white')
        self.fig.set_title(self.method)
        self.fig.set_xlabel(self.xlabel, fontdict=self.fontdict)
        self.fig.set_ylabel(self.ylabel, fontdict=self.fontdict)
        for i, item in enumerate(self.data):
            self.fig.scatter(item[1][:, 0], item[1][:, 1], color=self.colorlist[i % len(self.colorlist)], s=70,
                             marker=self.marklist[i % len(self.marklist)], label='%s %s' % (self.prefix, item[0]),
                             edgecolor="w")
        self.fig.legend()
        labels = self.fig.get_xticklabels() + self.fig.get_yticklabels()
        [label.set_color('#282828') for label in labels]
        [label.set_fontname('Arial') for label in labels]
        [label.set_size(16) for label in labels]
        self.fig.spines['left'].set_color('black')
        self.fig.spines['bottom'].set_color('black')

# Histogram
class HistogramWidget(QWidget):
    def __init__(self):
        super(HistogramWidget, self).__init__()
        self.fontdict = {
            'family': 'Arial',
            'size': 16,
            'color': '#282828',
        }
        self.color =['#FF0099', '#008080', '#FF9900', '#660033']
        self.initUI()

    def initUI(self):
        self.setWindowTitle('iLearnPlus Histogram')
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.resize(800, 600)
        layout = QVBoxLayout(self)
        self.figureCanvas = MyFigureCanvas()
        layout.addWidget(self.figureCanvas)
        self.navigationToolbar = NavigationToolbar2QT(self.figureCanvas, self)
        hLayout = QHBoxLayout()
        hLayout.addStretch(1)
        hLayout.addWidget(self.navigationToolbar)
        hLayout.addStretch(1)
        layout.addLayout(hLayout)

    def init_data(self, title, data):        
        self.title = title
        # if data is too large, random part samples
        if data.shape[0] * (data.shape[1]-1) > 32000:
            sel_num = int(32000 / (data.shape[1]-1))
            random_index = random.sample(range(0, len(data)), sel_num)
            self.data = data[random_index]
        else:
            self.data = data
        self.categories = sorted(set(self.data[:, 0]))
        self.fig = self.figureCanvas.figure.add_subplot(111)
        self.__draw_figure__()

    def __draw_figure__(self):
        try:
            self.fig.cla()
            self.fig.set_title(self.title)
            self.fig.set_xlabel("Value bins", fontdict=self.fontdict)
            self.fig.set_ylabel("Density", fontdict=self.fontdict)
            max_value = max(self.data[:, 1:].reshape(-1))
            min_value = min(self.data[:, 1:].reshape(-1))
            if max_value == min_value and max_value == 0:
                pass
            else:
                bins = np.linspace(min_value, max_value, 10)
                for i, c in enumerate(self.categories):
                    tmp_data = self.data[np.where(self.data[:, 0]==c)][:, 1:].reshape(-1)
                    self.fig.hist(tmp_data, bins=bins, stacked=True, density=True, facecolor=self.color[i%len(self.color)], alpha=0.5)
                    X_plot = np.linspace(min_value, max_value, 100)[:, np.newaxis]
                    bandwidth = (max_value - min_value) / 20.0
                    if bandwidth <= 0:
                        bandwidth = 0.1
                    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(tmp_data.reshape((-1, 1)))
                    log_dens = kde.score_samples(X_plot)
                    self.fig.plot(X_plot[:, 0], np.exp(log_dens), color=self.color[i%len(self.color)], label='Category %s' %int(c))
                self.fig.legend()        
            labels = self.fig.get_xticklabels() + self.fig.get_yticklabels()
            [label.set_color('#282828') for label in labels]
            [label.set_fontname('Arial') for label in labels]
            [label.set_size(16) for label in labels]
        except Exception as e:
            QMessageBox.critical(self, 'Plot error', str(e), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

# ROC and PRC curve
class CurveWidget(QWidget):
    def __init__(self):
        super(CurveWidget, self).__init__()
        self.colorlist = ['#377EB8', '#FF1493', '#984EA3', '#FF7F00', '#A65628', '#F781BF', '#999999', '#4DAF4A',
                          '#D2691E', '#DEB887']
        self.fontdict = {
            'family': 'Arial',
            'size': 16,
            'color': '#282828',
        }
        self.initUI()

    def initUI(self):
        self.setWindowTitle('iLearnPlus Curve')
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.resize(800, 600)
        layout = QVBoxLayout(self)
        self.figureCanvas = MyFigureCanvas()
        layout.addWidget(self.figureCanvas)
        self.navigationToolbar = NavigationToolbar2QT(self.figureCanvas, self)
        hLayout = QHBoxLayout()
        hLayout.addStretch(1)
        hLayout.addWidget(self.navigationToolbar)
        hLayout.addStretch(1)
        layout.addLayout(hLayout)

    def init_data(self, type, title, fold_data=None, mean_data=None, ind_data=None):
        self.title = title
        self.fold_data = fold_data
        self.mean_data = mean_data
        self.ind_data = ind_data
        self.type = type
        if type == 0:
            self.x = 'fpr'
            self.y = 'tpr'
        else:
            self.x = 'recall'
            self.y = 'precision'
        self.fig = self.figureCanvas.figure.add_subplot(111)
        self.__draw_figure__()

    def __draw_figure__(self):
        self.fig.cla()
        self.fig.set_facecolor('white')
        self.fig.set_xlim((-0.05, 1.05))
        self.fig.set_ylim((-0.05, 1.05))
        self.fig.set_title(self.title)
        if self.title == 'ROC curve':
            x_label = 'False positive rate'
            y_label = 'True positive rate'
        else:
            x_label = 'Recall'
            y_label = 'Precision'
        self.fig.set_xlabel(x_label, fontdict=self.fontdict)
        self.fig.set_ylabel(y_label, fontdict=self.fontdict)
        self.fig.set_xticks(np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
        self.fig.set_yticks(np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
        if not self.fold_data is None:
            for i, item in enumerate(self.fold_data):
                self.fig.plot(item[1][self.x], item[1][self.y], label=item[0], color=self.colorlist[i%len(self.colorlist)], lw=2, alpha=1.0)
        if not self.mean_data is None:
            self.fig.plot(self.mean_data[1][self.x], self.mean_data[1][self.y], label=self.mean_data[0], lw=2,
                          alpha=0.8, color='b')
        if not self.ind_data is None:
            self.fig.plot(self.ind_data[1][self.x], self.ind_data[1][self.y], label=self.ind_data[0], lw=2, alpha=0.5,
                          color='r')
        if self.type == 0:
            self.fig.plot([0, 1], [0, 1], label='Random', lw=2, alpha=0.5, linestyle='dashed', color='#696969')
        self.fig.legend()
        labels = self.fig.get_xticklabels() + self.fig.get_yticklabels()
        [label.set_color('#282828') for label in labels]
        [label.set_fontname('Arial') for label in labels]
        [label.set_size(16) for label in labels]
        self.fig.spines['left'].set_color('black')
        self.fig.spines['bottom'].set_color('black')


class CurvesWidget(QWidget):
    def __init__(self):
        super(CurvesWidget, self).__init__()
        self.colorlist = ['#377EB8', '#FF1493', '#984EA3', '#FF7F00', '#A65628', '#F781BF', '#999999', '#4DAF4A',
                          '#D2691E', '#DEB887']
        self.fontdict = {
            'family': 'Arial',
            'size': 16,
            'color': '#282828',
        }
        self.initUI()

    def initUI(self):
        self.setWindowTitle('iLearnPlus Curve')
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.resize(800, 600)
        layout = QVBoxLayout(self)
        self.figureCanvas = MyFigureCanvas()
        layout.addWidget(self.figureCanvas)
        self.navigationToolbar = NavigationToolbar2QT(self.figureCanvas, self)
        spacer = QSpacerItem(20, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.comboBox = QComboBox()
        self.comboBox.setFont(QFont('Arial', 8))
        self.comboBox.addItems(['ROC', 'PRC'])
        self.pvalueBtn = QPushButton(' P values ')
        self.pvalueBtn.clicked.connect(self.calculate_pvalue)
        hLayout = QHBoxLayout()
        hLayout.addStretch(1)
        hLayout.addWidget(self.navigationToolbar)
        hLayout.addItem(spacer)
        hLayout.addWidget(self.comboBox)
        hLayout.addWidget(self.pvalueBtn)
        hLayout.addStretch(1)
        layout.addLayout(hLayout)

    def init_roc_data(self, type, title, data):
        self.title = title
        self.data = data
        self.type = type
        if type == 0:
            self.x = 'fpr'
            self.y = 'tpr'
        else:
            self.x = 'recall'
            self.y = 'precision'
        self.fig = self.figureCanvas.figure.add_subplot(121)
        self.__draw_roc_figure__()

    def __draw_roc_figure__(self):
        self.fig.cla()
        self.fig.set_facecolor('white')
        self.fig.set_title(self.title, fontfamily='Arial', fontsize=18)
        self.fig.set_xlabel("False positive rate", fontdict=self.fontdict)
        self.fig.set_ylabel("True positive rate", fontdict=self.fontdict)
        self.fig.set_xticks(np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
        self.fig.set_yticks(np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
        for i, item in enumerate(self.data.keys()):
            self.fig.plot(self.data[item][1][self.x], self.data[item][1][self.y], label='%s %s' %(item, self.data[item][0]), color=self.colorlist[i%len(self.colorlist)], lw=2, alpha=1.0)
        if self.type == 0:
            self.fig.plot([0, 1], [0, 1], label='Random', lw=2, alpha=0.5, linestyle='dashed', color='#696969')
        self.fig.legend()
        labels = self.fig.get_xticklabels() + self.fig.get_yticklabels()
        [label.set_color('#282828') for label in labels]
        [label.set_fontname('Arial') for label in labels]
        [label.set_size(16) for label in labels]
        self.fig.spines['left'].set_color('black')
        self.fig.spines['bottom'].set_color('black')

    def init_prc_data(self, type, title, data):
        self.title = title
        self.prc_data = data
        self.prc_type = type
        if self.prc_type == 0:
            self.prc_x = 'fpr'
            self.prc_y = 'tpr'
        else:
            self.prc_x = 'recall'
            self.prc_y = 'precision'
        self.prc = self.figureCanvas.figure.add_subplot(122)
        self.__draw_prc_figure__()

    def __draw_prc_figure__(self):
        self.prc.cla()
        self.prc.set_facecolor('white')
        self.prc.set_title(self.title, fontfamily='Arial', fontsize=18)
        self.prc.set_xlim((-0.05, 1.05))
        self.prc.set_ylim((-0.05, 1.05))
        self.prc.set_xlabel("Recall", fontdict=self.fontdict)
        self.prc.set_ylabel("Precision", fontdict=self.fontdict)
        self.prc.set_xticks(np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
        self.prc.set_yticks(np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
        for i, item in enumerate(self.prc_data.keys()):
            self.prc.plot(self.prc_data[item][1][self.prc_x], self.prc_data[item][1][self.prc_y], label='%s %s' %(item, self.prc_data[item][0]), color=self.colorlist[i%len(self.colorlist)], lw=2, alpha=1.0)
        self.prc.legend()
        labels = self.prc.get_xticklabels() + self.prc.get_yticklabels()
        [label.set_color('#282828') for label in labels]
        [label.set_fontname('Arial') for label in labels]
        [label.set_size(16) for label in labels]
        self.prc.spines['left'].set_color('black')
        self.prc.spines['bottom'].set_color('black')

    def init_prediction_scores(self, task, prediction_score):
        self.task = task
        self.prediction_scores = prediction_score

    def calculate_pvalue(self):
        try:
            if self.task == 'binary':
                method, bootstrap_n, ok = InputDialog.QStaticsInput.getValues()
                type = self.comboBox.currentText()
                if ok:
                    self.subWin = BootstrapTestWidget(self.prediction_scores, bootstrap_n, type)
                    self.subWin.setWindowTitle('Calculating p values ... ')
                    t = threading.Thread(target=self.subWin.bootstrapTest)
                    t.start()
                    self.subWin.show()
            else:
                QMessageBox.warning(self, 'Warning', 'Only be used in binary classification task.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)


class CustomCurveWidget(QWidget):
    def __init__(self):
        super(CustomCurveWidget, self).__init__()
        self.colorlist = ['#377EB8', '#FF1493', '#984EA3', '#FF7F00', '#A65628', '#F781BF', '#999999', '#4DAF4A',
                          '#D2691E', '#DEB887']
        self.fontdict = {
            'family': 'Arial',
            'size': 16,
            'color': '#282828',
        }
        self.initUI()

    def initUI(self):
        self.setWindowTitle('iLearnPlus Curve')
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.resize(800, 600)
        layout = QVBoxLayout(self)
        self.figureCanvas = MyFigureCanvas()
        layout.addWidget(self.figureCanvas)
        self.navigationToolbar = NavigationToolbar2QT(self.figureCanvas, self)
        hLayout = QHBoxLayout()
        hLayout.addStretch(1)
        hLayout.addWidget(self.navigationToolbar)
        hLayout.addStretch(1)
        layout.addLayout(hLayout)

    def init_data(self, type, title, fold_data):
        self.title = title
        self.fold_data = fold_data
        self.type = type
        if type == 0:
            self.x = 'fpr'
            self.y = 'tpr'
        else:
            self.x = 'recall'
            self.y = 'precision'
        self.fig = self.figureCanvas.figure.add_subplot(111)
        self.__draw_figure__()

    def __draw_figure__(self):
        self.fig.cla()
        self.fig.set_facecolor('white')
        self.fig.set_xlim((-0.05, 1.05))
        self.fig.set_ylim((-0.05, 1.05))
        self.fig.set_title(self.title)
        if self.title == 'ROC curve':
            x_label = 'False positive rate'
            y_label = 'True positive rate'
        else:
            x_label = 'Recall'
            y_label = 'Precision'
        self.fig.set_xlabel(x_label, fontdict=self.fontdict)
        self.fig.set_ylabel(y_label, fontdict=self.fontdict)
        self.fig.set_xticks(np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
        self.fig.set_yticks(np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
        for i, item in enumerate(self.fold_data):
            self.fig.plot(item[1][self.x], item[1][self.y], label=item[0], lw=item[2], linestyle=item[3], color=item[4], alpha=1.0)
        if self.type == 0:
            self.fig.plot([0, 1], [0, 1], label='Random', lw=2, alpha=0.5, linestyle='dashed', color='#696969')
        self.fig.legend()
        labels = self.fig.get_xticklabels() + self.fig.get_yticklabels()
        [label.set_color('#282828') for label in labels]
        [label.set_fontname('Arial') for label in labels]
        [label.set_size(16) for label in labels]
        self.fig.spines['left'].set_color('black')
        self.fig.spines['bottom'].set_color('black')

# Boxplot
class BoxplotWidget(QWidget):
    def __init__(self):
        super(BoxplotWidget, self).__init__()
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.fontdict = {
            'family': 'Arial',
            'size': 14,
            'color': '#282828',
        }
        self.colorlist = ['#3274A1', '#E1812C', '#3A923A', '#C03D3E', '#9372B2', '#845B53', '#D684BD', '#7F7F7F',
                          '#A9AA35', '#2EABB8']
        self.name_dict = {
                'Sn': 'Sensitivity (%)',
                'Sp': 'Specificity (%)',
                'Pre': 'Precision (%)',
                'Acc': 'Accuracy (%)',
                'MCC': 'Matthews correlation coefficient',
                'F1': 'F1-Score',
                'AUROC': 'AUROC',
                'AUPRC': 'AUPRC'
            }
        self.name_dict_1 = {
            'Sensitivity': 'Sn',
            'Specificity': 'Sp',
            'Precision': 'Pre',
            'Accuracy': 'Acc',
            'MCC': 'MCC',
            'F1-Score': 'F1',
            'AUROC': 'AUROC',
            'AUPRC': 'AUPRC'
        }
        self.initUI()

    def initUI(self):
        self.setWindowTitle('iLearnPlus Boxplot')
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.resize(800, 600)
        layout = QVBoxLayout(self)
        self.figureCanvas = MyFigureCanvas()
        layout.addWidget(self.figureCanvas)
        self.navigationToolbar = NavigationToolbar2QT(self.figureCanvas, self)
        spacer = QSpacerItem(20, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.boxcomboBox = QComboBox()
        self.boxcomboBox.setFont(QFont('Arial', 8))
        self.boxcomboBox.addItems(
            ['Sensitivity', 'Specificity', 'Precision', 'Accuracy', 'MCC', 'F1-Score', 'AUROC', 'AUPRC'])
        self.boxAdjustBtn = QPushButton(' Display full size image ')
        self.boxAdjustBtn.setFont(QFont('Arial', 8))
        self.boxAdjustBtn.clicked.connect(self.display_singel_boxplot)
        hLayout = QHBoxLayout()
        hLayout.addStretch(1)
        hLayout.addWidget(self.navigationToolbar)
        hLayout.addItem(spacer)
        hLayout.addWidget(self.boxcomboBox)
        hLayout.addWidget(self.boxAdjustBtn)
        hLayout.addStretch(1)
        layout.addLayout(hLayout)

    def init_data(self, metrics_header_list, metrics_dict):  # metrics_header_list is the metric index like Sn, Sp, ...    metrics_dict {model: pandas DataFrame}
        self.metrics_header_list = metrics_header_list
        self.data_list = []
        self.models = metrics_dict.keys()
        for metric in metrics_header_list:
            tmp_data = np.hstack(tuple(metrics_dict[model].loc[:, metric].values.reshape((-1, 1)) for model in self.models))
            tmp_data = pd.DataFrame(tmp_data, columns=self.models)
            self.data_list.append([metric, tmp_data])
        self.__draw_figure__()

    def __draw_figure__(self):
        if len(self.metrics_header_list) == 1:
            fig = self.figureCanvas.figure.add_subplot(111)
            fig.cla()
            fig.set_facecolor('white')
            fig.set_xlabel("Models", fontdict=self.fontdict)
            fig.set_ylabel(self.data_list[0][0], fontdict=self.fontdict)
            fig.spines['left'].set_color('#585858')
            fig.spines['bottom'].set_color('#585858')
            bp = fig.boxplot(self.data_list[0][1].values.astype(float), widths=0.5, boxprops=dict(lw=1.5), medianprops=dict(lw=1.5),
                             capprops=dict(lw=1.5), whiskerprops=dict(lw=1.5, linestyle='dashed'), showmeans=True,
                             patch_artist=True)

            for k, box in enumerate(bp['boxes']):
                box.set(edgecolor=self.colorlist[k%len(self.colorlist)], facecolor='#FFFFFF')
            for k, median in enumerate(bp['medians']):
                median.set(color=self.colorlist[k%len(self.colorlist)])

            for elem in ['whiskers', 'caps']:
                for k, item in enumerate(bp[elem]):
                    j = k // 2
                    item.set(color=self.colorlist[j%len(self.colorlist)])
            for k, flier in enumerate(bp['fliers']):
                flier.set(marker='o', markeredgecolor=self.colorlist[k%len(self.colorlist)], markeredgewidth=1)
            for k, mean in enumerate(bp['means']):
                mean.set(marker='^', markeredgecolor=self.colorlist[k%len(self.colorlist)], markeredgewidth=1, markerfacecolor='#FFFFFF')

            scale_ls = list(range(1, len(self.data_list[0][1].columns)+1))
            index_ls = self.data_list[0][1].columns
            index_dict = {}
            for i in range(len(scale_ls)):
                index_dict[index_ls[i]] = scale_ls[i]
            fig.set_xticklabels(index_dict, rotation=45)
            labels = fig.get_xticklabels() + fig.get_yticklabels()
            [label.set_color('#282828') for label in labels]
            [label.set_fontname('Arial') for label in labels]
            [label.set_size(11) for label in labels]
        else:
            for i in range(len(self.data_list)):
                position = 1 + i
                fig = self.figureCanvas.figure.add_subplot(2, 4, position)
                fig.cla()
                fig.set_facecolor('white')
                fig.set_xlabel("Models", fontdict=self.fontdict)
                fig.set_ylabel(self.name_dict[self.data_list[i][0]], fontdict=self.fontdict)
                fig.spines['left'].set_color('#585858')
                fig.spines['bottom'].set_color('#585858')
                bp = fig.boxplot(self.data_list[i][1].values, widths=0.5, boxprops=dict(lw=1.5), medianprops=dict(lw=1.5),
                                 capprops=dict(lw=1.5), whiskerprops=dict(lw=1.5, linestyle='dashed'), showmeans=True,
                                 patch_artist=True)

                for k, box in enumerate(bp['boxes']):
                    box.set(edgecolor=self.colorlist[k%len(self.colorlist)], facecolor='#FFFFFF')
                for k, median in enumerate(bp['medians']):
                    # print(median.get_data())
                    median.set(color=self.colorlist[k%len(self.colorlist)])

                for elem in ['whiskers', 'caps']:
                    for k, item in enumerate(bp[elem]):
                        j = k // 2
                        item.set(color=self.colorlist[j%len(self.colorlist)])
                for k, flier in enumerate(bp['fliers']):
                    flier.set(marker='o', markeredgecolor=self.colorlist[k%len(self.colorlist)], markeredgewidth=1)
                for k, mean in enumerate(bp['means']):
                    mean.set(marker='^', markeredgecolor=self.colorlist[k%len(self.colorlist)], markeredgewidth=1, markerfacecolor='#FFFFFF')

                scale_ls = list(range(1, len(self.data_list[i][1].columns)+1))
                index_ls = self.data_list[i][1].columns
                index_dict = {}
                for i in range(len(scale_ls)):
                    index_dict[index_ls[i]] = scale_ls[i]
                fig.set_xticklabels(index_dict, rotation=45)

                labels = fig.get_xticklabels() + fig.get_yticklabels()
                [label.set_color('#282828') for label in labels]
                [label.set_fontname('Arial') for label in labels]
                [label.set_size(11) for label in labels]
            self.figureCanvas.figure.subplots_adjust(left=0.08, right=0.95, bottom=0.15, top=0.95, wspace=0.5, hspace=0.5)

    def display_singel_boxplot(self):
        comboBoxIndex = self.boxcomboBox.currentIndex()
        if len(self.data_list) == 1:
            comboBoxIndex = 0
        try:
            self.subWin = SingleBoxplotWidget(self.data_list[comboBoxIndex])
            self.subWin.show()
        except Exception as e:
            QMessageBox.warning(self, 'Warning', str(e), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)


class SingleBoxplotWidget(QWidget):
    def __init__(self, data):
        super(SingleBoxplotWidget, self).__init__()
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.fontdict = {
            'family': 'Arial',
            'size': 14,
            'color': '#282828',
        }
        self.colorlist = ['#3274A1', '#E1812C', '#3A923A', '#C03D3E', '#9372B2', '#845B53', '#D684BD', '#7F7F7F',
                          '#A9AA35', '#2EABB8']
        self.name_dict = {
            'Sn': 'Sensitivity (%)',
            'Sp': 'Specificity (%)',
            'Pre': 'Precision (%)',
            'Acc': 'Accuracy (%)',
            'MCC': 'Matthews correlation coefficient',
            'F1': 'F1-Score',
            'AUROC': 'AUROC',
            'AUPRC': 'AUPRC'
        }
        self.metric = self.name_dict[data[0]]
        self.dataframe = data[1]
        self.models = self.dataframe.columns
        self.line_dict = None
        self.pvalue_dict = {}
        self.bar_len = 1.0
        self.x_set = set()
        self.y_set = set()
        self.y_ticks = {}
        if data[0] in ['Sn', 'Sp', 'Pre', 'Acc']:
            self.bar_len = 1.0
        else:
            self.bar_len = 0.01
        self.initUI()
        self.__draw_figure__()
        self.__draw_heatmap__()

    def initUI(self):
        self.setWindowTitle('iLearnPlus Boxplot')
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.resize(900, 600)
        layout = QVBoxLayout(self)
        self.figureCanvas = MyFigureCanvas()
        layout.addWidget(self.figureCanvas)
        self.navigationToolbar = NavigationToolbar2QT(self.figureCanvas, self)
        spacer = QSpacerItem(20, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.comboBox_model1 = QComboBox()
        self.comboBox_model1.setFont(QFont('Arial', 8))
        self.comboBox_model1.addItems(self.models)
        self.comboBox_model2 = QComboBox()
        self.comboBox_model2.setFont(QFont('Arial', 8))
        self.comboBox_model2.addItems(self.models)
        self.pvalueBtn = QPushButton(' Add P_value ')
        self.pvalueBtn.setFont(QFont('Arial', 8))
        self.pvalueBtn.clicked.connect(self.re_drawplot)
        hLayout = QHBoxLayout()
        hLayout.addStretch(1)
        hLayout.addWidget(self.navigationToolbar)
        hLayout.addItem(spacer)
        hLayout.addWidget(self.comboBox_model1)
        hLayout.addWidget(self.comboBox_model2)
        hLayout.addWidget(self.pvalueBtn)
        hLayout.addStretch(1)
        layout.addLayout(hLayout)
        self.fig = self.figureCanvas.figure.add_subplot(121)

    def __draw_figure__(self):
        self.fig.cla()
        self.fig.set_facecolor('white')
        self.fig.set_title('Performance of different models', fontfamily='Arial', fontsize=12)
        self.fig.set_xlabel("Models", fontdict=self.fontdict)
        self.fig.set_ylabel(self.metric, fontdict=self.fontdict)
        self.fig.spines['left'].set_color('#585858')
        self.fig.spines['bottom'].set_color('#585858')
        bp = self.fig.boxplot(self.dataframe.values.astype(float), widths=0.5, boxprops=dict(lw=1.5), medianprops=dict(lw=1.5),
                         capprops=dict(lw=1.5), whiskerprops=dict(lw=1.5, linestyle='dashed'), showmeans=True,
                         patch_artist=True)

        for k, box in enumerate(bp['boxes']):
            box.set(edgecolor=self.colorlist[k%len(self.colorlist)], facecolor='#FFFFFF')
        for k, median in enumerate(bp['medians']):
            median.set(color=self.colorlist[k%len(self.colorlist)])

        for elem in ['whiskers', 'caps']:
            for k, item in enumerate(bp[elem]):
                j = k // 2
                item.set(color=self.colorlist[j%len(self.colorlist)])
        for k, flier in enumerate(bp['fliers']):
            flier.set(marker='o', markeredgecolor=self.colorlist[k%len(self.colorlist)], markeredgewidth=1)
        for k, mean in enumerate(bp['means']):
            mean.set(marker='^', markeredgecolor=self.colorlist[k%len(self.colorlist)], markeredgewidth=1, markerfacecolor='#FFFFFF')

        # calculate max value for each position
        if self.line_dict is None:
            self.line_dict = {}
            for cap in bp['caps']:
                x = sum(cap.get_data()[0])/2
                y = cap.get_data()[1][0]
                if x in self.line_dict:
                    if y > self.line_dict[x]:
                        self.line_dict[x] = y
                else:
                    self.line_dict[x] = y
            for flier in bp['fliers']:
                if len(flier.get_data()[0]) != 0:
                    for k in range(len(flier.get_data()[1])):
                        x = flier.get_data()[0][k]
                        y = flier.get_data()[1][k]
                        if x in self.line_dict:
                            if y > self.line_dict[x]:
                                self.line_dict[x] = y
                        else:
                            self.line_dict[x] = y
        # add p_value
        for key in self.pvalue_dict:
            array = [float(i) for i in key.split('@')]
            # y1 = max(list([self.line_dict[item] for item in range(int(array[0]), int(array[1])+1)])) + self.bar_len
            # y2 = y1 + self.bar_len
            # if y2 in self.y_set:
            #     y2 += self.bar_len
            # self.y_set.add(y2)
            y1 = self.y_ticks[key][0]
            y2 = self.y_ticks[key][1]

            self.fig.plot([array[0], array[0]], [self.line_dict[array[0]]+self.bar_len, y2], color='#101010', lw=1.5)
            self.fig.plot([array[1], array[1]], [self.line_dict[array[1]]+self.bar_len, y2], color='#101010', lw=1.5)
            self.fig.plot([array[0], array[1]], [y2, y2], color='#101010', lw=1.5)
            if self.pvalue_dict[key] < 0.001:
                pvalue = '%.2e'%self.pvalue_dict[key]
            else:
                pvalue = '%.3f'%self.pvalue_dict[key]
            self.fig.text((array[0]+array[1])/2 - 0.15, y2 + self.bar_len, 'p = %s' %pvalue, family='Arial', fontsize=11, style='italic', color='#101010')

        scale_ls = list(range(1, len(self.dataframe.columns)+1))
        index_ls = self.dataframe.columns
        index_dict = {}
        for i in range(len(scale_ls)):
            index_dict[index_ls[i]] = scale_ls[i]
        self.fig.set_xticklabels(index_dict, rotation=45)
        labels = self.fig.get_xticklabels() + self.fig.get_yticklabels()
        [label.set_color('#282828') for label in labels]
        [label.set_fontname('Arial') for label in labels]
        [label.set_size(11) for label in labels]

    def __draw_heatmap__(self):
        model_num = len(self.dataframe.columns)
        pvalue_matrix = np.zeros((model_num, model_num))
        for i in range(model_num):
            pvalue_matrix[i][i] = 1
            for j in range(i+1, model_num):
                tTest = stats.ttest_rel(self.dataframe.iloc[:, i].values.astype(float), self.dataframe.iloc[:, j].values.astype(float))
                if tTest.pvalue < 0.001:
                    pvalue = '%.2e' % tTest.pvalue
                else:
                    pvalue = '%.3f' % tTest.pvalue
                pvalue_matrix[i][j] = pvalue
                pvalue_matrix[j][i] = pvalue

        self.heatmap = self.figureCanvas.figure.add_subplot(122)
        self.heatmap.cla()
        self.heatmap.set_facecolor('white')
        self.heatmap.grid(False)
        self.heatmap.set_title('P values calculated by paired Student’s t-test', pad=35, fontfamily='Arial', fontsize=12)
        self.heatmap.tick_params(axis=u'both', which=u'both', length=0)  # tick length = 0
        tick = range(len(self.dataframe.columns))
        self.heatmap.set_yticks(tick)
        self.heatmap.set_yticklabels(self.dataframe.columns)
        self.heatmap.set_xticks(tick)
        self.heatmap.set_xticklabels(self.dataframe.columns, rotation=45)
        im = self.heatmap.imshow(pvalue_matrix, cmap=plt.cm.winter, alpha=0.6)
        self.figureCanvas.figure.colorbar(im)

        if pvalue_matrix.shape[0] <= 5:
            fontsize = 10
        elif 5 < pvalue_matrix.shape[0] <= 10:
            fontsize = 8
        else:
            fontsize = 5
        for i in range(pvalue_matrix.shape[0]):
            for j in range(i):
                if pvalue_matrix[i][j] < 0.001:
                    vs = '%.2e' %pvalue_matrix[i][j]
                else:
                    vs = '%.4f' %pvalue_matrix[i][j]
                self.heatmap.text(j, i, vs, fontsize=fontsize, family='Arial', style='italic', color='#101010', ha='center', va='center')
        labels = self.heatmap.get_xticklabels() + self.heatmap.get_yticklabels()
        [label.set_color('#282828') for label in labels]
        [label.set_fontname('Arial') for label in labels]
        [label.set_size(11) for label in labels]
        self.figureCanvas.figure.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.95, wspace=0.5, hspace=0.5)

    def re_drawplot(self):
        index_1 = int(self.comboBox_model1.currentIndex())
        index_2 = int(self.comboBox_model2.currentIndex())
        if index_1 == index_2:
            QMessageBox.critical(self, 'Error', 'P_value must be calculated between two different models.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        else:
            tTest = stats.ttest_rel(self.dataframe.iloc[:, index_1], self.dataframe.iloc[:, index_2])
            array = [str(i) for i in sorted([index_1 +1, index_2 + 1])]
            self.pvalue_dict['@'.join(array)] = tTest.pvalue

            y1 = max(list([self.line_dict[item] for item in range(int(array[0]), int(array[1]) + 1)])) + self.bar_len
            y2 = y1 + self.bar_len
            if y2 in self.y_set:
                y2 += self.bar_len
            self.y_set.add(y2)
            self.y_ticks['@'.join(array)] = (y1, y2)

        self.__draw_figure__()
        width = self.geometry().width()
        height = self.geometry().height()
        self.resize(width-1, height-1)


class CustomSingleBoxplotWidget(QWidget):
    def __init__(self, data, x_label, y_label):
        super(CustomSingleBoxplotWidget, self).__init__()
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.fontdict = {
            'family': 'Arial',
            'size': 14,
            'color': '#282828',
        }
        self.colorlist = ['#3274A1', '#E1812C', '#3A923A', '#C03D3E', '#9372B2', '#845B53', '#D684BD', '#7F7F7F',
                          '#A9AA35', '#2EABB8']
        self.dataframe = data
        self.models = self.dataframe.columns
        self.x_label = x_label
        self.y_label = y_label
        self.initUI()
        self.__draw_figure__()

    def initUI(self):
        self.setWindowTitle('iLearnPlus Boxplot')
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.resize(600, 400)
        layout = QVBoxLayout(self)
        self.figureCanvas = MyFigureCanvas()
        layout.addWidget(self.figureCanvas)
        self.navigationToolbar = NavigationToolbar2QT(self.figureCanvas, self)
        hLayout = QHBoxLayout()
        hLayout.addStretch(1)
        hLayout.addWidget(self.navigationToolbar)
        hLayout.addStretch(1)
        layout.addLayout(hLayout)
        self.fig = self.figureCanvas.figure.add_subplot(111)

    def __draw_figure__(self):
        self.fig.cla()
        self.fig.set_facecolor('white')
        # self.fig.set_title('Performance of different models', fontfamily='Arial', fontsize=12)
        self.fig.set_xlabel(self.x_label, fontdict=self.fontdict)
        self.fig.set_ylabel(self.y_label, fontdict=self.fontdict)
        self.fig.spines['left'].set_color('#585858')
        self.fig.spines['bottom'].set_color('#585858')
        bp = self.fig.boxplot(self.dataframe.values.astype(float), widths=0.5, boxprops=dict(lw=1.5), medianprops=dict(lw=1.5),
                              capprops=dict(lw=1.5), whiskerprops=dict(lw=1.5, linestyle='dashed'), showmeans=True,
                              patch_artist=True)

        for k, box in enumerate(bp['boxes']):
            box.set(edgecolor=self.colorlist[k%len(self.colorlist)], facecolor='#FFFFFF')
        for k, median in enumerate(bp['medians']):
            median.set(color=self.colorlist[k%len(self.colorlist)])

        for elem in ['whiskers', 'caps']:
            for k, item in enumerate(bp[elem]):
                j = k // 2
                item.set(color=self.colorlist[j%len(self.colorlist)])
        for k, flier in enumerate(bp['fliers']):
            flier.set(marker='o', markeredgecolor=self.colorlist[k%len(self.colorlist)], markeredgewidth=1)
        for k, mean in enumerate(bp['means']):
            mean.set(marker='^', markeredgecolor=self.colorlist[k%len(self.colorlist)], markeredgewidth=1, markerfacecolor='#FFFFFF')

        scale_ls = list(range(1, len(self.dataframe.columns)+1))
        index_ls = self.dataframe.columns
        index_dict = {}
        for i in range(len(scale_ls)):
            index_dict[index_ls[i]] = scale_ls[i]
        self.fig.set_xticklabels(index_dict, rotation=45)
        labels = self.fig.get_xticklabels() + self.fig.get_yticklabels()
        [label.set_color('#282828') for label in labels]
        [label.set_fontname('Arial') for label in labels]
        [label.set_size(11) for label in labels]


# Heatmap for correlation
class HeatmapWidget(QWidget):
    def __init__(self, dataframe):
        super(HeatmapWidget, self).__init__()
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.dataframe = dataframe
        self.initUI()
        self.__draw_heatmap__()

    def initUI(self):
        self.setWindowTitle('iLearnPlus Heatmap')
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.resize(800, 600)
        layout = QVBoxLayout(self)
        self.figureCanvas = MyFigureCanvas()
        self.navigationToolbar = NavigationToolbar2QT(self.figureCanvas, self)
        hLayout = QHBoxLayout()
        hLayout.addStretch(1)
        hLayout.addWidget(self.navigationToolbar)
        hLayout.addStretch(1)
        layout.addWidget(self.figureCanvas)
        layout.addLayout(hLayout)

    def __draw_heatmap__(self):
        data = self.dataframe.T
        correlation_matrix = data.corr(method='pearson').values
        self.heatmap = self.figureCanvas.figure.add_subplot(111)
        self.heatmap.cla()
        self.heatmap.set_facecolor('white')
        self.heatmap.grid(False)
        self.heatmap.set_title('Correlation between models', fontfamily='Arial', fontsize=12)
        self.heatmap.tick_params(axis=u'both', which=u'both', length=0)  # tick length = 0
        tick = range(len(data.columns))
        self.heatmap.set_yticks(tick)
        self.heatmap.set_yticklabels(data.columns)
        self.heatmap.set_xticks(tick)
        self.heatmap.set_xticklabels(data.columns, rotation=45)
        im = self.heatmap.imshow(correlation_matrix, cmap=plt.cm.winter, alpha=0.6)
        self.figureCanvas.figure.colorbar(im)

        if correlation_matrix.shape[0] <= 5:
            fontsize = 10
        elif 5 < correlation_matrix.shape[0] <= 10:
            fontsize = 8
        else:
            fontsize = 5
        for i in range(correlation_matrix.shape[0]):
            for j in range(i):
                self.heatmap.text(j, i, '%.4f' %correlation_matrix[i][j], fontsize=fontsize, family='Arial', style='italic', color='#101010', ha='center', va='center')
        labels = self.heatmap.get_xticklabels() + self.heatmap.get_yticklabels()
        [label.set_color('#282828') for label in labels]
        [label.set_fontname('Arial') for label in labels]
        [label.set_size(11) for label in labels]
        self.figureCanvas.figure.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.95, wspace=0.5, hspace=0.5)


class CustomHeatmapWidget(QWidget):
    def __init__(self, dataframe, x_label, y_label):
        super(CustomHeatmapWidget, self).__init__()
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.fontdict = {
            'family': 'Arial',
            'size': 14,
            'color': '#282828',
        }
        self.dataframe = dataframe
        self.x_label = x_label
        self.y_label = y_label
        self.initUI()
        self.__draw_heatmap__()

    def initUI(self):
        self.setWindowTitle('iLearnPlus Heatmap')
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.resize(800, 600)
        layout = QVBoxLayout(self)
        self.figureCanvas = MyFigureCanvas()
        self.navigationToolbar = NavigationToolbar2QT(self.figureCanvas, self)
        hLayout = QHBoxLayout()
        hLayout.addStretch(1)
        hLayout.addWidget(self.navigationToolbar)
        hLayout.addStretch(1)
        layout.addWidget(self.figureCanvas)
        layout.addLayout(hLayout)

    def __draw_heatmap__(self):
        # data = self.dataframe.T
        # correlation_matrix = data.corr(method='pearson').values
        self.heatmap = self.figureCanvas.figure.add_subplot(111)
        self.heatmap.cla()
        self.heatmap.set_facecolor('white')
        self.heatmap.grid(False)
        self.heatmap.set_xlabel(self.x_label, fontdict=self.fontdict)
        self.heatmap.set_ylabel(self.y_label, fontdict=self.fontdict)
        # self.heatmap.set_title('Correlation between models', fontfamily='Arial', fontsize=12)
        self.heatmap.tick_params(axis=u'both', which=u'both', length=0)  # tick length = 0
        tick = range(len(self.dataframe.index))
        self.heatmap.set_yticks(tick)
        self.heatmap.set_yticklabels(self.dataframe.index)
        tick = range(len(self.dataframe.columns))
        self.heatmap.set_xticks(tick)
        self.heatmap.set_xticklabels(self.dataframe.columns, rotation=45)
        im = self.heatmap.imshow(self.dataframe.values, cmap=plt.cm.autumn, alpha=0.6)
        self.figureCanvas.figure.colorbar(im)


        labels = self.heatmap.get_xticklabels() + self.heatmap.get_yticklabels()
        [label.set_color('#282828') for label in labels]
        [label.set_fontname('Arial') for label in labels]
        [label.set_size(11) for label in labels]
        self.figureCanvas.figure.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.95, wspace=0.5, hspace=0.5)


class BootstrapTestWidget(QWidget):
    resizeWindowSignal = pyqtSignal()
    def __init__(self, data, boost_n, type):
        super(BootstrapTestWidget, self).__init__()
        self.resizeWindowSignal.connect(self.resizeWindow)
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.data = data
        self.boost_n = boost_n
        self.type = type
        self.dataframe = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('iLearnPlus Bootstrap test')
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.resize(800, 600)
        layout = QVBoxLayout(self)
        self.figureCanvas = MyFigureCanvas()
        self.navigationToolbar = NavigationToolbar2QT(self.figureCanvas, self)
        hLayout = QHBoxLayout()
        hLayout.addStretch(1)
        hLayout.addWidget(self.navigationToolbar)
        hLayout.addStretch(1)
        layout.addWidget(self.figureCanvas)
        layout.addLayout(hLayout)

    def bootstrapTest(self):
        orig_auc = {}
        auc_dict = {}
        for key in self.data:
            # calculate original AUC
            if self.type == 'ROC':
                fpr, tpr, _ = roc_curve(self.data[key].iloc[:, 1], self.data[key].iloc[:, 3])
                orig_auc[key] = auc(fpr, tpr)
            else:
                precision, recall, _ = precision_recall_curve(self.data[key].iloc[:, 1], self.data[key].iloc[:, 3])
                orig_auc[key] = auc(recall, precision)

            # calculate the bootstrap resampled AUCs
            curve_areas = []
            sample_number = self.data[key].values.shape[0]
            for i in range(self.boost_n):
                random_index = np.random.choice(list(range(sample_number)), sample_number, replace=True)
                new_df = self.data[key].iloc[random_index, [1, 3]]
                new_df.columns = ['Label', 'Score']
                if self.type == 'ROC':
                    fpr, tpr, _ = roc_curve(new_df.iloc[:, 0], new_df.iloc[:, 1])
                    curve_areas.append(auc(fpr, tpr))
                else:
                    precision, recall, _ = precision_recall_curve(new_df.iloc[:, 0], new_df.iloc[:, 1])
                    curve_areas.append(auc(recall, precision))
            auc_dict[key] = np.array(curve_areas)

        model_list = list(self.data.keys())
        pvalues = np.zeros((len(model_list), len(model_list)))

        for i in range(len(model_list)):
            pvalues[i][i] = 1
            for j in range(i+1, len(model_list)):
                D = (orig_auc[model_list[i]] - orig_auc[model_list[j]]) / np.std(auc_dict[model_list[i]] - auc_dict[model_list[j]])
                p = stats.norm.cdf(D)
                pvalue = 2 - 2 * p if D > 0 else 2 * p
                pvalues[i][j] = pvalue
                pvalues[j][i] = pvalue
        self.dataframe = pd.DataFrame(pvalues, columns=model_list, index=model_list)
        self.displayPlot()

    def displayPlot(self):
        self.setWindowTitle('iLearnPlus Bootstrap test')
        if not self.dataframe is None:
            self.heatmap = self.figureCanvas.figure.add_subplot(111)
            self.heatmap.cla()
            self.heatmap.set_facecolor('white')
            self.heatmap.grid(False)
            self.heatmap.set_title('P values between models', fontfamily='Arial', fontsize=12)
            self.heatmap.tick_params(axis=u'both', which=u'both', length=0)  # tick length = 0
            tick = range(len(self.dataframe.columns))
            self.heatmap.set_yticks(tick)
            self.heatmap.set_yticklabels(self.dataframe.columns)
            self.heatmap.set_xticks(tick)
            self.heatmap.set_xticklabels(self.dataframe.columns, rotation=45)
            im = self.heatmap.imshow(self.dataframe.values, cmap=plt.cm.winter, alpha=0.6)
            self.figureCanvas.figure.colorbar(im)

            if self.dataframe.values.shape[0] <= 5:
                fontsize = 10
            elif 5 < self.dataframe.values.shape[0] <= 10:
                fontsize = 8
            else:
                fontsize = 5
            for i in range(self.dataframe.values.shape[0]):
                for j in range(i):
                    if self.dataframe.values[i][j] < 0.001:
                        vs = '%.2e' %self.dataframe.values[i][j]
                    else:
                        vs = '%.4f' %self.dataframe.values[i][j]
                    self.heatmap.text(j, i, vs, fontsize=fontsize, family='Arial', style='italic', color='#101010', ha='center', va='center')
            labels = self.heatmap.get_xticklabels() + self.heatmap.get_yticklabels()
            [label.set_color('#282828') for label in labels]
            [label.set_fontname('Arial') for label in labels]
            [label.set_size(11) for label in labels]
            self.figureCanvas.figure.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.95, wspace=0.5, hspace=0.5)
        self.resizeWindowSignal.emit()

    def resizeWindow(self):
        width = self.geometry().width()
        height = self.geometry().height()
        self.resize(width-1, height-1)




if __name__ == "__main__":
    app = QApplication(sys.argv)
    example = BoxplotWidget()
    df1 = pd.read_csv('../boxplot1.txt', sep='\t', header=0)
    df2 = pd.read_csv('../boxplot2.txt', sep='\t', header=0)
    test = {
        'AAC': df1,
        'EAAC': df2,
    }
    example.init_data(['Sn', 'Sp', 'Pre', 'Acc', 'MCC', 'F1', 'AUROC', 'AUPRC'], test)
    # example.init_data(['AUROC'], test)
    example.show()
    sys.exit(app.exec_())
