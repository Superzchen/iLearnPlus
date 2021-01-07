#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import sys, os
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QAction, QMessageBox, QFileDialog)
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QIcon, QDesktopServices
import PlotWidgets, InputDialog
import threading
import sip
import pandas as pd
import numpy as np


class PlotCurve(QMainWindow):
    def __init__(self, curve='ROC'):
        super(PlotCurve, self).__init__()
        self.initUI()
        self.curve = curve
        self.data = []
        self.raw_data = {}

    def initUI(self):
        bar = self.menuBar()
        file = bar.addMenu('File')
        add = QAction('Add curve', self)
        add.triggered.connect(self.importData)
        comp = QAction('Compare curves', self)
        comp.triggered.connect(self.compareCurves)
        quit = QAction('Exit', self)
        quit.triggered.connect(self.close)
        file.addAction(add)
        file.addAction(comp)
        file.addSeparator()
        file.addAction(quit)

        help = bar.addMenu('Help')
        about = QAction('Document', self)
        about.triggered.connect(self.openDocumentUrl)
        help.addAction(about)

        self.setWindowTitle('iLearnPlus Plot curve')
        self.resize(600, 600)
        self.setWindowIcon(QIcon('images/logo.ico'))
        curveWidget = QWidget()
        self.curveLayout = QVBoxLayout(curveWidget)
        self.curveGraph = PlotWidgets.CustomCurveWidget()
        self.curveLayout.addWidget(self.curveGraph)
        self.setCentralWidget(curveWidget)

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message', 'Are you sure to quit?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()

    def importData(self):
        auc, dot, color, lineWidth, lineStyle, prefix, raw_data, ok = InputDialog.QPlotInput.getValues(self.curve)
        if not auc is None and not prefix is None and ok:
            self.raw_data[prefix] = raw_data
            if self.curve == 'ROC':
                self.data.append(['%s AUC = %s'%(prefix, auc), dot, lineWidth, lineStyle, color])
            else:
                self.data.append(['%s AUPRC = %s'%(prefix, auc), dot, lineWidth, lineStyle, color])
            self.curveLayout.removeWidget(self.curveGraph)
            sip.delete(self.curveGraph)
            self.curveGraph = PlotWidgets.CustomCurveWidget()
            if self.curve == 'ROC':
                self.curveGraph.init_data(0, 'ROC curve', self.data)
            else:
                self.curveGraph.init_data(1, 'PRC curve', self.data)
            self.curveLayout.addWidget(self.curveGraph)

    def compareCurves(self):
        if len(self.raw_data) >= 2:
            method, bootstrap_n, ok = InputDialog.QStaticsInput.getValues()
            if ok:
                self.subWin = PlotWidgets.BootstrapTestWidget(self.raw_data, bootstrap_n, self.curve)
                self.subWin.setWindowTitle('Calculating p values ... ')
                self.subWin.resize(600, 600)
                t = threading.Thread(target=self.subWin.bootstrapTest)
                t.start()
                self.subWin.show()
        else:
            QMessageBox.critical(self, 'Error', 'Two or more curve could be compared!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

    def openDocumentUrl(self):
        QDesktopServices.openUrl(QUrl('https://ilearnplus.erc.monash.edu/'))

class ScatterPlot(QMainWindow):
    def __init__(self):
        super(ScatterPlot, self).__init__()
        self.initUI()       

    def initUI(self):
        bar = self.menuBar()
        file = bar.addMenu('File')
        add = QAction('Open file', self)
        add.triggered.connect(self.importData)        
        quit = QAction('Exit', self)
        quit.triggered.connect(self.close)
        file.addAction(add)        
        file.addSeparator()
        file.addAction(quit)       

        self.setWindowTitle('iLearnPlus Scatter Plot')
        self.resize(600, 600)
        self.setWindowIcon(QIcon('images/logo.ico'))
        plotWidget = QWidget()
        self.plotLayout = QVBoxLayout(plotWidget)
        self.plotGraph = PlotWidgets.ClusteringDiagramMatplotlib()
        self.plotLayout.addWidget(self.plotGraph)
        self.setCentralWidget(plotWidget)
    
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message', 'Are you sure to quit?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()

    def importData(self):
        try:
            file, ok = QFileDialog.getOpenFileName(self, 'Open', './data', 'TSV (*.tsv)')
            if ok:
                data = pd.read_csv(file, delimiter='\t', header=0, dtype=float)                
                category =[int(i) for i in sorted(set(data.iloc[:, 0]))]
                new_data = []
                for c in category:
                    index = np.where(data.values[:,0] == c)[0]
                    new_data.append([c, data.values[index, 1:]])
                self.plotLayout.removeWidget(self.plotGraph)
                sip.delete(self.plotGraph)
                self.plotGraph = PlotWidgets.ClusteringDiagramMatplotlib()
                self.plotGraph.init_data('Scatter Plot', new_data, data.columns[1], data.columns[2])
                self.plotLayout.addWidget(self.plotGraph)
        except Exception as e:
            QMessageBox.critical(self, 'Error', 'Please check your input.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

class Hist(QMainWindow):
    def __init__(self):
        super(Hist, self).__init__()
        self.initUI()        

    def initUI(self):
        bar = self.menuBar()
        file = bar.addMenu('File')
        add = QAction('Open file', self)
        add.triggered.connect(self.importData)        
        quit = QAction('Exit', self)
        quit.triggered.connect(self.close)
        file.addAction(add)        
        file.addSeparator()
        file.addAction(quit)       

        self.setWindowTitle('iLearnPlus Histogram and Kernel density plot')
        self.resize(600, 600)
        self.setWindowIcon(QIcon('images/logo.ico'))
        plotWidget = QWidget()
        self.plotLayout = QVBoxLayout(plotWidget)
        self.plotGraph = PlotWidgets.HistogramWidget()
        self.plotLayout.addWidget(self.plotGraph)
        self.setCentralWidget(plotWidget)
    
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message', 'Are you sure to quit?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()

    def importData(self):
        try:
            file, ok = QFileDialog.getOpenFileName(self, 'Open', './data', 'TSV (*.tsv)')
            if ok:
                data = np.loadtxt(file, dtype=float, delimiter='\t')
                fill_zero = np.zeros((data.shape[0], 1))            
                data = np.hstack((np.zeros((data.shape[0], 1)), data))
                self.plotLayout.removeWidget(self.plotGraph)
                sip.delete(self.plotGraph)
                self.plotGraph = PlotWidgets.HistogramWidget()
                self.plotGraph.init_data('Data distribution', data)
                self.plotLayout.addWidget(self.plotGraph)
        except Exception as e:
            QMessageBox.critical(self, 'Error', 'Please check your input.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)








