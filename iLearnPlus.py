#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QDesktopWidget, QLabel, QHBoxLayout, QMessageBox, QAction, QFileDialog)
from PyQt5.QtGui import QIcon, QFont, QPixmap, QCloseEvent, QDesktopServices
from PyQt5.QtCore import Qt, QUrl
from util import Modules, InputDialog, PlotWidgets, MachineLearning
import iLearnPlusBasic, iLearnPlusEstimator, iLearnPlusAutoML, iLearnPlusLoadModel
import qdarkstyle
import threading
import pandas as pd
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.initUI()

    def initUI(self):
        # initialize GUI
        self.setWindowTitle('iLearnPlus')
        # self.resize(750, 500)
        self.setMaximumSize(600, 400)
        self.setMinimumSize(600, 400)
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.setFont(QFont('Arial'))
        bar = self.menuBar()
        app = bar.addMenu('Applications')
        basic = QAction('iLearnPlus Basic', self)
        basic.triggered.connect(self.openBasicWindow)
        estimator = QAction('iLearnPlus Estimator', self)
        estimator.triggered.connect(self.openEstimatorWindow)
        autoML = QAction('iLearnPlus AutoML', self)
        autoML.triggered.connect(self.openMLWindow)
        loadModel = QAction('Load model(s)', self)
        loadModel.triggered.connect(self.openLoadModelWindow)
        quit = QAction('Exit', self)
        quit.triggered.connect(self.closeEvent)
        app.addAction(basic)
        app.addAction(estimator)
        app.addAction(autoML)
        app.addSeparator()
        app.addAction(loadModel)
        app.addSeparator()
        app.addAction(quit)

        visual = bar.addMenu('Visualization')
        roc = QAction('Plot ROC curve', self)
        roc.triggered.connect(lambda: self.plotCurve('ROC'))
        prc = QAction('Plot PRC curve', self)
        prc.triggered.connect(lambda: self.plotCurve('PRC'))
        boxplot = QAction('Boxplot', self)
        boxplot.triggered.connect(self.drawBoxplot)
        heatmap = QAction('Heatmap', self)
        heatmap.triggered.connect(self.drawHeatmap)
        scatterPlot = QAction('Scatter plot', self)
        scatterPlot.triggered.connect(self.scatterPlot)
        data = QAction('Distribution visualization', self)
        data.triggered.connect(self.displayHist)
        visual.addActions([roc, prc, boxplot, heatmap, scatterPlot, data])

        tools = bar.addMenu('Tools')        
        fileTF = QAction('File format transformation', self)
        fileTF.triggered.connect(self.openFileTF)
        mergeFile = QAction('Merge feature set files into one', self)
        mergeFile.triggered.connect(self.mergeCodingFiles)
        tools.addActions([fileTF, mergeFile])

        help = bar.addMenu('Help')
        document = QAction('Document', self)
        document.triggered.connect(self.openDocumentUrl)
        about = QAction('About', self)
        about.triggered.connect(self.openAbout)
        help.addActions([document, about])

        # move window to center
        self.moveCenter()

        self.widget = QWidget()
        hLayout = QHBoxLayout(self.widget)
        hLayout.setAlignment(Qt.AlignCenter)
        label = QLabel()
        # label.setMaximumWidth(600)
        label.setPixmap(QPixmap('images/logo.png'))
        hLayout.addWidget(label)
        self.setCentralWidget(self.widget)

    def moveCenter(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        newLeft = (screen.width() - size.width()) / 2
        newTop = (screen.height() - size.height()) / 2
        self.move(int(newLeft), int(newTop))

    def openBasicWindow(self):
        self.basicWin = iLearnPlusBasic.ILearnPlusBasic()
        self.basicWin.setFont(QFont('Arial', 10))
        self.basicWin.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.basicWin.close_signal.connect(self.recover)
        self.basicWin.show()
        self.setDisabled(True)
        self.setVisible(False)

    def openEstimatorWindow(self):
        self.estimatorWin = iLearnPlusEstimator.ILearnPlusEstimator()
        self.estimatorWin.setFont(QFont('Arial', 10))
        self.estimatorWin.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.estimatorWin.close_signal.connect(self.recover)
        self.estimatorWin.show()
        self.setDisabled(True)
        self.setVisible(False)

    def openMLWindow(self):
        self.mlWin = iLearnPlusAutoML.ILearnPlusAutoML()
        self.mlWin.setFont(QFont('Arial', 10))
        self.mlWin.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.mlWin.close_signal.connect(self.recover)
        self.mlWin.show()
        self.setDisabled(True)
        self.setVisible(False)

    def openLoadModelWindow(self):
        self.loadWin = iLearnPlusLoadModel.iLearnPlusLoadModel()
        # self.loadWin.setFont(QFont('Arial', 10))
        self.loadWin.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.loadWin.close_signal.connect(self.recover)
        self.loadWin.show()
        self.setDisabled(True)
        self.setVisible(False)

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Confirm Exit', 'Are you sure want to quit iLearnPlus?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            sys.exit(0)
        else:
            if event:
                event.ignore()

    def plotCurve(self, curve='ROC'):
        self.curveWin = Modules.PlotCurve(curve)
        self.curveWin.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.curveWin.show()

    def displayHist(self):
        self.hist = Modules.Hist()
        self.hist.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.hist.show()

    def scatterPlot(self):
        self.scatter = Modules.ScatterPlot()
        self.scatter.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.scatter.show()

    def openFileTF(self):
        try:
            fileName, ok = InputDialog.QFileTransformation.getValues()
            if ok:
                kw = {}
                TFData = MachineLearning.ILearnMachineLearning(kw)            
                TFData.load_data(fileName, target='Training')
                if not TFData.training_dataframe is None:
                    saved_file, ok = QFileDialog.getSaveFileName(self, 'Save to', './data', 'CSV Files (*.csv);;TSV Files (*.tsv);;SVM Files(*.svm);;Weka Files (*.arff)')
                    if ok:
                        ok1 = TFData.save_coder(saved_file, 'training')
                        if not ok1:
                            QMessageBox.critical(self, 'Error', str(self.TFData.error_msg), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

    def mergeCodingFiles(self):
        try:
            coding_files, ok = QFileDialog.getOpenFileNames(self, 'Open coding files (more file can be selected)', './data', 'CSV Files (*.csv);;TSV Files (*.tsv);;SVM Files(*.svm);;Weka Files (*.arff)')
            merged_codings = None
            labels = None
            if len(coding_files) > 0:
                dataframe, datalabel = None, None
                for file in coding_files:
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
                
                    if merged_codings is None:
                        merged_codings = np.hstack((datalabel.reshape((-1, 1)), dataframe.values))
                    else:
                        merged_codings = np.hstack((merged_codings, dataframe.values))
            if merged_codings is not None:
                saved_file, ok = QFileDialog.getSaveFileName(self, 'Save to', './data', 'CSV Files (*.csv);;TSV Files (*.tsv);;SVM Files(*.svm);;Weka Files (*.arff)')
                data = merged_codings
                if saved_file.endswith('.csv'):
                    np.savetxt(saved_file, data, fmt="%s", delimiter=',')
                if saved_file.endswith('.tsv'):
                    np.savetxt(saved_file, data, fmt="%s", delimiter='\t')
                if saved_file.endswith('.svm'):
                    with open(saved_file, 'w') as f:
                        for line in data:
                            f.write('%s' % line[0])
                            for i in range(1, len(line)):
                                f.write('  %d:%s' % (i, line[i]))
                            f.write('\n')
                if saved_file.endswith('.arff'):
                    with open(saved_file, 'w') as f:
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
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

    def openDocumentUrl(self):
        QDesktopServices.openUrl(QUrl('https://ilearnplus.erc.monash.edu/'))

    def openAbout(self):
        QMessageBox.information(self, 'iLearnPlus', 'Version: 1.0\nAuthor: Zhen Chen\nE-mail: chenzhen-win2009@163.com', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

    def drawBoxplot(self):
        try:
            x, y, data, ok = InputDialog.QBoxPlotInput.getValues()
            if ok:
                self.boxWin = PlotWidgets.CustomSingleBoxplotWidget(data, x, y)
                self.boxWin.show()
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

    def drawHeatmap(self):
        try:
            x, y, data, ok = InputDialog.QHeatmapInput.getValues()
            if ok:
                self.heatWin = PlotWidgets.CustomHeatmapWidget(data, x, y)
                self.heatWin.show()
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

    def recover(self, module):
        try:
            if module == 'Basic':                
                del self.basicWin               
            elif module == 'Estimator':
                del self.estimatorWin
            elif module == 'AutoML':
                del self.mlWin
            elif module == 'LoadModel':
                del self.loadWin
            else:
                pass
        except Exception as e:
            pass
        self.setDisabled(False)
        self.setVisible(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window.show()
    sys.exit(app.exec_())
