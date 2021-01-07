#!/usr/bin/env python
# _*_ coding: utf-8 _*_
import sys
from PyQt5.QtWidgets import (QDialog, QFormLayout, QLineEdit, QDialogButtonBox, QInputDialog, QMessageBox, QComboBox,
                             QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QTableWidgetSelectionRange,
                             QAbstractItemView, QGridLayout, QTreeWidget, QTreeWidgetItem, QCheckBox, QApplication,
                             QFileDialog, QColorDialog, QPushButton)
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import pyqtSignal, Qt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import numpy as np
import pandas as pd
import qdarkstyle
from multiprocessing import cpu_count



class MyLineEdit(QLineEdit):
    clicked = pyqtSignal()
    def mouseReleaseEvent(self, QMouseEvent):
        if QMouseEvent.button() == Qt.LeftButton:
            self.clicked.emit()


class QSOrderInput(QDialog):
    def __init__(self, nlag_range):
        super(QSOrderInput, self).__init__()
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.nlag_range = nlag_range
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowTitle('QSOrder setting')
        self.resize(400, 110)
        self.setFont(QFont('Arial', 10))
        layout = QFormLayout(self)
        self.lagLineEdit = MyLineEdit("2")
        self.lagLineEdit.clicked.connect(self.setLagValue)
        self.weightLineEdit = MyLineEdit("0.05")
        layout.addRow('Lag value: ', self.lagLineEdit)
        layout.addRow('Weight factor (0~1): ', self.weightLineEdit)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def setLagValue(self):
        num, ok = QInputDialog.getInt(self, 'Lag value', 'Get lag value', 3, 1, self.nlag_range, 1)
        if ok:
            self.lagLineEdit.setText(str(num))

    def getLagValue(self):
        if self.lagLineEdit.text != '':
            if int(self.lagLineEdit.text()) > self.nlag_range:
                return 2
            else:
                return int(self.lagLineEdit.text())
        else:
            return 2

    def getWeightValue(self):
        weight = 0.05
        if self.weightLineEdit.text() != '':
            weight = float(self.weightLineEdit.text())
            if 0 < weight < 1:
                return weight
            else:
                return 0.05
        else:
            return 0.05

    @staticmethod
    def getValues(nlag_range):
        dialog = QSOrderInput(nlag_range)
        result = dialog.exec_()
        lag = dialog.getLagValue()
        weight = dialog.getWeightValue()
        return lag, weight, result == QDialog.Accepted


class QPAACInput(QDialog):
    def __init__(self, lambda_range):
        super(QPAACInput, self).__init__()
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.lambda_range = lambda_range
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowTitle('PAAC & APAAC setting')
        self.setFont(QFont('Arial', 10))
        self.resize(400, 110)
        layout = QFormLayout(self)
        self.lambdaLineEdit = MyLineEdit("2")
        self.lambdaLineEdit.clicked.connect(self.setLambdaValue)
        self.weightLineEdit = MyLineEdit("0.05")
        layout.addRow('Lambda value: ', self.lambdaLineEdit)
        layout.addRow('Weight factor (0~1): ', self.weightLineEdit)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def setLambdaValue(self):
        num, ok = QInputDialog.getInt(self, 'Lambda value', 'Get lambda value', 3, 1, self.lambda_range, 1)
        if ok:
            self.lambdaLineEdit.setText(str(num))

    def getLambdaValue(self):
        return int(self.lambdaLineEdit.text()) if self.lambdaLineEdit.text() != '' else 2

    def getWeightValue(self):
        weight = 0.05
        if self.weightLineEdit.text() != '':
            weight = float(self.weightLineEdit.text())
            if 0 < weight < 1:
                return weight
            else:
                return 0.05
        else:
            return 0.05

    @staticmethod
    def getValues(lambda_range):
        dialog = QPAACInput(lambda_range)
        result = dialog.exec_()
        lambda_value = dialog.getLambdaValue()
        weight = dialog.getWeightValue()
        return lambda_value, weight, result == QDialog.Accepted


class QPseKRAACInput(QDialog):
    def __init__(self, clust_type):
        super(QPseKRAACInput, self).__init__()
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.clust_type = clust_type
        self.types = {
            'PseKRAAC type 1': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'PseKRAAC type 2': [2, 3, 4, 5, 6, 8, 15, 20],
            'PseKRAAC type 3A': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'PseKRAAC type 3B': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'PseKRAAC type 4': [5, 8, 9, 11, 13, 20],
            'PseKRAAC type 5': [3, 4, 8, 10, 15, 20],
            'PseKRAAC type 6A': [4, 5, 20],
            'PseKRAAC type 6B': [5, ],
            'PseKRAAC type 6C': [5, ],
            'PseKRAAC type 7': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'PseKRAAC type 8': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'PseKRAAC type 9': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'PseKRAAC type 10': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'PseKRAAC type 11': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'PseKRAAC type 12': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20],
            'PseKRAAC type 13': [4, 12, 17, 20],
            'PseKRAAC type 14': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'PseKRAAC type 15': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20],
            'PseKRAAC type 16': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20],
        }
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowTitle('PseKRAAC setting')
        self.setFont(QFont('Arial', 10))
        layout = QFormLayout(self)
        self.modelComboBox = QComboBox()
        self.modelComboBox.addItems(['g-gap', 'lambda-correlation'])
        self.modelComboBox.setCurrentIndex(0)
        self.modelComboBox.currentIndexChanged.connect(self.modelSelection)
        self.gapComboBox = QComboBox()
        self.gapComboBox.addItems(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        self.gapComboBox.setCurrentIndex(2)
        self.lambdaComoBox = QComboBox()
        self.lambdaComoBox.addItems((['1', '2', '3', '4', '5', '6', '7', '8', '9']))
        self.lambdaComoBox.setCurrentIndex(2)
        self.lambdaComoBox.setEnabled(False)
        self.ktupleComoBox = QComboBox()
        self.ktupleComoBox.addItems(['1', '2', '3'])
        self.ktupleComoBox.setCurrentIndex(1)
        self.raac_cluster_comboBox = QComboBox()
        self.raac_cluster_comboBox.addItems([str(i) for i in self.types[self.clust_type]])
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow("RAAC subtype model:", self.modelComboBox)
        layout.addRow("Gap value:", self.gapComboBox)
        layout.addRow("Lambda value", self.lambdaComoBox)
        layout.addRow("K-tuple value:", self.ktupleComoBox)
        layout.addRow("RAAC cluster", self.raac_cluster_comboBox)
        layout.addWidget(self.buttons)

    def modelSelection(self, index):
        if index == 0:
            self.lambdaComoBox.setEnabled(False)
            self.gapComboBox.setEnabled(True)
        else:
            self.lambdaComoBox.setEnabled(True)
            self.gapComboBox.setEnabled(False)

    @staticmethod
    def getValues(clust_type):
        dialog = QPseKRAACInput(clust_type)
        result = dialog.exec_()
        model = dialog.modelComboBox.currentText()
        gap = dialog.gapComboBox.currentText()
        lambdaValue = dialog.lambdaComoBox.currentText()
        ktuple = dialog.ktupleComoBox.currentText()
        clust = dialog.raac_cluster_comboBox.currentText()
        return model, gap, lambdaValue, ktuple, clust, result == QDialog.Accepted


class QDNAACC2Input(QDialog):
    def __init__(self, lag_range):
        super(QDNAACC2Input, self).__init__()
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.lag_range = lag_range
        self.didna_list = ['Twist', 'Tilt', 'Roll', 'Shift', 'Slide', 'Rise',
                           'Base stacking', 'Protein induced deformability', 'B-DNA twist', 'Dinucleotide GC Content',
                           'A-philicity',
                           'Propeller twist', 'Duplex stability:(freeenergy)',
                           'Duplex tability(disruptenergy)', 'DNA denaturation', 'Bending stiffness',
                           'Protein DNA twist',
                           'Stabilising energy of Z-DNA', 'Aida_BA_transition', 'Breslauer_dG', 'Breslauer_dH',
                           'Breslauer_dS', 'Electron_interaction', 'Hartman_trans_free_energy', 'Helix-Coil_transition',
                           'Ivanov_BA_transition', 'Lisser_BZ_transition', 'Polar_interaction', 'SantaLucia_dG',
                           'SantaLucia_dH', 'SantaLucia_dS', 'Sarai_flexibility', 'Stability', 'Stacking_energy',
                           'Sugimoto_dG', 'Sugimoto_dH', 'Sugimoto_dS', 'Watson-Crick_interaction',
                           'Clash Strength', 'Roll_roll', 'Twist stiffness', 'Tilt stiffness', 'Shift_rise',
                           'Adenine content', 'Direction', 'Twist_shift', 'Enthalpy1', 'Twist_twist', 'Roll_shift',
                           'Shift_slide', 'Shift2', 'Tilt3', 'Tilt1', 'Tilt4', 'Tilt2', 'Slide (DNA-protein complex)1',
                           'Tilt_shift', 'Twist_tilt', 'Twist (DNA-protein complex)1', 'Tilt_rise', 'Roll_rise',
                           'Stacking energy', 'Stacking energy1', 'Stacking energy2', 'Stacking energy3',
                           'Propeller Twist',
                           'Roll11', 'Rise (DNA-protein complex)', 'Tilt_tilt', 'Roll4', 'Roll2', 'Roll3', 'Roll1',
                           'Minor Groove Size', 'GC content', 'Slide_slide', 'Enthalpy', 'Shift_shift',
                           'Slide stiffness',
                           'Melting Temperature1', 'Flexibility_slide', 'Minor Groove Distance',
                           'Rise (DNA-protein complex)1', 'Tilt (DNA-protein complex)', 'Guanine content',
                           'Roll (DNA-protein complex)1', 'Entropy', 'Cytosine content', 'Major Groove Size',
                           'Twist_rise',
                           'Major Groove Distance', 'Twist (DNA-protein complex)', 'Purine (AG) content',
                           'Melting Temperature', 'Free energy', 'Tilt_slide', 'Major Groove Width',
                           'Major Groove Depth',
                           'Wedge', 'Free energy8', 'Free energy6', 'Free energy7', 'Free energy4', 'Free energy5',
                           'Free energy2', 'Free energy3', 'Free energy1', 'Twist_roll', 'Shift (DNA-protein complex)',
                           'Rise_rise', 'Flexibility_shift', 'Shift (DNA-protein complex)1', 'Thymine content',
                           'Slide_rise',
                           'Tilt_roll', 'Tip', 'Keto (GT) content', 'Roll stiffness', 'Minor Groove Width',
                           'Inclination',
                           'Entropy1', 'Roll_slide', 'Slide (DNA-protein complex)', 'Twist1', 'Twist3', 'Twist2',
                           'Twist5',
                           'Twist4', 'Twist7', 'Twist6', 'Tilt (DNA-protein complex)1', 'Twist_slide',
                           'Minor Groove Depth',
                           'Roll (DNA-protein complex)', 'Rise2', 'Persistance Length', 'Rise3', 'Shift stiffness',
                           'Probability contacting nucleosome core', 'Mobility to bend towards major groove', 'Slide3',
                           'Slide2', 'Slide1', 'Shift1', 'Bend', 'Rise1', 'Rise stiffness',
                           'Mobility to bend towards minor groove', '', '']
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowTitle('DNA Auto-Correlation setting')
        self.resize(650, 400)
        layout = QGridLayout(self)
        lagLabel = QLabel('Lag value:')
        self.lagLineEdit = MyLineEdit("2")
        self.lagLineEdit.clicked.connect(self.setLagValue)
        label = QLabel('Di-DNA physicochemical indices: (Press [Ctrl] to select more)')
        self.tablewidget = QTableWidget(25, 6)
        self.tablewidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tablewidget.verticalHeader().setVisible(False)
        self.tablewidget.horizontalHeader().setVisible(False)
        self.tablewidget.setFont(QFont('Arial', 8))
        rect = QTableWidgetSelectionRange(0, 0, 0, 5)
        self.tablewidget.setRangeSelected(rect, True)
        data = np.array(self.didna_list).reshape((25, 6))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                cell = QTableWidgetItem(data[i][j])
                self.tablewidget.setItem(i, j, cell)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(lagLabel, 0, 0)
        layout.addWidget(self.lagLineEdit, 0, 1)
        layout.addWidget(label, 1, 0, 1, 2)
        layout.addWidget(self.tablewidget, 2, 0, 1, 2)
        layout.addWidget(self.buttons, 3, 0, 1, 2)

    def setLagValue(self):
        num, ok = QInputDialog.getInt(self, 'Lag', 'Get lag value', 2, 1, self.lag_range, 1)
        if ok:
            self.lagLineEdit.setText(str(num))

    def getLagValue(self):
        return int(self.lagLineEdit.text()) if self.lagLineEdit.text() != '' else 2

    def getProperty(self):
        content = self.tablewidget.selectedItems()
        property = ''
        for item in content:
            if item.text() != '':
                property += ';' + item.text()
        if property == '':
            property = ';Twist;Tilt;Roll;Shift;Slide;Rise'
        return property[1:]

    @staticmethod
    def getValues(nlag_range):
        dialog = QDNAACC2Input(nlag_range)
        result = dialog.exec_()
        lag = dialog.getLagValue()
        property = dialog.getProperty()
        return lag, property, result == QDialog.Accepted


class QDNADPCPInput(QDialog):
    def __init__(self):
        super(QDNADPCPInput, self).__init__()
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.didna_list = ['Twist', 'Tilt', 'Roll', 'Shift', 'Slide', 'Rise',
                           'Base stacking', 'Protein induced deformability', 'B-DNA twist', 'Dinucleotide GC Content',
                           'A-philicity',
                           'Propeller twist', 'Duplex stability:(freeenergy)',
                           'Duplex tability(disruptenergy)', 'DNA denaturation', 'Bending stiffness',
                           'Protein DNA twist',
                           'Stabilising energy of Z-DNA', 'Aida_BA_transition', 'Breslauer_dG', 'Breslauer_dH',
                           'Breslauer_dS', 'Electron_interaction', 'Hartman_trans_free_energy', 'Helix-Coil_transition',
                           'Ivanov_BA_transition', 'Lisser_BZ_transition', 'Polar_interaction', 'SantaLucia_dG',
                           'SantaLucia_dH', 'SantaLucia_dS', 'Sarai_flexibility', 'Stability', 'Stacking_energy',
                           'Sugimoto_dG', 'Sugimoto_dH', 'Sugimoto_dS', 'Watson-Crick_interaction',
                           'Clash Strength', 'Roll_roll', 'Twist stiffness', 'Tilt stiffness', 'Shift_rise',
                           'Adenine content', 'Direction', 'Twist_shift', 'Enthalpy1', 'Twist_twist', 'Roll_shift',
                           'Shift_slide', 'Shift2', 'Tilt3', 'Tilt1', 'Tilt4', 'Tilt2', 'Slide (DNA-protein complex)1',
                           'Tilt_shift', 'Twist_tilt', 'Twist (DNA-protein complex)1', 'Tilt_rise', 'Roll_rise',
                           'Stacking energy', 'Stacking energy1', 'Stacking energy2', 'Stacking energy3',
                           'Propeller Twist',
                           'Roll11', 'Rise (DNA-protein complex)', 'Tilt_tilt', 'Roll4', 'Roll2', 'Roll3', 'Roll1',
                           'Minor Groove Size', 'GC content', 'Slide_slide', 'Enthalpy', 'Shift_shift',
                           'Slide stiffness',
                           'Melting Temperature1', 'Flexibility_slide', 'Minor Groove Distance',
                           'Rise (DNA-protein complex)1', 'Tilt (DNA-protein complex)', 'Guanine content',
                           'Roll (DNA-protein complex)1', 'Entropy', 'Cytosine content', 'Major Groove Size',
                           'Twist_rise',
                           'Major Groove Distance', 'Twist (DNA-protein complex)', 'Purine (AG) content',
                           'Melting Temperature', 'Free energy', 'Tilt_slide', 'Major Groove Width',
                           'Major Groove Depth',
                           'Wedge', 'Free energy8', 'Free energy6', 'Free energy7', 'Free energy4', 'Free energy5',
                           'Free energy2', 'Free energy3', 'Free energy1', 'Twist_roll', 'Shift (DNA-protein complex)',
                           'Rise_rise', 'Flexibility_shift', 'Shift (DNA-protein complex)1', 'Thymine content',
                           'Slide_rise',
                           'Tilt_roll', 'Tip', 'Keto (GT) content', 'Roll stiffness', 'Minor Groove Width',
                           'Inclination',
                           'Entropy1', 'Roll_slide', 'Slide (DNA-protein complex)', 'Twist1', 'Twist3', 'Twist2',
                           'Twist5',
                           'Twist4', 'Twist7', 'Twist6', 'Tilt (DNA-protein complex)1', 'Twist_slide',
                           'Minor Groove Depth',
                           'Roll (DNA-protein complex)', 'Rise2', 'Persistance Length', 'Rise3', 'Shift stiffness',
                           'Probability contacting nucleosome core', 'Mobility to bend towards major groove', 'Slide3',
                           'Slide2', 'Slide1', 'Shift1', 'Bend', 'Rise1', 'Rise stiffness',
                           'Mobility to bend towards minor groove', '', '']
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowTitle('DNA DPCP setting')
        self.resize(650, 400)
        layout = QGridLayout(self)
        label = QLabel('Di-DNA physicochemical indices: (Press [Ctrl] to select more)')
        self.tablewidget = QTableWidget(25, 6)
        self.tablewidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tablewidget.verticalHeader().setVisible(False)
        self.tablewidget.horizontalHeader().setVisible(False)
        self.tablewidget.setFont(QFont('Arial', 8))
        rect = QTableWidgetSelectionRange(0, 0, 0, 5)
        self.tablewidget.setRangeSelected(rect, True)
        data = np.array(self.didna_list).reshape((25, 6))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                cell = QTableWidgetItem(data[i][j])
                self.tablewidget.setItem(i, j, cell)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(label, 0, 0, 1, 2)
        layout.addWidget(self.tablewidget, 1, 0, 1, 2)
        layout.addWidget(self.buttons, 2, 0, 1, 2)

    def getProperty(self):
        content = self.tablewidget.selectedItems()
        property = ''
        for item in content:
            if item.text() != '':
                property += ';' + item.text()
        if property == '':
            property = ';Twist;Tilt;Roll;Shift;Slide;Rise'
        return property[1:]

    @staticmethod
    def getValues():
        dialog = QDNADPCPInput()
        result = dialog.exec_()
        property = dialog.getProperty()
        return property, result == QDialog.Accepted


class QDNAACC3Input(QDialog):
    def __init__(self, lag_range):
        super(QDNAACC3Input, self).__init__()
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.lag_range = lag_range
        self.tridna_list = ['Dnase I', 'Bendability (DNAse)', 'Bendability (consensus)', 'Trinucleotide GC Content',
                            'Nucleosome positioning', 'Consensus_roll', 'Consensus-Rigid', 'Dnase I-Rigid',
                            'MW-Daltons',
                            'MW-kg', 'Nucleosome', 'Nucleosome-Rigid']
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowTitle('DNA Auto-Correlation setting')
        self.resize(430, 300)
        layout = QGridLayout(self)
        lagLabel = QLabel('Lag value:')
        self.lagLineEdit = MyLineEdit("2")
        self.lagLineEdit.clicked.connect(self.setLagValue)
        label = QLabel('Tri-DNA physicochemical indices: (Press [Ctrl] to select more)')
        self.tablewidget = QTableWidget(3, 4)
        self.tablewidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tablewidget.verticalHeader().setVisible(False)
        self.tablewidget.horizontalHeader().setVisible(False)
        self.tablewidget.setFont(QFont('Arial', 8))
        rect = QTableWidgetSelectionRange(0, 0, 0, 1)
        self.tablewidget.setRangeSelected(rect, True)
        data = np.array(self.tridna_list).reshape((3, 4))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                cell = QTableWidgetItem(data[i][j])
                self.tablewidget.setItem(i, j, cell)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(lagLabel, 0, 0)
        layout.addWidget(self.lagLineEdit, 0, 1)
        layout.addWidget(label, 1, 0, 1, 2)
        layout.addWidget(self.tablewidget, 2, 0, 1, 2)
        layout.addWidget(self.buttons, 3, 0, 1, 2)

    def setLagValue(self):
        num, ok = QInputDialog.getInt(self, 'Lag', 'Get lag value', 2, 1, self.lag_range, 1)
        if ok:
            self.lagLineEdit.setText(str(num))

    def getLagValue(self):
        return int(self.lagLineEdit.text()) if self.lagLineEdit.text() != '' else 2

    def getProperty(self):
        content = self.tablewidget.selectedItems()
        property = ''
        for item in content:
            if item.text() != '':
                property += ';' + item.text()
        if property == '':
            property = ';Dnase I;Bendability (DNAse)'
        return property[1:]

    @staticmethod
    def getValues(nlag_range):
        dialog = QDNAACC3Input(nlag_range)
        result = dialog.exec_()
        lag = dialog.getLagValue()
        property = dialog.getProperty()
        return lag, property, result == QDialog.Accepted


class QDNATPCPInput(QDialog):
    def __init__(self):
        super(QDNATPCPInput, self).__init__()
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.tridna_list = ['Dnase I', 'Bendability (DNAse)', 'Bendability (consensus)', 'Trinucleotide GC Content',
                            'Nucleosome positioning', 'Consensus_roll', 'Consensus-Rigid', 'Dnase I-Rigid',
                            'MW-Daltons',
                            'MW-kg', 'Nucleosome', 'Nucleosome-Rigid']
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowTitle('DNA TPCP setting')
        self.resize(430, 300)
        layout = QGridLayout(self)
        label = QLabel('Tri-DNA physicochemical indices: (Press [Ctrl] to select more)')
        self.tablewidget = QTableWidget(3, 4)
        self.tablewidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tablewidget.verticalHeader().setVisible(False)
        self.tablewidget.horizontalHeader().setVisible(False)
        self.tablewidget.setFont(QFont('Arial', 8))
        rect = QTableWidgetSelectionRange(0, 0, 0, 1)
        self.tablewidget.setRangeSelected(rect, True)
        data = np.array(self.tridna_list).reshape((3, 4))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                cell = QTableWidgetItem(data[i][j])
                self.tablewidget.setItem(i, j, cell)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(label, 0, 0, 1, 2)
        layout.addWidget(self.tablewidget, 1, 0, 1, 2)
        layout.addWidget(self.buttons, 2, 0, 1, 2)

    def getProperty(self):
        content = self.tablewidget.selectedItems()
        property = ''
        for item in content:
            if item.text() != '':
                property += ';' + item.text()
        if property == '':
            property = ';Dnase I;Bendability (DNAse)'
        return property[1:]

    @staticmethod
    def getValues():
        dialog = QDNATPCPInput()
        result = dialog.exec_()
        property = dialog.getProperty()
        return property, result == QDialog.Accepted


class QRNAACC2Input(QDialog):
    def __init__(self, lag_range):
        super(QRNAACC2Input, self).__init__()
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.lag_range = lag_range
        self.dirna_list = ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)',
                           'Entropy (RNA)', 'Adenine content', 'Purine (AG) content', 'Hydrophilicity (RNA)',
                           'Enthalpy (RNA)1', 'GC content',
                           'Entropy (RNA)1', 'Hydrophilicity (RNA)1', 'Free energy (RNA)', 'Keto (GT) content',
                           'Free energy (RNA)1', 'Enthalpy (RNA)',
                           'Stacking energy (RNA)', 'Guanine content', 'Cytosine content', 'Thymine content', '', '']
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowTitle('RNA Auto-Correlation setting')
        self.resize(430, 300)
        layout = QGridLayout(self)
        lagLabel = QLabel('Lag value:')
        self.lagLineEdit = MyLineEdit("2")
        self.lagLineEdit.clicked.connect(self.setLagValue)
        label = QLabel('Di-DNA physicochemical indices: (Press [Ctrl] to select more)')
        self.tablewidget = QTableWidget(6, 4)
        self.tablewidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tablewidget.verticalHeader().setVisible(False)
        self.tablewidget.horizontalHeader().setVisible(False)
        self.tablewidget.setFont(QFont('Arial', 8))
        rect = QTableWidgetSelectionRange(0, 0, 0, 3)
        self.tablewidget.setRangeSelected(rect, True)
        data = np.array(self.dirna_list).reshape((6, 4))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                cell = QTableWidgetItem(data[i][j])
                self.tablewidget.setItem(i, j, cell)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(lagLabel, 0, 0)
        layout.addWidget(self.lagLineEdit, 0, 1)
        layout.addWidget(label, 1, 0, 1, 2)
        layout.addWidget(self.tablewidget, 2, 0, 1, 2)
        layout.addWidget(self.buttons, 3, 0, 1, 2)

    def setLagValue(self):
        num, ok = QInputDialog.getInt(self, 'Lag', 'Get lag value', 2, 1, self.lag_range, 1)
        if ok:
            self.lagLineEdit.setText(str(num))

    def getLagValue(self):
        return int(self.lagLineEdit.text()) if self.lagLineEdit.text() != '' else 2

    def getProperty(self):
        content = self.tablewidget.selectedItems()
        property = ''
        for item in content:
            if item.text() != '':
                property += ';' + item.text()
        if property == '':
            property = ';Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)'
        return property[1:]

    @staticmethod
    def getValues(nlag_range):
        dialog = QRNAACC2Input(nlag_range)
        result = dialog.exec_()
        lag = dialog.getLagValue()
        property = dialog.getProperty()
        return lag, property, result == QDialog.Accepted


class QRNADPCPInput(QDialog):
    def __init__(self):
        super(QRNADPCPInput, self).__init__()
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.dirna_list = ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)',
                           'Entropy (RNA)', 'Adenine content', 'Purine (AG) content', 'Hydrophilicity (RNA)',
                           'Enthalpy (RNA)1', 'GC content',
                           'Entropy (RNA)1', 'Hydrophilicity (RNA)1', 'Free energy (RNA)', 'Keto (GT) content',
                           'Free energy (RNA)1', 'Enthalpy (RNA)',
                           'Stacking energy (RNA)', 'Guanine content', 'Cytosine content', 'Thymine content', '', '']
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowTitle('RNA DPCP setting')
        self.resize(430, 300)
        layout = QGridLayout(self)
        label = QLabel('Di-RNA physicochemical indices: (Press [Ctrl] to select more)')
        self.tablewidget = QTableWidget(6, 4)
        self.tablewidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tablewidget.verticalHeader().setVisible(False)
        self.tablewidget.horizontalHeader().setVisible(False)
        self.tablewidget.setFont(QFont('Arial', 8))
        rect = QTableWidgetSelectionRange(0, 0, 0, 3)
        self.tablewidget.setRangeSelected(rect, True)
        data = np.array(self.dirna_list).reshape((6, 4))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                cell = QTableWidgetItem(data[i][j])
                self.tablewidget.setItem(i, j, cell)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(label, 0, 0, 1, 2)
        layout.addWidget(self.tablewidget, 1, 0, 1, 2)
        layout.addWidget(self.buttons, 2, 0, 1, 2)

    def getProperty(self):
        content = self.tablewidget.selectedItems()
        property = ''
        for item in content:
            if item.text() != '':
                property += ';' + item.text()
        if property == '':
            property = ';Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)'
        return property[1:]

    @staticmethod
    def getValues():
        dialog = QRNADPCPInput()
        result = dialog.exec_()
        property = dialog.getProperty()
        return property, result == QDialog.Accepted


class QDNAPse2Input(QDialog):
    def __init__(self, lag_range):
        super(QDNAPse2Input, self).__init__()
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.lag_range = lag_range
        self.didna_list = ['Twist', 'Tilt', 'Roll', 'Shift', 'Slide', 'Rise',
                           'Base stacking', 'Protein induced deformability', 'B-DNA twist', 'Dinucleotide GC Content',
                           'A-philicity',
                           'Propeller twist', 'Duplex stability:(freeenergy)',
                           'Duplex tability(disruptenergy)', 'DNA denaturation', 'Bending stiffness',
                           'Protein DNA twist',
                           'Stabilising energy of Z-DNA', 'Aida_BA_transition', 'Breslauer_dG', 'Breslauer_dH',
                           'Breslauer_dS', 'Electron_interaction', 'Hartman_trans_free_energy', 'Helix-Coil_transition',
                           'Ivanov_BA_transition', 'Lisser_BZ_transition', 'Polar_interaction', 'SantaLucia_dG',
                           'SantaLucia_dH', 'SantaLucia_dS', 'Sarai_flexibility', 'Stability', 'Stacking_energy',
                           'Sugimoto_dG', 'Sugimoto_dH', 'Sugimoto_dS', 'Watson-Crick_interaction',
                           'Clash Strength', 'Roll_roll', 'Twist stiffness', 'Tilt stiffness', 'Shift_rise',
                           'Adenine content', 'Direction', 'Twist_shift', 'Enthalpy1', 'Twist_twist', 'Roll_shift',
                           'Shift_slide', 'Shift2', 'Tilt3', 'Tilt1', 'Tilt4', 'Tilt2', 'Slide (DNA-protein complex)1',
                           'Tilt_shift', 'Twist_tilt', 'Twist (DNA-protein complex)1', 'Tilt_rise', 'Roll_rise',
                           'Stacking energy', 'Stacking energy1', 'Stacking energy2', 'Stacking energy3',
                           'Propeller Twist',
                           'Roll11', 'Rise (DNA-protein complex)', 'Tilt_tilt', 'Roll4', 'Roll2', 'Roll3', 'Roll1',
                           'Minor Groove Size', 'GC content', 'Slide_slide', 'Enthalpy', 'Shift_shift',
                           'Slide stiffness',
                           'Melting Temperature1', 'Flexibility_slide', 'Minor Groove Distance',
                           'Rise (DNA-protein complex)1', 'Tilt (DNA-protein complex)', 'Guanine content',
                           'Roll (DNA-protein complex)1', 'Entropy', 'Cytosine content', 'Major Groove Size',
                           'Twist_rise',
                           'Major Groove Distance', 'Twist (DNA-protein complex)', 'Purine (AG) content',
                           'Melting Temperature', 'Free energy', 'Tilt_slide', 'Major Groove Width',
                           'Major Groove Depth',
                           'Wedge', 'Free energy8', 'Free energy6', 'Free energy7', 'Free energy4', 'Free energy5',
                           'Free energy2', 'Free energy3', 'Free energy1', 'Twist_roll', 'Shift (DNA-protein complex)',
                           'Rise_rise', 'Flexibility_shift', 'Shift (DNA-protein complex)1', 'Thymine content',
                           'Slide_rise',
                           'Tilt_roll', 'Tip', 'Keto (GT) content', 'Roll stiffness', 'Minor Groove Width',
                           'Inclination',
                           'Entropy1', 'Roll_slide', 'Slide (DNA-protein complex)', 'Twist1', 'Twist3', 'Twist2',
                           'Twist5',
                           'Twist4', 'Twist7', 'Twist6', 'Tilt (DNA-protein complex)1', 'Twist_slide',
                           'Minor Groove Depth',
                           'Roll (DNA-protein complex)', 'Rise2', 'Persistance Length', 'Rise3', 'Shift stiffness',
                           'Probability contacting nucleosome core', 'Mobility to bend towards major groove', 'Slide3',
                           'Slide2', 'Slide1', 'Shift1', 'Bend', 'Rise1', 'Rise stiffness',
                           'Mobility to bend towards minor groove', '', '']
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowTitle('DNA Pseudo Nucleic Acid Composition')
        self.resize(650, 400)
        layout = QGridLayout(self)
        lagLabel = QLabel('Lambda value:')
        self.lagLineEdit = MyLineEdit("2")
        self.lagLineEdit.clicked.connect(self.setLagValue)
        weightLabel = QLabel('Weight factor (0~1):')
        self.weightLineEdit = MyLineEdit("0.1")

        label = QLabel('Di-DNA physicochemical indices: (Press [Ctrl] to select more)')
        self.tablewidget = QTableWidget(25, 6)
        self.tablewidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tablewidget.verticalHeader().setVisible(False)
        self.tablewidget.horizontalHeader().setVisible(False)
        self.tablewidget.setFont(QFont('Arial', 8))
        rect = QTableWidgetSelectionRange(0, 0, 0, 5)
        self.tablewidget.setRangeSelected(rect, True)
        data = np.array(self.didna_list).reshape((25, 6))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                cell = QTableWidgetItem(data[i][j])
                self.tablewidget.setItem(i, j, cell)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(lagLabel, 0, 0)
        layout.addWidget(self.lagLineEdit, 0, 1)
        layout.addWidget(weightLabel, 1, 0)
        layout.addWidget(self.weightLineEdit, 1, 1)
        layout.addWidget(label, 2, 0, 1, 2)
        layout.addWidget(self.tablewidget, 3, 0, 1, 2)
        layout.addWidget(self.buttons, 4, 0, 1, 2)

    def setLagValue(self):
        num, ok = QInputDialog.getInt(self, 'Lambda', 'Get lambda value', 2, 1, self.lag_range, 1)
        if ok:
            self.lagLineEdit.setText(str(num))

    def getLagValue(self):
        return int(self.lagLineEdit.text()) if self.lagLineEdit.text() != '' else 2

    def getWeightValue(self):
        weight = 0.1
        if self.weightLineEdit.text() != '':
            weight = float(self.weightLineEdit.text())
            if 0 < weight < 1:
                return weight
            else:
                return 0.1
        else:
            return 0.1

    def getProperty(self):
        content = self.tablewidget.selectedItems()
        property = ''
        for item in content:
            if item.text() != '':
                property += ';' + item.text()
        if property == '':
            property = ';Twist;Tilt;Roll;Shift;Slide;Rise'
        return property[1:]

    @staticmethod
    def getValues(nlag_range):
        dialog = QDNAPse2Input(nlag_range)
        result = dialog.exec_()
        lambdaValue = dialog.getLagValue()
        weight = dialog.getWeightValue()
        property = dialog.getProperty()
        return lambdaValue, weight, property, result == QDialog.Accepted


class QDNAPse3Input(QDialog):
    def __init__(self, lag_range):
        super(QDNAPse3Input, self).__init__()
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.lag_range = lag_range
        self.tridna_list = ['Dnase I', 'Bendability (DNAse)', 'Bendability (consensus)', 'Trinucleotide GC Content',
                            'Nucleosome positioning', 'Consensus_roll', 'Consensus-Rigid', 'Dnase I-Rigid',
                            'MW-Daltons',
                            'MW-kg', 'Nucleosome', 'Nucleosome-Rigid']
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowTitle('DNA Pseudo Nucleic Acid Composition')
        self.resize(430, 300)
        layout = QGridLayout(self)
        lagLabel = QLabel('Lambda value:')
        self.lagLineEdit = MyLineEdit("2")
        self.lagLineEdit.clicked.connect(self.setLagValue)
        weightLabel = QLabel('Weight factor (0~1):')
        self.weightLineEdit = MyLineEdit("0.1")

        label = QLabel('Tri-DNA physicochemical indices: (Press [Ctrl] to select more)')
        self.tablewidget = QTableWidget(3, 4)
        self.tablewidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tablewidget.verticalHeader().setVisible(False)
        self.tablewidget.horizontalHeader().setVisible(False)
        self.tablewidget.setFont(QFont('Arial', 8))
        rect = QTableWidgetSelectionRange(0, 0, 0, 3)
        self.tablewidget.setRangeSelected(rect, True)
        data = np.array(self.tridna_list).reshape((3, 4))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                cell = QTableWidgetItem(data[i][j])
                self.tablewidget.setItem(i, j, cell)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(lagLabel, 0, 0)
        layout.addWidget(self.lagLineEdit, 0, 1)
        layout.addWidget(weightLabel, 1, 0)
        layout.addWidget(self.weightLineEdit, 1, 1)
        layout.addWidget(label, 2, 0, 1, 2)
        layout.addWidget(self.tablewidget, 3, 0, 1, 2)
        layout.addWidget(self.buttons, 4, 0, 1, 2)

    def setLagValue(self):
        num, ok = QInputDialog.getInt(self, 'Lambda', 'Get lambda value', 2, 1, self.lag_range, 1)
        if ok:
            self.lagLineEdit.setText(str(num))

    def getLagValue(self):
        return int(self.lagLineEdit.text()) if self.lagLineEdit.text() != '' else 2

    def getWeightValue(self):
        weight = 0.1
        if self.weightLineEdit.text() != '':
            weight = float(self.weightLineEdit.text())
            if 0 < weight < 1:
                return weight
            else:
                return 0.1
        else:
            return 0.1

    def getProperty(self):
        content = self.tablewidget.selectedItems()
        property = ''
        for item in content:
            if item.text() != '':
                property += ';' + item.text()
        if property == '':
            property = ';Dnase I;Bendability (DNAse)'
        return property[1:]

    @staticmethod
    def getValues(nlag_range):
        dialog = QDNAPse3Input(nlag_range)
        result = dialog.exec_()
        lambdaValue = dialog.getLagValue()
        weight = dialog.getWeightValue()
        property = dialog.getProperty()
        return lambdaValue, weight, property, result == QDialog.Accepted


class QRNAPse2Input(QDialog):
    def __init__(self, lag_range):
        super(QRNAPse2Input, self).__init__()
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.lag_range = lag_range
        self.dirna_list = ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)',
                           'Entropy (RNA)', 'Adenine content', 'Purine (AG) content', 'Hydrophilicity (RNA)',
                           'Enthalpy (RNA)1', 'GC content',
                           'Entropy (RNA)1', 'Hydrophilicity (RNA)1', 'Free energy (RNA)', 'Keto (GT) content',
                           'Free energy (RNA)1', 'Enthalpy (RNA)',
                           'Stacking energy (RNA)', 'Guanine content', 'Cytosine content', 'Thymine content', '', '']
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowTitle('RNA Pseudo Nucleic Acid Composition')
        self.resize(430, 350)
        layout = QGridLayout(self)
        lagLabel = QLabel('Lambda value:')
        self.lagLineEdit = MyLineEdit("2")
        self.lagLineEdit.clicked.connect(self.setLagValue)
        weightLabel = QLabel('Weight factor (0~1):')
        self.weightLineEdit = MyLineEdit("0.1")

        label = QLabel('Di-DNA physicochemical indices: (Press [Ctrl] to select more)')
        self.tablewidget = QTableWidget(6, 4)
        self.tablewidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tablewidget.verticalHeader().setVisible(False)
        self.tablewidget.horizontalHeader().setVisible(False)
        self.tablewidget.setFont(QFont('Arial', 8))
        rect = QTableWidgetSelectionRange(0, 0, 0, 3)
        self.tablewidget.setRangeSelected(rect, True)
        data = np.array(self.dirna_list).reshape((6, 4))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                cell = QTableWidgetItem(data[i][j])
                self.tablewidget.setItem(i, j, cell)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(lagLabel, 0, 0)
        layout.addWidget(self.lagLineEdit, 0, 1)
        layout.addWidget(weightLabel, 1, 0)
        layout.addWidget(self.weightLineEdit, 1, 1)
        layout.addWidget(label, 2, 0, 1, 2)
        layout.addWidget(self.tablewidget, 3, 0, 1, 2)
        layout.addWidget(self.buttons, 4, 0, 1, 2)

    def setLagValue(self):
        num, ok = QInputDialog.getInt(self, 'Lambda', 'Get lambda value', 2, 1, self.lag_range, 1)
        if ok:
            self.lagLineEdit.setText(str(num))

    def getLagValue(self):
        return int(self.lagLineEdit.text()) if self.lagLineEdit.text() != '' else 2

    def getWeightValue(self):
        weight = 0.1
        if self.weightLineEdit.text() != '':
            weight = float(self.weightLineEdit.text())
            if 0 < weight < 1:
                return weight
            else:
                return 0.1
        else:
            return 0.1

    def getProperty(self):
        content = self.tablewidget.selectedItems()
        property = ''
        for item in content:
            if item.text() != '':
                property += ';' + item.text()
        if property == '':
            property = ';Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)'
        return property[1:]

    @staticmethod
    def getValues(nlag_range):
        dialog = QRNAPse2Input(nlag_range)
        result = dialog.exec_()
        lambdaValue = dialog.getLagValue()
        weight = dialog.getWeightValue()
        property = dialog.getProperty()
        return lambdaValue, weight, property, result == QDialog.Accepted


class QDNAPseKNCInput(QDialog):
    def __init__(self, lag_range):
        super(QDNAPseKNCInput, self).__init__()
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.lag_range = lag_range
        self.didna_list = ['Twist', 'Tilt', 'Roll', 'Shift', 'Slide', 'Rise']
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowTitle('DNA Pseudo Nucleic Acid Composition')
        layout = QGridLayout(self)
        lagLabel = QLabel('Lambda value:')
        self.lagLineEdit = MyLineEdit("2")
        self.lagLineEdit.clicked.connect(self.setLagValue)
        weightLabel = QLabel('Weight factor (0~1):')
        self.weightLineEdit = MyLineEdit("0.1")
        kmerLabel = QLabel('Kmer size:')
        self.kmerLineEdit = MyLineEdit("3")
        self.kmerLineEdit.clicked.connect(self.setKmerSize)
        label = QLabel('Di-DNA physicochemical indices: (Press [Ctrl] to select more)')
        self.tablewidget = QTableWidget(2, 3)
        self.tablewidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tablewidget.verticalHeader().setVisible(False)
        self.tablewidget.horizontalHeader().setVisible(False)
        self.tablewidget.setFont(QFont('Arial', 8))
        rect = QTableWidgetSelectionRange(0, 0, 1, 2)
        self.tablewidget.setRangeSelected(rect, True)
        data = np.array(self.didna_list).reshape((2, 3))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                cell = QTableWidgetItem(data[i][j])
                self.tablewidget.setItem(i, j, cell)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(lagLabel, 0, 0)
        layout.addWidget(self.lagLineEdit, 0, 1)
        layout.addWidget(weightLabel, 1, 0)
        layout.addWidget(self.weightLineEdit, 1, 1)
        layout.addWidget(kmerLabel, 2, 0)
        layout.addWidget(self.kmerLineEdit, 2, 1)
        layout.addWidget(label, 3, 0, 1, 2)
        layout.addWidget(self.tablewidget, 4, 0, 1, 2)
        layout.addWidget(self.buttons, 5, 0, 1, 2)

    def setLagValue(self):
        num, ok = QInputDialog.getInt(self, 'Lambda', 'Get lambda value', 2, 1, self.lag_range, 1)
        if ok:
            self.lagLineEdit.setText(str(num))

    def getLagValue(self):
        return int(self.lagLineEdit.text()) if self.lagLineEdit.text() != '' else 2

    def setKmerSize(self):
        num, ok = QInputDialog.getInt(self, 'Kmer size', 'Get kmer size', 3, 1, 6, 1)
        if ok:
            self.kmerLineEdit.setText(str(num))

    def getKmerSize(self):
        return int(self.kmerLineEdit.text()) if self.kmerLineEdit.text() != '' else 3

    def getWeightValue(self):
        weight = 0.1
        if self.weightLineEdit.text() != '':
            weight = float(self.weightLineEdit.text())
            if 0 < weight < 1:
                return weight
            else:
                return 0.1
        else:
            return 0.1

    def getProperty(self):
        content = self.tablewidget.selectedItems()
        property = ''
        for item in content:
            if item.text() != '':
                property += ';' + item.text()
        if property == '':
            property = ';Twist;Tilt;Roll;Shift;Slide;Rise'
        return property[1:]

    @staticmethod
    def getValues(nlag_range):
        dialog = QDNAPseKNCInput(nlag_range)
        result = dialog.exec_()
        lambdaValue = dialog.getLagValue()
        weight = dialog.getWeightValue()
        kmer = dialog.getKmerSize()
        property = dialog.getProperty()
        return lambdaValue, weight, kmer, property, result == QDialog.Accepted


class QRNAPseKNCInput(QDialog):
    def __init__(self, lag_range):
        super(QRNAPseKNCInput, self).__init__()
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.lag_range = lag_range
        self.dirna_list = ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowTitle('RNA Pseudo Nucleic Acid Composition')
        self.resize(430, 350)
        layout = QGridLayout(self)
        lagLabel = QLabel('Lambda value:')
        self.lagLineEdit = MyLineEdit("2")
        self.lagLineEdit.clicked.connect(self.setLagValue)
        weightLabel = QLabel('Weight factor (0~1):')
        self.weightLineEdit = MyLineEdit("0.1")
        kmerLabel = QLabel('Kmer size:')
        self.kmerLineEdit = MyLineEdit("3")
        self.kmerLineEdit.clicked.connect(self.setKmerSize)
        label = QLabel('Di-DNA physicochemical indices: (Press [Ctrl] to select more)')
        self.tablewidget = QTableWidget(2, 3)
        self.tablewidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tablewidget.verticalHeader().setVisible(False)
        self.tablewidget.horizontalHeader().setVisible(False)
        self.tablewidget.setFont(QFont('Arial', 8))
        rect = QTableWidgetSelectionRange(0, 0, 1, 2)
        self.tablewidget.setRangeSelected(rect, True)
        data = np.array(self.dirna_list).reshape((2, 3))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                cell = QTableWidgetItem(data[i][j])
                self.tablewidget.setItem(i, j, cell)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(lagLabel, 0, 0)
        layout.addWidget(self.lagLineEdit, 0, 1)
        layout.addWidget(weightLabel, 1, 0)
        layout.addWidget(self.weightLineEdit, 1, 1)
        layout.addWidget(kmerLabel, 2, 0)
        layout.addWidget(self.kmerLineEdit, 2, 1)
        layout.addWidget(label, 3, 0, 1, 2)
        layout.addWidget(self.tablewidget, 4, 0, 1, 2)
        layout.addWidget(self.buttons, 5, 0, 1, 2)

    def setLagValue(self):
        num, ok = QInputDialog.getInt(self, 'Lambda', 'Get lambda value', 2, 1, self.lag_range, 1)
        if ok:
            self.lagLineEdit.setText(str(num))

    def getLagValue(self):
        return int(self.lagLineEdit.text()) if self.lagLineEdit.text() != '' else 2

    def setKmerSize(self):
        num, ok = QInputDialog.getInt(self, 'Kmer size', 'Get kmer size', 3, 1, 6, 1)
        if ok:
            self.kmerLineEdit.setText(str(num))

    def getKmerSize(self):
        return int(self.kmerLineEdit.text()) if self.kmerLineEdit.text() != '' else 3

    def getWeightValue(self):
        weight = 0.1
        if self.weightLineEdit.text() != '':
            weight = float(self.weightLineEdit.text())
            if 0 < weight < 1:
                return weight
            else:
                return 0.1
        else:
            return 0.1

    def getProperty(self):
        content = self.tablewidget.selectedItems()
        property = ''
        for item in content:
            if item.text() != '':
                property += ';' + item.text()
        if property == '':
            property = ';Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)'
        return property[1:]

    @staticmethod
    def getValues(nlag_range):
        dialog = QRNAPseKNCInput(nlag_range)
        result = dialog.exec_()
        lambdaValue = dialog.getLagValue()
        weight = dialog.getWeightValue()
        kmer = dialog.getKmerSize()
        property = dialog.getProperty()
        return lambdaValue, weight, kmer, property, result == QDialog.Accepted


class QAAindexInput(QDialog):
    def __init__(self):
        super(QAAindexInput, self).__init__()
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.AAindex = ['ANDN920101', 'ARGP820101', 'ARGP820102', 'ARGP820103', 'BEGF750101', 'BEGF750102',
                        'BEGF750103', 'BHAR880101', 'BIGC670101', 'BIOV880101', 'BIOV880102', 'BROC820101',
                        'BROC820102', 'BULH740101', 'BULH740102', 'BUNA790101', 'BUNA790102', 'BUNA790103',
                        'BURA740101', 'BURA740102', 'CHAM810101', 'CHAM820101', 'CHAM820102', 'CHAM830101',
                        'CHAM830102', 'CHAM830103', 'CHAM830104', 'CHAM830105', 'CHAM830106', 'CHAM830107',
                        'CHAM830108', 'CHOC750101', 'CHOC760101', 'CHOC760102', 'CHOC760103', 'CHOC760104',
                        'CHOP780101', 'CHOP780201', 'CHOP780202', 'CHOP780203', 'CHOP780204', 'CHOP780205',
                        'CHOP780206', 'CHOP780207', 'CHOP780208', 'CHOP780209', 'CHOP780210', 'CHOP780211',
                        'CHOP780212', 'CHOP780213', 'CHOP780214', 'CHOP780215', 'CHOP780216', 'CIDH920101',
                        'CIDH920102', 'CIDH920103', 'CIDH920104', 'CIDH920105', 'COHE430101', 'CRAJ730101',
                        'CRAJ730102', 'CRAJ730103', 'DAWD720101', 'DAYM780101', 'DAYM780201', 'DESM900101',
                        'DESM900102', 'EISD840101', 'EISD860101', 'EISD860102', 'EISD860103', 'FASG760101',
                        'FASG760102', 'FASG760103', 'FASG760104', 'FASG760105', 'FAUJ830101', 'FAUJ880101',
                        'FAUJ880102', 'FAUJ880103', 'FAUJ880104', 'FAUJ880105', 'FAUJ880106', 'FAUJ880107',
                        'FAUJ880108', 'FAUJ880109', 'FAUJ880110', 'FAUJ880111', 'FAUJ880112', 'FAUJ880113',
                        'FINA770101', 'FINA910101', 'FINA910102', 'FINA910103', 'FINA910104', 'GARJ730101',
                        'GEIM800101', 'GEIM800102', 'GEIM800103', 'GEIM800104', 'GEIM800105', 'GEIM800106',
                        'GEIM800107', 'GEIM800108', 'GEIM800109', 'GEIM800110', 'GEIM800111', 'GOLD730101',
                        'GOLD730102', 'GRAR740101', 'GRAR740102', 'GRAR740103', 'GUYH850101', 'HOPA770101',
                        'HOPT810101', 'HUTJ700101', 'HUTJ700102', 'HUTJ700103', 'ISOY800101', 'ISOY800102',
                        'ISOY800103', 'ISOY800104', 'ISOY800105', 'ISOY800106', 'ISOY800107', 'ISOY800108',
                        'JANJ780101', 'JANJ780102', 'JANJ780103', 'JANJ790101', 'JANJ790102', 'JOND750101',
                        'JOND750102', 'JOND920101', 'JOND920102', 'JUKT750101', 'JUNJ780101', 'KANM800101',
                        'KANM800102', 'KANM800103', 'KANM800104', 'KARP850101', 'KARP850102', 'KARP850103',
                        'KHAG800101', 'KLEP840101', 'KRIW710101', 'KRIW790101', 'KRIW790102', 'KRIW790103',
                        'KYTJ820101', 'LAWE840101', 'LEVM760101', 'LEVM760102', 'LEVM760103', 'LEVM760104',
                        'LEVM760105', 'LEVM760106', 'LEVM760107', 'LEVM780101', 'LEVM780102', 'LEVM780103',
                        'LEVM780104', 'LEVM780105', 'LEVM780106', 'LEWP710101', 'LIFS790101', 'LIFS790102',
                        'LIFS790103', 'MANP780101', 'MAXF760101', 'MAXF760102', 'MAXF760103', 'MAXF760104',
                        'MAXF760105', 'MAXF760106', 'MCMT640101', 'MEEJ800101', 'MEEJ800102', 'MEEJ810101',
                        'MEEJ810102', 'MEIH800101', 'MEIH800102', 'MEIH800103', 'MIYS850101', 'NAGK730101',
                        'NAGK730102', 'NAGK730103', 'NAKH900101', 'NAKH900102', 'NAKH900103', 'NAKH900104',
                        'NAKH900105', 'NAKH900106', 'NAKH900107', 'NAKH900108', 'NAKH900109', 'NAKH900110',
                        'NAKH900111', 'NAKH900112', 'NAKH900113', 'NAKH920101', 'NAKH920102', 'NAKH920103',
                        'NAKH920104', 'NAKH920105', 'NAKH920106', 'NAKH920107', 'NAKH920108', 'NISK800101',
                        'NISK860101', 'NOZY710101', 'OOBM770101', 'OOBM770102', 'OOBM770103', 'OOBM770104',
                        'OOBM770105', 'OOBM850101', 'OOBM850102', 'OOBM850103', 'OOBM850104', 'OOBM850105',
                        'PALJ810101', 'PALJ810102', 'PALJ810103', 'PALJ810104', 'PALJ810105', 'PALJ810106',
                        'PALJ810107', 'PALJ810108', 'PALJ810109', 'PALJ810110', 'PALJ810111', 'PALJ810112',
                        'PALJ810113', 'PALJ810114', 'PALJ810115', 'PALJ810116', 'PARJ860101', 'PLIV810101',
                        'PONP800101', 'PONP800102', 'PONP800103', 'PONP800104', 'PONP800105', 'PONP800106',
                        'PONP800107', 'PONP800108', 'PRAM820101', 'PRAM820102', 'PRAM820103', 'PRAM900101',
                        'PRAM900102', 'PRAM900103', 'PRAM900104', 'PTIO830101', 'PTIO830102', 'QIAN880101',
                        'QIAN880102', 'QIAN880103', 'QIAN880104', 'QIAN880105', 'QIAN880106', 'QIAN880107',
                        'QIAN880108', 'QIAN880109', 'QIAN880110', 'QIAN880111', 'QIAN880112', 'QIAN880113',
                        'QIAN880114', 'QIAN880115', 'QIAN880116', 'QIAN880117', 'QIAN880118', 'QIAN880119',
                        'QIAN880120', 'QIAN880121', 'QIAN880122', 'QIAN880123', 'QIAN880124', 'QIAN880125',
                        'QIAN880126', 'QIAN880127', 'QIAN880128', 'QIAN880129', 'QIAN880130', 'QIAN880131',
                        'QIAN880132', 'QIAN880133', 'QIAN880134', 'QIAN880135', 'QIAN880136', 'QIAN880137',
                        'QIAN880138', 'QIAN880139', 'RACS770101', 'RACS770102', 'RACS770103', 'RACS820101',
                        'RACS820102', 'RACS820103', 'RACS820104', 'RACS820105', 'RACS820106', 'RACS820107',
                        'RACS820108', 'RACS820109', 'RACS820110', 'RACS820111', 'RACS820112', 'RACS820113',
                        'RACS820114', 'RADA880101', 'RADA880102', 'RADA880103', 'RADA880104', 'RADA880105',
                        'RADA880106', 'RADA880107', 'RADA880108', 'RICJ880101', 'RICJ880102', 'RICJ880103',
                        'RICJ880104', 'RICJ880105', 'RICJ880106', 'RICJ880107', 'RICJ880108', 'RICJ880109',
                        'RICJ880110', 'RICJ880111', 'RICJ880112', 'RICJ880113', 'RICJ880114', 'RICJ880115',
                        'RICJ880116', 'RICJ880117', 'ROBB760101', 'ROBB760102', 'ROBB760103', 'ROBB760104',
                        'ROBB760105', 'ROBB760106', 'ROBB760107', 'ROBB760108', 'ROBB760109', 'ROBB760110',
                        'ROBB760111', 'ROBB760112', 'ROBB760113', 'ROBB790101', 'ROSG850101', 'ROSG850102',
                        'ROSM880101', 'ROSM880102', 'ROSM880103', 'SIMZ760101', 'SNEP660101', 'SNEP660102',
                        'SNEP660103', 'SNEP660104', 'SUEM840101', 'SUEM840102', 'SWER830101', 'TANS770101',
                        'TANS770102', 'TANS770103', 'TANS770104', 'TANS770105', 'TANS770106', 'TANS770107',
                        'TANS770108', 'TANS770109', 'TANS770110', 'VASM830101', 'VASM830102', 'VASM830103',
                        'VELV850101', 'VENT840101', 'VHEG790101', 'WARP780101', 'WEBA780101', 'WERD780101',
                        'WERD780102', 'WERD780103', 'WERD780104', 'WOEC730101', 'WOLR810101', 'WOLS870101',
                        'WOLS870102', 'WOLS870103', 'YUTK870101', 'YUTK870102', 'YUTK870103', 'YUTK870104',
                        'ZASB820101', 'ZIMJ680101', 'ZIMJ680102', 'ZIMJ680103', 'ZIMJ680104', 'ZIMJ680105',
                        'AURR980101', 'AURR980102', 'AURR980103', 'AURR980104', 'AURR980105', 'AURR980106',
                        'AURR980107', 'AURR980108', 'AURR980109', 'AURR980110', 'AURR980111', 'AURR980112',
                        'AURR980113', 'AURR980114', 'AURR980115', 'AURR980116', 'AURR980117', 'AURR980118',
                        'AURR980119', 'AURR980120', 'ONEK900101', 'ONEK900102', 'VINM940101', 'VINM940102',
                        'VINM940103', 'VINM940104', 'MUNV940101', 'MUNV940102', 'MUNV940103', 'MUNV940104',
                        'MUNV940105', 'WIMW960101', 'KIMC930101', 'MONM990101', 'BLAM930101', 'PARS000101',
                        'PARS000102', 'KUMS000101', 'KUMS000102', 'KUMS000103', 'KUMS000104', 'TAKK010101',
                        'FODM020101', 'NADH010101', 'NADH010102', 'NADH010103', 'NADH010104', 'NADH010105',
                        'NADH010106', 'NADH010107', 'MONM990201', 'KOEP990101', 'KOEP990102', 'CEDJ970101',
                        'CEDJ970102', 'CEDJ970103', 'CEDJ970104', 'CEDJ970105', 'FUKS010101', 'FUKS010102',
                        'FUKS010103', 'FUKS010104', 'FUKS010105', 'FUKS010106', 'FUKS010107', 'FUKS010108',
                        'FUKS010109', 'FUKS010110', 'FUKS010111', 'FUKS010112', 'MITS020101', 'TSAJ990101',
                        'TSAJ990102', 'COSI940101', 'PONP930101', 'WILM950101', 'WILM950102', 'WILM950103',
                        'WILM950104', 'KUHL950101', 'GUOD860101', 'JURD980101', 'BASU050101', 'BASU050102',
                        'BASU050103', 'SUYM030101', 'PUNT030101', 'PUNT030102', 'GEOR030101', 'GEOR030102',
                        'GEOR030103', 'GEOR030104', 'GEOR030105', 'GEOR030106', 'GEOR030107', 'GEOR030108',
                        'GEOR030109', 'ZHOH040101', 'ZHOH040102', 'ZHOH040103', 'BAEK050101', 'HARY940101',
                        'PONJ960101', 'DIGM050101', 'WOLR790101', 'OLSK800101', 'KIDA850101', 'GUYH850102',
                        'GUYH850104', 'GUYH850105', 'JACR890101', 'COWR900101', 'BLAS910101', 'CASG920101',
                        'CORJ870101', 'CORJ870102', 'CORJ870103', 'CORJ870104', 'CORJ870105', 'CORJ870106',
                        'CORJ870107', 'CORJ870108', 'MIYS990101', 'MIYS990102', 'MIYS990103', 'MIYS990104',
                        'MIYS990105', 'ENGD860101', 'FASG890101', '', '', '', '', '']
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowTitle('AAIndex')
        self.resize(850, 500)
        layout = QGridLayout(self)
        label = QLabel('AAIndex: (Press [Ctrl] to select more)')
        self.tablewidget = QTableWidget(67, 8)
        self.tablewidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tablewidget.verticalHeader().setVisible(False)
        self.tablewidget.horizontalHeader().setVisible(False)
        self.tablewidget.setFont(QFont('Arial', 8))
        rect = QTableWidgetSelectionRange(0, 0, 0, 7)
        self.tablewidget.setRangeSelected(rect, True)
        data = np.array(self.AAindex).reshape((67, 8))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                cell = QTableWidgetItem(data[i][j])
                self.tablewidget.setItem(i, j, cell)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(label, 0, 0, 1, 2)
        layout.addWidget(self.tablewidget, 1, 0, 1, 2)
        layout.addWidget(self.buttons, 2, 0, 1, 2)

    def getProperty(self):
        content = self.tablewidget.selectedItems()
        property = ''
        for item in content:
            if item.text() != '':
                property += ';' + item.text()
        if property == '':
            property = ';ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101'
        return property[1:]

    @staticmethod
    def getValues():
        dialog = QAAindexInput()
        result = dialog.exec_()
        property = dialog.getProperty()
        return property, result == QDialog.Accepted


class QAAindex2Input(QDialog):
    def __init__(self, nlag_range):
        super(QAAindex2Input, self).__init__()
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.nlag_range = nlag_range
        self.AAindex = ['ANDN920101', 'ARGP820101', 'ARGP820102', 'ARGP820103', 'BEGF750101', 'BEGF750102',
                        'BEGF750103', 'BHAR880101', 'BIGC670101', 'BIOV880101', 'BIOV880102', 'BROC820101',
                        'BROC820102', 'BULH740101', 'BULH740102', 'BUNA790101', 'BUNA790102', 'BUNA790103',
                        'BURA740101', 'BURA740102', 'CHAM810101', 'CHAM820101', 'CHAM820102', 'CHAM830101',
                        'CHAM830102', 'CHAM830103', 'CHAM830104', 'CHAM830105', 'CHAM830106', 'CHAM830107',
                        'CHAM830108', 'CHOC750101', 'CHOC760101', 'CHOC760102', 'CHOC760103', 'CHOC760104',
                        'CHOP780101', 'CHOP780201', 'CHOP780202', 'CHOP780203', 'CHOP780204', 'CHOP780205',
                        'CHOP780206', 'CHOP780207', 'CHOP780208', 'CHOP780209', 'CHOP780210', 'CHOP780211',
                        'CHOP780212', 'CHOP780213', 'CHOP780214', 'CHOP780215', 'CHOP780216', 'CIDH920101',
                        'CIDH920102', 'CIDH920103', 'CIDH920104', 'CIDH920105', 'COHE430101', 'CRAJ730101',
                        'CRAJ730102', 'CRAJ730103', 'DAWD720101', 'DAYM780101', 'DAYM780201', 'DESM900101',
                        'DESM900102', 'EISD840101', 'EISD860101', 'EISD860102', 'EISD860103', 'FASG760101',
                        'FASG760102', 'FASG760103', 'FASG760104', 'FASG760105', 'FAUJ830101', 'FAUJ880101',
                        'FAUJ880102', 'FAUJ880103', 'FAUJ880104', 'FAUJ880105', 'FAUJ880106', 'FAUJ880107',
                        'FAUJ880108', 'FAUJ880109', 'FAUJ880110', 'FAUJ880111', 'FAUJ880112', 'FAUJ880113',
                        'FINA770101', 'FINA910101', 'FINA910102', 'FINA910103', 'FINA910104', 'GARJ730101',
                        'GEIM800101', 'GEIM800102', 'GEIM800103', 'GEIM800104', 'GEIM800105', 'GEIM800106',
                        'GEIM800107', 'GEIM800108', 'GEIM800109', 'GEIM800110', 'GEIM800111', 'GOLD730101',
                        'GOLD730102', 'GRAR740101', 'GRAR740102', 'GRAR740103', 'GUYH850101', 'HOPA770101',
                        'HOPT810101', 'HUTJ700101', 'HUTJ700102', 'HUTJ700103', 'ISOY800101', 'ISOY800102',
                        'ISOY800103', 'ISOY800104', 'ISOY800105', 'ISOY800106', 'ISOY800107', 'ISOY800108',
                        'JANJ780101', 'JANJ780102', 'JANJ780103', 'JANJ790101', 'JANJ790102', 'JOND750101',
                        'JOND750102', 'JOND920101', 'JOND920102', 'JUKT750101', 'JUNJ780101', 'KANM800101',
                        'KANM800102', 'KANM800103', 'KANM800104', 'KARP850101', 'KARP850102', 'KARP850103',
                        'KHAG800101', 'KLEP840101', 'KRIW710101', 'KRIW790101', 'KRIW790102', 'KRIW790103',
                        'KYTJ820101', 'LAWE840101', 'LEVM760101', 'LEVM760102', 'LEVM760103', 'LEVM760104',
                        'LEVM760105', 'LEVM760106', 'LEVM760107', 'LEVM780101', 'LEVM780102', 'LEVM780103',
                        'LEVM780104', 'LEVM780105', 'LEVM780106', 'LEWP710101', 'LIFS790101', 'LIFS790102',
                        'LIFS790103', 'MANP780101', 'MAXF760101', 'MAXF760102', 'MAXF760103', 'MAXF760104',
                        'MAXF760105', 'MAXF760106', 'MCMT640101', 'MEEJ800101', 'MEEJ800102', 'MEEJ810101',
                        'MEEJ810102', 'MEIH800101', 'MEIH800102', 'MEIH800103', 'MIYS850101', 'NAGK730101',
                        'NAGK730102', 'NAGK730103', 'NAKH900101', 'NAKH900102', 'NAKH900103', 'NAKH900104',
                        'NAKH900105', 'NAKH900106', 'NAKH900107', 'NAKH900108', 'NAKH900109', 'NAKH900110',
                        'NAKH900111', 'NAKH900112', 'NAKH900113', 'NAKH920101', 'NAKH920102', 'NAKH920103',
                        'NAKH920104', 'NAKH920105', 'NAKH920106', 'NAKH920107', 'NAKH920108', 'NISK800101',
                        'NISK860101', 'NOZY710101', 'OOBM770101', 'OOBM770102', 'OOBM770103', 'OOBM770104',
                        'OOBM770105', 'OOBM850101', 'OOBM850102', 'OOBM850103', 'OOBM850104', 'OOBM850105',
                        'PALJ810101', 'PALJ810102', 'PALJ810103', 'PALJ810104', 'PALJ810105', 'PALJ810106',
                        'PALJ810107', 'PALJ810108', 'PALJ810109', 'PALJ810110', 'PALJ810111', 'PALJ810112',
                        'PALJ810113', 'PALJ810114', 'PALJ810115', 'PALJ810116', 'PARJ860101', 'PLIV810101',
                        'PONP800101', 'PONP800102', 'PONP800103', 'PONP800104', 'PONP800105', 'PONP800106',
                        'PONP800107', 'PONP800108', 'PRAM820101', 'PRAM820102', 'PRAM820103', 'PRAM900101',
                        'PRAM900102', 'PRAM900103', 'PRAM900104', 'PTIO830101', 'PTIO830102', 'QIAN880101',
                        'QIAN880102', 'QIAN880103', 'QIAN880104', 'QIAN880105', 'QIAN880106', 'QIAN880107',
                        'QIAN880108', 'QIAN880109', 'QIAN880110', 'QIAN880111', 'QIAN880112', 'QIAN880113',
                        'QIAN880114', 'QIAN880115', 'QIAN880116', 'QIAN880117', 'QIAN880118', 'QIAN880119',
                        'QIAN880120', 'QIAN880121', 'QIAN880122', 'QIAN880123', 'QIAN880124', 'QIAN880125',
                        'QIAN880126', 'QIAN880127', 'QIAN880128', 'QIAN880129', 'QIAN880130', 'QIAN880131',
                        'QIAN880132', 'QIAN880133', 'QIAN880134', 'QIAN880135', 'QIAN880136', 'QIAN880137',
                        'QIAN880138', 'QIAN880139', 'RACS770101', 'RACS770102', 'RACS770103', 'RACS820101',
                        'RACS820102', 'RACS820103', 'RACS820104', 'RACS820105', 'RACS820106', 'RACS820107',
                        'RACS820108', 'RACS820109', 'RACS820110', 'RACS820111', 'RACS820112', 'RACS820113',
                        'RACS820114', 'RADA880101', 'RADA880102', 'RADA880103', 'RADA880104', 'RADA880105',
                        'RADA880106', 'RADA880107', 'RADA880108', 'RICJ880101', 'RICJ880102', 'RICJ880103',
                        'RICJ880104', 'RICJ880105', 'RICJ880106', 'RICJ880107', 'RICJ880108', 'RICJ880109',
                        'RICJ880110', 'RICJ880111', 'RICJ880112', 'RICJ880113', 'RICJ880114', 'RICJ880115',
                        'RICJ880116', 'RICJ880117', 'ROBB760101', 'ROBB760102', 'ROBB760103', 'ROBB760104',
                        'ROBB760105', 'ROBB760106', 'ROBB760107', 'ROBB760108', 'ROBB760109', 'ROBB760110',
                        'ROBB760111', 'ROBB760112', 'ROBB760113', 'ROBB790101', 'ROSG850101', 'ROSG850102',
                        'ROSM880101', 'ROSM880102', 'ROSM880103', 'SIMZ760101', 'SNEP660101', 'SNEP660102',
                        'SNEP660103', 'SNEP660104', 'SUEM840101', 'SUEM840102', 'SWER830101', 'TANS770101',
                        'TANS770102', 'TANS770103', 'TANS770104', 'TANS770105', 'TANS770106', 'TANS770107',
                        'TANS770108', 'TANS770109', 'TANS770110', 'VASM830101', 'VASM830102', 'VASM830103',
                        'VELV850101', 'VENT840101', 'VHEG790101', 'WARP780101', 'WEBA780101', 'WERD780101',
                        'WERD780102', 'WERD780103', 'WERD780104', 'WOEC730101', 'WOLR810101', 'WOLS870101',
                        'WOLS870102', 'WOLS870103', 'YUTK870101', 'YUTK870102', 'YUTK870103', 'YUTK870104',
                        'ZASB820101', 'ZIMJ680101', 'ZIMJ680102', 'ZIMJ680103', 'ZIMJ680104', 'ZIMJ680105',
                        'AURR980101', 'AURR980102', 'AURR980103', 'AURR980104', 'AURR980105', 'AURR980106',
                        'AURR980107', 'AURR980108', 'AURR980109', 'AURR980110', 'AURR980111', 'AURR980112',
                        'AURR980113', 'AURR980114', 'AURR980115', 'AURR980116', 'AURR980117', 'AURR980118',
                        'AURR980119', 'AURR980120', 'ONEK900101', 'ONEK900102', 'VINM940101', 'VINM940102',
                        'VINM940103', 'VINM940104', 'MUNV940101', 'MUNV940102', 'MUNV940103', 'MUNV940104',
                        'MUNV940105', 'WIMW960101', 'KIMC930101', 'MONM990101', 'BLAM930101', 'PARS000101',
                        'PARS000102', 'KUMS000101', 'KUMS000102', 'KUMS000103', 'KUMS000104', 'TAKK010101',
                        'FODM020101', 'NADH010101', 'NADH010102', 'NADH010103', 'NADH010104', 'NADH010105',
                        'NADH010106', 'NADH010107', 'MONM990201', 'KOEP990101', 'KOEP990102', 'CEDJ970101',
                        'CEDJ970102', 'CEDJ970103', 'CEDJ970104', 'CEDJ970105', 'FUKS010101', 'FUKS010102',
                        'FUKS010103', 'FUKS010104', 'FUKS010105', 'FUKS010106', 'FUKS010107', 'FUKS010108',
                        'FUKS010109', 'FUKS010110', 'FUKS010111', 'FUKS010112', 'MITS020101', 'TSAJ990101',
                        'TSAJ990102', 'COSI940101', 'PONP930101', 'WILM950101', 'WILM950102', 'WILM950103',
                        'WILM950104', 'KUHL950101', 'GUOD860101', 'JURD980101', 'BASU050101', 'BASU050102',
                        'BASU050103', 'SUYM030101', 'PUNT030101', 'PUNT030102', 'GEOR030101', 'GEOR030102',
                        'GEOR030103', 'GEOR030104', 'GEOR030105', 'GEOR030106', 'GEOR030107', 'GEOR030108',
                        'GEOR030109', 'ZHOH040101', 'ZHOH040102', 'ZHOH040103', 'BAEK050101', 'HARY940101',
                        'PONJ960101', 'DIGM050101', 'WOLR790101', 'OLSK800101', 'KIDA850101', 'GUYH850102',
                        'GUYH850104', 'GUYH850105', 'JACR890101', 'COWR900101', 'BLAS910101', 'CASG920101',
                        'CORJ870101', 'CORJ870102', 'CORJ870103', 'CORJ870104', 'CORJ870105', 'CORJ870106',
                        'CORJ870107', 'CORJ870108', 'MIYS990101', 'MIYS990102', 'MIYS990103', 'MIYS990104',
                        'MIYS990105', 'ENGD860101', 'FASG890101', '', '', '', '', '']
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowTitle('Autocorrelation')
        self.resize(850, 500)
        layout = QGridLayout(self)
        lagLabel = QLabel('Lag value:')
        self.lagLineEdit = MyLineEdit("2")
        self.lagLineEdit.clicked.connect(self.setLagValue)
        label = QLabel('AAIndex: (Press [Ctrl] to select more)')
        self.tablewidget = QTableWidget(67, 8)
        self.tablewidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tablewidget.verticalHeader().setVisible(False)
        self.tablewidget.horizontalHeader().setVisible(False)
        self.tablewidget.setFont(QFont('Arial', 8))
        rect = QTableWidgetSelectionRange(0, 0, 0, 7)
        self.tablewidget.setRangeSelected(rect, True)
        data = np.array(self.AAindex).reshape((67, 8))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                cell = QTableWidgetItem(data[i][j])
                self.tablewidget.setItem(i, j, cell)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(lagLabel, 0, 0)
        layout.addWidget(self.lagLineEdit, 0, 1)
        layout.addWidget(label, 1, 0, 1, 2)
        layout.addWidget(self.tablewidget, 2, 0, 1, 2)
        layout.addWidget(self.buttons, 3, 0, 1, 2)

    def setLagValue(self):
        num, ok = QInputDialog.getInt(self, 'Lag value', 'Get lag value', 3, 1, self.nlag_range, 1)
        if ok:
            self.lagLineEdit.setText(str(num))

    def getLagValue(self):
        return int(self.lagLineEdit.text()) if self.lagLineEdit.text() != '' else 2

    def getProperty(self):
        content = self.tablewidget.selectedItems()
        property = ''
        for item in content:
            if item.text() != '':
                property += ';' + item.text()
        if property == '':
            property = ';ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101'
        return property[1:]

    @staticmethod
    def getValues(nlag_range):
        dialog = QAAindex2Input(nlag_range)
        result = dialog.exec_()
        lag = dialog.getLagValue()
        property = dialog.getProperty()
        return lag, property, result == QDialog.Accepted


class QMismatchInput(QDialog):
    def __init__(self):
        super(QMismatchInput, self).__init__()
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowTitle('Mismatch setting')
        layout = QGridLayout(self)
        kmerLabel = QLabel('Kmer size:')
        self.kmerLineEdit = MyLineEdit("2")
        self.kmerLineEdit.clicked.connect(self.setKmerValue)
        mismatchLabel = QLabel('Mismatch value (should < Kmer size):')
        self.mismatchLineEdit = MyLineEdit("1")
        self.mismatchLineEdit.clicked.connect(self.setMismatchValue)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(kmerLabel, 0, 0)
        layout.addWidget(self.kmerLineEdit, 0, 1)
        layout.addWidget(mismatchLabel, 1, 0)
        layout.addWidget(self.mismatchLineEdit, 1, 1)
        layout.addWidget(self.buttons, 2, 0, 1, 2)

    def setKmerValue(self):
        num, ok = QInputDialog.getInt(self, 'Kmer size', 'Get Kmer size', 2, 2, 6, 1)
        if ok:
            self.kmerLineEdit.setText(str(num))

    def setMismatchValue(self):
        num, ok = QInputDialog.getInt(self, 'Mismatch value', 'Get mismatch value', 1, 1, 5, 1)
        try:
            max_value = int(self.kmerLineEdit.text())
            if num >= max_value:
                self.mismatchLineEdit.setText('1')
            else:
                self.mismatchLineEdit.setText(str(num))
        except Exception as e:
            QMessageBox.critical(self, 'Error', 'Please set the kmer size at first!', QMessageBox.Ok | QMessageBox.No,
                                 QMessageBox.Ok)

    def getKmerValue(self):
        return int(self.kmerLineEdit.text()) if self.kmerLineEdit.text() != '' else 2

    def getMismatchValue(self):
        return int(self.mismatchLineEdit.text()) if self.mismatchLineEdit.text() != '' else 1

    @staticmethod
    def getValues():
        dialog = QMismatchInput()
        result = dialog.exec_()
        kmer = dialog.getKmerValue()
        mismatch = dialog.getMismatchValue()
        return kmer, mismatch, result == QDialog.Accepted


class QSubsequenceInput(QDialog):
    def __init__(self):
        super(QSubsequenceInput, self).__init__()
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowTitle('Subsequence setting')
        layout = QGridLayout(self)
        kmerLabel = QLabel('Kmer size:')
        self.kmerLineEdit = MyLineEdit("2")
        self.kmerLineEdit.clicked.connect(self.setKmerValue)
        deltaLabel = QLabel('Delta value [0, 1]:')
        self.deltaLineEdit = MyLineEdit("0")
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(kmerLabel, 0, 0)
        layout.addWidget(self.kmerLineEdit, 0, 1)
        layout.addWidget(deltaLabel, 1, 0)
        layout.addWidget(self.deltaLineEdit, 1, 1)
        layout.addWidget(self.buttons, 2, 0, 1, 2)

    def setKmerValue(self):
        num, ok = QInputDialog.getInt(self, 'Kmer size', 'Get Kmer size', 2, 1, 3, 1)
        if ok:
            self.kmerLineEdit.setText(str(num))

    def getKmerValue(self):
        return int(self.kmerLineEdit.text()) if self.kmerLineEdit.text() != '' else 2

    def getDeltaValue(self):
        delta = float(self.deltaLineEdit.text()) if self.deltaLineEdit.text() != '' else 0
        if delta < 0 or delta > 1:
            delta = 0
        return delta

    @staticmethod
    def getValues():
        dialog = QSubsequenceInput()
        result = dialog.exec_()
        kmer = dialog.getKmerValue()
        delta = dialog.getDeltaValue()
        return kmer, delta, result == QDialog.Accepted


class QDistancePairInput(QDialog):
    def __init__(self):
        super(QDistancePairInput, self).__init__()
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowTitle('DistancePair setting')
        layout = QGridLayout(self)
        distanceLabel = QLabel('Maximum distance:')
        self.distanceLineEdit = MyLineEdit("0")
        self.distanceLineEdit.clicked.connect(self.setDistanceValue)
        cpLabel = QLabel('Reduced alphabet scheme:')
        self.cpComboBox = QComboBox()
        self.cpComboBox.addItems(['cp(20)', 'cp(19)', 'cp(14)', 'cp(13)'])
        self.cpComboBox.setCurrentIndex(0)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(distanceLabel, 0, 0)
        layout.addWidget(self.distanceLineEdit, 0, 1)
        layout.addWidget(cpLabel, 1, 0)
        layout.addWidget(self.cpComboBox, 1, 1)
        layout.addWidget(self.buttons, 2, 0, 1, 2)

    def setDistanceValue(self):
        num, ok = QInputDialog.getInt(self, 'Distance', 'Set maximum distance', 0, 0, 6, 1)
        if ok:
            self.distanceLineEdit.setText(str(num))

    def getDistanceValue(self):
        return int(self.distanceLineEdit.text()) if self.distanceLineEdit.text() != '' else 0

    def getCpValue(self):
        return self.cpComboBox.currentText()

    @staticmethod
    def getValues():
        dialog = QDistancePairInput()
        result = dialog.exec_()
        distance = dialog.getDistanceValue()
        cp = dialog.getCpValue()
        return distance, cp, result == QDialog.Accepted


class QMCLInput(QDialog):
    def __init__(self):
        super(QMCLInput, self).__init__()
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowTitle('Markov cluster setting')
        self.setFont(QFont('Arial', 10))
        layout = QFormLayout(self)
        self.expand_factor_lineEdit = MyLineEdit("2")
        self.expand_factor_lineEdit.clicked.connect(self.setExpand)
        self.inflate_factor_lineEdit = MyLineEdit("2.0")
        self.inflate_factor_lineEdit.clicked.connect(self.setInflate)
        self.multiply_factor_lineEdit = MyLineEdit('2.0')
        self.multiply_factor_lineEdit.clicked.connect(self.setMultiply)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow("Expand factor:", self.expand_factor_lineEdit)
        layout.addRow("Inflate factor (float):", self.inflate_factor_lineEdit)
        layout.addRow("Multiply factor (float)", self.multiply_factor_lineEdit)
        layout.addWidget(self.buttons)

    def setExpand(self):
        num, ok = QInputDialog.getInt(self, 'Expand factor', 'Get expand factor', 2, 1, 10, 1)
        if ok:
            self.expand_factor_lineEdit.setText(str(num))

    def getExpand(self):
        return int(self.expand_factor_lineEdit.text()) if self.expand_factor_lineEdit.text() != '' else 2

    def setInflate(self):
        num, ok = QInputDialog.getDouble(self, 'Inflate factor', 'Get inflate factor', 2.0, 1.0, 6.0)
        if ok:
            self.inflate_factor_lineEdit.setText(str(num))

    def getInflate(self):
        return float(self.inflate_factor_lineEdit.text()) if self.inflate_factor_lineEdit.text() != '' else 2.0

    def setMultiply(self):
        num, ok = QInputDialog.getDouble(self, 'Inflate factor', 'Get inflate factor', 2.0, 1.0, 6.0)
        if ok:
            self.multiply_factor_lineEdit.setText(str(num))

    def getMultiply(self):
        return float(self.multiply_factor_lineEdit.text()) if self.multiply_factor_lineEdit.text() != '' else 2.0

    @staticmethod
    def getValues():
        dialog = QMCLInput()
        result = dialog.exec_()
        expand = dialog.getExpand()
        inflate = dialog.getInflate()
        multiply = dialog.getMultiply()
        return expand, inflate, multiply, result == QDialog.Accepted


class QDataSelection(QDialog):
    def __init__(self, descriptor=None, selection=None, machinelearning=None, reduction=None):
        super(QDataSelection, self).__init__()
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.descriptor = descriptor
        self.reduction = reduction
        self.selection = selection
        self.machinelearning = machinelearning
        self.data_source = None
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowTitle('Data selection')
        self.setFont(QFont('Arial', 8))
        self.resize(430, 260)

        layout = QVBoxLayout(self)
        self.dataTreeWidget = QTreeWidget()
        self.dataTreeWidget.setFont(QFont('Arial', 8))
        self.dataTreeWidget.setColumnCount(2)
        self.dataTreeWidget.setColumnWidth(0, 300)
        self.dataTreeWidget.setColumnWidth(1, 100)
        self.dataTreeWidget.clicked.connect(self.treeClicked)
        self.dataTreeWidget.setHeaderLabels(['Source', 'Shape'])
        if not self.descriptor is None and 'encoding_array' in dir(self.descriptor) and len(self.descriptor.encoding_array) > 0:
            descriptor = QTreeWidgetItem(self.dataTreeWidget)
            descriptor.setText(0, 'Descriptor data')
            descriptor.setText(1, '(%s, %s)' % (self.descriptor.row, self.descriptor.column - 2))        
        if not self.reduction is None and not self.reduction.dimension_reduction_result is None:
            reduction = QTreeWidgetItem(self.dataTreeWidget)
            reduction.setText(0, 'Dimensionality reduction data')
            shape = self.reduction.dimension_reduction_result.shape
            reduction.setText(1, '(%s, %s)' % (shape[0], shape[1]))
        if not self.selection is None and not self.selection.feature_selection_data is None:
            selection = QTreeWidgetItem(self.dataTreeWidget)
            selection.setText(0, 'Feature selection data')
            shape = self.selection.feature_selection_data.values.shape
            selection.setText(1, '(%s, %s)' % (shape[0], shape[1] - 1))
        if not self.selection is None and not self.selection.feature_normalization_data is None:
            normalization = QTreeWidgetItem(self.dataTreeWidget)
            normalization.setText(0, 'Feature normalization data')
            shape = self.selection.feature_normalization_data.values.shape
            normalization.setText(1, '(%s, %s)' % (shape[0], shape[1] - 1))
        if not self.machinelearning is None and not self.machinelearning.training_dataframe is None:
            ml_training_data = QTreeWidgetItem(self.dataTreeWidget)
            ml_training_data.setText(0, 'Machine learning training data')
            shape = self.machinelearning.training_dataframe.values.shape
            ml_training_data.setText(1, str(shape))
        if not self.machinelearning is None and not self.machinelearning.testing_dataframe is None:
            ml_testing_data = QTreeWidgetItem(self.dataTreeWidget)
            ml_testing_data.setText(0, 'Machine learning testing data')
            shape = self.machinelearning.testing_dataframe.values.shape
            ml_testing_data.setText(1, str(shape))
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.dataTreeWidget)
        layout.addWidget(self.buttons)

    def treeClicked(self, index):
        item = self.dataTreeWidget.currentItem()
        if item and item.text(0) in ['Descriptor data', 'Feature selection data', 'Feature normalization data',
                                     'Machine learning training data', 'Machine learning testing data',
                                     'Dimensionality reduction data']:
            self.data_source = item.text(0)

    def getDataSource(self):
        return self.data_source

    @staticmethod
    def getValues(descriptor=None, selection=None, machinelearning=None, reduction=None):
        dialog = QDataSelection(descriptor, selection, machinelearning, reduction)
        result = dialog.exec_()
        data_source = dialog.getDataSource()
        return data_source, result == QDialog.Accepted


class QRandomForestInput(QDialog):
    def __init__(self):
        super(QRandomForestInput, self).__init__()
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.setWindowTitle('Random Forest')
        self.setFont(QFont('Arial', 8))
        self.resize(430, 260)
        layout = QFormLayout(self)
        self.tree_number = MyLineEdit('100')
        self.tree_number.clicked.connect(self.setTreeNumber)
        self.cpu_number = MyLineEdit('1')
        self.cpu_number.clicked.connect(self.setCpuNumber)
        self.auto = QCheckBox('Auto optimization')
        self.auto.stateChanged.connect(self.checkBoxStatus)
        self.start_tree_num = MyLineEdit('50')
        self.start_tree_num.setDisabled(True)
        self.end_tree_num = MyLineEdit('500')
        self.end_tree_num.setDisabled(True)
        self.step = MyLineEdit('50')
        self.step.setDisabled(True)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow('Tree number', self.tree_number)
        layout.addRow('Number of threads', self.cpu_number)
        layout.addRow(self.auto)
        layout.addRow('Trees range from', self.start_tree_num)
        layout.addRow('Trees range to', self.end_tree_num)
        layout.addRow('Tree step', self.step)
        layout.addRow(self.buttons)

    def setTreeNumber(self):
        number, ok = QInputDialog.getInt(self, 'Tree number', 'Set tree number', 100, 50, 3000, 50)
        if ok:
            self.tree_number.setText(str(number))

    def getTreeNumber(self):
        try:
            number = int(self.tree_number.text())
            return number
        except Exception as e:
            return 100

    def setCpuNumber(self):
        number, ok = QInputDialog.getInt(self, 'Cpu number', 'Set Cpu number', 1, 1, cpu_count(), 1)
        if ok:
            self.cpu_number.setText(str(number))

    def getCpuNumber(self):
        try:
            number = int(self.cpu_number.text())
            return number
        except Exception as e:
            return 1

    def checkBoxStatus(self):
        if self.auto.isChecked():
            self.tree_number.setDisabled(True)
            self.start_tree_num.setDisabled(False)
            self.end_tree_num.setDisabled(False)
            self.step.setDisabled(False)
        else:
            self.tree_number.setDisabled(False)
            self.start_tree_num.setDisabled(True)
            self.end_tree_num.setDisabled(True)
            self.step.setDisabled(True)

    def getTreeRange(self):
        try:
            start_number = int(self.start_tree_num.text())
            end_number = int(self.end_tree_num.text())
            step = int(self.step.text())
            return (start_number, end_number, step)
        except Exception as e:
            return (100, 1000, 100)

    def getState(self):
        return self.auto.isChecked()

    @staticmethod
    def getValues():
        dialog = QRandomForestInput()
        result = dialog.exec_()
        tree_number = dialog.getTreeNumber()
        tree_range = dialog.getTreeRange()
        cpu_number = dialog.getCpuNumber()
        state = dialog.getState()
        return tree_number, tree_range, cpu_number, state, result == QDialog.Accepted


class QSupportVectorMachineInput(QDialog):
    def __init__(self):
        super(QSupportVectorMachineInput, self).__init__()
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.setWindowTitle('Support Vector Machine')
        self.setFont(QFont('Arial', 8))
        self.resize(430, 260)
        layout = QFormLayout(self)
        self.kernelComoBox = QComboBox()
        self.kernelComoBox.addItems(['linear', 'rbf', 'poly', 'sigmoid'])
        self.kernelComoBox.setCurrentIndex(1)
        self.kernelComoBox.currentIndexChanged.connect(self.selectionChange)
        self.penaltyLineEdit = MyLineEdit('1.0')
        self.GLineEdit = MyLineEdit('auto')
        self.auto = QCheckBox('Auto optimization')
        self.auto.stateChanged.connect(self.checkBoxStatus)
        self.penaltyFromLineEdit = MyLineEdit('1.0')
        self.penaltyFromLineEdit.setDisabled(True)
        self.penaltyToLineEdit = MyLineEdit('15.0')
        self.penaltyToLineEdit.setDisabled(True)
        self.GFromLineEdit = MyLineEdit('-10.0')
        self.GFromLineEdit.setDisabled(True)
        self.GToLineEdit = MyLineEdit('5.0')
        self.GToLineEdit.setDisabled(True)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow('Kernel function:', self.kernelComoBox)
        layout.addRow('Penalty:', self.penaltyLineEdit)
        layout.addRow('Gamma:', self.GLineEdit)
        layout.addRow(self.auto)
        layout.addRow('Penalty from:', self.penaltyFromLineEdit)
        layout.addRow('Penalty to:', self.penaltyToLineEdit)
        layout.addRow('Gamma from (2^x):', self.GFromLineEdit)
        layout.addRow('Gamma to (2^x):', self.GToLineEdit)
        layout.addWidget(self.buttons)

    def selectionChange(self, i):
        svm_kernel = self.kernelComoBox.itemText(i)
        if svm_kernel == 'linear' and not self.auto.isChecked():
            self.GLineEdit.setDisabled(True)
        else:
            self.GLineEdit.setDisabled(False)

    def checkBoxStatus(self):
        if self.auto.isChecked():
            self.penaltyLineEdit.setDisabled(True)
            self.GLineEdit.setDisabled(True)
            self.GFromLineEdit.setDisabled(False)
            self.GToLineEdit.setDisabled(False)
            self.penaltyFromLineEdit.setDisabled(False)
            self.penaltyToLineEdit.setDisabled(False)
        else:
            self.penaltyLineEdit.setDisabled(False)
            self.GLineEdit.setDisabled(False)
            self.GFromLineEdit.setDisabled(True)
            self.GToLineEdit.setDisabled(True)
            self.penaltyFromLineEdit.setDisabled(True)
            self.penaltyToLineEdit.setDisabled(True)

    def getKernel(self):
        return self.kernelComoBox.currentText()

    def getPenality(self):
        return float(self.penaltyLineEdit.text()) if self.penaltyLineEdit.text() != '' else 1.0

    def getGamma(self):
        if self.GLineEdit.text() != '' and self.GLineEdit.text() != 'auto':            
            return float(self.GLineEdit.text())
        else:
            return 'auto'

    def getAutoStatus(self):
        return self.auto.isChecked()

    def getPenalityRange(self):
        fromValue = float(self.penaltyFromLineEdit.text()) if self.penaltyFromLineEdit.text() != '' else 1.0
        toValue = float(self.penaltyToLineEdit.text()) if self.penaltyToLineEdit.text() != '' else 15.0
        return (fromValue, toValue)

    def getGammaRange(self):
        fromValue = float(self.GFromLineEdit.text()) if self.GFromLineEdit.text() != '' else 1.0
        toValue = float(self.GToLineEdit.text()) if self.GToLineEdit.text() != '' else 15.0
        return (fromValue, toValue)

    @staticmethod
    def getValues():
        try:
            dialog = QSupportVectorMachineInput()
            result = dialog.exec_()
            kernel = dialog.getKernel()
            penality = dialog.getPenality()
            gamma = dialog.getGamma()
            auto = dialog.getAutoStatus()
            penalityRange = dialog.getPenalityRange()
            gammaRange = dialog.getGammaRange()
            return kernel, penality, gamma, auto, penalityRange, gammaRange, result == QDialog.Accepted
        except Exception as e:
            QMessageBox.critical(dialog, 'Error', 'Invalided parameter(s), use the default parameter!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
            return 'rbf', 1.0, 'auto', False, (1.0, 15.0), (-10.0, 5.0), result == QDialog.Accepted


class QMultiLayerPerceptronInput(QDialog):
    def __init__(self):
        super(QMultiLayerPerceptronInput, self).__init__()
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.setWindowTitle('Multi-layer Perceptron')
        self.setFont(QFont('Arial', 8))
        self.resize(430, 200)
        layout = QFormLayout(self)
        self.layoutLineEdit = MyLineEdit('32;32')
        self.epochsLineEdit = MyLineEdit('200')
        self.activationComboBox = QComboBox()
        self.activationComboBox.addItems(['identity', 'logistic', 'tanh', 'relu'])
        self.activationComboBox.setCurrentIndex(3)
        self.optimizerComboBox = QComboBox()
        self.optimizerComboBox.addItems(['lbfgs', 'sgd', 'adam'])
        self.optimizerComboBox.setCurrentIndex(2)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow('Hidden layer size:', self.layoutLineEdit)
        layout.addRow('Epochs:', self.epochsLineEdit)
        layout.addRow('Activation:', self.activationComboBox)
        layout.addRow('Optimizer:', self.optimizerComboBox)
        layout.addRow(self.buttons)

    def getLayer(self):
        return self.layoutLineEdit.text() if self.layoutLineEdit.text() != '' else '32;32'

    def getActivation(self):
        return self.activationComboBox.currentText()

    def getOptimizer(self):
        return self.optimizerComboBox.currentText()

    def getEpochs(self):
        return int(self.epochsLineEdit.text()) if self.epochsLineEdit.text() != '' else 200

    @staticmethod
    def getValues():
        dialog = QMultiLayerPerceptronInput()
        result = dialog.exec_()
        layer = dialog.getLayer()

        epochs = dialog.getEpochs()
        optimizer = dialog.getOptimizer()
        activation = dialog.getActivation()
        return layer, epochs, activation, optimizer, result == QDialog.Accepted


class QKNeighborsInput(QDialog):
    def __init__(self):
        super(QKNeighborsInput, self).__init__()
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.setWindowTitle('KNN')
        self.setFont(QFont('Arial', 8))
        self.resize(430, 50)
        layout = QFormLayout(self)
        self.kValueLineEdit = MyLineEdit('3')
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow('Top K values:', self.kValueLineEdit)
        layout.addRow(self.buttons)

    def getTopKNNValue(self):
        return int(self.kValueLineEdit.text()) if self.kValueLineEdit.text() != '' else 3

    @staticmethod
    def getValues():
        dialog = QKNeighborsInput()
        result = dialog.exec_()
        topKValue = dialog.getTopKNNValue()
        return topKValue, result == QDialog.Accepted


class QLightGBMInput(QDialog):
    def __init__(self):
        super(QLightGBMInput, self).__init__()
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.setWindowTitle('LightGBM')
        self.setFont(QFont('Arial', 8))
        self.resize(500, 260)
        layout = QFormLayout(self)
        self.boostingTypeComboBox = QComboBox()
        self.boostingTypeComboBox.addItems(['gbdt', 'dart', 'goss', 'rf'])
        self.boostingTypeComboBox.setCurrentIndex(0)
        self.numLeavesLineEdit = MyLineEdit('31')
        self.maxDepthLineEdit = MyLineEdit('-1')
        self.learningRateLineEdit = MyLineEdit('0.1')
        self.cpu_number = MyLineEdit('1')
        self.cpu_number.clicked.connect(self.setCpuNumber)
        self.auto = QCheckBox('Auto optimization')
        self.auto.stateChanged.connect(self.checkBoxStatus)
        self.leavesRangeLineEdit = MyLineEdit('20:100:10')
        self.leavesRangeLineEdit.setDisabled(True)
        self.depthRangeLineEdit = MyLineEdit('15:55:10')
        self.depthRangeLineEdit.setDisabled(True)
        self.rateRangeLineEdit = MyLineEdit('0.01:0.15:0.02')
        self.rateRangeLineEdit.setDisabled(True)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow('Boosting type:', self.boostingTypeComboBox)
        layout.addRow('Number of leaves:', self.numLeavesLineEdit)
        layout.addRow('Max depth:', self.maxDepthLineEdit)
        layout.addRow('Learning rate:', self.learningRateLineEdit)
        layout.addRow('Number of threads:', self.cpu_number)
        layout.addRow(self.auto)
        layout.addRow('Leaves range (from:to:step)', self.leavesRangeLineEdit)
        layout.addRow('Depth range', self.depthRangeLineEdit)
        layout.addRow('Learning rate range:', self.rateRangeLineEdit)
        layout.addRow(self.buttons)

    def setCpuNumber(self):
        number, ok = QInputDialog.getInt(self, 'Cpu number', 'Set Cpu number', 1, 1, cpu_count(), 1)
        if ok:
            self.cpu_number.setText(str(number))

    def getCpuNumber(self):
        try:
            number = int(self.cpu_number.text())
            return number
        except Exception as e:
            return 1

    def checkBoxStatus(self):
        if self.auto.isChecked():
            self.numLeavesLineEdit.setDisabled(True)
            self.leavesRangeLineEdit.setDisabled(False)
            self.maxDepthLineEdit.setDisabled(True)
            self.depthRangeLineEdit.setDisabled(False)
            self.learningRateLineEdit.setDisabled(True)
            self.rateRangeLineEdit.setDisabled(False)
        else:
            self.numLeavesLineEdit.setDisabled(False)
            self.leavesRangeLineEdit.setDisabled(True)
            self.maxDepthLineEdit.setDisabled(False)
            self.depthRangeLineEdit.setDisabled(True)
            self.learningRateLineEdit.setDisabled(False)
            self.rateRangeLineEdit.setDisabled(True)

    def getState(self):
        return self.auto.isChecked()

    def getBoostingType(self):
        return self.boostingTypeComboBox.currentText()

    def getLeaves(self):
        return int(self.numLeavesLineEdit.text()) if self.numLeavesLineEdit.text() != '' else 31

    def getMaxDepth(self):
        return int(self.maxDepthLineEdit.text()) if self.maxDepthLineEdit.text() != '' else -1

    def getLearningRate(self):
        return float(self.learningRateLineEdit.text()) if self.learningRateLineEdit.text() != '' else 0.1

    def getLeavesRange(self):
        return tuple([int(i) for i in
                      self.leavesRangeLineEdit.text().split(':')]) if self.leavesRangeLineEdit.text() != '' else (
            20, 100, 10)

    def getDepthRange(self):
        return tuple(
            [int(i) for i in self.depthRangeLineEdit.text().split(':')]) if self.depthRangeLineEdit.text() != '' else (
            15, 55, 10)

    def getRateRange(self):
        return tuple(
            [float(i) for i in self.rateRangeLineEdit.text().split(':')]) if self.rateRangeLineEdit.text() != '' else (
            0.01, 0.15, 0.02)

    @staticmethod
    def getValues():
        dialog = QLightGBMInput()
        result = dialog.exec_()
        threads = dialog.getCpuNumber()
        state = dialog.getState()
        type = dialog.getBoostingType()
        leaves = dialog.getLeaves()
        depth = dialog.getMaxDepth()
        rate = dialog.getLearningRate()
        leavesRange = dialog.getLeavesRange()
        depthRange = dialog.getDepthRange()
        rateRange = dialog.getRateRange()
        return type, leaves, depth, rate, leavesRange, depthRange, rateRange, threads, state, result == QDialog.Accepted


class QXGBoostInput(QDialog):
    def __init__(self):
        super(QXGBoostInput, self).__init__()
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.setWindowTitle('XGBoost')
        self.setFont(QFont('Arial', 8))
        self.resize(500, 260)
        layout = QFormLayout(self)
        self.boosterComboBox = QComboBox()
        self.boosterComboBox.addItems(['gbtree', 'gblinear'])
        self.boosterComboBox.setCurrentIndex(0)
        self.cpu_number = MyLineEdit('1')
        self.cpu_number.clicked.connect(self.setCpuNumber)
        self.maxDepthLineEdit = MyLineEdit('6')
        self.n_estimatorLineEdit = QLineEdit('100')
        self.learningRateLineEdit = MyLineEdit('0.3')
        self.colsample_bytreeLineEdit = MyLineEdit('0.8')
        self.auto = QCheckBox('Auto optimization')
        self.auto.stateChanged.connect(self.checkBoxStatus)
        self.depthRangeLineEdit = MyLineEdit('3:10:1')
        self.depthRangeLineEdit.setDisabled(True)
        self.rateRangeLineEdit = MyLineEdit('0.01:0.3:0.05')
        self.rateRangeLineEdit.setDisabled(True)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow('Booster:', self.boosterComboBox)
        layout.addRow('Number of threads:', self.cpu_number)
        layout.addRow('Max depth (3~10):', self.maxDepthLineEdit)
        layout.addRow('Learnning rate:', self.learningRateLineEdit)
        # layout.addRow('n_estimator:', self.n_estimatorLineEdit)
        layout.addRow('colsample_bytree', self.colsample_bytreeLineEdit)
        layout.addRow(self.auto)
        layout.addRow('Depth range', self.depthRangeLineEdit)
        layout.addRow('Learning rate range:', self.rateRangeLineEdit)
        layout.addRow(self.buttons)

    def getBooster(self):
        return self.boosterComboBox.currentText()

    def setCpuNumber(self):
        number, ok = QInputDialog.getInt(self, 'Cpu number', 'Set Cpu number', 1, 1, cpu_count(), 1)
        if ok:
            self.cpu_number.setText(str(number))

    def getCpuNumber(self):
        try:
            number = int(self.cpu_number.text())
            return number
        except Exception as e:
            return 1

    def checkBoxStatus(self):
        if self.auto.isChecked():
            self.rateRangeLineEdit.setDisabled(False)
            self.depthRangeLineEdit.setDisabled(False)
            self.maxDepthLineEdit.setDisabled(True)
            self.learningRateLineEdit.setDisabled(True)
        else:
            self.rateRangeLineEdit.setDisabled(True)
            self.depthRangeLineEdit.setDisabled(True)
            self.maxDepthLineEdit.setDisabled(False)
            self.learningRateLineEdit.setDisabled(False)

    def getMaxDepth(self):
        return int(self.maxDepthLineEdit.text()) if self.maxDepthLineEdit.text() != '' else -1

    def getLearningRate(self):
        return float(self.learningRateLineEdit.text()) if self.learningRateLineEdit.text() != '' else 0.1

    def getNEstimator(self):
        return int(self.n_estimatorLineEdit.text()) if self.n_estimatorLineEdit.text() != '' else 100

    def getColsample(self):
        return float(self.colsample_bytreeLineEdit.text()) if self.colsample_bytreeLineEdit.text() != '' else 0.8

    def getState(self):
        return self.auto.isChecked()

    def getLearningRate(self):
        return float(self.learningRateLineEdit.text()) if self.learningRateLineEdit.text() != '' else 0.1

    def getRateRange(self):
        return tuple(
            [float(i) for i in self.rateRangeLineEdit.text().split(':')]) if self.rateRangeLineEdit.text() != '' else (
            0.01, 0.15, 0.02)

    def getDepthRange(self):
        return tuple(
            [int(i) for i in self.depthRangeLineEdit.text().split(':')]) if self.depthRangeLineEdit.text() != '' else (
            15, 55, 10)

    @staticmethod
    def getValues():
        dialog = QXGBoostInput()
        result = dialog.exec_()
        state = dialog.getState()
        threads = dialog.getCpuNumber()
        booster = dialog.getBooster()
        maxdepth = dialog.getMaxDepth()
        rate = dialog.getLearningRate()
        estimator = dialog.getNEstimator()
        colsample = dialog.getColsample()
        depthRange = dialog.getDepthRange()
        rateRange = dialog.getRateRange()
        return booster, maxdepth, rate, estimator, colsample, depthRange, rateRange, threads, state, result == QDialog.Accepted


class QBaggingInput(QDialog):
    def __init__(self):
        super(QBaggingInput, self).__init__()
        self.initUI()

    def initUI(self):
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.setWindowTitle('Bagging')
        self.setFont(QFont('Arial', 8))
        self.resize(500, 100)
        layout = QFormLayout(self)
        self.n_estimators = MyLineEdit('10')
        self.cpu_number = MyLineEdit('1')
        self.cpu_number.clicked.connect(self.setCpuNumber)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow('n_estimators', self.n_estimators)
        layout.addRow('CPU number', self.cpu_number)
        layout.addRow(self.buttons)

    def setCpuNumber(self):
        number, ok = QInputDialog.getInt(self, 'Cpu number', 'Set Cpu number', 1, 1, cpu_count(), 1)
        if ok:
            self.cpu_number.setText(str(number))

    def getCpuNumber(self):
        try:
            number = int(self.cpu_number.text())
            return number
        except Exception as e:
            return 1

    def getEstimator(self):
        try:
            if self.n_estimators.text() != '':
                n_estimator = int(self.n_estimators.text())
                if 0 < n_estimator <= 1000:
                    return int(self.n_estimators.text())
                else:
                    return 10
        except Exception as e:
            return 10        

    @staticmethod
    def getValues():
        dialog = QBaggingInput()
        result = dialog.exec_()
        n_estimators = dialog.getEstimator()
        threads = dialog.getCpuNumber()
        return n_estimators, threads, result == QDialog.Accepted


class QStaticsInput(QDialog):
    def __init__(self):
        super(QStaticsInput, self).__init__()
        self.initUI()

    def initUI(self):
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.setWindowTitle('iLearnPlus Statics')
        self.setFont(QFont('Arial', 8))
        self.resize(300, 100)
        layout = QFormLayout(self)
        self.method = QComboBox()
        self.method.addItems(['bootstrap'])
        self.bootstrap_num = MyLineEdit('500')
        self.bootstrap_num.clicked.connect(self.setBootstrapNum)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow('Method:', self.method)
        layout.addRow('Bootstrap number:', self.bootstrap_num)
        layout.addWidget(self.buttons)

    def setBootstrapNum(self):
        number, ok = QInputDialog.getInt(self, 'Bootstrap number', 'SBootstrap number', 500, 100, 2000, 100)
        if ok:
            self.bootstrap_num.setText(str(number))

    def getMethod(self):
        return self.method.currentText()

    def getBootstrapNum(self):
        return int(self.bootstrap_num.text())

    @staticmethod
    def getValues():
        dialog = QStaticsInput()
        result = dialog.exec_()
        method = dialog.getMethod()
        bootstrap_n = dialog.getBootstrapNum()
        return method, bootstrap_n, result == QDialog.Accepted


class QNetInput_1(QDialog):
    def __init__(self, dim):
        super(QNetInput_1, self).__init__()
        self.dim = dim
        self.initUI()

    def initUI(self):
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.setWindowTitle('iLearnPlus Deeplearning')
        self.setFont(QFont('Arial', 8))
        self.resize(600, 200)
        layout = QFormLayout(self)
        self.in_channels = MyLineEdit('1')
        self.in_channels.clicked.connect(self.set_in_channel)
        self.in_channels.textChanged.connect(self.check_input_channel)
        self.in_length = MyLineEdit(str(self.dim))
        self.in_length.clicked.connect(self.set_in_length)
        self.in_length.textChanged.connect(self.check_input_length)
        label = QLabel('Note: Input channels X Input length = Feature number (i.e. %s)' % self.dim)
        label.setAlignment(Qt.AlignLeft)
        label.setFont(QFont('Arial', 7))
        self.out_channels = MyLineEdit('64')
        self.padding = MyLineEdit('2')
        self.kernel_size = MyLineEdit('5')
        self.drop_out = MyLineEdit('0.5')
        self.learning_rate = MyLineEdit('0.001')
        self.epochs = MyLineEdit('1000')
        self.early_stopping = MyLineEdit('100')
        self.batch_size = MyLineEdit('64')
        self.fc_size = MyLineEdit('64')
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow('Input channels (integer):', self.in_channels)
        layout.addRow('Input length (integer):', self.in_length)
        layout.addWidget(label)
        layout.addRow('Output channels (integer):', self.out_channels)
        layout.addRow('Padding (integer):', self.padding)
        layout.addRow('Kernel size (integer):', self.kernel_size)
        layout.addRow('FC layer size (integer):', self.fc_size)
        layout.addRow('Dropout rate (0~1):', self.drop_out)
        layout.addRow('Learning rate (0~1):', self.learning_rate)
        layout.addRow('Epochs (integer):', self.epochs)
        layout.addRow('Early stopping (integer):', self.early_stopping)
        layout.addRow('Batch size (integer):', self.batch_size)
        layout.addWidget(self.buttons)

    def set_in_channel(self):
        number, ok = QInputDialog.getInt(self, 'Input channels', 'Input channels', 1, 1, 100, 1)
        if ok:
            self.in_channels.setText(str(number))

    def set_in_length(self):
        number, ok = QInputDialog.getInt(self, 'Input length', 'Input length', self.dim, 1, self.dim, 1)
        if ok:
            self.in_length.setText(str(number))

    def check_input_channel(self, text):
        try:
            if self.dim % int(text) == 0:
                self.in_length.setText(str(int(self.dim / int(text))))
            else:
                self.in_channels.setText('1')
                self.in_length.setText(str(int(self.dim)))
        except Exception as e:
            self.in_channels.setText('1')
            QMessageBox.critical(self, 'Error', 'Invalid parameter value.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

    def check_input_length(self, text):
        try:
            if self.dim % int(text) == 0:
                self.in_channels.setText(str(int(self.dim / int(text))))
            else:
                self.in_channels.setText('1')
                self.in_length.setText(str(int(self.dim)))
        except Exception as e:
            self.in_length.setText(str(int(self.dim)))
            QMessageBox.critical(self, 'Error', 'Invalid parameter value.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

    def getChannels(self):
        try:
            if self.in_channels != '':
                return int(self.in_channels.text())
            else:
                return 1
        except Exception as e:
            return 1

    def getLength(self):
        try:
            if self.in_length != '':
                return int(self.in_length.text())
            else:
                return self.dim
        except Exception as e:
            return self.dim

    def getOutputChannel(self):
        try:
            if self.out_channels.text() != '':
                channel = int(self.out_channels.text())
                if 0 < channel <= 1024:
                    return channel
                else:
                    return 64               
            else:
                return 64
        except Exception as e:
            return 64

    def getPadding(self):
        try:
            if self.padding.text() != '':
                if 0 < int(self.padding.text()) <= 64:
                    return int(self.padding.text())
                else:
                    return 2
            else:
                return 2
        except Exception as e:
            return 2

    def getKernelSize(self):
        try:
            if self.kernel_size.text() != '':
                if 0 < int(self.kernel_size.text()):
                    return int(self.kernel_size.text())
                else:
                    return 5
            else:
                return 5
        except Exception as e:
            return 5

    def getLearningRate(self):
        try:
            if self.learning_rate.text() != '':
                lr = float(self.learning_rate.text())
                if 0 < lr < 1:
                    return lr
                else:
                    return 0.001
            else:
                return 0.001
        except Exception as e:
            return 0.001

    def getDroprate(self):
        try:
            if self.drop_out.text() != '':
                dropout = float(self.drop_out.text())
                if 0 < dropout < 1:
                    return dropout
                else:
                    return 0.5
            else:
                return 0.5
        except Exception as e:
            return 0.5

    def getEpochs(self):
        try:
            if self.epochs.text() != '':
                if int(self.epochs.text()) > 0:
                    return int(self.epochs.text())
                else:
                    return 1000
            else:
                return 1000
        except Exception as e:
            return 1000

    def getEarlyStopping(self):
        try:
            if self.early_stopping.text() != '':
                if int(self.early_stopping.text()) > 0:
                    return int(self.early_stopping.text())
                else:
                    return 100
            else:
                return 100
        except Exception as e:
            return 100

    def getBatchSize(self):
        try:
            if self.batch_size.text() != '':
                if int(self.batch_size.text()) > 0:
                    return int(self.batch_size.text())
                else:
                    return 64
            else:
                return 64
        except Exception as e:
            return 64

    def getFCSize(self):
        try:
            if self.fc_size.text() != '':
                if int(self.fc_size.text()) > 0:
                    return int(self.fc_size.text())
                else:
                    return 64
            else:
                return 64
        except Exception as e:
            return 64

    @staticmethod
    def getValues(dim):
        try:
            dialog = QNetInput_1(dim)
            result = dialog.exec_()
            input_channel = dialog.getChannels()
            input_length = dialog.getLength()
            output_channel = dialog.getOutputChannel()
            padding = dialog.getPadding()
            kernel_size = dialog.getKernelSize()
            dropout = dialog.getDroprate()
            learning_rate = dialog.getLearningRate()
            epochs = dialog.getEpochs()
            early_stopping = dialog.getEarlyStopping()
            batch_size = dialog.getBatchSize()
            fc_size = dialog.getFCSize()
            return input_channel, input_length, output_channel, padding, kernel_size, dropout, learning_rate, epochs, early_stopping, batch_size, fc_size, result == QDialog.Accepted
        except Exception as e:
            return 1, dim, 64, 2, 5, 0.5, 0.001, 1000, 100, 64, 64, False


class QNetInput_2(QDialog):
    def __init__(self, dim):
        super(QNetInput_2, self).__init__()
        self.dim = dim
        self.initUI()

    def initUI(self):
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.setWindowTitle('iLearnPlus Deeplearning')
        self.setFont(QFont('Arial', 8))
        self.resize(600, 200)
        layout = QFormLayout(self)
        self.in_channels = MyLineEdit('1')
        self.in_channels.clicked.connect(self.set_in_channel)
        self.in_channels.textChanged.connect(self.check_input_channel)
        self.in_length = MyLineEdit(str(self.dim))
        self.in_length.clicked.connect(self.set_in_length)
        self.in_length.textChanged.connect(self.check_input_length)
        label = QLabel('Note: Input channels X Input length = Feature number (i.e. %s)' % self.dim)
        label.setAlignment(Qt.AlignLeft)
        label.setFont(QFont('Arial', 7))
        self.hidden_size = MyLineEdit('32')
        self.num_layers = MyLineEdit('1')
        self.fc_size = MyLineEdit('64')
        self.drop_out = MyLineEdit('0.5')
        self.learning_rate = MyLineEdit('0.001')
        self.epochs = MyLineEdit('1000')
        self.early_stopping = MyLineEdit('100')
        self.batch_size = MyLineEdit('64')
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow('Input channels (integer):', self.in_channels)
        layout.addRow('Input length (integer):', self.in_length)
        layout.addWidget(label)
        layout.addRow('Hidden size (integer):', self.hidden_size)
        layout.addRow('Number of recurrent layers (integer):', self.num_layers)
        layout.addRow('FC layer size (integer):', self.fc_size)
        layout.addRow('Dropout rate (0~1):', self.drop_out)
        layout.addRow('Learning rate (0~1):', self.learning_rate)
        layout.addRow('Epochs (integer):', self.epochs)
        layout.addRow('Early stopping (integer):', self.early_stopping)
        layout.addRow('Batch size (integer):', self.batch_size)
        layout.addWidget(self.buttons)

    def set_in_channel(self):
        number, ok = QInputDialog.getInt(self, 'Input channels', 'Input channels', 1, 1, 100, 1)
        if ok:
            self.in_channels.setText(str(number))

    def set_in_length(self):
        number, ok = QInputDialog.getInt(self, 'Input length', 'Input length', self.dim, 1, self.dim, 1)
        if ok:
            self.in_length.setText(str(number))

    def check_input_channel(self, text):
            try:
                if self.dim % int(text) == 0:
                    self.in_length.setText(str(int(self.dim / int(text))))
                else:
                    self.in_channels.setText('1')
                    self.in_length.setText(str(int(self.dim)))
            except Exception as e:
                self.in_channels.setText('1')
                QMessageBox.critical(self, 'Error', 'Invalid parameter value.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

    def check_input_length(self, text):
        try:
            if self.dim % int(text) == 0:
                self.in_channels.setText(str(int(self.dim / int(text))))
            else:
                self.in_channels.setText('1')
                self.in_length.setText(str(int(self.dim)))
        except Exception as e:
            self.in_length.setText(str(int(self.dim)))
            QMessageBox.critical(self, 'Error', 'Invalid parameter value.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

    def getChannels(self):
        try:
            if self.in_channels != '':
                if int(self.in_channels.text()) > 0:
                    return int(self.in_channels.text())
                else:
                    return 1
            else:
                return 1
        except Exception as e:
            return 1

    def getLength(self):
        try:
            if self.in_length != '':
                return int(self.in_length.text())
            else:
                return self.dim
        except Exception as e:
            return self.dim

    def getHiddenSize(self):
        try:
            if self.hidden_size != '':
                if 0 < int(self.hidden_size.text()) <= 512:
                    return int(self.hidden_size.text())
                else:
                    return 32
            else:
                return 32
        except Exception as e:
            return 32

    def getRnnLayers(self):
        try:
            if self.num_layers.text() != '':
                if 0 < int(self.num_layers.text()) <=32:
                    return int(self.num_layers.text())
                else:
                    return 1
            else:
                return 1
        except Exception as e:
            return 1

    def getLearningRate(self):
        try:
            if self.learning_rate.text() != '':
                lr = float(self.learning_rate.text())
                if 0 < lr < 1:
                    return lr
                else:
                    return 0.001
            else:
                return 0.001
        except Exception as e:
            return 0.001

    def getDroprate(self):
        try:
            if self.drop_out.text() != '':
                dropout = float(self.drop_out.text())
                if 0 < dropout < 1:
                    return dropout
                else:
                    return 0.5
            else:
                return 0.5
        except Exception as e:
            return 0.5

    def getEpochs(self):
        try:
            if self.epochs.text() != '':
                if int(self.epochs.text()) > 0:
                    return int(self.epochs.text())
                else:
                    return 1000
            else:
                return 1000
        except Exception as e:
            return 1000

    def getEarlyStopping(self):
        try:
            if self.early_stopping.text() != '':
                if int(self.early_stopping.text()) > 0:
                    return int(self.early_stopping.text())
                else:
                    return 100
            else:
                return 100
        except Exception as e:
            return 100

    def getBatchSize(self):
        try:
            if self.batch_size.text() != '':
                if int(self.batch_size.text()) > 0:
                    return int(self.batch_size.text())
                else:
                    return 64
            else:
                return 64
        except Exception as e:
            return 64

    def getFCSize(self):
        try:
            if self.fc_size.text() != '':
                if int(self.fc_size.text()) > 0:
                    return int(self.fc_size.text())
                else:
                    return 64
            else:
                return 64
        except Exception as e:
            return 64

    @staticmethod
    def getValues(dim):
        dialog = QNetInput_2(dim)
        result = dialog.exec_()
        input_channel = dialog.getChannels()
        input_length = dialog.getLength()
        hidden_size = dialog.getHiddenSize()
        num_layers = dialog.getRnnLayers()
        dropout = dialog.getDroprate()
        learning_rate = dialog.getLearningRate()
        epochs = dialog.getEpochs()
        early_stopping = dialog.getEarlyStopping()
        batch_size = dialog.getBatchSize()
        fc_size = dialog.getFCSize()
        return input_channel, input_length, hidden_size, num_layers, dropout, learning_rate, epochs, early_stopping, batch_size, fc_size, result == QDialog.Accepted


class QNetInput_4(QDialog):
    def __init__(self, dim):
        super(QNetInput_4, self).__init__()
        self.dim = dim
        self.initUI()

    def initUI(self):
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.setWindowTitle('iLearnPlus Deeplearning')
        self.setFont(QFont('Arial', 8))
        self.resize(600, 200)
        layout = QFormLayout(self)
        self.in_channels = MyLineEdit('1')
        self.in_channels.clicked.connect(self.set_in_channel)
        self.in_channels.textChanged.connect(self.check_input_channel)
        self.in_length = MyLineEdit(str(self.dim))
        self.in_length.clicked.connect(self.set_in_length)
        self.in_length.textChanged.connect(self.check_input_length)
        label = QLabel('Note: Input channels X Input length = Feature number (i.e. %s)' % self.dim)
        label.setAlignment(Qt.AlignLeft)
        label.setFont(QFont('Arial', 7))
        self.drop_out = MyLineEdit('0.5')
        self.learning_rate = MyLineEdit('0.001')
        self.epochs = MyLineEdit('1000')
        self.early_stopping = MyLineEdit('100')
        self.batch_size = MyLineEdit('64')
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow('Input channels (integer):', self.in_channels)
        layout.addRow('Input length (integer):', self.in_length)
        layout.addWidget(label)
        layout.addRow('Dropout rate (0~1):', self.drop_out)
        layout.addRow('Learning rate (0~1):', self.learning_rate)
        layout.addRow('Epochs (integer):', self.epochs)
        layout.addRow('Early stopping (integer):', self.early_stopping)
        layout.addRow('Batch size (integer):', self.batch_size)
        layout.addWidget(self.buttons)

    def set_in_channel(self):
        number, ok = QInputDialog.getInt(self, 'Input channels', 'Input channels', 1, 1, 100, 1)
        if ok:
            self.in_channels.setText(str(number))

    def set_in_length(self):
        number, ok = QInputDialog.getInt(self, 'Input length', 'Input length', self.dim, 1, self.dim, 1)
        if ok:
            self.in_length.setText(str(number))

    def check_input_channel(self, text):
        try:
            if self.dim % int(text) == 0:
                self.in_length.setText(str(int(self.dim / int(text))))
            else:
                self.in_channels.setText('1')
                self.in_length.setText(str(int(self.dim)))
        except Exception as e:
            self.in_channels.setText('1')
            QMessageBox.critical(self, 'Error', 'Invalid parameter value.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

    def check_input_length(self, text):
        try:
            if self.dim % int(text) == 0:
                self.in_channels.setText(str(int(self.dim / int(text))))
            else:
                self.in_channels.setText('1')
                self.in_length.setText(str(int(self.dim)))
        except Exception as e:
            self.in_length.setText(str(int(self.dim)))
            QMessageBox.critical(self, 'Error', 'Invalid parameter value.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

    def getChannels(self):
        try:
            if self.in_channels != '':
                return int(self.in_channels.text())
            else:
                return 1
        except Exception as e:
            return 1

    def getLength(self):
        try:
            if self.in_length != '':
                return int(self.in_length.text())
            else:
                return self.dim
        except Exception as e:
            return self.dim

    def getLearningRate(self):
        try:
            if self.learning_rate.text() != '':
                lr = float(self.learning_rate.text())
                if 0 < lr < 1:
                    return lr
                else:
                    return 0.001
            else:
                return 0.001
        except Exception as e:
            return 0.001

    def getDroprate(self):
        try:
            if self.drop_out.text() != '':
                dropout = float(self.drop_out.text())
                if 0 < dropout < 1:
                    return dropout
                else:
                    return 0.5
            else:
                return 0.5
        except Exception as e:
            return 0.5

    def getEpochs(self):
        try:
            if self.epochs.text() != '':
                if int(self.epochs.text()) > 0:
                    return int(self.epochs.text())
                else:
                    return 1000
            else:
                return 1000
        except Exception as e:
            return 1000

    def getEarlyStopping(self):
        try:
            if self.early_stopping.text() != '':
                if int(self.early_stopping.text()) > 0:
                    return int(self.early_stopping.text())
                else:
                    return 100
            else:
                return 100
        except Exception as e:
            return 100

    def getBatchSize(self):
        try:
            if self.batch_size.text() != '':
                if int(self.batch_size.text()) > 0:
                    return int(self.batch_size.text())
                else:
                    return 64
            else:
                return 64
        except Exception as e:
            return 64

    @staticmethod
    def getValues(dim):
        dialog = QNetInput_4(dim)
        result = dialog.exec_()
        input_channel = dialog.getChannels()
        input_length = dialog.getLength()
        dropout = dialog.getDroprate()
        learning_rate = dialog.getLearningRate()
        epochs = dialog.getEpochs()
        early_stopping = dialog.getEarlyStopping()
        batch_size = dialog.getBatchSize()
        return input_channel, input_length, dropout, learning_rate, epochs, early_stopping, batch_size, result == QDialog.Accepted


class QNetInput_5(QDialog):
    def __init__(self, dim):
        super(QNetInput_5, self).__init__()
        self.dim = dim
        self.initUI()

    def initUI(self):
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.setWindowTitle('iLearnPlus Deeplearning')
        self.setFont(QFont('Arial', 8))
        self.resize(600, 200)
        layout = QFormLayout(self)
        self.in_channels = MyLineEdit('1')
        self.in_channels.clicked.connect(self.set_in_channel)
        self.in_channels.textChanged.connect(self.check_input_channel)
        self.in_length = MyLineEdit(str(self.dim))
        self.in_length.clicked.connect(self.set_in_length)
        self.in_length.textChanged.connect(self.check_input_length)
        label = QLabel('Note: Input channels X Input length = Feature number (i.e. %s)' % self.dim)
        label.setAlignment(Qt.AlignLeft)
        label.setFont(QFont('Arial', 7))
        self.learning_rate = MyLineEdit('0.001')
        self.epochs = MyLineEdit('1000')
        self.early_stopping = MyLineEdit('100')
        self.batch_size = MyLineEdit('64')
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow('Input channels (integer):', self.in_channels)
        layout.addRow('Input length (integer):', self.in_length)
        layout.addWidget(label)
        layout.addRow('Learning rate (0~1):', self.learning_rate)
        layout.addRow('Epochs (integer):', self.epochs)
        layout.addRow('Early stopping (integer):', self.early_stopping)
        layout.addRow('Batch size (integer):', self.batch_size)
        layout.addWidget(self.buttons)

    def set_in_channel(self):
        number, ok = QInputDialog.getInt(self, 'Input channels', 'Input channels', 1, 1, 100, 1)
        if ok:
            self.in_channels.setText(str(number))

    def set_in_length(self):
        number, ok = QInputDialog.getInt(self, 'Input length', 'Input length', self.dim, 1, self.dim, 1)
        if ok:
            self.in_length.setText(str(number))

    def check_input_channel(self, text):
        try:
            if self.dim % int(text) == 0:
                self.in_length.setText(str(int(self.dim / int(text))))
            else:
                self.in_channels.setText('1')
                self.in_length.setText(str(int(self.dim)))
        except Exception as e:
            self.in_channels.setText('1')
            QMessageBox.critical(self, 'Error', 'Invalid parameter value.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

    def check_input_length(self, text):
        try:
            if self.dim % int(text) == 0:
                self.in_channels.setText(str(int(self.dim / int(text))))
            else:
                self.in_channels.setText('1')
                self.in_length.setText(str(int(self.dim)))
        except Exception as e:
            self.in_length.setText(str(int(self.dim)))
            QMessageBox.critical(self, 'Error', 'Invalid parameter value.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

    def getChannels(self):
        if self.in_channels != '':
            return int(self.in_channels.text())
        else:
            return 1

    def getLength(self):
        if self.in_length != '':
            return int(self.in_length.text())
        else:
            return self.dim

    def getLearningRate(self):
        try:
            if self.learning_rate.text() != '':
                lr = float(self.learning_rate.text())
                if 0 < lr < 1:
                    return lr
                else:
                    return 0.001
            else:
                return 0.001
        except Exception as e:
            return 0.001
    
    def getEpochs(self):
        try:
            if self.epochs.text() != '':
                if int(self.epochs.text()) > 0:
                    return int(self.epochs.text())
                else:
                    return 1000
            else:
                return 1000
        except Exception as e:
            return 1000

    def getEarlyStopping(self):
        try:
            if self.early_stopping.text() != '':
                if int(self.early_stopping.text()) > 0:
                    return int(self.early_stopping.text())
                else:
                    return 100
            else:
                return 100
        except Exception as e:
            return 100

    def getBatchSize(self):
        try:
            if self.batch_size.text() != '':
                if int(self.batch_size.text()) > 0:
                    return int(self.batch_size.text())
                else:
                    return 64
            else:
                return 64
        except Exception as e:
            return 64

    @staticmethod
    def getValues(dim):
        dialog = QNetInput_5(dim)
        result = dialog.exec_()
        input_channel = dialog.getChannels()
        input_length = dialog.getLength()
        learning_rate = dialog.getLearningRate()
        epochs = dialog.getEpochs()
        early_stopping = dialog.getEarlyStopping()
        batch_size = dialog.getBatchSize()
        return input_channel, input_length, learning_rate, epochs, early_stopping, batch_size, result == QDialog.Accepted


class QNetInput_6(QDialog):
    def __init__(self, dim):
        super(QNetInput_6, self).__init__()
        self.dim = dim
        self.initUI()

    def initUI(self):
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.setWindowTitle('iLearnPlus Deeplearning')
        self.setFont(QFont('Arial', 8))
        self.resize(600, 200)
        layout = QFormLayout(self)
        self.in_channels = MyLineEdit(str(self.dim))
        self.in_channels.setEnabled(False)
        self.drop_out = MyLineEdit('0.5')
        self.learning_rate = MyLineEdit('0.001')
        self.epochs = MyLineEdit('1000')
        self.early_stopping = MyLineEdit('100')
        self.batch_size = MyLineEdit('64')
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow('Input size (integer):', self.in_channels)
        layout.addRow('Dropout rate (0~1):', self.drop_out)
        layout.addRow('Learning rate (0~1):', self.learning_rate)
        layout.addRow('Epochs (integer):', self.epochs)
        layout.addRow('Early stopping (integer):', self.early_stopping)
        layout.addRow('Batch size (integer):', self.batch_size)
        layout.addWidget(self.buttons)

    def getChannels(self):
        if self.in_channels != '':
            return int(self.in_channels.text())
        else:
            return 1

    def getLearningRate(self):
        try:
            if self.learning_rate.text() != '':
                lr = float(self.learning_rate.text())
                if 0 < lr < 1:
                    return lr
                else:
                    return 0.001
            else:
                return 0.001
        except Exception as e:
            return 0.001

    def getDroprate(self):
        try:
            if self.drop_out.text() != '':
                dropout = float(self.drop_out.text())
                if 0 < dropout < 1:
                    return dropout
                else:
                    return 0.5
            else:
                return 0.5
        except Exception as e:
            return 0.5

    def getEpochs(self):
        try:
            if self.epochs.text() != '':
                if int(self.epochs.text()) > 0:
                    return int(self.epochs.text())
                else:
                    return 1000
            else:
                return 1000
        except Exception as e:
            return 1000

    def getEarlyStopping(self):
        try:
            if self.early_stopping.text() != '':
                if int(self.early_stopping.text()) > 0:
                    return int(self.early_stopping.text())
                else:
                    return 100
            else:
                return 100
        except Exception as e:
            return 100

    def getBatchSize(self):
        try:
            if self.batch_size.text() != '':
                if int(self.batch_size.text()) > 0:
                    return int(self.batch_size.text())
                else:
                    return 64
            else:
                return 64
        except Exception as e:
            return 64

    @staticmethod
    def getValues(dim):
        dialog = QNetInput_6(dim)
        result = dialog.exec_()
        input_channel = dialog.getChannels()
        dropout = dialog.getDroprate()
        learning_rate = dialog.getLearningRate()
        epochs = dialog.getEpochs()
        early_stopping = dialog.getEarlyStopping()
        batch_size = dialog.getBatchSize()
        return input_channel, dropout, learning_rate, epochs, early_stopping, batch_size, result == QDialog.Accepted


class QPlotInput(QDialog):
    def __init__(self, curve='ROC'):
        super(QPlotInput, self).__init__()
        self.initUI()
        self.curve = curve
        self.auc = None
        self.dot = None
        self.color = '#000000'
        self.lineWidth = 1
        self.lineStyle = 'solid'
        self.raw_data = None

    def initUI(self):
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setFont(QFont('Arial', 8))
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.setWindowTitle('iLearnPlus PlotCurve')
        self.resize(500, 200)
        layout = QFormLayout(self)
        self.file = MyLineEdit()
        self.file.clicked.connect(self.getFileName)
        self.colorLineEdit = MyLineEdit('#000000')
        self.colorLineEdit.clicked.connect(self.setColor)
        self.widthLineEdit = MyLineEdit('1')
        self.widthLineEdit.clicked.connect(self.setWidth)
        self.styleLineEdit = QComboBox()
        self.styleLineEdit.addItems(['solid', 'dashed', 'dashdot', 'dotted'])
        self.styleLineEdit.currentIndexChanged.connect(self.getLineStyle)
        self.legendLineEdit = MyLineEdit()
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow('Data file:', self.file)
        layout.addRow('Set color:', self.colorLineEdit)
        layout.addRow('Set line Width:', self.widthLineEdit)
        layout.addRow('Set line style:', self.styleLineEdit)
        layout.addRow('Set legend prefix:', self.legendLineEdit)
        layout.addWidget(self.buttons)

    def getFileName(self):
        file, ok = QFileDialog.getOpenFileName(self, 'Open', './data', 'CSV Files (*.csv);;TSV Files (*.tsv)')
        if ok:
            if file.endswith('.csv'):
                df = pd.read_csv(file, delimiter=',', header=0)
            elif file.endswith('.tsv'):
                df = pd.read_csv(file, delimiter='\t', header=0)
            else:
                QMessageBox.critical(self, 'Error', 'Incorrect file format!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

            if 'Label' in df.columns and 'Score' in df.columns:
                self.raw_data = pd.DataFrame({'col0': df.Label, 'col1': df.Label, 'col2': df.Score, 'col3': df.Score})
                if self.curve == 'ROC':
                    fpr, tpr, _ = roc_curve(df.Label.astype(int), df.Score.astype(float))
                    self.auc = round(auc(fpr, tpr), 4)
                    self.dot = pd.DataFrame(np.hstack((fpr.reshape((-1, 1)), tpr.reshape((-1, 1)))), columns=['fpr', 'tpr'])
                    self.file.setText(file)
                if self.curve == 'PRC':
                    precision, recall, _ = precision_recall_curve(df.Label.astype(int), df.Score.astype(float))
                    self.auc = round(auc(recall, precision), 4)
                    self.dot = pd.DataFrame(np.hstack((recall.reshape((-1, 1)), precision.reshape((-1, 1)))), columns=['recall', 'precision'])
                    self.file.setText(file)
            else:
                QMessageBox.critical(self, 'Error', 'Incorrect file format!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

    def setColor(self):
        self.color = QColorDialog.getColor().name()
        self.colorLineEdit.setText(self.color)

    def setWidth(self):
        lw, ok = QInputDialog.getInt(self, 'Line width', 'Get line width', 1, 1, 6, 1)
        if ok:
            self.lineWidth = lw
            self.widthLineEdit.setText(str(lw))

    def getLineStyle(self):
        self.lineStyle = self.styleLineEdit.currentText()

    def getPrefix(self):
        return self.legendLineEdit.text()

    @staticmethod
    def getValues(curve):
        dialog = QPlotInput(curve)
        result = dialog.exec_()
        prefix = dialog.getPrefix()
        if prefix != '':
            return dialog.auc, dialog.dot, dialog.color, dialog.lineWidth, dialog.lineStyle, prefix, dialog.raw_data, result == QDialog.Accepted
        else:
            QMessageBox.critical(dialog, 'Error', 'Empty field!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
            return dialog.auc, dialog.dot, dialog.color, dialog.lineWidth, dialog.lineStyle, prefix, dialog.raw_data, False


class QBoxPlotInput(QDialog):
    def __init__(self):
        super(QBoxPlotInput, self).__init__()
        self.initUI()
        self.dataframe = None

    def initUI(self):
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setFont(QFont('Arial', 8))
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.setWindowTitle('iLearnPlus Boxplot')
        self.resize(500, 50)
        layout = QFormLayout(self)
        self.file = MyLineEdit()
        self.file.clicked.connect(self.getFileName)
        self.x_label = MyLineEdit('X label name')
        self.y_label = MyLineEdit('Y label name')
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow('Data file:', self.file)
        layout.addRow('X label:', self.x_label)
        layout.addRow('Y label:', self.y_label)
        layout.addWidget(self.buttons)

    def getFileName(self):
        file, ok = QFileDialog.getOpenFileName(self, 'Open', './data', 'CSV Files (*.csv);;TSV Files (*.tsv)')        
        if ok:
            if file.endswith('.csv'):
                self.dataframe = pd.read_csv(file, delimiter=',', header=0)
            elif file.endswith('.tsv'):
                self.dataframe = pd.read_csv(file, delimiter='\t', header=0)
            else:
                QMessageBox.critical(self, 'Error', 'Incorrect file format!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
            self.file.setText(file)

    def getLabelNames(self):
        return self.x_label.text(), self.y_label.text()

    @staticmethod
    def getValues():
        dialog = QBoxPlotInput()
        result = dialog.exec_()
        labels = dialog.getLabelNames()
        return labels[0], labels[1], dialog.dataframe, result == QDialog.Accepted


class QFileTransformation(QDialog):
    def __init__(self):
        super(QFileTransformation, self).__init__()
        self.initUI()        

    def initUI(self):
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setFont(QFont('Arial', 8))
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.setWindowTitle('iLearnPlus File Transformer')
        self.resize(500, 50)
        layout = QFormLayout(self)
        self.file = MyLineEdit()
        self.file.clicked.connect(self.getFileName)       
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow('Data file:', self.file)        
        layout.addWidget(self.buttons)

    def getFileName(self):
        file, ok = QFileDialog.getOpenFileName(self, 'Open', './data', 'CSV Files (*.csv);;TSV Files (*.tsv);;SVM Files(*.svm);;Weka Files (*.arff)')
        self.file.setText(file)        
        
    def getName(self):
        return self.file.text()
    
    @staticmethod
    def getValues():
        dialog = QFileTransformation()
        result = dialog.exec_()
        fileName = dialog.getName()
        return fileName, result == QDialog.Accepted


class QHeatmapInput(QDialog):
    def __init__(self):
        super(QHeatmapInput, self).__init__()
        self.initUI()
        self.dataframe = None

    def initUI(self):
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setFont(QFont('Arial', 8))
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.setWindowTitle('iLearnPlus Boxplot')
        self.resize(500, 50)
        layout = QFormLayout(self)
        self.file = MyLineEdit()
        self.file.clicked.connect(self.getFileName)
        self.x_label = MyLineEdit('X label name')
        self.y_label = MyLineEdit('Y label name')
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow('Data file:', self.file)
        layout.addRow('X label:', self.x_label)
        layout.addRow('Y label:', self.y_label)
        layout.addWidget(self.buttons)

    def getFileName(self):
        file, ok = QFileDialog.getOpenFileName(self, 'Open', './data', 'CSV Files (*.csv);;TSV Files (*.tsv)')
        if ok:
            if file.endswith('.csv'):
                self.dataframe = pd.read_csv(file, delimiter=',', header=0, index_col=0)
            elif file.endswith('.tsv'):
                self.dataframe = pd.read_csv(file, delimiter='\t', header=0, index_col=0)
            else:
                QMessageBox.critical(self, 'Error', 'Incorrect file format!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
            self.file.setText(file)

    def getLabelNames(self):
        return self.x_label.text(), self.y_label.text()

    @staticmethod
    def getValues():
        dialog = QHeatmapInput()
        result = dialog.exec_()
        labels = dialog.getLabelNames()

        return labels[0], labels[1], dialog.dataframe, result == QDialog.Accepted


class QSelectModel(QDialog):
    def __init__(self, model_list):
        super(QSelectModel, self).__init__()
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.model_list = model_list
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.resize(400, 100)
        self.setWindowTitle('Select')
        self.setFont(QFont('Arial', 10))
        layout = QFormLayout(self)
        self.modelComboBox = QComboBox()
        self.modelComboBox.addItems(self.model_list)
        self.modelComboBox.setCurrentIndex(0)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow("Select model to save:", self.modelComboBox)
        layout.addWidget(self.buttons)

    @staticmethod
    def getValues(model_list):
        dialog = QSelectModel(model_list)
        result = dialog.exec_()
        model = dialog.modelComboBox.currentText()
        return model, result == QDialog.Accepted


class QSCombineModelDialog(QDialog):
    def __init__(self, model_list):
        super(QSCombineModelDialog, self).__init__()
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.model_list = model_list
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowTitle('Combine models')
        self.resize(500, 90)
        self.setFont(QFont('Arial', 10))
        layout = QFormLayout(self)
        self.modelComboBox = QComboBox()
        self.modelComboBox.addItems(['LR', 'RF','SVM', 'DecisionTree', 'LightGBM', 'XGBoost', 'KNN', 'LDA', 'QDA', 'SGD', 'NaiveBayes', 'Bagging', 'AdaBoost', 'GBDT'])
        self.modelComboBox.setCurrentIndex(0)        
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow("Use what algorithm to combine models:", self.modelComboBox)        
        layout.addWidget(self.buttons)   

    @staticmethod
    def getValues(model_list):
        dialog = QSCombineModelDialog(model_list)
        result = dialog.exec_()
        model = dialog.modelComboBox.currentText()        
        return model, result == QDialog.Accepted






if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = QFileTransformation()
    win.show()
    sys.exit(app.exec_())
