#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import sys, os
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QFileDialog, QLabel, QHBoxLayout, QGroupBox, QTextEdit,
                             QVBoxLayout, QLineEdit, QTreeWidget, QTreeWidgetItem, QSplitter, QTableWidget, QTabWidget,
                             QTableWidgetItem, QInputDialog, QMessageBox, QFormLayout, QHeaderView, QAbstractItemView)
from PyQt5.QtGui import QIcon, QFont, QMovie
from PyQt5.QtCore import Qt, pyqtSignal
from util import (FileProcessing, InputDialog, CheckAccPseParameter, MachineLearning, ModelMetrics, PlotWidgets)
import qdarkstyle
import numpy as np
import pandas as pd
from itertools import combinations
import copy
import threading
import datetime
import sip
import joblib
import re

class ILearnPlusEstimator(QWidget):
    # signal
    display_signal = pyqtSignal(list)
    display_curves_signal = pyqtSignal()
    append_msg_signal = pyqtSignal(str)
    close_signal = pyqtSignal(str)

    def __init__(self):
        super(ILearnPlusEstimator, self).__init__()
        # signal
        self.display_signal.connect(self.display_metrics)
        self.display_curves_signal.connect(self.display_curves)
        self.append_msg_signal.connect(self.append_message)

        # status bar
        self.gif = QMovie('images/progress_bar.gif')
        self.fasta_file = None                                                 # fasta file
        self.descriptor = None                                                 # descriptor object
        self.para_dict = {
            'EAAC': {'sliding_window': 5},
            'CKSAAP': {'kspace': 3},
            'EGAAC': {'sliding_window': 5},
            'CKSAAGP': {'kspace': 3},
            'AAIndex': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101'},
            'NMBroto': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 'nlag': 3, 'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise',},
            'Moran': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 'nlag': 3, 'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise',},
            'Geary': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 'nlag': 3, 'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise',},
            'KSCTriad': {'kspace': 3},
            'SOCNumber': {'nlag': 3},
            'QSOrder': {'nlag': 3, 'weight': 0.05},
            'PAAC': {'weight': 0.05, 'lambdaValue': 3},
            'APAAC': {'weight': 0.05, 'lambdaValue': 3},
            'DistancePair': {'distance': 0, 'cp': 'cp(20)',},
            'AC': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 'nlag': 3},
            'CC': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 'nlag': 3},
            'ACC': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 'nlag': 3},
            'PseKRAAC type 1': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 2': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 3A': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 3B': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 4': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 5': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 6A': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 6B': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 6C': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 7': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 8': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 9': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 10': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 11': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 12': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 13': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 14': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 15': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'PseKRAAC type 16': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 1},
            'Kmer': {'kmer': 3},
            'RCKmer': {'kmer': 3},
            'Mismatch': {'kmer': 3, 'mismatch': 1},
            'Subsequence': {'kmer': 3, 'delta': 0},
            'ENAC': {'sliding_window': 5},
            'CKSNAP': {'kspace': 3},
            'DPCP': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)'},
            'DPCP type2': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)'},
            'TPCP': {'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)'},
            'TPCP type2': {'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)'},
            'DAC': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)', 'nlag': 3},
            'DCC': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)', 'nlag': 3},
            'DACC': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)', 'nlag': 3},
            'TAC': {'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)', 'nlag': 3},
            'TCC': {'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)', 'nlag': 3},
            'TACC': {'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)', 'nlag': 3},
            'PseDNC': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)', 'weight': 0.05, 'lambdaValue': 3},
            'PseKNC': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)', 'weight': 0.05, 'lambdaValue': 3, 'kmer': 3},
            'PCPseDNC': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)', 'weight': 0.05, 'lambdaValue': 3},
            'PCPseTNC': {'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)', 'weight': 0.05, 'lambdaValue': 3},
            'SCPseDNC': {'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise', 'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)', 'weight': 0.05, 'lambdaValue': 3},
            'SCPseTNC': {'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)', 'weight': 0.05, 'lambdaValue': 3},
        }                                                 # single descriptor parameters
        self.desc_default_para = {             # default parameter for descriptors
            'sliding_window': 5,
            'kspace': 3,
            'props': ['CIDH920105', 'BHAR880101', 'CHAM820101', 'CHAM820102', 'CHOC760101', 'BIGC670101', 'CHAM810101', 'DAYM780201'],
            'nlag': 3,
            'weight': 0.05,
            'lambdaValue': 3,
            'PseKRAAC_model': 'g-gap',
            'g-gap': 2,
            'k-tuple': 2,
            'RAAC_clust': 1,
            'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101',
            'kmer': 3,
            'mismatch': 1,
            'delta': 0,
            'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise',
            'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)',
            'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)',
            'distance': 0,
            'cp': 'cp(20)',
        }                                         # descriptor parameters
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
        }                                          # machine learning parameters

        # for combination of models       
        self.bestPerformance = 0
        self.bestCombinations = ()
        self.bestMetrics = None
        self.bestAUC = None
        self.bestPRC = None
        self.bestModels = None
        self.bestTrainingScore = None

        self.desc_selected = set([])                                           # selected descriptors
        self.MLAlgorithm = None                                                # selected machine learning algorithm
        self.MLData = None                                                     #
        self.metrics = ModelMetrics.ModelMetrics()                             # data for display result
        self.current_data_index = 0
        self.boxplot_data = {}                                                 # dict: key -> model name, value -> metrics DataFrame
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
        self.initUI()

    def initUI(self):
        self.setWindowTitle('iLearnPlus Estimator')
        self.resize(800, 600)
        self.setWindowState(Qt.WindowMaximized)
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.setup_UI()

    def setup_UI(self):
        # file
        topGroupBox = QGroupBox('Choose file in special FASTA format', self)
        topGroupBox.setFont(QFont('Arial', 10))
        topGroupBoxLayout = QHBoxLayout()
        self.file_lineEdit = QLineEdit()
        self.file_lineEdit.setFont(QFont('Arial', 8))
        self.file_button = QPushButton('Open')
        self.file_button.setFont(QFont('Arial', 10))
        self.file_button.clicked.connect(self.load_fasta)
        topGroupBoxLayout.addWidget(self.file_lineEdit)
        topGroupBoxLayout.addWidget(self.file_button)
        topGroupBox.setLayout(topGroupBoxLayout)

        # encoding list -> treeGroupBox
        treeGroupBox = QGroupBox('Select descriptors', self)
        treeGroupBox.setFont(QFont('Arial', 10))
        treeLayout = QHBoxLayout()
        self.desc_treeWidget = QTreeWidget()
        self.desc_treeWidget.setColumnCount(2)
        self.desc_treeWidget.setMinimumWidth(300)
        self.desc_treeWidget.setColumnWidth(0, 150)
        self.desc_treeWidget.setFont(QFont('Arial', 8))
        self.desc_treeWidget.setHeaderLabels(['Codings', 'Definition'])
        """ Protein descriptors """
        self.Protein = QTreeWidgetItem(self.desc_treeWidget)
        self.Protein.setExpanded(True)  # set node expanded
        self.Protein.setText(0, 'Protein')
        self.AAC = QTreeWidgetItem(self.Protein)
        self.AAC.setText(0, 'AAC')
        self.AAC.setText(1, 'Amino Acids Content')
        self.AAC.setCheckState(0, Qt.Unchecked)
        self.AAC.setToolTip(1, 'The AAC encoding calculates the frequency of each amino acid\n type in a protein or peptide sequence.')
        self.EAAC = QTreeWidgetItem(self.Protein)
        self.EAAC.setText(0, 'EAAC')
        self.EAAC.setText(1, 'Enhanced Amino Acids Content')
        self.EAAC.setCheckState(0, Qt.Unchecked)
        self.EAAC.setToolTip(1, 'The EAAC feature calculates the AAC based on the sequence window\n of fixed length that continuously slides from the N- to\n C-terminus of each peptide and can be usually applied to\n encode the peptides with an equal length.')
        CKSAAP = QTreeWidgetItem(self.Protein)
        CKSAAP.setText(0, 'CKSAAP')
        CKSAAP.setText(1, 'Composition of k-spaced Amino Acid Pairs')
        CKSAAP.setCheckState(0, Qt.Unchecked)
        CKSAAP.setToolTip(1, 'The CKSAAP feature encoding calculates the frequency of amino\n acid pairs separated by any k residues.')
        self.DPC = QTreeWidgetItem(self.Protein)
        self.DPC.setText(0, 'DPC')
        self.DPC.setText(1, 'Di-Peptide Composition')
        self.DPC.setCheckState(0, Qt.Unchecked)
        self.DPC.setToolTip(1, 'The DPC descriptor calculate the frequency of di-peptides.')
        DDE = QTreeWidgetItem(self.Protein)
        DDE.setText(0, 'DDE')
        DDE.setText(1, 'Dipeptide Deviation from Expected Mean')
        DDE.setCheckState(0, Qt.Unchecked)
        DDE.setToolTip(1, 'The Dipeptide Deviation from Expected Mean feature vector is\n constructed by computing three parameters, i.e. dipeptide composition (Dc),\n theoretical mean (Tm), and theoretical variance (Tv).')
        self.TPC = QTreeWidgetItem(self.Protein)
        self.TPC.setText(0, 'TPC')
        self.TPC.setText(1, 'Tripeptide Composition')
        self.TPC.setCheckState(0, Qt.Unchecked)
        self.TPC.setToolTip(1, 'The TPC descriptor calculate the frequency of tri-peptides.')
        self.binary = QTreeWidgetItem(self.Protein)
        self.binary.setText(0, 'binary')
        self.binary.setText(1, 'binary')
        self.binary.setCheckState(0, Qt.Unchecked)
        self.binary.setToolTip(1, 'In the binary encoding, each amino acid is represented by a 20-dimensional binary vector.')
        self.binary_6bit = QTreeWidgetItem(self.Protein)
        self.binary_6bit.setToolTip(1, 'The descriptor need fasta sequences with equal length.')
        self.binary_6bit.setText(0, 'binary_6bit')
        self.binary_6bit.setText(1, 'binary (6 bit)')
        self.binary_6bit.setCheckState(0, Qt.Unchecked)
        self.binary_6bit.setToolTip(1, 'In the binary encoding, each amino acid is represented by a 6-dimensional binary vector.')
        self.binary_5bit_type1 = QTreeWidgetItem(self.Protein)
        self.binary_5bit_type1.setToolTip(1, 'The descriptor need fasta sequences with equal length.')
        self.binary_5bit_type1.setText(0, 'binary_5bit type 1')
        self.binary_5bit_type1.setText(1, 'binary (5 bit type 1)')
        self.binary_5bit_type1.setCheckState(0, Qt.Unchecked)
        self.binary_5bit_type1.setToolTip(1, 'In the binary encoding, each amino acid is represented by a 5-dimensional binary vector.')
        self.binary_5bit_type2 = QTreeWidgetItem(self.Protein)
        self.binary_5bit_type2.setToolTip(1, 'The descriptor need fasta sequences with equal length.')
        self.binary_5bit_type2.setText(0, 'binary_5bit type 2')
        self.binary_5bit_type2.setText(1, 'binary (5 bit type 2)')
        self.binary_5bit_type2.setCheckState(0, Qt.Unchecked)
        self.binary_5bit_type2.setToolTip(1, 'In the binary encoding, each amino acid is represented by a 5-dimensional binary vector.')
        self.binary_3bit_type1 = QTreeWidgetItem(self.Protein)
        self.binary_3bit_type1.setToolTip(1, 'The descriptor need fasta sequences with equal length.')
        self.binary_3bit_type1.setText(0, 'binary_3bit type 1')
        self.binary_3bit_type1.setText(1, 'binary (3 bit type 1 - Hydrophobicity)')
        self.binary_3bit_type1.setCheckState(0, Qt.Unchecked)
        self.binary_3bit_type1.setToolTip(1, 'In the binary encoding, each amino acid is represented by a 3-dimensional binary vector.')
        self.binary_3bit_type2 = QTreeWidgetItem(self.Protein)
        self.binary_3bit_type2.setToolTip(1, 'The descriptor need fasta sequences with equal length.')
        self.binary_3bit_type2.setText(0, 'binary_3bit type 2')
        self.binary_3bit_type2.setText(1, 'binary (3 bit type 2 - Normalized Van der Waals volume)')
        self.binary_3bit_type2.setCheckState(0, Qt.Unchecked)
        self.binary_3bit_type2.setToolTip(1, 'In the binary encoding, each amino acid is represented by a 3-dimensional binary vector.')
        self.binary_3bit_type3 = QTreeWidgetItem(self.Protein)
        self.binary_3bit_type3.setToolTip(1, 'The descriptor need fasta sequences with equal length.')
        self.binary_3bit_type3.setText(0, 'binary_3bit type 3')
        self.binary_3bit_type3.setText(1, 'binary (3 bit type 3 - Polarity)')
        self.binary_3bit_type3.setCheckState(0, Qt.Unchecked)
        self.binary_3bit_type3.setToolTip(1, 'In the binary encoding, each amino acid is represented by a 3-dimensional binary vector.')
        self.binary_3bit_type4 = QTreeWidgetItem(self.Protein)
        self.binary_3bit_type4.setToolTip(1, 'The descriptor need fasta sequences with equal length.')
        self.binary_3bit_type4.setText(0, 'binary_3bit type 4')
        self.binary_3bit_type4.setText(1, 'binary (3 bit type 4 - Polarizibility)')
        self.binary_3bit_type4.setCheckState(0, Qt.Unchecked)
        self.binary_3bit_type4.setToolTip(1, 'In the binary encoding, each amino acid is represented by a 3-dimensional binary vector.')
        self.binary_3bit_type5 = QTreeWidgetItem(self.Protein)
        self.binary_3bit_type5.setToolTip(1, 'The descriptor need fasta sequences with equal length.')
        self.binary_3bit_type5.setText(0, 'binary_3bit type 5')
        self.binary_3bit_type5.setText(1, 'binary (3 bit type 5 - Charge)')
        self.binary_3bit_type5.setCheckState(0, Qt.Unchecked)
        self.binary_3bit_type5.setToolTip(1, 'In the binary encoding, each amino acid is represented by a 3-dimensional binary vector.')
        self.binary_3bit_type6 = QTreeWidgetItem(self.Protein)
        self.binary_3bit_type6.setToolTip(1, 'The descriptor need fasta sequences with equal length.')
        self.binary_3bit_type6.setText(0, 'binary_3bit type 6')
        self.binary_3bit_type6.setText(1, 'binary (3 bit type 6 - Secondary structures)')
        self.binary_3bit_type6.setCheckState(0, Qt.Unchecked)
        self.binary_3bit_type6.setToolTip(1, 'In the binary encoding, each amino acid is represented by a 3-dimensional binary vector.')
        self.binary_3bit_type7 = QTreeWidgetItem(self.Protein)
        self.binary_3bit_type7.setToolTip(1, 'The descriptor need fasta sequences with equal length.')
        self.binary_3bit_type7.setText(0, 'binary_3bit type 7')
        self.binary_3bit_type7.setText(1, 'binary (3 bit type 7 - Solvent accessibility)')
        self.binary_3bit_type7.setCheckState(0, Qt.Unchecked)
        self.binary_3bit_type7.setToolTip(1, 'In the binary encoding, each amino acid is represented by a 3-dimensional binary vector.')
        self.AESNN3 = QTreeWidgetItem(self.Protein)
        self.AESNN3.setText(0, 'AESNN3')
        self.AESNN3.setText(1, 'Learn from alignments')
        self.AESNN3.setCheckState(0, Qt.Unchecked)
        self.AESNN3.setToolTip(1, 'For this descriptor, each amino acid type is described using\n a three-dimensional vector. Values are taken from the three\n hidden units from the neural network trained on structure alignments.')
        self.GAAC = QTreeWidgetItem(self.Protein)
        self.GAAC.setText(0, 'GAAC')
        self.GAAC.setText(1, 'Grouped Amino Acid Composition')
        self.GAAC.setCheckState(0, Qt.Unchecked)
        self.GAAC.setToolTip(1, 'In the GAAC encoding, the 20 amino acid types are further categorized\n into five classes according to their physicochemical properties. It calculate the frequency for each class.')
        self.EGAAC = QTreeWidgetItem(self.Protein)
        self.EGAAC.setText(0, 'EGAAC')
        self.EGAAC.setText(1, 'Enhanced Grouped Amino Acid Composition')
        self.EGAAC.setCheckState(0, Qt.Unchecked)
        self.EGAAC.setToolTip(1, 'It calculates GAAC in windows of fixed length continuously sliding\n from the N- to C-terminal of each peptide and is usually applied\n to peptides with an equal length.')
        CKSAAGP = QTreeWidgetItem(self.Protein)
        CKSAAGP.setText(0, 'CKSAAGP')
        CKSAAGP.setText(1, 'Composition of k-Spaced Amino Acid Group Pairs')
        CKSAAGP.setCheckState(0, Qt.Unchecked)
        CKSAAGP.setToolTip(1, ' It calculates the frequency of amino acid group pairs separated by any k residues.')
        self.GDPC = QTreeWidgetItem(self.Protein)
        self.GDPC.setText(0, 'GDPC')
        self.GDPC.setText(1, 'Grouped Di-Peptide Composition')
        self.GDPC.setCheckState(0, Qt.Unchecked)
        self.GDPC.setToolTip(1, 'GDPC calculate the frequency of amino acid group pairs.')
        self.GTPC = QTreeWidgetItem(self.Protein)
        self.GTPC.setText(0, 'GTPC')
        self.GTPC.setText(1, 'Grouped Tri-Peptide Composition')
        self.GTPC.setCheckState(0, Qt.Unchecked)
        self.GTPC.setToolTip(1, 'GTPC calculate the frequency of grouped tri-peptides.')
        self.AAIndex = QTreeWidgetItem(self.Protein)
        self.AAIndex.setText(0, 'AAIndex')
        self.AAIndex.setText(1, 'AAIndex')
        self.AAIndex.setCheckState(0, Qt.Unchecked)
        self.AAIndex.setToolTip(1, 'The amino acids is respresented by the physicochemical property value in AAindex database.')
        self.ZScale = QTreeWidgetItem(self.Protein)
        self.ZScale.setText(0, 'ZScale')
        self.ZScale.setText(1, 'ZScale')
        self.ZScale.setCheckState(0, Qt.Unchecked)
        self.ZScale.setToolTip(1, 'Each amino acid is characterized by five physicochemical descriptor variables, which were developed by Sandberg et al. in 1998.')
        self.BLOSUM62 = QTreeWidgetItem(self.Protein)
        self.BLOSUM62.setText(0, 'BLOSUM62')
        self.BLOSUM62.setText(1, 'BLOSUM62')
        self.BLOSUM62.setCheckState(0, Qt.Unchecked)
        self.BLOSUM62.setToolTip(1, 'In this descriptor, the BLOSUM62 matrix is employed to represent the\n protein primary sequence information as the basic feature set.')
        NMBroto = QTreeWidgetItem(self.Protein)
        NMBroto.setText(0, 'NMBroto')
        NMBroto.setText(1, 'Normalized Moreau-Broto Autocorrelation')
        NMBroto.setCheckState(0, Qt.Unchecked)
        NMBroto.setToolTip(1, 'The autocorrelation descriptors are defined based on the distribution\n of amino acid properties along the sequence.')
        Moran = QTreeWidgetItem(self.Protein)
        Moran.setText(0, 'Moran')
        Moran.setText(1, 'Moran correlation')        
        Moran.setCheckState(0, Qt.Unchecked)
        Moran.setToolTip(1, 'The autocorrelation descriptors are defined based on the distribution\n of amino acid properties along the sequence.')
        Geary = QTreeWidgetItem(self.Protein)
        Geary.setText(0, 'Geary')
        Geary.setText(1, 'Geary correlation')
        Geary.setCheckState(0, Qt.Unchecked)
        Geary.setToolTip(1, 'The autocorrelation descriptors are defined based on the distribution\n of amino acid properties along the sequence.')
        CTDC = QTreeWidgetItem(self.Protein)
        CTDC.setText(0, 'CTDC')
        CTDC.setText(1, 'Composition')
        CTDC.setCheckState(0, Qt.Unchecked)
        CTDC.setToolTip(1, 'The Composition, Transition and Distribution (CTD) features represent\n the amino acid distribution patterns of a specific structural\n or physicochemical property in a protein or peptide sequence.')
        CTDT = QTreeWidgetItem(self.Protein)
        CTDT.setText(0, 'CTDT')
        CTDT.setText(1, 'Transition')
        CTDT.setCheckState(0, Qt.Unchecked)
        CTDT.setToolTip(1, 'The Composition, Transition and Distribution (CTD) features represent\n the amino acid distribution patterns of a specific structural\n or physicochemical property in a protein or peptide sequence.')
        CTDD = QTreeWidgetItem(self.Protein)
        CTDD.setText(0, 'CTDD')
        CTDD.setText(1, 'Distribution')
        CTDD.setCheckState(0, Qt.Unchecked)
        CTDD.setToolTip(1, 'The Composition, Transition and Distribution (CTD) features represent\n the amino acid distribution patterns of a specific structural\n or physicochemical property in a protein or peptide sequence.')
        CTriad = QTreeWidgetItem(self.Protein)
        CTriad.setText(0, 'CTriad')
        CTriad.setText(1, 'Conjoint Triad')
        CTriad.setCheckState(0, Qt.Unchecked)
        CTriad.setToolTip(1, 'The CTriad considers the properties of one amino acid and its\n vicinal amino acids by regarding any three continuous amino\n acids as a single unit.')
        self.KSCTriad = QTreeWidgetItem(self.Protein)
        self.KSCTriad.setText(0, 'KSCTriad')
        self.KSCTriad.setText(1, 'k-Spaced Conjoint Triad')
        self.KSCTriad.setCheckState(0, Qt.Unchecked)
        self.KSCTriad.setToolTip(1, 'The KSCTriad descriptor is based on the Conjoint CTriad descriptor,\n which not only calculates the numbers of three continuous amino acid units,\n but also considers the continuous amino acid units that are separated by any k residues.')
        SOCNumber = QTreeWidgetItem(self.Protein)
        SOCNumber.setText(0, 'SOCNumber')
        SOCNumber.setText(1, 'Sequence-Order-Coupling Number')
        SOCNumber.setCheckState(0, Qt.Unchecked)
        SOCNumber.setToolTip(1, 'The SOCNumber descriptor consider the sequence order coupling number information.')
        QSOrder = QTreeWidgetItem(self.Protein)
        QSOrder.setText(0, 'QSOrder')
        QSOrder.setText(1, 'Quasi-sequence-order')
        QSOrder.setCheckState(0, Qt.Unchecked)
        QSOrder.setToolTip(1, 'Qsorder descriptor coonsider the quasi sequence order information.')
        PAAC = QTreeWidgetItem(self.Protein)
        PAAC.setText(0, 'PAAC')
        PAAC.setText(1, 'Pseudo-Amino Acid Composition')
        PAAC.setCheckState(0, Qt.Unchecked)
        PAAC.setToolTip(1, 'The PAAC descriptor is a combination of a set of discrete sequence correlation\n factors and the 20 components of the conventional amino acid composition.')
        APAAC = QTreeWidgetItem(self.Protein)
        APAAC.setText(0, 'APAAC')
        APAAC.setText(1, 'Amphiphilic Pseudo-Amino Acid Composition')
        APAAC.setCheckState(0, Qt.Unchecked)
        APAAC.setToolTip(1, 'The descriptor contains 20 + 2 lambda discrete numbers:\n the first 20 numbers are the components of the conventional amino acid composition;\n the next 2 lambda numbers are a set of correlation factors that reflect different\n hydrophobicity and hydrophilicity distribution patterns along a protein chain.')
        self.OPF_10bit = QTreeWidgetItem(self.Protein)
        self.OPF_10bit.setText(0, 'OPF_10bit')
        self.OPF_10bit.setText(1, 'Overlapping Property Features (10 bit)')
        self.OPF_10bit.setCheckState(0, Qt.Unchecked)
        self.OPF_10bit.setToolTip(1, 'For this descriptor, the amino acids are classified into 10 groups based their physicochemical properties.')
        self.OPF_7bit_type1 = QTreeWidgetItem(self.Protein)
        self.OPF_7bit_type1.setText(0, 'OPF_7bit type 1')
        self.OPF_7bit_type1.setText(1, 'Overlapping Property Features (7 bit type 1)')
        self.OPF_7bit_type1.setCheckState(0, Qt.Unchecked)
        self.OPF_7bit_type1.setToolTip(1, 'For this descriptor, the amino acids are classified into 7 groups based their physicochemical properties.')
        self.OPF_7bit_type2 = QTreeWidgetItem(self.Protein)
        self.OPF_7bit_type2.setText(0, 'OPF_7bit type 2')
        self.OPF_7bit_type2.setText(1, 'Overlapping Property Features (7 bit type 2)')
        self.OPF_7bit_type2.setCheckState(0, Qt.Unchecked)
        self.OPF_7bit_type2.setToolTip(1, 'For this descriptor, the amino acids are classified into 7 groups based their physicochemical properties.')
        self.OPF_7bit_type3 = QTreeWidgetItem(self.Protein)
        self.OPF_7bit_type3.setText(0, 'OPF_7bit type 3')
        self.OPF_7bit_type3.setText(1, 'Overlapping Property Features (7 bit type 3)')
        self.OPF_7bit_type3.setCheckState(0, Qt.Unchecked)
        self.OPF_7bit_type3.setToolTip(1, 'For this descriptor, the amino acids are classified into 7 groups based their physicochemical properties.')
        pASDC = QTreeWidgetItem(self.Protein)
        pASDC.setText(0, 'ASDC')
        pASDC.setText(1, 'Adaptive skip dipeptide composition')
        pASDC.setCheckState(0, Qt.Unchecked)
        pASDC.setToolTip(1, 'The adaptive skip dipeptide composition is a modified dipeptide composition,\n which sufficiently considers the correlation information present not only between\n adjacent residues but also between intervening residues.')
        self.proteinKNN = QTreeWidgetItem(self.Protein)
        self.proteinKNN.setText(0, 'KNN')
        self.proteinKNN.setText(1, 'K-nearest neighbor')
        self.proteinKNN.setCheckState(0, Qt.Unchecked)
        self.proteinKNN.setToolTip(1, 'The KNN descriptor depicts how much one query sample resembles other samples.')
        DistancePair = QTreeWidgetItem(self.Protein)
        DistancePair.setText(0, 'DistancePair')
        DistancePair.setText(1, 'PseAAC of Distance-Pairs and Reduced Alphabet')
        DistancePair.setCheckState(0, Qt.Unchecked)
        DistancePair.setToolTip(1, 'The descriptor incorporates the amino acid distance pair coupling information \nand the amino acid reduced alphabet profile into the general pseudo amino acid composition vector.')
        self.proteinAC = QTreeWidgetItem(self.Protein)
        self.proteinAC.setText(0, 'AC')
        self.proteinAC.setText(1, 'Auto covariance')
        self.proteinAC.setCheckState(0, Qt.Unchecked)
        self.proteinAC.setToolTip(1, 'The AC descriptor measures the correlation of the same physicochemical \nindex between two amino acids separated by a distance of lag along the sequence. ')
        self.proteinCC = QTreeWidgetItem(self.Protein)
        self.proteinCC.setText(0, 'CC')
        self.proteinCC.setText(1, 'Cross covariance')
        self.proteinCC.setCheckState(0, Qt.Unchecked)
        self.proteinCC.setToolTip(1, 'The CC descriptor measures the correlation of two different physicochemical \nindices between two amino acids separated by lag nucleic acids along the sequence.')
        self.proteinACC = QTreeWidgetItem(self.Protein)
        self.proteinACC.setText(0, 'ACC')
        self.proteinACC.setText(1, 'Auto-cross covariance')
        self.proteinACC.setCheckState(0, Qt.Unchecked)
        self.proteinACC.setToolTip(1, 'The Dinucleotide-based Auto-Cross Covariance (ACC) encoding is a combination of AC and CC encoding.')
        PseKRAAC_type1 = QTreeWidgetItem(self.Protein)
        PseKRAAC_type1.setText(0, 'PseKRAAC type 1')
        PseKRAAC_type1.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 1')
        PseKRAAC_type1.setCheckState(0, Qt.Unchecked)
        PseKRAAC_type1.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type2 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type2.setText(0, 'PseKRAAC type 2')
        self.PseKRAAC_type2.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 2')
        self.PseKRAAC_type2.setCheckState(0, Qt.Unchecked)
        self.PseKRAAC_type2.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type3A = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type3A.setText(0, 'PseKRAAC type 3A')
        self.PseKRAAC_type3A.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 3A')
        self.PseKRAAC_type3A.setCheckState(0, Qt.Unchecked)
        self.PseKRAAC_type3A.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type3B = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type3B.setText(0, 'PseKRAAC type 3B')
        self.PseKRAAC_type3B.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 3B')
        self.PseKRAAC_type3B.setCheckState(0, Qt.Unchecked)
        self.PseKRAAC_type3B.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type4 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type4.setText(0, 'PseKRAAC type 4')
        self.PseKRAAC_type4.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 4')
        self.PseKRAAC_type4.setCheckState(0, Qt.Unchecked)
        self.PseKRAAC_type4.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type5 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type5.setText(0, 'PseKRAAC type 5')
        self.PseKRAAC_type5.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 5')
        self.PseKRAAC_type5.setCheckState(0, Qt.Unchecked)
        self.PseKRAAC_type5.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type6A = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type6A.setText(0, 'PseKRAAC type 6A')
        self.PseKRAAC_type6A.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 6A')
        self.PseKRAAC_type6A.setCheckState(0, Qt.Unchecked)
        self.PseKRAAC_type6A.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type6B = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type6B.setText(0, 'PseKRAAC type 6B')
        self.PseKRAAC_type6B.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 6B')
        self.PseKRAAC_type6B.setCheckState(0, Qt.Unchecked)
        self.PseKRAAC_type6B.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type6C = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type6C.setText(0, 'PseKRAAC type 6C')
        self.PseKRAAC_type6C.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 6C')
        self.PseKRAAC_type6C.setCheckState(0, Qt.Unchecked)
        self.PseKRAAC_type6C.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type7 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type7.setText(0, 'PseKRAAC type 7')
        self.PseKRAAC_type7.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 7')
        self.PseKRAAC_type7.setCheckState(0, Qt.Unchecked)
        self.PseKRAAC_type7.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type8 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type8.setText(0, 'PseKRAAC type 8')
        self.PseKRAAC_type8.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 8')
        self.PseKRAAC_type8.setCheckState(0, Qt.Unchecked)
        self.PseKRAAC_type8.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type9 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type9.setText(0, 'PseKRAAC type 9')
        self.PseKRAAC_type9.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 9')
        self.PseKRAAC_type9.setCheckState(0, Qt.Unchecked)
        self.PseKRAAC_type9.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type10 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type10.setText(0, 'PseKRAAC type 10')
        self.PseKRAAC_type10.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 10')
        self.PseKRAAC_type10.setCheckState(0, Qt.Unchecked)
        self.PseKRAAC_type10.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type11 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type11.setText(0, 'PseKRAAC type 11')
        self.PseKRAAC_type11.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 11')
        self.PseKRAAC_type11.setCheckState(0, Qt.Unchecked)
        self.PseKRAAC_type11.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type12 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type12.setText(0, 'PseKRAAC type 12')
        self.PseKRAAC_type12.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 12')
        self.PseKRAAC_type12.setCheckState(0, Qt.Unchecked)
        self.PseKRAAC_type12.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type13 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type13.setText(0, 'PseKRAAC type 13')
        self.PseKRAAC_type13.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 13')
        self.PseKRAAC_type13.setCheckState(0, Qt.Unchecked)
        self.PseKRAAC_type13.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type14 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type14.setText(0, 'PseKRAAC type 14')
        self.PseKRAAC_type14.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 14')
        self.PseKRAAC_type14.setCheckState(0, Qt.Unchecked)
        self.PseKRAAC_type14.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type15 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type15.setText(0, 'PseKRAAC type 15')
        self.PseKRAAC_type15.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 15')
        self.PseKRAAC_type15.setCheckState(0, Qt.Unchecked)
        self.PseKRAAC_type15.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type16 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type16.setText(0, 'PseKRAAC type 16')
        self.PseKRAAC_type16.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 16')
        self.PseKRAAC_type16.setCheckState(0, Qt.Unchecked)
        self.PseKRAAC_type16.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        """ DNA descriptors """
        self.DNA = QTreeWidgetItem(self.desc_treeWidget)
        self.DNA.setText(0, 'DNA')
        Kmer = QTreeWidgetItem(self.DNA)
        Kmer.setText(0, 'Kmer')
        Kmer.setText(1, 'The occurrence frequencies of k neighboring nucleic acids')
        Kmer.setCheckState(0, Qt.Unchecked)
        Kmer.setToolTip(1, 'For kmer descriptor, the DNA or RNA sequences are represented\n as the occurrence frequencies of k neighboring nucleic acids.')
        RCKmer = QTreeWidgetItem(self.DNA)
        RCKmer.setText(0, 'RCKmer')
        RCKmer.setText(1, 'Reverse Compliment Kmer')
        RCKmer.setCheckState(0, Qt.Unchecked)
        RCKmer.setToolTip(1, 'The RCKmer descriptor is a variant of kmer descriptor,\n in which the kmers are not expected to be strand-specific. ')
        dnaMismatch = QTreeWidgetItem(self.DNA)
        dnaMismatch.setText(0, 'Mismatch')
        dnaMismatch.setText(1, 'Mismatch profile')
        dnaMismatch.setCheckState(0, Qt.Unchecked)
        dnaMismatch.setToolTip(1, 'The mismatch profile also calculates the occurrences of kmers,\n but allows max m inexact matching (m < k).')
        dnaSubsequence = QTreeWidgetItem(self.DNA)
        dnaSubsequence.setText(0, 'Subsequence')
        dnaSubsequence.setText(1, 'Subsequence profile')
        dnaSubsequence.setCheckState(0, Qt.Unchecked)
        dnaSubsequence.setToolTip(1, 'The subsequence descriptor allows non-contiguous matching.')
        self.NAC = QTreeWidgetItem(self.DNA)
        self.NAC.setText(0, 'NAC')
        self.NAC.setText(1, 'Nucleic Acid Composition')
        self.NAC.setCheckState(0, Qt.Unchecked)
        self.NAC.setToolTip(1, 'The NAC encoding calculates the frequency of each nucleic acid type in a nucleotide sequence.')
        # DNC = QTreeWidgetItem(self.DNA)
        # DNC.setText(0, 'DNC')
        # DNC.setText(1, 'Di-Nucleotide Composition')
        # DNC.setCheckState(0, Qt.Unchecked)
        # TNC = QTreeWidgetItem(self.DNA)
        # TNC.setText(0, 'TNC')
        # TNC.setText(1, 'Tri-Nucleotide Composition')
        # TNC.setCheckState(0, Qt.Unchecked)
        self.ANF = QTreeWidgetItem(self.DNA)
        self.ANF.setText(0, 'ANF')
        self.ANF.setText(1, 'Accumulated Nucleotide Frequency')
        self.ANF.setCheckState(0, Qt.Unchecked)
        self.ANF.setToolTip(1, 'The ANF encoding include the nucleotide frequency information and the distribution of each nucleotide in the RNA sequence.')
        self.ENAC = QTreeWidgetItem(self.DNA)
        self.ENAC.setText(0, 'ENAC')
        self.ENAC.setText(1, 'Enhanced Nucleic Acid Composition')
        self.ENAC.setCheckState(0, Qt.Unchecked)
        self.ENAC.setToolTip(1, 'The ENAC descriptor calculates the NAC based on the sequence window\n of fixed length that continuously slides from the 5\' to 3\' terminus\n of each nucleotide sequence and can be usually applied to encode the\n nucleotide sequence with an equal length.')
        self.DNAbinary = QTreeWidgetItem(self.DNA)
        self.DNAbinary.setText(0, 'binary')
        self.DNAbinary.setText(1, 'DNA binary')
        self.DNAbinary.setCheckState(0, Qt.Unchecked)
        self.DNAbinary.setToolTip(1, 'In the Binary encoding, each amino acid is represented by a 4-dimensional binary vector.')
        self.dnaPS2 = QTreeWidgetItem(self.DNA)
        self.dnaPS2.setText(0, 'PS2')
        self.dnaPS2.setText(1, 'Position-specific of two nucleotides')
        self.dnaPS2.setCheckState(0, Qt.Unchecked)
        self.dnaPS2.setToolTip(1, 'There are 4 x 4 = 16 pairs of adjacent pairwise nucleotides, \nthus a single variable representing one such pair gets one-hot\n (i.e. binary) encoded into 16 binary variables.')
        self.dnaPS3 = QTreeWidgetItem(self.DNA)
        self.dnaPS3.setText(0, 'PS3')
        self.dnaPS3.setText(1, 'Position-specific of three nucleotides')
        self.dnaPS3.setCheckState(0, Qt.Unchecked)
        self.dnaPS3.setToolTip(1, 'The PS3 descriptor is encoded for three adjacent nucleotides in a similar way with PS2.')
        self.dnaPS4 = QTreeWidgetItem(self.DNA)
        self.dnaPS4.setText(0, 'PS4')
        self.dnaPS4.setText(1, 'Position-specific of four nucleotides')
        self.dnaPS4.setCheckState(0, Qt.Unchecked)
        self.dnaPS4.setToolTip(1, 'The PS4 descriptor is encoded for four adjacent nucleotides in a similar way with PS2.')
        CKSNAP = QTreeWidgetItem(self.DNA)
        CKSNAP.setText(0, 'CKSNAP')
        CKSNAP.setText(1, 'Composition of k-spaced Nucleic Acid Pairs')
        CKSNAP.setCheckState(0, Qt.Unchecked)
        CKSNAP.setToolTip(1, 'The CKSNAP feature encoding calculates the frequency of nucleic acid pairs separated by any k nucleic acid.')
        self.NCP = QTreeWidgetItem(self.DNA)
        self.NCP.setText(0, 'NCP')
        self.NCP.setText(1, 'Nucleotide Chemical Property')
        self.NCP.setCheckState(0, Qt.Unchecked)
        self.NCP.setToolTip(1, 'Based on chemical properties, A can be represented by coordinates (1, 1, 1), \nC can be represented by coordinates (0, 1, 0), G can be represented by coordinates (1, 0, 0), \nU can be represented by coordinates (0, 0, 1). ')
        self.PSTNPss = QTreeWidgetItem(self.DNA)
        self.PSTNPss.setText(0, 'PSTNPss')
        self.PSTNPss.setText(1, 'Position-specific trinucleotide propensity based on single-strand')
        self.PSTNPss.setCheckState(0, Qt.Unchecked)
        self.PSTNPss.setToolTip(1, 'The PSTNPss descriptor usie a statistical strategy based on single-stranded characteristics of DNA or RNA.')
        self.PSTNPds = QTreeWidgetItem(self.DNA)
        self.PSTNPds.setText(0, 'PSTNPds')
        self.PSTNPds.setText(1, 'Position-specific trinucleotide propensity based on double-strand')
        self.PSTNPds.setCheckState(0, Qt.Unchecked)
        self.PSTNPds.setToolTip(1, 'The PSTNPds descriptor use a statistical strategy based on double-stranded characteristics of DNA according to complementary base pairing.')
        self.EIIP = QTreeWidgetItem(self.DNA)
        self.EIIP.setText(0, 'EIIP')
        self.EIIP.setText(1, 'Electron-ion interaction pseudopotentials of trinucleotide')
        self.EIIP.setCheckState(0, Qt.Unchecked)
        self.EIIP.setToolTip(1, 'The EIIP directly use the EIIP value represent the nucleotide in the DNA sequence.')
        PseEIIP = QTreeWidgetItem(self.DNA)
        PseEIIP.setText(0, 'PseEIIP')
        PseEIIP.setText(1, 'Electron-ion interaction pseudopotentials of trinucleotide')
        PseEIIP.setCheckState(0, Qt.Unchecked)
        PseEIIP.setToolTip(1, 'Electron-ion interaction pseudopotentials of trinucleotide.')
        DNAASDC = QTreeWidgetItem(self.DNA)
        DNAASDC.setText(0, 'ASDC')
        DNAASDC.setText(1, 'Adaptive skip dinucleotide composition')
        DNAASDC.setCheckState(0, Qt.Unchecked)
        DNAASDC.setToolTip(1, 'The adaptive skip dipeptide composition is a modified dinucleotide composition, \nwhich sufficiently considers the correlation information present not only between \nadjacent residues but also between intervening residues.')
        self.dnaDBE = QTreeWidgetItem(self.DNA)
        self.dnaDBE.setText(0, 'DBE')
        self.dnaDBE.setText(1, 'Dinucleotide binary encoding')
        self.dnaDBE.setCheckState(0, Qt.Unchecked)
        self.dnaDBE.setToolTip(1, 'The DBE descriptor encapsulates the positional information of the dinucleotide at each position in the sequence.')
        self.dnaLPDF = QTreeWidgetItem(self.DNA)
        self.dnaLPDF.setText(0, 'LPDF')
        self.dnaLPDF.setText(1, 'Local position-specific dinucleotide frequency')
        self.dnaLPDF.setCheckState(0, Qt.Unchecked)
        self.dnaLPDF.setToolTip(1, 'The LPDF descriptor calculate the local position-specific dinucleotide frequency.')
        dnaDPCP = QTreeWidgetItem(self.DNA)
        dnaDPCP.setText(0, 'DPCP')
        dnaDPCP.setText(1, 'Dinucleotide physicochemical properties')
        dnaDPCP.setCheckState(0, Qt.Unchecked)
        dnaDPCP.setToolTip(1, 'The DPCP descriptor calculate the value of frequency of dinucleotide multiplied by dinucleotide physicochemical properties.')
        self.dnaDPCP2 = QTreeWidgetItem(self.DNA)
        self.dnaDPCP2.setText(0, 'DPCP type2')
        self.dnaDPCP2.setText(1, 'Dinucleotide physicochemical properties type 2')
        self.dnaDPCP2.setCheckState(0, Qt.Unchecked)
        self.dnaDPCP2.setToolTip(1, 'The DPCP2 descriptor calculate the position specific dinucleotide physicochemical properties.')
        dnaTPCP = QTreeWidgetItem(self.DNA)
        dnaTPCP.setText(0, 'TPCP')
        dnaTPCP.setText(1, 'Trinucleotide physicochemical properties')
        dnaTPCP.setCheckState(0, Qt.Unchecked)
        dnaTPCP.setToolTip(1, 'The TPCP descriptor calculate the value of frequency of trinucleotide multiplied by trinucleotide physicochemical properties.')
        self.dnaTPCP2 = QTreeWidgetItem(self.DNA)
        self.dnaTPCP2.setText(0, 'TPCP type2')
        self.dnaTPCP2.setText(1, 'Trinucleotide physicochemical properties type 2')
        self.dnaTPCP2.setCheckState(0, Qt.Unchecked)
        self.dnaTPCP2.setToolTip(1, 'The TPCP2 descriptor calculate the position specific trinucleotide physicochemical properties.')
        dnaMMI = QTreeWidgetItem(self.DNA)
        dnaMMI.setText(0, 'MMI')
        dnaMMI.setText(1, 'Multivariate mutual information')
        dnaMMI.setCheckState(0, Qt.Unchecked)
        self.dnaKNN = QTreeWidgetItem(self.DNA)
        self.dnaKNN.setText(0, 'KNN')
        self.dnaKNN.setText(1, 'K-nearest neighbor')
        self.dnaKNN.setCheckState(0, Qt.Unchecked)
        self.dnaKNN.setToolTip(1, 'The K-nearest neighbor descriptor depicts how much one query sample resembles other samples.')
        dnazcurve9bit = QTreeWidgetItem(self.DNA)
        dnazcurve9bit.setText(0, 'Z_curve_9bit')
        dnazcurve9bit.setText(1, 'The Z curve parameters for frequencies of phase-specific mononucleotides')
        dnazcurve9bit.setCheckState(0, Qt.Unchecked)
        dnazcurve9bit.setToolTip(1, 'The Z curve parameters for frequencies of phase-specific mononucleotides.')
        self.dnazcurve12bit = QTreeWidgetItem(self.DNA)
        self.dnazcurve12bit.setText(0, 'Z_curve_12bit')
        self.dnazcurve12bit.setText(1, 'The Z curve parameters for frequencies of phaseindependent di-nucleotides')
        self.dnazcurve12bit.setCheckState(0, Qt.Unchecked)
        self.dnazcurve12bit.setToolTip(1, 'The Z curve parameters for frequencies of phaseindependent di-nucleotides')
        self.dnazcurve36bit = QTreeWidgetItem(self.DNA)
        self.dnazcurve36bit.setText(0, 'Z_curve_36bit')
        self.dnazcurve36bit.setText(1, 'The Z curve parameters for frequencies of phase-specific di-nucleotides')
        self.dnazcurve36bit.setCheckState(0, Qt.Unchecked)
        self.dnazcurve36bit.setToolTip(1, 'The Z curve parameters for frequencies of phase-specific di-nucleotides')
        self.dnazcurve48bit = QTreeWidgetItem(self.DNA)
        self.dnazcurve48bit.setText(0, 'Z_curve_48bit')
        self.dnazcurve48bit.setText(1, 'The Z curve parameters for frequencies of phaseindependent tri-nucleotides')
        self.dnazcurve48bit.setCheckState(0, Qt.Unchecked)
        self.dnazcurve48bit.setToolTip(1, 'The Z curve parameters for frequencies of phaseindependent tri-nucleotides')
        self.dnazcurve144bit = QTreeWidgetItem(self.DNA)
        self.dnazcurve144bit.setText(0, 'Z_curve_144bit')
        self.dnazcurve144bit.setText(1, 'The Z curve parameters for frequencies of phase-specific tri-nucleotides')
        self.dnazcurve144bit.setCheckState(0, Qt.Unchecked)
        self.dnazcurve144bit.setToolTip(1, 'The Z curve parameters for frequencies of phase-specific tri-nucleotides')
        dnaNMBroto = QTreeWidgetItem(self.DNA)
        dnaNMBroto.setText(0, 'NMBroto')
        dnaNMBroto.setText(1, 'Normalized Moreau-Broto Autocorrelation')
        dnaNMBroto.setCheckState(0, Qt.Unchecked)
        dnaNMBroto.setToolTip(1, 'The autocorrelation descriptors are defined based on the distribution\n of amino acid properties along the sequence.')
        dnaMoran = QTreeWidgetItem(self.DNA)
        dnaMoran.setText(0, 'Moran')
        dnaMoran.setText(1, 'Moran correlation')
        dnaMoran.setCheckState(0, Qt.Unchecked)
        dnaMoran.setToolTip(1, 'The autocorrelation descriptors are defined based on the distribution\n of amino acid properties along the sequence.')
        dnaGeary = QTreeWidgetItem(self.DNA)
        dnaGeary.setText(0, 'Geary')
        dnaGeary.setText(1, 'Geary correlation')
        dnaGeary.setCheckState(0, Qt.Unchecked)
        dnaGeary.setToolTip(1, 'The autocorrelation descriptors are defined based on the distribution\n of amino acid properties along the sequence.')
        self.DAC = QTreeWidgetItem(self.DNA)
        self.DAC.setText(0, 'DAC')
        self.DAC.setText(1, 'Dinucleotide-based Auto Covariance')
        self.DAC.setCheckState(0, Qt.Unchecked)
        self.DAC.setToolTip(1, 'The DAC descriptor measures the correlation of the same physicochemical \nindex between two dinucleotides separated by a distance of lag along the sequence.')
        self.DCC = QTreeWidgetItem(self.DNA)
        self.DCC.setText(0, 'DCC')
        self.DCC.setText(1, 'Dinucleotide-based Cross Covariance')
        self.DCC.setCheckState(0, Qt.Unchecked)
        self.DCC.setToolTip(1, 'The DCC descriptor measures the correlation of two different physicochemical \nindices between two dinucleotides separated by lag nucleic acids along the sequence.')
        DACC = QTreeWidgetItem(self.DNA)
        DACC.setText(0, 'DACC')
        DACC.setText(1, 'Dinucleotide-based Auto-Cross Covariance')
        DACC.setCheckState(0, Qt.Unchecked)
        DACC.setToolTip(1, 'The DACC encoding is a combination of DAC and DCC encoding.')
        self.TAC = QTreeWidgetItem(self.DNA)
        self.TAC.setText(0, 'TAC')
        self.TAC.setText(1, 'Trinucleotide-based Auto Covariance')
        self.TAC.setCheckState(0, Qt.Unchecked)
        self.TAC.setToolTip(1, 'The TAC descriptor measures the correlation of the same physicochemical \nindex between two trinucleotides separated by a distance of lag along the sequence.')
        self.TCC = QTreeWidgetItem(self.DNA)
        self.TCC.setText(0, 'TCC')
        self.TCC.setText(1, 'Trinucleotide-based Cross Covariance')
        self.TCC.setCheckState(0, Qt.Unchecked)
        self.TCC.setToolTip(1, 'The DCC descriptor measures the correlation of two different physicochemical \nindices between two trinucleotides separated by lag nucleic acids along the sequence.')
        TACC = QTreeWidgetItem(self.DNA)
        TACC.setText(0, 'TACC')
        TACC.setText(1, 'Trinucleotide-based Auto-Cross Covariance')
        TACC.setCheckState(0, Qt.Unchecked)
        TACC.setToolTip(1, 'The TACC encoding is a combination of TAC and TCC encoding.')
        PseDNC = QTreeWidgetItem(self.DNA)
        PseDNC.setText(0, 'PseDNC')
        PseDNC.setText(1, 'Pseudo Dinucleotide Composition')
        PseDNC.setCheckState(0, Qt.Unchecked)
        PseDNC.setToolTip(1, 'The PseDNC encodings incorporate contiguous local sequence-order information and the global sequence-order information into the feature vector of the nucleotide sequence.')
        PseKNC = QTreeWidgetItem(self.DNA)
        PseKNC.setText(0, 'PseKNC')
        PseKNC.setText(1, 'Pseudo k-tupler Composition')
        PseKNC.setCheckState(0, Qt.Unchecked)
        PseKNC.setToolTip(1, 'The PseKNC descriptor incorporate the k-tuple nucleotide composition.')
        PCPseDNC = QTreeWidgetItem(self.DNA)
        PCPseDNC.setText(0, 'PCPseDNC')
        PCPseDNC.setText(1, 'Parallel Correlation Pseudo Dinucleotide Composition')
        PCPseDNC.setCheckState(0, Qt.Unchecked)
        PCPseDNC.setToolTip(1, 'The PCPseDNC descriptor consider parallel correlation pseudo trinucleotide composition information.')
        PCPseTNC = QTreeWidgetItem(self.DNA)
        PCPseTNC.setText(0, 'PCPseTNC')
        PCPseTNC.setText(1, 'Parallel Correlation Pseudo Trinucleotide Composition')
        PCPseTNC.setCheckState(0, Qt.Unchecked)
        PCPseTNC.setToolTip(1, 'The PCPseTNC descriptor consider parallel correlation pseudo trinucleotide composition information.')
        SCPseDNC = QTreeWidgetItem(self.DNA)
        SCPseDNC.setText(0, 'SCPseDNC')
        SCPseDNC.setText(1, 'Series Correlation Pseudo Dinucleotide Composition')
        SCPseDNC.setCheckState(0, Qt.Unchecked)
        SCPseDNC.setToolTip(1, 'The SCPseDNC descriptor consider series correlation pseudo dinucleotide composition information.')
        SCPseTNC = QTreeWidgetItem(self.DNA)
        SCPseTNC.setText(0, 'SCPseTNC')
        SCPseTNC.setText(1, 'Series Correlation Pseudo Trinucleotide Composition')
        SCPseTNC.setCheckState(0, Qt.Unchecked)
        SCPseTNC.setToolTip(1, 'The SCPseTNC descriptor consider series correlation pseudo trinucleotide composition.')

        """ RNA descriptors """
        self.RNA = QTreeWidgetItem(self.desc_treeWidget)
        self.RNA.setText(0, 'RNA')
        RNAKmer = QTreeWidgetItem(self.RNA)
        RNAKmer.setText(0, 'Kmer')
        RNAKmer.setText(1, 'The occurrence frequencies of k neighboring nucleic acids')
        RNAKmer.setCheckState(0, Qt.Unchecked)
        RNAKmer.setToolTip(1, 'For kmer descriptor, the DNA or RNA sequences are represented\n as the occurrence frequencies of k neighboring nucleic acids.')
        rnaMismatch = QTreeWidgetItem(self.RNA)
        rnaMismatch.setText(0, 'Mismatch')
        rnaMismatch.setText(1, 'Mismatch profile')
        rnaMismatch.setCheckState(0, Qt.Unchecked)
        rnaMismatch.setToolTip(1, 'The mismatch profile also calculates the occurrences of kmers,\n but allows max m inexact matching (m < k).')
        rnaSubsequence = QTreeWidgetItem(self.RNA)
        rnaSubsequence.setText(0, 'Subsequence')
        rnaSubsequence.setText(1, 'Subsequence profile')
        rnaSubsequence.setCheckState(0, Qt.Unchecked)
        rnaSubsequence.setToolTip(1, 'The subsequence descriptor allows non-contiguous matching.')
        self.RNANAC = QTreeWidgetItem(self.RNA)
        self.RNANAC.setText(0, 'NAC')
        self.RNANAC.setText(1, 'Nucleic Acid Composition')
        self.RNANAC.setCheckState(0, Qt.Unchecked)
        self.RNANAC.setToolTip(1, 'The NAC encoding calculates the frequency of each nucleic acid type in a nucleotide sequence.')
        self.RNAENAC = QTreeWidgetItem(self.RNA)
        self.RNAENAC.setText(0, 'ENAC')
        self.RNAENAC.setText(1, 'Enhanced Nucleic Acid Composition')
        self.RNAENAC.setCheckState(0, Qt.Unchecked)
        self.RNAENAC.setToolTip(1, 'The ENAC descriptor calculates the NAC based on the sequence window\n of fixed length that continuously slides from the 5\' to 3\' terminus\n of each nucleotide sequence and can be usually applied to encode the\n nucleotide sequence with an equal length.')
        # RNADNC = QTreeWidgetItem(self.RNA)
        # RNADNC.setText(0, 'DNC')
        # RNADNC.setText(1, 'Di-Nucleotide Composition')
        # RNADNC.setCheckState(0, Qt.Unchecked)
        # RNATNC = QTreeWidgetItem(self.RNA)
        # RNATNC.setText(0, 'TNC')
        # RNATNC.setText(1, 'Tri-Nucleotide Composition')
        # RNATNC.setCheckState(0, Qt.Unchecked)
        self.RNAANF = QTreeWidgetItem(self.RNA)
        self.RNAANF.setText(0, 'ANF')
        self.RNAANF.setText(1, 'Accumulated Nucleotide Frequency')
        self.RNAANF.setCheckState(0, Qt.Unchecked)
        self.RNAANF.setToolTip(1, 'The ANF encoding include the nucleotide frequency information and the distribution of each nucleotide in the RNA sequence.')
        self.RNANCP = QTreeWidgetItem(self.RNA)
        self.RNANCP.setText(0, 'NCP')
        self.RNANCP.setText(1, 'Nucleotide Chemical Property')
        self.RNANCP.setCheckState(0, Qt.Unchecked)
        self.RNANCP.setToolTip(1, 'Based on chemical properties, A can be represented by coordinates (1, 1, 1), \nC can be represented by coordinates (0, 1, 0), G can be represented by coordinates (1, 0, 0), \nU can be represented by coordinates (0, 0, 1). ')
        self.RNAPSTNPss = QTreeWidgetItem(self.RNA)
        self.RNAPSTNPss.setText(0, 'PSTNPss')
        self.RNAPSTNPss.setText(1, 'Position-specific trinucleotide propensity based on single-strand')
        self.RNAPSTNPss.setCheckState(0, Qt.Unchecked)
        self.RNAPSTNPss.setToolTip(1, 'The PSTNPss descriptor usie a statistical strategy based on single-stranded characteristics of DNA or RNA.')
        self.RNAbinary = QTreeWidgetItem(self.RNA)
        self.RNAbinary.setText(0, 'binary')
        self.RNAbinary.setText(1, 'RNA binary')
        self.RNAbinary.setCheckState(0, Qt.Unchecked)
        self.RNAbinary.setToolTip(1, 'In the Binary encoding, each amino acid is represented by a 4-dimensional binary vector.')
        self.rnaPS2 = QTreeWidgetItem(self.RNA)
        self.rnaPS2.setText(0, 'PS2')
        self.rnaPS2.setText(1, 'Position-specific of two nucleotides')
        self.rnaPS2.setCheckState(0, Qt.Unchecked)
        self.rnaPS2.setToolTip(1, 'There are 4 x 4 = 16 pairs of adjacent pairwise nucleotides, \nthus a single variable representing one such pair gets one-hot\n (i.e. binary) encoded into 16 binary variables.')
        self.rnaPS3 = QTreeWidgetItem(self.RNA)
        self.rnaPS3.setText(0, 'PS3')
        self.rnaPS3.setText(1, 'Position-specific of three nucleotides')
        self.rnaPS3.setCheckState(0, Qt.Unchecked)
        self.rnaPS3.setToolTip(1, 'The PS3 descriptor is encoded for three adjacent nucleotides in a similar way with PS2.')
        self.rnaPS4 = QTreeWidgetItem(self.RNA)
        self.rnaPS4.setText(0, 'PS4')
        self.rnaPS4.setText(1, 'Position-specific of four nucleotides')
        self.rnaPS4.setCheckState(0, Qt.Unchecked)
        self.rnaPS4.setToolTip(1, 'The PS4 descriptor is encoded for four adjacent nucleotides in a similar way with PS2.')
        RNACKSNAP = QTreeWidgetItem(self.RNA)
        RNACKSNAP.setText(0, 'CKSNAP')
        RNACKSNAP.setText(1, 'Composition of k-spaced Nucleic Acid Pairs')
        RNACKSNAP.setCheckState(0, Qt.Unchecked)
        RNACKSNAP.setToolTip(1, 'The CKSNAP feature encoding calculates the frequency of nucleic acid pairs separated by any k nucleic acid.')
        RNAASDC = QTreeWidgetItem(self.RNA)
        RNAASDC.setText(0, 'ASDC')
        RNAASDC.setText(1, 'Adaptive skip di-nucleotide composition')
        RNAASDC.setCheckState(0, Qt.Unchecked)
        RNAASDC.setToolTip(1, 'The adaptive skip dipeptide composition is a modified dinucleotide composition, \nwhich sufficiently considers the correlation information present not only between \nadjacent residues but also between intervening residues.')
        self.rnaDBE = QTreeWidgetItem(self.RNA)
        self.rnaDBE.setText(0, 'DBE')
        self.rnaDBE.setText(1, 'Dinucleotide binary encoding')
        self.rnaDBE.setCheckState(0, Qt.Unchecked)
        self.rnaDBE.setToolTip(1, 'The DBE descriptor encapsulates the positional information of the dinucleotide at each position in the sequence.')
        self.rnaLPDF = QTreeWidgetItem(self.RNA)
        self.rnaLPDF.setText(0, 'LPDF')
        self.rnaLPDF.setText(1, 'Local position-specific dinucleotide frequency')
        self.rnaLPDF.setCheckState(0, Qt.Unchecked)
        self.rnaLPDF.setToolTip(1, 'The LPDF descriptor calculate the local position-specific dinucleotide frequency.')
        rnaDPCP = QTreeWidgetItem(self.RNA)
        rnaDPCP.setText(0, 'DPCP')
        rnaDPCP.setText(1, 'Dinucleotide physicochemical properties')
        rnaDPCP.setCheckState(0, Qt.Unchecked)
        rnaDPCP.setToolTip(1, 'The DPCP descriptor calculate the value of frequency of dinucleotide multiplied by dinucleotide physicochemical properties.')
        self.rnaDPCP2 = QTreeWidgetItem(self.RNA)
        self.rnaDPCP2.setText(0, 'DPCP type2')
        self.rnaDPCP2.setText(1, 'Dinucleotide physicochemical properties type 2')
        self.rnaDPCP2.setCheckState(0, Qt.Unchecked)
        self.rnaDPCP2.setToolTip(1, 'The DPCP2 descriptor calculate the position specific dinucleotide physicochemical properties.')
        rnaMMI = QTreeWidgetItem(self.RNA)
        rnaMMI.setText(0, 'MMI')
        rnaMMI.setText(1, 'Multivariate mutual information')
        rnaMMI.setCheckState(0, Qt.Unchecked)
        rnaMMI.setToolTip(1, 'The MMI descriptor calculate multivariate mutual information on a DNA/RNA sequence.')
        self.rnaKNN = QTreeWidgetItem(self.RNA)
        self.rnaKNN.setText(0, 'KNN')
        self.rnaKNN.setText(1, 'K-nearest neighbor')
        self.rnaKNN.setCheckState(0, Qt.Unchecked)
        self.rnaKNN.setToolTip(1, 'The K-nearest neighbor descriptor depicts how much one query sample resembles other samples.')
        rnazcurve9bit = QTreeWidgetItem(self.RNA)
        rnazcurve9bit.setText(0, 'Z_curve_9bit')
        rnazcurve9bit.setText(1, 'The Z curve parameters for frequencies of phase-specific mononucleotides')
        rnazcurve9bit.setCheckState(0, Qt.Unchecked)
        rnazcurve9bit.setToolTip(1, 'The Z curve parameters for frequencies of phase-specific mononucleotides.')
        self.rnazcurve12bit = QTreeWidgetItem(self.RNA)
        self.rnazcurve12bit.setText(0, 'Z_curve_12bit')
        self.rnazcurve12bit.setText(1, 'The Z curve parameters for frequencies of phaseindependent di-nucleotides')
        self.rnazcurve12bit.setCheckState(0, Qt.Unchecked)
        self.rnazcurve12bit.setToolTip(1, 'The Z curve parameters for frequencies of phaseindependent di-nucleotides')
        self.rnazcurve36bit = QTreeWidgetItem(self.RNA)
        self.rnazcurve36bit.setText(0, 'Z_curve_36bit')
        self.rnazcurve36bit.setText(1, 'The Z curve parameters for frequencies of phase-specific di-nucleotides')
        self.rnazcurve36bit.setCheckState(0, Qt.Unchecked)
        self.rnazcurve36bit.setToolTip(1, 'The Z curve parameters for frequencies of phase-specific di-nucleotides')
        self.rnazcurve48bit = QTreeWidgetItem(self.RNA)
        self.rnazcurve48bit.setText(0, 'Z_curve_48bit')
        self.rnazcurve48bit.setText(1, 'The Z curve parameters for frequencies of phaseindependent tri-nucleotides')
        self.rnazcurve48bit.setCheckState(0, Qt.Unchecked)
        self.rnazcurve48bit.setToolTip(1, 'The Z curve parameters for frequencies of phaseindependent tri-nucleotides')
        self.rnazcurve144bit = QTreeWidgetItem(self.RNA)
        self.rnazcurve144bit.setText(0, 'Z_curve_144bit')
        self.rnazcurve144bit.setText(1, 'The Z curve parameters for frequencies of phase-specific tri-nucleotides')
        self.rnazcurve144bit.setCheckState(0, Qt.Unchecked)
        self.rnazcurve144bit.setToolTip(1, 'The Z curve parameters for frequencies of phase-specific tri-nucleotides')
        rnaNMBroto = QTreeWidgetItem(self.RNA)
        rnaNMBroto.setText(0, 'NMBroto')
        rnaNMBroto.setText(1, 'Normalized Moreau-Broto Autocorrelation')
        rnaNMBroto.setCheckState(0, Qt.Unchecked)
        rnaNMBroto.setToolTip(1, 'The autocorrelation descriptors are defined based on the distribution\n of amino acid properties along the sequence.')
        rnaMoran = QTreeWidgetItem(self.RNA)
        rnaMoran.setText(0, 'Moran')
        rnaMoran.setText(1, 'Moran correlation')
        rnaMoran.setCheckState(0, Qt.Unchecked)
        rnaMoran.setToolTip(1, 'The autocorrelation descriptors are defined based on the distribution\n of amino acid properties along the sequence.')
        rnaGeary = QTreeWidgetItem(self.RNA)
        rnaGeary.setText(0, 'Geary')
        rnaGeary.setText(1, 'Geary correlation')
        rnaGeary.setCheckState(0, Qt.Unchecked)
        rnaGeary.setToolTip(1, 'The autocorrelation descriptors are defined based on the distribution\n of amino acid properties along the sequence.')
        self.RNADAC = QTreeWidgetItem(self.RNA)
        self.RNADAC.setText(0, 'DAC')
        self.RNADAC.setText(1, 'Dinucleotide-based Auto Covariance')
        self.RNADAC.setCheckState(0, Qt.Unchecked)
        self.RNADAC.setToolTip(1, 'The DAC descriptor measures the correlation of the same physicochemical \nindex between two dinucleotides separated by a distance of lag along the sequence.')
        self.RNADCC = QTreeWidgetItem(self.RNA)
        self.RNADCC.setText(0, 'DCC')
        self.RNADCC.setText(1, 'Dinucleotide-based Cross Covariance')
        self.RNADCC.setCheckState(0, Qt.Unchecked)
        self.RNADCC.setToolTip(1, 'The DCC descriptor measures the correlation of two different physicochemical \nindices between two dinucleotides separated by lag nucleic acids along the sequence.')
        RNADACC = QTreeWidgetItem(self.RNA)
        RNADACC.setText(0, 'DACC')
        RNADACC.setText(1, 'Dinucleotide-based Auto-Cross Covariance')        
        RNADACC.setCheckState(0, Qt.Unchecked)
        RNADACC.setToolTip(1, 'The DACC encoding is a combination of DAC and DCC encoding.')
        RNAPseDNC = QTreeWidgetItem(self.RNA)
        RNAPseDNC.setText(0, 'PseDNC')
        RNAPseDNC.setText(1, 'Pseudo Nucleic Acid Composition')
        RNAPseDNC.setCheckState(0, Qt.Unchecked)
        RNAPseDNC.setToolTip(1, 'The PseDNC encodings incorporate contiguous local sequence-order information and the global sequence-order information into the feature vector of the nucleotide sequence.')
        RNAPseKNC = QTreeWidgetItem(self.RNA)
        RNAPseKNC.setText(0, 'PseKNC')
        RNAPseKNC.setText(1, 'Pseudo k-tupler Composition')
        RNAPseKNC.setCheckState(0, Qt.Unchecked)
        RNAPseKNC.setToolTip(1, 'The PseKNC descriptor incorporate the k-tuple nucleotide composition.')
        RNAPCPseDNC = QTreeWidgetItem(self.RNA)
        RNAPCPseDNC.setText(0, 'PCPseDNC')
        RNAPCPseDNC.setText(1, 'Parallel Correlation Pseudo Dinucleotide Composition')
        RNAPCPseDNC.setCheckState(0, Qt.Unchecked)
        RNAPCPseDNC.setToolTip(1, 'The PCPseDNC descriptor consider parallel correlation pseudo trinucleotide composition information.')
        RNASCPseDNC = QTreeWidgetItem(self.RNA)
        RNASCPseDNC.setText(0, 'SCPseDNC')
        RNASCPseDNC.setText(1, 'Series Correlation Pseudo Dinucleotide Composition')
        RNASCPseDNC.setCheckState(0, Qt.Unchecked)
        RNASCPseDNC.setToolTip(1, 'The SCPseDNC descriptor consider series correlation pseudo dinucleotide composition information.')

        treeLayout.addWidget(self.desc_treeWidget)
        treeGroupBox.setLayout(treeLayout)
        self.Protein.setDisabled(True)
        self.DNA.setDisabled(True)
        self.RNA.setDisabled(True)
        self.desc_treeWidget.clicked.connect(self.desc_tree_clicked)
        self.desc_treeWidget.itemChanged.connect(self.desc_tree_checkState)

        # machine learning algorithms
        ml_treeGroupBox = QGroupBox('Select ML algorithms', self)
        ml_treeGroupBox.setFont(QFont('Arial', 10))
        ml_treeLayout = QHBoxLayout()
        self.ml_treeWidget = QTreeWidget()
        self.ml_treeWidget.setColumnCount(2)
        self.ml_treeWidget.setMinimumWidth(300)
        self.ml_treeWidget.setColumnWidth(0, 150)
        self.ml_treeWidget.setFont(QFont('Arial', 8))
        self.ml_treeWidget.setHeaderLabels(['Methods', 'Definition'])
        self.machineLearningAlgorighms = QTreeWidgetItem(self.ml_treeWidget)
        self.machineLearningAlgorighms.setExpanded(True)  # set node expanded
        self.machineLearningAlgorighms.setText(0, 'Machine learning algorithms')
        self.ml_treeWidget.clicked.connect(self.ml_tree_clicked)
        rf = QTreeWidgetItem(self.machineLearningAlgorighms)
        rf.setText(0, 'RF')
        rf.setText(1, 'Random Forest')
        dtree = QTreeWidgetItem(self.machineLearningAlgorighms)
        dtree.setText(0, 'DecisionTree')
        dtree.setText(1, 'Decision Tree')
        lightgbm = QTreeWidgetItem(self.machineLearningAlgorighms)
        lightgbm.setText(0, 'LightGBM')
        lightgbm.setText(1, 'LightGBM')
        svm = QTreeWidgetItem(self.machineLearningAlgorighms)
        svm.setText(0, 'SVM')
        svm.setText(1, 'Support Verctor Machine')
        mlp = QTreeWidgetItem(self.machineLearningAlgorighms)
        mlp.setText(0, 'MLP')
        mlp.setText(1, 'Multi-layer Perceptron')
        xgboost = QTreeWidgetItem(self.machineLearningAlgorighms)
        xgboost.setText(0, 'XGBoost')
        xgboost.setText(1, 'XGBoost')
        knn = QTreeWidgetItem(self.machineLearningAlgorighms)
        knn.setText(0, 'KNN')
        knn.setText(1, 'K-Nearest Neighbour')
        lr = QTreeWidgetItem(self.machineLearningAlgorighms)
        lr.setText(0, 'LR')
        lr.setText(1, 'Logistic Regression')
        lda = QTreeWidgetItem(self.machineLearningAlgorighms)
        lda.setText(0, 'LDA')
        lda.setText(1, 'Linear Discriminant Analysis')
        qda = QTreeWidgetItem(self.machineLearningAlgorighms)
        qda.setText(0, 'QDA')
        qda.setText(1, 'Quadratic Discriminant Analysis')
        sgd = QTreeWidgetItem(self.machineLearningAlgorighms)
        sgd.setText(0, 'SGD')
        sgd.setText(1, 'Stochastic Gradient Descent')
        bayes = QTreeWidgetItem(self.machineLearningAlgorighms)
        bayes.setText(0, 'NaiveBayes')
        bayes.setText(1, 'NaiveBayes')
        bagging = QTreeWidgetItem(self.machineLearningAlgorighms)
        bagging.setText(0, 'Bagging')
        bagging.setText(1, 'Bagging')
        adaboost = QTreeWidgetItem(self.machineLearningAlgorighms)
        adaboost.setText(0, 'AdaBoost')
        adaboost.setText(1, 'AdaBoost')
        gbdt = QTreeWidgetItem(self.machineLearningAlgorighms)
        gbdt.setText(0, 'GBDT')
        gbdt.setText(1, 'Gradient Tree Boosting')
        ml_treeLayout.addWidget(self.ml_treeWidget)
        ml_treeGroupBox.setLayout(ml_treeLayout)
        # deep learning algorighms
        # net1 = QTreeWidgetItem(self.machineLearningAlgorighms)
        # net1.setText(0, 'Net_1_CNN')
        # net1.setText(1, 'Convolutional Neural Network')
        # net2 = QTreeWidgetItem(self.machineLearningAlgorighms)
        # net2.setText(0, 'Net_2_RNN')
        # net2.setText(1, 'Recurrent Neural Network')
        # net3 = QTreeWidgetItem(self.machineLearningAlgorighms)
        # net3.setText(0, 'Net_3_BRNN')
        # net3.setText(1, 'Bidirectional Recurrent Neural Network')
        # net4 = QTreeWidgetItem(self.machineLearningAlgorighms)
        # net4.setText(0, 'Net_4_ABCNN')
        # net4.setText(1, 'Attention Based Convolutional Neural Network')
        # net5 = QTreeWidgetItem(self.machineLearningAlgorighms)
        # net5.setText(0, 'Net_5_ResNet')
        # net5.setText(1, 'Deep Residual Network')
        # net6 = QTreeWidgetItem(self.machineLearningAlgorighms)
        # net6.setText(0, 'Net_6_AE')
        # net6.setText(1, 'AutoEncoder')

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
        self.start_button.clicked.connect(self.run_estimator)
        self.start_button.setFont(QFont('Arial', 10))
        self.desc_slim_button = QPushButton('Show descriptor slims')
        self.desc_slim_button.clicked.connect(self.showDescriptorSlims)
        self.desc_slim_button.setFont(QFont('Arial', 10))

        startLayout.addWidget(self.start_button)
        startLayout.addWidget(self.desc_slim_button)

        ### layout
        left_vertical_layout = QVBoxLayout()
        left_vertical_layout.addWidget(topGroupBox)
        left_vertical_layout.addWidget(treeGroupBox)
        left_vertical_layout.addWidget(ml_treeGroupBox)
        left_vertical_layout.addWidget(paraGroupBox)
        left_vertical_layout.addWidget(startGroupBox)

        #### widget
        leftWidget = QWidget()
        leftWidget.setLayout(left_vertical_layout)

        resultTabWidget = QWidget()
        resultTabLayout = QVBoxLayout(resultTabWidget)
        resultTabControlLayout = QHBoxLayout()
        self.resultSaveBtn = QPushButton(' Save metrics ')
        self.resultSaveBtn.clicked.connect(self.save_result)
        self.modelSaveBtn = QPushButton(' Save model ')
        self.modelSaveBtn.clicked.connect(self.save_model)
        self.displayCorrBtn = QPushButton(' Display correlation ')
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
        self.metricsTableWidget.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.metricsTableWidget.resizeRowsToContents()
        resultTabLayout.addWidget(self.metricsTableWidget)
        resultTabLayout.addLayout(resultTabControlLayout)

        """ boxplot using matplotlib """
        # self.rocGraph = pg.GraphicsLayoutWidget()
        # self.prcGraph = pg.GraphicsLayoutWidget()
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

    """ Events """
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

    def descriptor_length_check(self, sequence_type, status):
        """
        Protein: 'AAIndex', 'binary', 'BLOSUM62', 'EAAC', 'EGAAC', 'ZSCALE'
        DNA: 'ANF', 'DNAbinary', 'EIIP', 'ENAC', 'NCP', 'PSTNPds', 'PSTNPss'
        RNA: 'RNAANF', 'RNAbinary', 'RNAEIIP', 'RNAENAC', 'RNANCP', 'RNAPSTNPds', 'RNAPSTNPss'
        """
        if sequence_type == 'Protein':
            if status:
                self.AAIndex.setDisabled(False)
                self.binary.setDisabled(False)
                self.BLOSUM62.setDisabled(False)
                self.EAAC.setDisabled(False)
                self.EGAAC.setDisabled(False)
                self.ZScale.setDisabled(False)
                self.binary_6bit.setDisabled(False)
                self.binary_5bit_type1.setDisabled(False)
                self.binary_5bit_type2.setDisabled(False)
                self.binary_3bit_type1.setDisabled(False)
                self.binary_3bit_type2.setDisabled(False)
                self.binary_3bit_type3.setDisabled(False)
                self.binary_3bit_type4.setDisabled(False)
                self.binary_3bit_type5.setDisabled(False)
                self.binary_3bit_type6.setDisabled(False)
                self.binary_3bit_type7.setDisabled(False)
                self.AESNN3.setDisabled(False)
                self.OPF_10bit.setDisabled(False)
                self.OPF_7bit_type1.setDisabled(False)
                self.OPF_7bit_type2.setDisabled(False)
                self.OPF_7bit_type3.setDisabled(False)
                self.proteinKNN.setDisabled(False)
            else:
                self.AAIndex.setDisabled(True)
                self.binary.setDisabled(True)
                self.BLOSUM62.setDisabled(True)
                self.EAAC.setDisabled(True)
                self.EGAAC.setDisabled(True)
                self.ZScale.setDisabled(True)
                self.binary_6bit.setDisabled(True)
                self.binary_5bit_type1.setDisabled(True)
                self.binary_5bit_type2.setDisabled(True)
                self.binary_3bit_type1.setDisabled(True)
                self.binary_3bit_type2.setDisabled(True)
                self.binary_3bit_type3.setDisabled(True)
                self.binary_3bit_type4.setDisabled(True)
                self.binary_3bit_type5.setDisabled(True)
                self.binary_3bit_type6.setDisabled(True)
                self.binary_3bit_type7.setDisabled(True)
                self.AESNN3.setDisabled(True)
                self.OPF_10bit.setDisabled(True)
                self.OPF_7bit_type1.setDisabled(True)
                self.OPF_7bit_type2.setDisabled(True)
                self.OPF_7bit_type3.setDisabled(True)
                self.proteinKNN.setDisabled(True)
        elif sequence_type == 'DNA':
            if status:
                self.ANF.setDisabled(False)
                self.DNAbinary.setDisabled(False)
                self.EIIP.setDisabled(False)
                self.ENAC.setDisabled(False)
                self.NCP.setDisabled(False)
                self.PSTNPds.setDisabled(False)
                self.PSTNPss.setDisabled(False)
                self.dnaPS2.setDisabled(False)
                self.dnaPS3.setDisabled(False)
                self.dnaPS4.setDisabled(False)
                self.dnaDBE.setDisabled(False)
                self.dnaLPDF.setDisabled(False)
                self.dnaDPCP2.setDisabled(False)
                self.dnaTPCP2.setDisabled(False)
                self.dnaKNN.setDisabled(False)
            else:
                self.ANF.setDisabled(True)
                self.DNAbinary.setDisabled(True)
                self.EIIP.setDisabled(True)
                self.ENAC.setDisabled(True)
                self.NCP.setDisabled(True)
                self.PSTNPds.setDisabled(True)
                self.PSTNPss.setDisabled(True)
                self.dnaPS2.setDisabled(True)
                self.dnaPS3.setDisabled(True)
                self.dnaPS4.setDisabled(True)
                self.dnaDBE.setDisabled(True)
                self.dnaLPDF.setDisabled(True)
                self.dnaDPCP2.setDisabled(True)
                self.dnaTPCP2.setDisabled(True)
                self.dnaKNN.setDisabled(True)
        elif sequence_type == 'RNA':
            if status:
                self.RNAANF.setDisabled(False)
                self.RNAbinary.setDisabled(False)
                self.RNAENAC.setDisabled(False)
                self.RNANCP.setDisabled(False)
                self.RNAPSTNPss.setDisabled(False)
                self.rnaPS2.setDisabled(False)
                self.rnaPS3.setDisabled(False)
                self.rnaPS4.setDisabled(False)
                self.rnaDBE.setDisabled(False)
                self.rnaLPDF.setDisabled(False)
                self.rnaDPCP2.setDisabled(False)
                self.rnaKNN.setDisabled(False)
            else:
                self.RNAANF.setDisabled(True)
                self.RNAbinary.setDisabled(True)
                self.RNAENAC.setDisabled(True)
                self.RNANCP.setDisabled(True)
                self.RNAPSTNPss.setDisabled(True)
                self.rnaPS2.setDisabled(True)
                self.rnaPS3.setDisabled(True)
                self.rnaPS4.setDisabled(True)
                self.rnaDBE.setDisabled(True)
                self.rnaLPDF.setDisabled(True)
                self.rnaDPCP2.setDisabled(True)
                self.rnaKNN.setDisabled(True)

    def load_fasta(self):
        try:
            self.panel_clear()
            self.desc_selected.clear()
            self.fasta_file, ok = QFileDialog.getOpenFileName(self, 'Open', './data', 'Plan text (*.*)')
            self.file_lineEdit.setText(self.fasta_file)
            if ok and self.fasta_file is not None and os.path.exists(self.fasta_file):
                self.status_label.setText('Open file ' + self.fasta_file)
                self.descriptor = FileProcessing.Descriptor(self.fasta_file, self.desc_default_para)
                self.logTextEdit.append('%s\tOpen file %s' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), self.fasta_file))
                self.logTextEdit.append('%s\tSequence type: %s' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), self.descriptor.sequence_type))
                self.logTextEdit.append('%s\tNumber of sequences: %s' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), self.descriptor.sequence_number))
                if self.descriptor.error_msg != '':
                    self.logTextEdit.append('%s\tError: %s' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), self.descriptor.error_msg))
                    QMessageBox.critical(self, 'Error', str(self.descriptor.error_msg), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
                else:
                    if self.descriptor.sequence_type == 'Protein':
                        self.Protein.setDisabled(False)
                        self.DNA.setDisabled(True)
                        self.RNA.setDisabled(True)
                        self.Protein.setExpanded(True)
                        self.DNA.setExpanded(False)
                        self.RNA.setExpanded(False)
                        self.status_label.setText('Sequence type is Protein.')
                    elif self.descriptor.sequence_type == 'DNA':
                        self.Protein.setDisabled(True)
                        self.DNA.setDisabled(False)
                        self.RNA.setDisabled(True)
                        self.Protein.setExpanded(False)
                        self.DNA.setExpanded(True)
                        self.RNA.setExpanded(False)
                        self.status_label.setText('Sequence type is DNA.')
                    elif self.descriptor.sequence_type == 'RNA':
                        self.DNA.setDisabled(True)
                        self.Protein.setDisabled(True)
                        self.RNA.setDisabled(False)
                        self.Protein.setExpanded(False)
                        self.DNA.setExpanded(False)
                        self.RNA.setExpanded(True)
                        self.status_label.setText('Sequence type is RNA.')
                    """ forbidden some descriptors """
                    self.descriptor_length_check(self.descriptor.sequence_type, self.descriptor.sequence_with_equal_length()[0])
            else:
                self.logTextEdit.append('%s\tOpen file failed.' %datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                self.Protein.setDisabled(True)
                self.DNA.setDisabled(True)
                self.RNA.setDisabled(True)
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

    def desc_tree_clicked(self, index):
        item = self.desc_treeWidget.currentItem()   # item = None if currentItem() is disabled
        if item and item.text(0) not in ['Protein', 'DNA', 'RNA']:
            # parameters with "sliding_window"
            if item.text(0) in ['EAAC', 'EGAAC', 'ENAC']:
                num, ok = QInputDialog.getInt(self, '%s setting' %item.text(0), 'Sliding window size', 5, 2, 10, 1)
                if ok:
                    self.para_dict[item.text(0)]['sliding_window'] = num
                    self.logTextEdit.append('%s\t%s coding setting (sliding_window_size=%s)' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), num))
            # parameter with "kspace"
            elif item.text(0) in ['CKSAAP', 'CKSAAGP', 'KSCTriad', 'CKSNAP']:
                num, ok = QInputDialog.getInt(self, '%s setting' %item.text(0), 'K-space number', 3, 0, 5, 1)
                if ok:
                    self.para_dict[item.text(0)]['kspace'] = num
                    self.logTextEdit.append('%s\t%s coding setting (k_space_size=%s)' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), num))
            elif item.text(0) in ['SOCNumber']:
                num, ok = QInputDialog.getInt(self, '%s setting' %item.text(0), 'lag value', 3, 1, self.descriptor.minimum_length_without_minus - 1, 1)
                if ok:
                    self.para_dict[item.text(0)]['nlag'] = num
                    self.logTextEdit.append('%s\t%s coding setting (n_lag=%s)' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), num))
            elif item.text(0) in ['QSOrder']:
                lag, weight, ok, = InputDialog.QSOrderInput.getValues(self.descriptor.minimum_length_without_minus - 1)
                if ok:
                    self.para_dict[item.text(0)]['nlag'] = lag
                    self.para_dict[item.text(0)]['weight'] = weight
                    self.logTextEdit.append('%s\t%s coding setting (n_lag=%s; weight=%s)' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), lag, weight))
            elif item.text(0) in ['AAIndex']:
                property, ok = InputDialog.QAAindexInput.getValues()
                if ok:
                    self.para_dict[item.text(0)]['aaindex'] = property
                    self.logTextEdit.append('%s\t%s coding setting (AAindex=%s)' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), property))
            elif item.text(0) in ['NMBroto', 'Moran', 'Geary', 'AC', 'CC', 'ACC']:
                if item.parent().text(0) == 'Protein':
                    lag, property, ok = InputDialog.QAAindex2Input.getValues(self.descriptor.minimum_length_without_minus - 1)
                    if ok:
                        self.para_dict[item.text(0)]['aaindex'] = property
                        self.para_dict[item.text(0)]['nlag'] = lag
                        self.logTextEdit.append('%s\t%s coding setting (n_lag=%s; AAindex=%s)' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), lag, property))
                if item.parent().text(0) == 'DNA':
                    num, property, ok = InputDialog.QDNAACC2Input.getValues(self.descriptor.minimum_length_without_minus - 2)
                    if ok:
                        self.para_dict[item.text(0)]['nlag'] = num
                        self.para_dict[item.text(0)]['Di-DNA-Phychem'] = property
                        self.logTextEdit.append('%s\t%s coding setting (n_lag=%s; Di-DNA-Phychem=%s)' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), num, property))
                if item.parent().text(0) == 'RNA':
                    num, property, ok = InputDialog.QRNAACC2Input.getValues(self.descriptor.minimum_length_without_minus - 2)
                    if ok:
                        self.para_dict[item.text(0)]['nlag'] = num
                        self.para_dict[item.text(0)]['Di-RNA-Phychem'] = property
                        self.logTextEdit.append('%s\t%s coding setting (n_lag=%s; Di-RNA-Phychem=%s)' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), num, property))
            elif item.text(0) in ['PAAC', 'APAAC']:
                lambdaValue, weight, ok, = InputDialog.QPAACInput.getValues(self.descriptor.minimum_length_without_minus - 1)
                if ok:
                    self.para_dict[item.text(0)]['lambdaValue'] = lambdaValue
                    self.para_dict[item.text(0)]['weight'] = weight
                    self.logTextEdit.append('%s\t%s coding setting (lambda=%s; weight=%s)' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), lambdaValue, weight))
            elif item.text(0) in ['PseKRAAC type 1', 'PseKRAAC type 2', 'PseKRAAC type 3A', 'PseKRAAC type 3B', 'PseKRAAC type 5',
                                  'PseKRAAC type 6A', 'PseKRAAC type 6B', 'PseKRAAC type 6C', 'PseKRAAC type 7', 'PseKRAAC type 8',
                                  'PseKRAAC type 9', 'PseKRAAC type 10', 'PseKRAAC type 11', 'PseKRAAC type 12', 'PseKRAAC type 13',
                                  'PseKRAAC type 14', 'PseKRAAC type 15', 'PseKRAAC type 16']:
                model, gap, lambdaValue, ktuple, clust, ok = InputDialog.QPseKRAACInput.getValues(item.text(0))
                if ok:
                    text = 'Model: %s ' %model
                    if model == 'g-gap':
                        text += 'g-gap: %s k-tuple: %s' %(gap, ktuple)
                    else:
                        text += 'lambda: %s k-tuple: %s' %(lambdaValue, ktuple)
                    self.para_dict[item.text(0)]['PseKRAAC_model'] = model
                    self.para_dict[item.text(0)]['g-gap'] = int(gap)
                    self.para_dict[item.text(0)]['lambdaValue'] = int(lambdaValue)
                    self.para_dict[item.text(0)]['k-tuple'] = int(ktuple)
                    self.para_dict[item.text(0)]['RAAC_clust'] = int(clust)
                    if model == 'g-gap':
                        self.logTextEdit.append('%s\t%s coding setting (Model=%s; g-gap=%s; k-tuple=%s; RAAC_clust=%s)' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), model, gap, ktuple, clust))
                    else:
                        self.logTextEdit.append('%s\t%s coding setting (Model=%s; lambda=%s; k-tuple=%s; RAAC_clust=%s)' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), model, lambdaValue, ktuple, clust))
            elif item.text(0) in ['Kmer', 'RCKmer']:
                num, ok = QInputDialog.getInt(self, '%s setting' % item.text(0), 'Kmer size', 3, 1, 6, 1)
                if ok:
                    self.para_dict[item.text(0)]['kmer'] = num
                    self.logTextEdit.append('%s\t%s coding setting (kmer=%s)' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), num))
            elif item.text(0) in ['DPCP', 'DPCP type2']:
                if item.parent().text(0) == 'DNA':
                    property, ok = InputDialog.QDNADPCPInput.getValues()
                    if ok:
                        self.para_dict[item.text(0)]['Di-DNA-Phychem'] = property
                        self.logTextEdit.append('%s\t%s coding setting (Di-DNA-Phychem=%s)' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), property))
                if item.parent().text(0) == 'RNA':
                    property, ok = InputDialog.QRNADPCPInput.getValues()
                    if ok:
                        self.para_dict[item.text(0)]['Di-RNA-Phychem'] = property
                        self.logTextEdit.append('%s\t%s coding setting (Di-RNA-Phychem=%s)' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), property))
            elif item.text(0) in ['TPCP', 'TPCP type2']:
                if item.parent().text(0) == 'DNA':
                    property, ok = InputDialog.QDNATPCPInput.getValues()
                    if ok:
                        self.para_dict[item.text(0)]['Tri-DNA-Phychem'] = property
                        self.logTextEdit.append('%s\t%s coding setting (Tri-DNA-Phychem=%s)' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), property))
            elif item.text(0) in ['DAC', 'DCC', 'DACC']:
                if self.descriptor.sequence_type == 'DNA':
                    num, property, ok = InputDialog.QDNAACC2Input.getValues(self.descriptor.minimum_length_without_minus - 2)
                    if ok:
                        self.para_dict[item.text(0)]['nlag'] = num
                        self.para_dict[item.text(0)]['Di-DNA-Phychem'] = property
                        self.logTextEdit.append('%s\t%s coding setting (Di-DNA-Phychem=%s; n_lag=%s)' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), property, num))
                else:
                    num, property, ok = InputDialog.QRNAACC2Input.getValues(self.descriptor.minimum_length_without_minus - 2)
                    if ok:
                        self.para_dict[item.text(0)]['nlag'] = num
                        self.para_dict[item.text(0)]['Di-RNA-Phychem'] = property
                        self.logTextEdit.append('%s\t%s coding setting (Di-RNA-Phychem=%s; n_lag=%s)' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), property, num))
            elif item.text(0) in ['TAC', 'TCC', 'TACC']:
                if self.descriptor.sequence_type == 'DNA':
                    num, property, ok = InputDialog.QDNAACC3Input.getValues(self.descriptor.minimum_length_without_minus - 3)
                    if ok:
                        self.para_dict[item.text(0)]['nlag'] = num
                        self.para_dict[item.text(0)]['Tri-DNA-Phychem'] = property
                        self.logTextEdit.append('%s\t%s coding setting (Tri-DNA-Phychem=%s; n_lag=%s)' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), property, num))
            elif item.text(0) in ['PseDNC', 'PCPseDNC', 'SCPseDNC']:
                if self.descriptor.sequence_type == 'DNA':
                    num, weight, property, ok = InputDialog.QDNAPse2Input.getValues(self.descriptor.minimum_length_without_minus - 2)
                    if ok:
                        self.para_dict[item.text(0)]['lambdaValue'] = num
                        self.para_dict[item.text(0)]['weight'] = weight
                        self.para_dict[item.text(0)]['Di-DNA-Phychem'] = property
                        self.logTextEdit.append('%s\t%s coding setting (Di-DNA-Phychem=%s; lambda=%s; weight=%s)' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), property, num, weight))
                else:
                    num, weight, property, ok = InputDialog.QRNAPse2Input.getValues(self.descriptor.minimum_length_without_minus - 2)
                    if ok:
                        self.para_dict[item.text(0)]['lambdaValue'] = num
                        self.para_dict[item.text(0)]['weight'] = weight
                        self.para_dict[item.text(0)]['Di-RNA-Phychem'] = property
                        self.logTextEdit.append('%s\t%s coding setting (Di-RNA-Phychem=%s; lambda=%s; weight=%s)' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), property, num, weight))
            elif item.text(0) in ['PCPseTNC', 'SCPseTNC']:
                if self.descriptor.sequence_type == 'DNA':
                    num, weight, property, ok = InputDialog.QDNAPse3Input.getValues(self.descriptor.minimum_length_without_minus - 3)
                    if ok:
                        self.para_dict[item.text(0)]['lambdaValue'] = num
                        self.para_dict[item.text(0)]['weight'] = weight
                        self.para_dict[item.text(0)]['Di-DNA-Phychem'] = property
                        self.logTextEdit.append('%s\t%s coding setting (Di-DNA-Phychem=%s; lambda=%s; weight=%s)' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), property, num, weight))
            elif item.text(0) in ['PseKNC']:
                if self.descriptor.sequence_type == 'DNA':
                    num, weight, kmer, property, ok = InputDialog.QDNAPseKNCInput.getValues(self.descriptor.minimum_length_without_minus - 2)
                    if ok:
                        self.para_dict[item.text(0)]['lambdaValue'] = num
                        self.para_dict[item.text(0)]['weight'] = weight
                        self.para_dict[item.text(0)]['kmer'] = kmer
                        self.para_dict[item.text(0)]['Di-DNA-Phychem'] = property
                        self.logTextEdit.append('%s\t%s coding setting (Di-RNA-Phychem=%s; lambda=%s; weight=%s; kmer=%s)' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), property, num, weight, kmer))
                else:
                    num, weight, kmer, property, ok = InputDialog.QRNAPseKNCInput.getValues(self.descriptor.minimum_length_without_minus - 2)
                    if ok:
                        self.para_dict[item.text(0)]['lambdaValue'] = num
                        self.para_dict[item.text(0)]['weight'] = weight
                        self.para_dict[item.text(0)]['kmer'] = kmer
                        self.para_dict[item.text(0)]['Di-RNA-Phychem'] = property
                        self.logTextEdit.append('%s\t%s coding setting (Di-RNA-Phychem=%s; lambda=%s; weight=%s; kmer=%s)' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), property, num, weight, kmer))
            elif item.text(0) in ['Mismatch']:
                num_k, num_m, ok = InputDialog.QMismatchInput.getValues()
                if ok:
                    if num_m >= num_k:
                        num_m = num_k - 1
                    self.para_dict[item.text(0)]['kmer'] = num_k
                    self.para_dict[item.text(0)]['mismatch'] = num_m
                    self.logTextEdit.append('%s\t%s coding setting (Kmer size=%s; Mismatch=%s)' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), num_k, num_m))
            elif item.text(0) in ['Subsequence']:
                num, delta, ok = InputDialog.QSubsequenceInput.getValues()
                if ok:
                    self.para_dict[item.text(0)]['kmer'] = num
                    self.para_dict[item.text(0)]['delta'] = delta
                    self.logTextEdit.append('%s\t%s coding setting (Kmer size=%s; delta=%s)' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), num, delta))
            elif item.text(0) in ['DistancePair']:
                num, cp, ok = InputDialog.QDistancePairInput.getValues()
                if ok:
                    self.para_dict[item.text(0)]['distance'] = num
                    self.para_dict[item.text(0)]['cp'] = cp
                    self.logTextEdit.append('%s\t%s coding setting (distance size=%s; cp=%s)' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), num, cp))
            else:
                pass

    def desc_tree_checkState(self, item, column):
        if self.descriptor is not None and self.descriptor.sequence_type in ['Protein', 'DNA', 'RNA'] and item and item.text(0) not in ['Protein', 'DNA', 'RNA']:
            self.desc_treeWidget.setCurrentItem(item)
            if self.descriptor.sequence_type == 'Protein':

                if item.parent().text(0) == 'Protein':
                    if item.checkState(column) == Qt.Checked:
                        self.desc_selected.add(item.text(0))
                    if item.checkState(column) == Qt.Unchecked:
                        self.desc_selected.discard(item.text(0))
                else:
                    item.setCheckState(0, Qt.Unchecked)
            elif self.descriptor.sequence_type == 'DNA':
                if item.parent().text(0) == 'DNA':
                    if item.checkState(column) == Qt.Checked:
                        self.desc_selected.add(item.text(0))
                    if item.checkState(column) == Qt.Unchecked:
                        self.desc_selected.discard(item.text(0))
                else:
                    item.setCheckState(0, Qt.Unchecked)
            elif self.descriptor.sequence_type == 'RNA':
                if item.parent().text(0) == 'RNA':
                    if item.checkState(column) == Qt.Checked:
                        self.desc_selected.add(item.text(0))
                    if item.checkState(column) == Qt.Unchecked:
                        self.desc_selected.discard(item.text(0))
                else:
                    item.setCheckState(0, Qt.Unchecked)

    def ml_tree_clicked(self, index):
        item = self.ml_treeWidget.currentItem()
        if item.text(0) in ['RF']:
            self.MLAlgorithm = item.text(0)
            num, range, cpu, auto, ok = InputDialog.QRandomForestInput.getValues()
            if ok:
                self.ml_defatult_para['n_trees'] = num
                self.ml_defatult_para['tree_range'] = range
                self.ml_defatult_para['auto'] = auto
                self.ml_defatult_para['cpu'] = cpu
                if auto:
                    self.logTextEdit.append('%s\tAlgorithm: %s\t[Auto-optimization\ttree_range=%s]' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), str(range)))
                else:
                    self.logTextEdit.append('%s\tAlgorithm: %s\t[tree number=%s]' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), num))
        elif item.text(0) in ['SVM']:
            self.MLAlgorithm = item.text(0)
            kernel, penality, gamma, auto, penalityRange, gammaRange, ok = InputDialog.QSupportVectorMachineInput.getValues()
            if ok:
                self.ml_defatult_para['kernel'] = kernel
                self.ml_defatult_para['penality'] = penality
                self.ml_defatult_para['gamma'] = gamma
                self.ml_defatult_para['auto'] = auto
                self.ml_defatult_para['penalityRange'] = penalityRange
                self.ml_defatult_para['gammaRange'] = gammaRange
                if auto:
                    self.logTextEdit.append('%s\tAlgorithm: %s\t[kernel=%s\tAuto-optimization\tpenality range=%s\tgamma range=%s]' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), kernel, str(penalityRange), str(gammaRange)))
                else:
                    self.logTextEdit.append('%s\tAlgorithm: %s\t[kernel=%s\tpenality=%s\tgamma=%s]' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), kernel, penality, gamma))
        elif item.text(0) in ['MLP']:
            self.MLAlgorithm = item.text(0)
            layer, epochs, activation, optimizer, ok = InputDialog.QMultiLayerPerceptronInput.getValues()
            if ok:
                self.ml_defatult_para['layer'] = layer
                self.ml_defatult_para['epochs'] = epochs
                self.ml_defatult_para['activation'] = activation
                self.ml_defatult_para['optimizer'] = optimizer
                self.logTextEdit.append('%s\tAlgorithm: %s\t[layer=%s\tepochs=%s\tactivation function=%s\toptimizer=%s]' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), layer, epochs, activation, optimizer))
        elif item.text(0) in ['LR', 'SGD', 'DecisionTree', 'NaiveBayes', 'AdaBoost', 'GBDT', 'LDA', 'QDA']:
            self.MLAlgorithm = item.text(0)
            self.logTextEdit.append('%s\tAlgorithm: %s' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0)))
        elif item.text(0) in ['KNN']:
            self.MLAlgorithm = item.text(0)
            topKValue, ok = InputDialog.QKNeighborsInput.getValues()
            if ok:
                self.ml_defatult_para['topKValue'] = topKValue
                self.logTextEdit.append('%s\tAlgorithm: %s\t[top K value=%s]' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), topKValue))
        elif item.text(0) in ['LightGBM']:
            self.MLAlgorithm = item.text(0)
            type, leaves, depth, rate, leavesRange, depthRange, rateRange, threads, auto, ok = InputDialog.QLightGBMInput.getValues()
            if ok:
                self.ml_defatult_para['boosting_type'] = type
                self.ml_defatult_para['num_leaves'] = leaves
                self.ml_defatult_para['max_depth'] = depth
                self.ml_defatult_para['learning_rate'] = rate
                self.ml_defatult_para['auto'] = auto
                self.ml_defatult_para['leaves_range'] = leavesRange
                self.ml_defatult_para['depth_range'] = depthRange
                self.ml_defatult_para['rate_range'] = rateRange
                self.ml_defatult_para['cpu'] = threads
                if auto:
                    self.logTextEdit.append('%s\tAlgorithm: %s\t[Auto optimization\tLeaves range=%s\tDepth range=%s\tLearning rate range=%s]' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), leavesRange, depthRange, rateRange))
                else:
                    self.logTextEdit.append('%s\tAlgorithm: %s\t[boosting type=%s\tnumber of leaves=%s\tmax depth=%s\tlearning rate=%s\tnumber of threads=%s]' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), type, leaves, depth, rate, threads))
        elif item.text(0) in ['XGBoost']:
            self.MLAlgorithm = item.text(0)
            booster, maxdepth, rate, estimator, colsample, depthRange, rateRange, threads, auto, ok = InputDialog.QXGBoostInput.getValues()
            self.ml_defatult_para['booster'] = booster
            self.ml_defatult_para['max_depth'] = maxdepth
            self.ml_defatult_para['learning_rate'] = rate
            self.ml_defatult_para['n_estimator'] = estimator
            self.ml_defatult_para['colsample_bytree'] = colsample
            self.ml_defatult_para['depth_range'] = depthRange
            self.ml_defatult_para['rate_range'] = rateRange
            self.ml_defatult_para['cpu'] = threads
            self.ml_defatult_para['auto'] = auto
            if auto:
                self.logTextEdit.append('%s\tAlgorithm: %s\t[Auto optimization\tDepth range=%s\tLearning rate range=%s]' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), depthRange, rateRange))
            else:
                self.logTextEdit.append('%s\tAlgorithm: %s\t[Booster=%s\tMax depth=%s\tLearning rate=%s\tn_estimator=%s\tcolsample_bytree=%s]' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), booster, maxdepth, rate, estimator, colsample))
        elif item.text(0) in ['Bagging']:
            self.MLAlgorithm = item.text(0)
            n_estimators, threads, ok = InputDialog.QBaggingInput.getValues()
            if ok:
                self.ml_defatult_para['n_estimator'] = n_estimators
                self.ml_defatult_para['cpu'] = threads
                self.logTextEdit.append('%s\tAlgorithm: %s\t[n_estimators=%s\tThreads=%s]' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), n_estimators, threads))
        # elif item.text(0) in ['Net_1_CNN']:
        #     self.MLAlgorithm = item.text(0)
        #     if not self.MLData is None:
        #         input_channel, input_length, output_channel, padding, kernel_size, dropout, learning_rate, epochs, early_stopping, batch_size, fc_size, ok = InputDialog.QNetInput_1.getValues(self.MLData.training_dataframe.values.shape[1])
        #         if ok:
        #             self.ml_defatult_para['input_channel'] = input_channel
        #             self.ml_defatult_para['input_length'] = input_length
        #             self.ml_defatult_para['output_channel'] = output_channel
        #             self.ml_defatult_para['padding'] = padding
        #             self.ml_defatult_para['kernel_size'] = kernel_size
        #             self.ml_defatult_para['dropout'] = dropout
        #             self.ml_defatult_para['learning_rate'] = learning_rate
        #             self.ml_defatult_para['epochs'] = epochs
        #             self.ml_defatult_para['early_stopping'] = early_stopping
        #             self.ml_defatult_para['batch_size'] = batch_size
        #             self.ml_defatult_para['fc_size'] = fc_size
        #             self.logTextEdit.append('%s\tAlgorithm: %s; Input channel=%s; Input_length=%s; Output_channel=%s; Padding=%s; Kernel_size=%s; Dropout=%s; lr=%s; Epochs=%s; Early_stopping=%s; Batch_size=%s' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), input_channel, input_length, output_channel, padding, kernel_size, dropout, learning_rate, epochs, early_stopping, batch_size))
        #             # self.ml_para_lineEdit.setText('Input channel=%s; Input_length=%s; Output_channel=%s; Padding=%s; Kernel_size=%s; Dropout=%s; lr=%s; Epochs=%s; Early_stopping=%s; Batch_size=%s' %(input_channel, input_length, output_channel, padding, kernel_size, dropout, learning_rate, epochs, early_stopping, batch_size))
        #     else:
        #         QMessageBox.warning(self, 'Warning', 'Please input training data at first!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        # elif item.text(0) in ['Net_2_RNN']:
        #     self.MLAlgorithm = item.text(0)
        #     if not self.MLData is None:
        #         input_channel, input_length, hidden_size, num_layers, dropout, learning_rate, epochs, early_stopping, batch_size, fc_size, ok = InputDialog.QNetInput_2.getValues(self.MLData.training_dataframe.values.shape[1])
        #         if ok:
        #             self.ml_defatult_para['input_channel'] = input_channel
        #             self.ml_defatult_para['input_length'] = input_length
        #             self.ml_defatult_para['rnn_hidden_size'] = hidden_size
        #             self.ml_defatult_para['rnn_hidden_layers'] = num_layers
        #             self.ml_defatult_para['rnn_bidirection'] = False
        #             self.ml_defatult_para['dropout'] = dropout
        #             self.ml_defatult_para['learning_rate'] = learning_rate
        #             self.ml_defatult_para['epochs'] = epochs
        #             self.ml_defatult_para['early_stopping'] = early_stopping
        #             self.ml_defatult_para['batch_size'] = batch_size
        #             self.ml_defatult_para['rnn_bidirectional'] = False
        #             self.logTextEdit.append('%s\tAlgorithm: %s; Input size=%s; Input_length=%s; Hidden_size=%s; Num_hidden_layers=%s; Dropout=%s; lr=%s; Epochs=%s; Early_stopping=%s; Batch_size=%s' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), input_channel, input_length, hidden_size, num_layers, dropout, learning_rate, epochs, early_stopping, batch_size))
        #             # self.ml_para_lineEdit.setText('Input size=%s; Input_length=%s; Hidden_size=%s; Num_hidden_layers=%s; Dropout=%s; lr=%s; Epochs=%s; Early_stopping=%s; Batch_size=%s' %(input_channel, input_length, hidden_size, num_layers, dropout, learning_rate, epochs, early_stopping, batch_size))
        #     else:
        #         QMessageBox.warning(self, 'Warning', 'Please input training data at first!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        # elif item.text(0) in ['Net_3_BRNN']:
        #     self.MLAlgorithm = item.text(0)
        #     if not self.MLData is None:
        #         input_channel, input_length, hidden_size, num_layers, dropout, learning_rate, epochs, early_stopping, batch_size, fc_size, ok = InputDialog.QNetInput_2.getValues(self.MLData.training_dataframe.values.shape[1])
        #         if ok:
        #             self.ml_defatult_para['input_channel'] = input_channel
        #             self.ml_defatult_para['input_length'] = input_length
        #             self.ml_defatult_para['rnn_hidden_size'] = hidden_size
        #             self.ml_defatult_para['rnn_hidden_layers'] = num_layers
        #             self.ml_defatult_para['rnn_bidirection'] = False
        #             self.ml_defatult_para['dropout'] = dropout
        #             self.ml_defatult_para['learning_rate'] = learning_rate
        #             self.ml_defatult_para['epochs'] = epochs
        #             self.ml_defatult_para['early_stopping'] = early_stopping
        #             self.ml_defatult_para['batch_size'] = batch_size
        #             self.ml_defatult_para['rnn_bidirectional'] = True
        #             self.logTextEdit.append('%s\tAlgorithm: %s; Input size=%s; Input_length=%s; Hidden_size=%s; Num_hidden_layers=%s; Dropout=%s; lr=%s; Epochs=%s; Early_stopping=%s; Batch_size=%s' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), input_channel, input_length, hidden_size, num_layers, dropout, learning_rate, epochs, early_stopping, batch_size))
        #             # self.ml_para_lineEdit.setText('Input size=%s; Input_length=%s; Hidden_size=%s; Num_hidden_layers=%s; Dropout=%s; lr=%s; Epochs=%s; Early_stopping=%s; Batch_size=%s' %(input_channel, input_length, hidden_size, num_layers, dropout, learning_rate, epochs, early_stopping, batch_size))
        #     else:
        #         QMessageBox.warning(self, 'Warning', 'Please input training data at first!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        # elif item.text(0) in ['Net_4_ABCNN']:
        #     self.MLAlgorithm = item.text(0)
        #     if not self.MLData is None:
        #         input_channel, input_length, dropout, learning_rate, epochs, early_stopping, batch_size, ok = InputDialog.QNetInput_4.getValues(self.MLData.training_dataframe.values.shape[1])
        #         if ok:
        #             self.ml_defatult_para['input_channel'] = input_channel
        #             self.ml_defatult_para['input_length'] = input_length
        #             self.ml_defatult_para['dropout'] = dropout
        #             self.ml_defatult_para['learning_rate'] = learning_rate
        #             self.ml_defatult_para['epochs'] = epochs
        #             self.ml_defatult_para['early_stopping'] = early_stopping
        #             self.ml_defatult_para['batch_size'] = batch_size
        #             self.logTextEdit.append('%s\tAlgorithm: %s; Input size=%s; Input_length=%s; Dropout=%s; lr=%s; Epochs=%s; Early_stopping=%s; Batch_size=%s' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), input_channel, input_length, dropout, learning_rate, epochs, early_stopping, batch_size))
        #             # self.ml_para_lineEdit.setText('Input size=%s; Input_length=%s; Dropout=%s; lr=%s; Epochs=%s; Early_stopping=%s; Batch_size=%s' %(input_channel, input_length, dropout, learning_rate, epochs, early_stopping, batch_size))
        #     else:
        #         QMessageBox.warning(self, 'Warning', 'Please input training data at first!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        # elif item.text(0) in ['Net_5_ResNet']:
        #     self.MLAlgorithm = item.text(0)
        #     if not self.MLData is None:
        #         input_channel, input_length, learning_rate, epochs, early_stopping, batch_size, ok = InputDialog.QNetInput_5.getValues(self.MLData.training_dataframe.values.shape[1])
        #         if ok:
        #             self.ml_defatult_para['input_channel'] = input_channel
        #             self.ml_defatult_para['input_length'] = input_length
        #             self.ml_defatult_para['learning_rate'] = learning_rate
        #             self.ml_defatult_para['epochs'] = epochs
        #             self.ml_defatult_para['early_stopping'] = early_stopping
        #             self.ml_defatult_para['batch_size'] = batch_size
        #             self.logTextEdit.append('%s\tAlgorithm: %s; Input size=%s; Input_length=%s; lr=%s; Epochs=%s; Early_stopping=%s; Batch_size=%s'  %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), input_channel, input_length, learning_rate, epochs, early_stopping, batch_size))
        #             # self.ml_para_lineEdit.setText('Input size=%s; Input_length=%s; lr=%s; Epochs=%s; Early_stopping=%s; Batch_size=%s' %(input_channel, input_length, learning_rate, epochs, early_stopping, batch_size))
        #     else:
        #         QMessageBox.warning(self, 'Warning', 'Please input training data at first!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        # elif item.text(0) in ['Net_6_AE']:
        #     self.MLAlgorithm = item.text(0)
        #     if not self.MLData is None:
        #         input_dim, dropout, learning_rate, epochs, early_stopping, batch_size, ok = InputDialog.QNetInput_6.getValues(self.MLData.training_dataframe.values.shape[1])
        #         if ok:
        #             self.ml_defatult_para['mlp_input_dim'] = input_dim
        #             self.ml_defatult_para['dropout'] = dropout
        #             self.ml_defatult_para['learning_rate'] = learning_rate
        #             self.ml_defatult_para['epochs'] = epochs
        #             self.ml_defatult_para['early_stopping'] = early_stopping
        #             self.ml_defatult_para['batch_size'] = batch_size
        #             self.logTextEdit.append('%s\tAlgorithm: %s; Input dimension=%s; Dropout=%s; lr=%s; Epochs=%s; Early_stopping=%s; Batch_size=%s' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item.text(0), input_dim, dropout, learning_rate, epochs, early_stopping, batch_size))
        #             # self.ml_para_lineEdit.setText('Input dimension=%s; Dropout=%s; lr=%s; Epochs=%s; Early_stopping=%s; Batch_size=%s' %(input_dim, dropout, learning_rate, epochs, early_stopping, batch_size))
        #     else:
        #         QMessageBox.warning(self, 'Warning', 'Please input training data at first!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

    def setFold(self):
        fold, ok = QInputDialog.getInt(self, 'Fold number', 'Setting K-fold cross-validation', 5, 2, 100, 1)
        if ok:
            self.fold_lineEdit.setText(str(fold))
            self.ml_defatult_para['FOLD'] = fold
            self.logTextEdit.append('%s\tSet %s-fold Cross-validation' %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), fold))

    def run_estimator(self):
        self.restart_init()
        if self.descriptor is None:
            QMessageBox.critical(self, 'Error', 'Please load fasta file', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        elif self.descriptor.sequence_number < 100:
            QMessageBox.critical(self, 'Error', 'Sequence number need > 100.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        elif len(self.desc_selected) == 0:
            QMessageBox.critical(self, 'Error', 'Please at least one descriptor.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        elif self.MLAlgorithm is None:
            QMessageBox.critical(self, 'Error', 'Please select a machine learning algorithm.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        elif self.ml_defatult_para['FOLD'] <= 1:
            QMessageBox.critical(self, 'Error', 'FOLD need > 1', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        else:
            self.logTextEdit.append('%s\tJobs start ...' %datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            self.progress_bar.setMovie(self.gif)
            self.gif.start()
            t = threading.Thread(target=self.estimator)
            t.start()

    def estimator(self):
        self.start_button.setDisabled(True)
        self.setDisabled(True)
        for desc in self.desc_selected:
            # copy parameters for each descriptor
            if desc in self.para_dict:
                for key in self.para_dict[desc]:
                    self.desc_default_para[key] = self.para_dict[desc][key]

            start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.status_label.setText('Calculating descriptor %s' %desc)
            msg = '%s\tCalculating and training model for descriptor %s ...' %(start_time, desc)
            self.append_msg_signal.emit(msg)
            if self.descriptor.sequence_type == 'Protein':
                descriptor_name = re.sub(' ', '_', desc)
                cmd = 'self.descriptor.' + self.descriptor.sequence_type + '_' + descriptor_name + '()'
                status = eval(cmd)
            else:
                descriptor_name = re.sub(' ', '_', desc)
                if desc in ['DAC', 'TAC']:
                    my_property_name, my_property_value, my_kmer, ok = CheckAccPseParameter.check_acc_arguments(desc, self.descriptor.sequence_type, self.desc_default_para)
                    status = self.descriptor.make_ac_vector(my_property_name, my_property_value, my_kmer)
                elif desc in ['DCC', 'TCC']:
                    my_property_name, my_property_value, my_kmer, ok = CheckAccPseParameter.check_acc_arguments(desc, self.descriptor.sequence_type, self.desc_default_para)
                    status = self.descriptor.make_cc_vector(my_property_name, my_property_value, my_kmer)
                elif desc in ['DACC', 'TACC']:
                    my_property_name, my_property_value, my_kmer, ok = CheckAccPseParameter.check_acc_arguments(desc, self.descriptor.sequence_type, self.desc_default_para)
                    status = self.descriptor.make_acc_vector(my_property_name, my_property_value, my_kmer)
                elif desc in ['PseDNC', 'PseKNC', 'PCPseDNC', 'PCPseTNC', 'SCPseDNC', 'SCPseTNC']:
                    my_property_name, my_property_value, ok = CheckAccPseParameter.check_Pse_arguments(desc, self.descriptor.sequence_type, self.desc_default_para)
                    cmd = 'self.descriptor.' + desc + '(my_property_name, my_property_value)'
                    status = eval(cmd)
                else:
                    cmd = 'self.descriptor.' + descriptor_name + '()'
                    status = eval(cmd)
            end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if status:
                self.status_label.setText('Descriptor %s calculating complete.' %desc)
                msg = '%s\tDescriptor %s calculating complete.' %(end_time, desc)
                self.append_msg_signal.emit(msg)
                df = pd.DataFrame(self.descriptor.encoding_array[1:, 2:].astype(float), columns=self.descriptor.encoding_array[0, 2:])
                label = self.descriptor.encoding_array[1:, 1].astype(int)
                self.MLData = MachineLearning.ILearnMachineLearning(self.ml_defatult_para)
                self.MLData.import_training_data(df, label)
                self.status_label.setText('Start training model for descriptor %s ...' %desc)
                ok = self.train_model()
                if ok:
                    end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    self.status_label.setText('Training model for descriptor %s complete.' %desc)
                    data_list = ['%s_%s_model'%(desc, self.MLAlgorithm), start_time, end_time] + list(self.MLData.metrics.values[-1])
                    item = pd.DataFrame([data_list], columns=['Id', 'StartTime', 'EndTime', 'Sn', 'Sp', 'Pre', 'Acc', 'MCC', 'F1', 'AUROC', 'AUPRC'])
                    self.metrics.insert_data(item, '%s_%s_model' %(desc, self.MLAlgorithm), self.MLData.meanAucData, self.MLData.meanPrcData, self.MLData.training_score, self.MLData.best_model)
                    self.boxplot_data[desc] = self.MLData.metrics.iloc[0:self.ml_defatult_para['FOLD'], :]
                    self.metrics.classification_task = self.MLData.task
                    self.display_signal.emit(data_list)
                else:
                    self.status_label.setText('Training model for descriptor %s failed.' %desc)
        self.display_curves_signal.emit()

    def train_model(self):
        if self.MLAlgorithm == 'RF':
            ok = self.MLData.RandomForest()
        elif self.MLAlgorithm == 'SVM':
            ok = self.MLData.SupportVectorMachine()
        elif self.MLAlgorithm == 'MLP':
            ok = self.MLData.MultiLayerPerceptron()
        elif self.MLAlgorithm == 'LR':
            ok = self.MLData.LogisticRegressionClassifier()
        elif self.MLAlgorithm == 'LDA':
            ok = self.MLData.LDAClassifier()
        elif self.MLAlgorithm == 'QDA':
            ok = self.MLData.QDAClassifier()
        elif self.MLAlgorithm == 'KNN':
            ok = self.MLData.KNeighbors()
        elif self.MLAlgorithm == 'LightGBM':
            ok = self.MLData.LightGBMClassifier()
        elif self.MLAlgorithm == 'XGBoost':
            ok = self.MLData.XGBoostClassifier()
        elif self.MLAlgorithm == 'SGD':
            ok = self.MLData.StochasticGradientDescentClassifier()
        elif self.MLAlgorithm == 'DecisionTree':
            ok = self.MLData.DecisionTree()
        elif self.MLAlgorithm == 'NaiveBayes':
            ok = self.MLData.GaussianNBClassifier()
        elif self.MLAlgorithm == 'AdaBoost':
            ok = self.MLData.AdaBoost()
        elif self.MLAlgorithm == 'Bagging':
            ok = self.MLData.Bagging()
        elif self.MLAlgorithm == 'GBDT':
            ok = self.MLData.GBDTClassifier()
        # self.MLData.calculate_boxplot_data()
        return ok

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
            QMessageBox.critical(self, 'Error', str(e), QMessageBox.Ok | QMessageBox.No,
                                 QMessageBox.Ok)

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
                                joblib.dump(model, model_name)
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

        ml_combination_para = copy.deepcopy(self.ml_defatult_para)
        ml_combination_para['auto'] = True
        for c in combinations_array:
            self.status_label.setText('Training combined model for combination %s with %s method.' %(str(c), combineAlgorithm))
            msg = '%s\tTraining model for combinations %s with %s method ...' %(start_time, str(c), combineAlgorithm)
            self.append_msg_signal.emit(msg)
            # if self.metrics.classification_task == 'binary':
            #     ensemble_data_X = np.concatenate(tuple(self.metrics.prediction_scores[m].values[:, 3].reshape((-1, 1)) for m in c), axis=1)
                # ensemble_data_X = np.concatenate(tuple(self.metrics.prediction_scores[m].values[:, 2:] for m in c), axis=1)
            # else:
            #     ensemble_data_X = np.concatenate(tuple(self.metrics.prediction_scores[m].values[:, 2:] for m in c), axis=1)
            ensemble_data_X = np.concatenate(tuple(self.metrics.prediction_scores[m].values[:, 2:] for m in c), axis=1)
            ensemble_data_y = self.metrics.prediction_scores[c[0]].values[:, 1]            
            df = pd.DataFrame(ensemble_data_X.astype(float))
            label = ensemble_data_y.astype(int)
            self.MLData = MachineLearning.ILearnMachineLearning(ml_combination_para)
            self.MLData.import_training_data(df, label)
            
            if combineAlgorithm == 'RF':
                ok = self.MLData.RandomForest()
            elif combineAlgorithm == 'SVM':
                ok = self.MLData.SupportVectorMachine()
            elif combineAlgorithm == 'MLP':
                ok = self.MLData.MultiLayerPerceptron()
            elif combineAlgorithm == 'LR':
                ok = self.MLData.LogisticRegressionClassifier()
            elif combineAlgorithm == 'LDA':
                ok = self.MLData.LDAClassifier()
            elif combineAlgorithm == 'QDA':
                ok = self.MLData.QDAClassifier()
            elif combineAlgorithm == 'KNN':
                ok = self.MLData.KNeighbors()
            elif combineAlgorithm == 'LightGBM':
                ok = self.MLData.LightGBMClassifier()
            elif combineAlgorithm == 'XGBoost':
                ok = self.MLData.XGBoostClassifier()
            elif combineAlgorithm == 'SGD':
                ok = self.MLData.StochasticGradientDescentClassifier()
            elif combineAlgorithm == 'DecisionTree':
                ok = self.MLData.DecisionTree()
            elif combineAlgorithm == 'NaiveBayes':
                ok = self.MLData.GaussianNBClassifier()
            elif combineAlgorithm == 'AdaBoost':
                ok = self.MLData.AdaBoost()
            elif combineAlgorithm == 'Bagging':
                ok = self.MLData.Bagging()
            elif combineAlgorithm == 'GBDT':
                ok = self.MLData.GBDTClassifier()

            end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')            
            if ok:
                self.status_label.setText('Training combined model for combination %s complete.' %str(c))
                msg = '%s\tTraining combined model for combination %s complete.' %(end_time, str(c))
                self.append_msg_signal.emit(msg)
                if self.metrics.classification_task == 'binary':
                    if self.MLData.metrics.loc['Mean', 'AUROC'] > self.bestPerformance:
                        self.bestPerformance = self.MLData.metrics.loc['Mean', 'AUROC']                           
                        self.bestMetrics = self.MLData.metrics
                        self.bestAUC = self.MLData.meanAucData
                        self.bestPRC = self.MLData.meanPrcData
                        self.bestModels = self.MLData.best_model
                        self.bestTrainingScore = self.MLData.training_score
                        self.boxplot_data['Combined'] = self.MLData.metrics.iloc[0:self.ml_defatult_para['FOLD'], :]

                else:
                    if self.MLData.metrics.loc['Mean', 'AUROC'] > self.bestPerformance:
                        self.bestPerformance = self.MLData.metrics.loc['Mean', 'Acc']
                        self.bestMetrics = self.MLData.metrics
                        self.bestAUC = self.MLData.meanAucData
                        self.bestPRC = self.MLData.meanPrcData
                        self.bestModels = self.MLData.best_model
                        self.bestTrainingScore = self.MLData.training_score
                        self.boxplot_data['Combined'] = self.MLData.metrics.iloc[0:self.ml_defatult_para['FOLD'], :]

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
        reply = QMessageBox.question(self, 'Confirm Exit', 'Are you sure want to quit iLearnPlus?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close_signal.emit('Estimator')            
            self.close()
        else:
            if event:
                event.ignore()

    def showDescriptorSlims(self):
        if self.desc_slim_button.text() == 'Show descriptor slims':
            self.AAC.setHidden(True)
            self.DPC.setHidden(True)
            self.TPC.setHidden(True)
            self.binary_6bit.setHidden(True)
            self.binary_5bit_type1.setHidden(True)
            self.binary_5bit_type2.setHidden(True)
            self.binary_3bit_type1.setHidden(True)
            self.binary_3bit_type2.setHidden(True)
            self.binary_3bit_type3.setHidden(True)
            self.binary_3bit_type4.setHidden(True)
            self.binary_3bit_type5.setHidden(True)
            self.binary_3bit_type6.setHidden(True)
            self.binary_3bit_type7.setHidden(True)
            self.GAAC.setHidden(True)
            self.GDPC.setHidden(True)
            self.GTPC.setHidden(True)
            self.KSCTriad.setHidden(True)
            self.OPF_7bit_type1.setHidden(True)
            self.OPF_7bit_type2.setHidden(True)
            self.OPF_7bit_type3.setHidden(True)
            self.proteinAC.setHidden(True)
            self.proteinCC.setHidden(True)
            self.PseKRAAC_type2.setHidden(True)
            self.PseKRAAC_type3A.setHidden(True)
            self.PseKRAAC_type3B.setHidden(True)
            self.PseKRAAC_type4.setHidden(True)
            self.PseKRAAC_type5.setHidden(True)
            self.PseKRAAC_type6A.setHidden(True)
            self.PseKRAAC_type6B.setHidden(True)
            self.PseKRAAC_type6C.setHidden(True)
            self.PseKRAAC_type7.setHidden(True)
            self.PseKRAAC_type8.setHidden(True)
            self.PseKRAAC_type9.setHidden(True)
            self.PseKRAAC_type10.setHidden(True)
            self.PseKRAAC_type11.setHidden(True)
            self.PseKRAAC_type12.setHidden(True)
            self.PseKRAAC_type13.setHidden(True)
            self.PseKRAAC_type14.setHidden(True)
            self.PseKRAAC_type15.setHidden(True)
            self.PseKRAAC_type16.setHidden(True)

            # DNA descriptor control
            self.NAC.setHidden(True)
            self.dnaPS3.setHidden(True)
            self.dnaPS4.setHidden(True)
            self.dnaDPCP2.setHidden(True)
            self.dnaTPCP2.setHidden(True)
            self.dnazcurve12bit.setHidden(True)
            self.dnazcurve36bit.setHidden(True)
            self.dnazcurve48bit.setHidden(True)
            self.dnazcurve144bit.setHidden(True)
            self.DAC.setHidden(True)
            self.DCC.setHidden(True)
            self.TAC.setHidden(True)
            self.TCC.setHidden(True)

            # RNA descriptor control
            self.RNANAC.setHidden(True)
            self.rnaPS3.setHidden(True)
            self.rnaPS4.setHidden(True)
            self.rnaDPCP2.setHidden(True)
            self.rnazcurve12bit.setHidden(True)
            self.rnazcurve36bit.setHidden(True)
            self.rnazcurve48bit.setHidden(True)
            self.rnazcurve144bit.setHidden(True)
            self.RNADAC.setHidden(True)
            self.RNADCC.setHidden(True)

            self.desc_slim_button.setText('Show all descriptors')
        else:
            self.AAC.setHidden(False)
            self.DPC.setHidden(False)
            self.TPC.setHidden(False)
            self.binary_6bit.setHidden(False)
            self.binary_5bit_type1.setHidden(False)
            self.binary_5bit_type2.setHidden(False)
            self.binary_3bit_type1.setHidden(False)
            self.binary_3bit_type2.setHidden(False)
            self.binary_3bit_type3.setHidden(False)
            self.binary_3bit_type4.setHidden(False)
            self.binary_3bit_type5.setHidden(False)
            self.binary_3bit_type6.setHidden(False)
            self.binary_3bit_type7.setHidden(False)
            self.GAAC.setHidden(False)
            self.GDPC.setHidden(False)
            self.GTPC.setHidden(False)
            self.KSCTriad.setHidden(False)
            self.OPF_7bit_type1.setHidden(False)
            self.OPF_7bit_type2.setHidden(False)
            self.OPF_7bit_type3.setHidden(False)
            self.proteinAC.setHidden(False)
            self.proteinCC.setHidden(False)
            self.PseKRAAC_type2.setHidden(False)
            self.PseKRAAC_type3A.setHidden(False)
            self.PseKRAAC_type3B.setHidden(False)
            self.PseKRAAC_type4.setHidden(False)
            self.PseKRAAC_type5.setHidden(False)
            self.PseKRAAC_type6A.setHidden(False)
            self.PseKRAAC_type6B.setHidden(False)
            self.PseKRAAC_type6C.setHidden(False)
            self.PseKRAAC_type7.setHidden(False)
            self.PseKRAAC_type8.setHidden(False)
            self.PseKRAAC_type9.setHidden(False)
            self.PseKRAAC_type10.setHidden(False)
            self.PseKRAAC_type11.setHidden(False)
            self.PseKRAAC_type12.setHidden(False)
            self.PseKRAAC_type13.setHidden(False)
            self.PseKRAAC_type14.setHidden(False)
            self.PseKRAAC_type15.setHidden(False)
            self.PseKRAAC_type16.setHidden(False)

            # DNA descriptor
            self.NAC.setHidden(False)
            self.dnaPS3.setHidden(False)
            self.dnaPS4.setHidden(False)
            self.dnaDPCP2.setHidden(False)
            self.dnaTPCP2.setHidden(False)
            self.dnazcurve12bit.setHidden(False)
            self.dnazcurve36bit.setHidden(False)
            self.dnazcurve48bit.setHidden(False)
            self.dnazcurve144bit.setHidden(False)
            self.DAC.setHidden(False)
            self.DCC.setHidden(False)
            self.TAC.setHidden(False)
            self.TCC.setHidden(False)

            # RNA descriptor
            self.RNANAC.setHidden(False)
            self.rnaPS3.setHidden(False)
            self.rnaPS4.setHidden(False)
            self.rnaDPCP2.setHidden(False)
            self.rnazcurve12bit.setHidden(False)
            self.rnazcurve36bit.setHidden(False)
            self.rnazcurve48bit.setHidden(False)
            self.rnazcurve144bit.setHidden(False)
            self.RNADAC.setHidden(False)
            self.RNADCC.setHidden(False)
            
            self.desc_slim_button.setText('Show descriptor slims')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setFont(QFont('Arial', 10))
    win = ILearnPlusEstimator()
    win.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    win.show()
    sys.exit(app.exec_())

