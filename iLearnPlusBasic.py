#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import sys, os, re
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QFileDialog, QLabel, QHBoxLayout, QGroupBox, QTextEdit,
                             QVBoxLayout, QLineEdit, QTreeWidget, QTreeWidgetItem, QSplitter, QTableWidget, QTabWidget,
                             QTableWidgetItem, QInputDialog, QMessageBox, QFormLayout, QGridLayout, QRadioButton,
                             QHeaderView, QAbstractItemView, QLabel)
from PyQt5.QtGui import QIcon, QFont, QMovie
from PyQt5.QtCore import Qt, pyqtSignal
from util import (FileProcessing, DataAnalysis, InputDialog, CheckAccPseParameter, MachineLearning, TableWidget,
                  PlotWidgets)
import numpy as np
import pandas as pd
import threading
import qdarkstyle
import sip
import joblib
import torch
import copy

class ILearnPlusBasic(QTabWidget):
    desc_signal = pyqtSignal()
    clust_signal = pyqtSignal()
    selection_signal = pyqtSignal()
    ml_signal = pyqtSignal()
    close_signal = pyqtSignal(str)

    def __init__(self):
        super(ILearnPlusBasic, self).__init__()

        # signal
        self.desc_signal.connect(self.set_table_content)
        self.clust_signal.connect(self.display_data_analysis)
        self.selection_signal.connect(self.display_selection_data)
        self.ml_signal.connect(self.display_ml_data)

        # graph setting
        # pg.setConfigOption('background', '#FFFFFF')
        # status bar
        self.gif = QMovie('images/progress_bar.gif')

        # default variable (start with "desc")
        """ Descriptor variable """
        self.desc_fasta_file = ''  # fasta sequences file
        self.desc_seq_type = ''  # sequences type: DNA, RNA or Protein
        self.desc_selected_descriptor = ''  # descriptor
        self.descriptor = None  # coding
        self.desc_running_status = False
        self.desc_default_para = {  # default parameter for descriptors
            'sliding_window': 5,
            'kspace': 3,
            'props': ['CIDH920105', 'BHAR880101', 'CHAM820101', 'CHAM820102', 'CHOC760101', 'BIGC670101', 'CHAM810101',
                      'DAYM780201'],
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
        }

        """ Cluster Variable """
        self.cluster_file = ''
        self.clust_data = None
        self.clust_analysis_type = ''
        self.clust_selected_algorithm = ''
        self.clust_default_para = {
            'nclusters': 2,
            'n_components': 2,
            'expand_factor': 2,
            'inflate_factor': 2.0,
            'multiply_factor': 2.0
        }

        self.clust_status = False
        self.clust_symbol = {0: 'o', 1: 's', 2: 't', 3: '+', 4: 'p', 5: 't2', 6: 'h', 7: 't3', 8: 'star', 9: 't1',
                             10: 't2'}

        """ Feature Selection Variable  """
        self.selection_file = ''
        self.selection_data = None
        self.selection_analysis_type = ''
        self.selection_selected_algorithm = ''
        self.selection_running_status = False
        self.selection_default_para = {
            'feature_number': 5,
        }

        """ Machine Learning Variable """
        self.MLData = None
        self.MLAlgorithm = None
        self.fold_num = 5
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
        }

        # initialize UI
        self.initUI()

    def initUI(self):
        self.setWindowTitle('iLearnPlus Basic')
        self.resize(800, 600)
        self.setWindowState(Qt.WindowMaximized)
        self.setWindowIcon(QIcon('images/logo.ico'))

        """ four QWidget """
        self.tab_descriptor = QWidget()
        self.tab_cluster = QWidget()
        self.tab_selection = QWidget()
        self.tab_machine = QWidget()
        self.addTab(self.tab_descriptor, " Descriptor ")
        self.addTab(self.tab_cluster, " Cluster / Dimensionality Reduction ")
        self.addTab(self.tab_selection, ' Feature Normalization/Selection ')
        self.addTab(self.tab_machine, ' Machine Learning ')

        """ Initialize tab """
        self.setup_tab_descriptor()
        self.setup_tab_cluster()
        self.setup_tab_selection()
        self.setup_tab_machinelearning()

    """ setup tab UI """

    def setup_tab_descriptor(self):
        # choose file -> topGroupBox
        topGroupBox = QGroupBox('Choose file in special FASTA format', self)
        topGroupBox.setFont(QFont('Arial', 10))
        topGroupBoxLayout = QHBoxLayout()
        self.desc_file_lineEdit = QLineEdit()
        self.desc_file_lineEdit.setFont(QFont('Arial', 8))
        self.desc_file_button = QPushButton('Open')
        self.desc_file_button.clicked.connect(self.get_fasta_file_name)
        self.desc_file_button.setFont(QFont('Arial', 10))
        topGroupBoxLayout.addWidget(self.desc_file_lineEdit)
        topGroupBoxLayout.addWidget(self.desc_file_button)
        topGroupBox.setLayout(topGroupBoxLayout)

        # encoding list -> treeGroupBox
        treeGroupBox = QGroupBox('Descriptors', self)
        treeGroupBox.setFont(QFont('Arial', 10))
        treeLayout = QVBoxLayout()
        self.desc_treeWidget = QTreeWidget()
        self.desc_treeWidget.setColumnCount(2)
        self.desc_treeWidget.setMinimumWidth(300)
        self.desc_treeWidget.setColumnWidth(0, 150)
        self.desc_treeWidget.setFont(QFont('Arial', 8))
        self.desc_treeWidget.setHeaderLabels(['Codings', 'Definition'])
        self.desc_treeWidget.clicked.connect(self.desc_tree_clicked)
        # Protein descriptors
        self.Protein = QTreeWidgetItem(self.desc_treeWidget)
        self.Protein.setExpanded(True)  # set node expanded
        self.Protein.setText(0, 'Protein')
        self.AAC = QTreeWidgetItem(self.Protein)
        self.AAC.setText(0, 'AAC')
        self.AAC.setText(1, 'Amino Acids Content')
        self.AAC.setToolTip(1, 'The AAC encoding calculates the frequency of each amino acid\n type in a protein or peptide sequence.')
        EAAC = QTreeWidgetItem(self.Protein)
        EAAC.setToolTip(1, 'The descriptor need fasta sequences with equal length.')
        EAAC.setText(0, 'EAAC')
        EAAC.setText(1, 'Enhanced Amino Acids Content')
        EAAC.setToolTip(1, 'The EAAC feature calculates the AAC based on the sequence window\n of fixed length that continuously slides from the N- to\n C-terminus of each peptide and can be usually applied to\n encode the peptides with an equal length.')
        CKSAAP = QTreeWidgetItem(self.Protein)
        CKSAAP.setText(0, 'CKSAAP')
        CKSAAP.setText(1, 'Composition of k-spaced Amino Acid Pairs')
        CKSAAP.setToolTip(1, 'The CKSAAP feature encoding calculates the frequency of amino\n acid pairs separated by any k residues.')
        self.DPC = QTreeWidgetItem(self.Protein)
        self.DPC.setText(0, 'DPC')
        self.DPC.setText(1, 'Di-Peptide Composition')
        self.DPC.setToolTip(1, 'The DPC descriptor calculate the frequency of di-peptides.')
        DDE = QTreeWidgetItem(self.Protein)
        DDE.setText(0, 'DDE')
        DDE.setText(1, 'Dipeptide Deviation from Expected Mean')
        DDE.setToolTip(1, 'The Dipeptide Deviation from Expected Mean feature vector is\n constructed by computing three parameters, i.e. dipeptide composition (Dc),\n theoretical mean (Tm), and theoretical variance (Tv).')
        self.TPC = QTreeWidgetItem(self.Protein)
        self.TPC.setText(0, 'TPC')
        self.TPC.setText(1, 'Tripeptide Composition')
        self.TPC.setToolTip(1, 'The TPC descriptor calculate the frequency of tri-peptides.')
        binary = QTreeWidgetItem(self.Protein)
        binary.setToolTip(1, 'In the binary encoding, each amino acid is represented by a 20-dimensional binary vector.')
        binary.setText(0, 'binary')
        binary.setText(1, 'binary (20 bit)')
        self.binary_6bit = QTreeWidgetItem(self.Protein)        
        self.binary_6bit.setText(0, 'binary_6bit')
        self.binary_6bit.setText(1, 'binary (6 bit)')
        self.binary_6bit.setToolTip(1, 'In the binary encoding, each amino acid is represented by a 6-dimensional binary vector.')
        self.binary_5bit_type1 = QTreeWidgetItem(self.Protein)
        self.binary_5bit_type1.setToolTip(1, 'The descriptor need fasta sequences with equal length.')
        self.binary_5bit_type1.setText(0, 'binary_5bit type 1')
        self.binary_5bit_type1.setText(1, 'binary (5 bit type 1)')
        self.binary_5bit_type1.setToolTip(1, 'In the binary encoding, each amino acid is represented by a 5-dimensional binary vector.')
        self.binary_5bit_type2 = QTreeWidgetItem(self.Protein)
        self.binary_5bit_type2.setToolTip(1, 'The descriptor need fasta sequences with equal length.')
        self.binary_5bit_type2.setText(0, 'binary_5bit type 2')
        self.binary_5bit_type2.setText(1, 'binary (5 bit type 2)')
        self.binary_5bit_type2.setToolTip(1, 'In the binary encoding, each amino acid is represented by a 5-dimensional binary vector.')
        self.binary_3bit_type1 = QTreeWidgetItem(self.Protein)
        self.binary_3bit_type1.setToolTip(1, 'The descriptor need fasta sequences with equal length.')
        self.binary_3bit_type1.setText(0, 'binary_3bit type 1')
        self.binary_3bit_type1.setText(1, 'binary (3 bit type 1 - Hydrophobicity)')
        self.binary_3bit_type1.setToolTip(1, 'In the binary encoding, each amino acid is represented by a 3-dimensional binary vector.')
        self.binary_3bit_type2 = QTreeWidgetItem(self.Protein)
        self.binary_3bit_type2.setToolTip(1, 'The descriptor need fasta sequences with equal length.')
        self.binary_3bit_type2.setText(0, 'binary_3bit type 2')
        self.binary_3bit_type2.setText(1, 'binary (3 bit type 2 - Normalized Van der Waals volume)')
        self.binary_3bit_type2.setToolTip(1, 'In the binary encoding, each amino acid is represented by a 3-dimensional binary vector.')
        self.binary_3bit_type3 = QTreeWidgetItem(self.Protein)
        self.binary_3bit_type3.setToolTip(1, 'The descriptor need fasta sequences with equal length.')
        self.binary_3bit_type3.setText(0, 'binary_3bit type 3')
        self.binary_3bit_type3.setText(1, 'binary (3 bit type 3 - Polarity)')
        self.binary_3bit_type3.setToolTip(1, 'In the binary encoding, each amino acid is represented by a 3-dimensional binary vector.')
        self.binary_3bit_type4 = QTreeWidgetItem(self.Protein)
        self.binary_3bit_type4.setToolTip(1, 'The descriptor need fasta sequences with equal length.')
        self.binary_3bit_type4.setText(0, 'binary_3bit type 4')
        self.binary_3bit_type4.setText(1, 'binary (3 bit type 4 - Polarizibility)')
        self.binary_3bit_type4.setToolTip(1, 'In the binary encoding, each amino acid is represented by a 3-dimensional binary vector.')
        self.binary_3bit_type5 = QTreeWidgetItem(self.Protein)
        self.binary_3bit_type5.setToolTip(1, 'The descriptor need fasta sequences with equal length.')
        self.binary_3bit_type5.setText(0, 'binary_3bit type 5')
        self.binary_3bit_type5.setText(1, 'binary (3 bit type 5 - Charge)')
        self.binary_3bit_type5.setToolTip(1, 'In the binary encoding, each amino acid is represented by a 3-dimensional binary vector.')
        self.binary_3bit_type6 = QTreeWidgetItem(self.Protein)
        self.binary_3bit_type6.setToolTip(1, 'The descriptor need fasta sequences with equal length.')
        self.binary_3bit_type6.setText(0, 'binary_3bit type 6')
        self.binary_3bit_type6.setText(1, 'binary (3 bit type 6 - Secondary structures)')
        self.binary_3bit_type6.setToolTip(1, 'In the binary encoding, each amino acid is represented by a 3-dimensional binary vector.')
        self.binary_3bit_type7 = QTreeWidgetItem(self.Protein)
        self.binary_3bit_type7.setToolTip(1, 'The descriptor need fasta sequences with equal length.')
        self.binary_3bit_type7.setText(0, 'binary_3bit type 7')
        self.binary_3bit_type7.setText(1, 'binary (3 bit type 7 - Solvent accessibility)')
        self.binary_3bit_type7.setToolTip(1, 'In the binary encoding, each amino acid is represented by a 3-dimensional binary vector.')
        AESNN3 = QTreeWidgetItem(self.Protein)
        AESNN3.setText(0, 'AESNN3')
        AESNN3.setText(1, 'Learn from alignments')
        AESNN3.setToolTip(1, 'For this descriptor, each amino acid type is described using\n a three-dimensional vector. Values are taken from the three\n hidden units from the neural network trained on structure alignments.')
        self.GAAC = QTreeWidgetItem(self.Protein)
        self.GAAC.setText(0, 'GAAC')
        self.GAAC.setText(1, 'Grouped Amino Acid Composition')
        self.GAAC.setToolTip(1, 'In the GAAC encoding, the 20 amino acid types are further categorized\n into five classes according to their physicochemical properties. It calculate the frequency for each class.')
        EGAAC = QTreeWidgetItem(self.Protein)
        EGAAC.setToolTip(1, 'The descriptor need fasta sequences with equal length.')
        EGAAC.setText(0, 'EGAAC')
        EGAAC.setText(1, 'Enhanced Grouped Amino Acid Composition')
        EGAAC.setToolTip(1, 'It calculates GAAC in windows of fixed length continuously sliding\n from the N- to C-terminal of each peptide and is usually applied\n to peptides with an equal length.')
        CKSAAGP = QTreeWidgetItem(self.Protein)
        CKSAAGP.setText(0, 'CKSAAGP')
        CKSAAGP.setText(1, 'Composition of k-Spaced Amino Acid Group Pairs')
        CKSAAGP.setToolTip(1, ' It calculates the frequency of amino acid group pairs separated by any k residues.')
        self.GDPC = QTreeWidgetItem(self.Protein)
        self.GDPC.setText(0, 'GDPC')
        self.GDPC.setText(1, 'Grouped Di-Peptide Composition')
        self.GDPC.setToolTip(1, 'GDPC calculate the frequency of amino acid group pairs.')
        self.GTPC = QTreeWidgetItem(self.Protein)
        self.GTPC.setText(0, 'GTPC')
        self.GTPC.setText(1, 'Grouped Tri-Peptide Composition')
        self.GTPC.setToolTip(1, 'GTPC calculate the frequency of grouped tri-peptides.')
        AAIndex = QTreeWidgetItem(self.Protein)
        AAIndex.setText(0, 'AAIndex')
        AAIndex.setText(1, 'AAIndex')
        AAIndex.setToolTip(1, 'The amino acids is respresented by the physicochemical property value in AAindex database.')
        ZScale = QTreeWidgetItem(self.Protein)
        ZScale.setText(0, 'ZScale')
        ZScale.setText(1, 'ZScale')
        ZScale.setToolTip(1, 'Each amino acid is characterized by five physicochemical descriptor variables, which were developed by Sandberg et al. in 1998.')
        BLOSUM62 = QTreeWidgetItem(self.Protein)
        BLOSUM62.setText(0, 'BLOSUM62')
        BLOSUM62.setText(1, 'BLOSUM62')
        BLOSUM62.setToolTip(1, 'In this descriptor, the BLOSUM62 matrix is employed to represent the\n protein primary sequence information as the basic feature set.')
        NMBroto = QTreeWidgetItem(self.Protein)
        NMBroto.setText(0, 'NMBroto')
        NMBroto.setText(1, 'Normalized Moreau-Broto Autocorrelation')
        NMBroto.setToolTip(1, 'The autocorrelation descriptors are defined based on the distribution\n of amino acid properties along the sequence.')
        Moran = QTreeWidgetItem(self.Protein)
        Moran.setText(0, 'Moran')
        Moran.setText(1, 'Moran correlation')
        Moran.setToolTip(1, 'The autocorrelation descriptors are defined based on the distribution\n of amino acid properties along the sequence.')
        Geary = QTreeWidgetItem(self.Protein)
        Geary.setText(0, 'Geary')
        Geary.setText(1, 'Geary correlation')
        Geary.setToolTip(1, 'The autocorrelation descriptors are defined based on the distribution\n of amino acid properties along the sequence.')
        CTDC = QTreeWidgetItem(self.Protein)
        CTDC.setText(0, 'CTDC')
        CTDC.setText(1, 'Composition')
        CTDC.setToolTip(1, 'The Composition, Transition and Distribution (CTD) features represent\n the amino acid distribution patterns of a specific structural\n or physicochemical property in a protein or peptide sequence.')
        CTDT = QTreeWidgetItem(self.Protein)
        CTDT.setText(0, 'CTDT')
        CTDT.setText(1, 'Transition')
        CTDT.setToolTip(1, 'The Composition, Transition and Distribution (CTD) features represent\n the amino acid distribution patterns of a specific structural\n or physicochemical property in a protein or peptide sequence.')
        CTDD = QTreeWidgetItem(self.Protein)
        CTDD.setText(0, 'CTDD')
        CTDD.setToolTip(1, 'The Composition, Transition and Distribution (CTD) features represent\n the amino acid distribution patterns of a specific structural\n or physicochemical property in a protein or peptide sequence.')
        CTDD.setText(1, 'Distribution')
        CTriad = QTreeWidgetItem(self.Protein)
        CTriad.setText(0, 'CTriad')
        CTriad.setText(1, 'Conjoint Triad')
        CTriad.setToolTip(1, 'The CTriad considers the properties of one amino acid and its\n vicinal amino acids by regarding any three continuous amino\n acids as a single unit.')
        self.KSCTriad = QTreeWidgetItem(self.Protein)
        self.KSCTriad.setText(0, 'KSCTriad')
        self.KSCTriad.setText(1, 'k-Spaced Conjoint Triad')
        self.KSCTriad.setToolTip(1, 'The KSCTriad descriptor is based on the Conjoint CTriad descriptor,\n which not only calculates the numbers of three continuous amino acid units,\n but also considers the continuous amino acid units that are separated by any k residues.')
        SOCNumber = QTreeWidgetItem(self.Protein)
        SOCNumber.setText(0, 'SOCNumber')
        SOCNumber.setText(1, 'Sequence-Order-Coupling Number')
        SOCNumber.setToolTip(1, 'The SOCNumber descriptor consider the sequence order coupling number information.')
        QSOrder = QTreeWidgetItem(self.Protein)
        QSOrder.setText(0, 'QSOrder')
        QSOrder.setText(1, 'Quasi-sequence-order')
        QSOrder.setToolTip(1, 'Qsorder descriptor coonsider the quasi sequence order information.')
        PAAC = QTreeWidgetItem(self.Protein)
        PAAC.setText(0, 'PAAC')
        PAAC.setText(1, 'Pseudo-Amino Acid Composition')
        PAAC.setToolTip(1, 'The PAAC descriptor is a combination of a set of discrete sequence correlation\n factors and the 20 components of the conventional amino acid composition.')
        APAAC = QTreeWidgetItem(self.Protein)
        APAAC.setText(0, 'APAAC')
        APAAC.setText(1, 'Amphiphilic Pseudo-Amino Acid Composition')
        APAAC.setToolTip(1, 'The descriptor contains 20 + 2 lambda discrete numbers:\n the first 20 numbers are the components of the conventional amino acid composition;\n the next 2 lambda numbers are a set of correlation factors that reflect different\n hydrophobicity and hydrophilicity distribution patterns along a protein chain.')
        OPF_10bit = QTreeWidgetItem(self.Protein)
        OPF_10bit.setText(0, 'OPF_10bit')
        OPF_10bit.setText(1, 'Overlapping Property Features (10 bit)')
        OPF_10bit.setToolTip(1, 'For this descriptor, the amino acids are classified into 10 groups based their physicochemical properties.')
        self.OPF_7bit_type1 = QTreeWidgetItem(self.Protein)
        self.OPF_7bit_type1.setText(0, 'OPF_7bit type 1')
        self.OPF_7bit_type1.setText(1, 'Overlapping Property Features (7 bit type 1)')
        self.OPF_7bit_type1.setToolTip(1, 'For this descriptor, the amino acids are classified into 7 groups based their physicochemical properties.')
        self.OPF_7bit_type2 = QTreeWidgetItem(self.Protein)
        self.OPF_7bit_type2.setText(0, 'OPF_7bit type 2')
        self.OPF_7bit_type2.setText(1, 'Overlapping Property Features (7 bit type 2)')
        self.OPF_7bit_type2.setToolTip(1, 'For this descriptor, the amino acids are classified into 7 groups based their physicochemical properties.')
        self.OPF_7bit_type3 = QTreeWidgetItem(self.Protein)
        self.OPF_7bit_type3.setText(0, 'OPF_7bit type 3')
        self.OPF_7bit_type3.setText(1, 'Overlapping Property Features (7 bit type 3)')
        self.OPF_7bit_type3.setToolTip(1, 'For this descriptor, the amino acids are classified into 7 groups based their physicochemical properties.')
        pASDC = QTreeWidgetItem(self.Protein)
        pASDC.setText(0, 'ASDC')
        pASDC.setText(1, 'Adaptive skip dipeptide composition')
        pASDC.setToolTip(1, 'The adaptive skip dipeptide composition is a modified dipeptide composition,\n which sufficiently considers the correlation information present not only between\n adjacent residues but also between intervening residues.')
        proteinKNN = QTreeWidgetItem(self.Protein)
        proteinKNN.setText(0, 'KNN')
        proteinKNN.setText(1, 'K-nearest neighbor')
        proteinKNN.setToolTip(1, 'The KNN descriptor depicts how much one query sample resembles other samples.')
        DistancePair = QTreeWidgetItem(self.Protein)
        DistancePair.setText(0, 'DistancePair')
        DistancePair.setText(1, 'PseAAC of Distance-Pairs and Reduced Alphabet')
        DistancePair.setToolTip(1, 'The descriptor incorporates the amino acid distance pair coupling information \nand the amino acid reduced alphabet profile into the general pseudo amino acid composition vector.')
        self.proteinAC = QTreeWidgetItem(self.Protein)
        self.proteinAC.setText(0, 'AC')
        self.proteinAC.setText(1, 'Auto covariance')
        self.proteinAC.setToolTip(1, 'The AC descriptor measures the correlation of the same physicochemical \nindex between two amino acids separated by a distance of lag along the sequence. ')
        self.proteinCC = QTreeWidgetItem(self.Protein)
        self.proteinCC.setText(0, 'CC')
        self.proteinCC.setText(1, 'Cross covariance')
        self.proteinCC.setToolTip(1, 'The CC descriptor measures the correlation of two different physicochemical \nindices between two amino acids separated by lag nucleic acids along the sequence.')
        proteinACC = QTreeWidgetItem(self.Protein)
        proteinACC.setText(0, 'ACC')
        proteinACC.setText(1, 'Auto-cross covariance')
        proteinACC.setToolTip(1, 'The Dinucleotide-based Auto-Cross Covariance (ACC) encoding is a combination of AC and CC encoding.')

        PseKRAAC_type1 = QTreeWidgetItem(self.Protein)
        PseKRAAC_type1.setText(0, 'PseKRAAC type 1')
        PseKRAAC_type1.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 1')
        PseKRAAC_type1.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type2 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type2.setText(0, 'PseKRAAC type 2')
        self.PseKRAAC_type2.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 2')
        self.PseKRAAC_type2.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type3A = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type3A.setText(0, 'PseKRAAC type 3A')
        self.PseKRAAC_type3A.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 3A')
        self.PseKRAAC_type3A.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type3B = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type3B.setText(0, 'PseKRAAC type 3B')
        self.PseKRAAC_type3B.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 3B')
        self.PseKRAAC_type3B.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type4 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type4.setText(0, 'PseKRAAC type 4')
        self.PseKRAAC_type4.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 4')
        self.PseKRAAC_type4.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type5 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type5.setText(0, 'PseKRAAC type 5')
        self.PseKRAAC_type5.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 5')
        self.PseKRAAC_type5.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type6A = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type6A.setText(0, 'PseKRAAC type 6A')
        self.PseKRAAC_type6A.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 6A')
        self.PseKRAAC_type6A.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type6B = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type6B.setText(0, 'PseKRAAC type 6B')
        self.PseKRAAC_type6B.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 6B')
        self.PseKRAAC_type6B.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type6C = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type6C.setText(0, 'PseKRAAC type 6C')
        self.PseKRAAC_type6C.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 6C')
        self.PseKRAAC_type6C.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type7 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type7.setText(0, 'PseKRAAC type 7')
        self.PseKRAAC_type7.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 7')
        self.PseKRAAC_type7.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type8 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type8.setText(0, 'PseKRAAC type 8')
        self.PseKRAAC_type8.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 8')
        self.PseKRAAC_type8.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type9 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type9.setText(0, 'PseKRAAC type 9')
        self.PseKRAAC_type9.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 9')
        self.PseKRAAC_type9.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type10 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type10.setText(0, 'PseKRAAC type 10')
        self.PseKRAAC_type10.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 10')
        self.PseKRAAC_type10.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type11 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type11.setText(0, 'PseKRAAC type 11')
        self.PseKRAAC_type11.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 11')
        self.PseKRAAC_type11.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type12 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type12.setText(0, 'PseKRAAC type 12')
        self.PseKRAAC_type12.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 12')
        self.PseKRAAC_type12.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type13 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type13.setText(0, 'PseKRAAC type 13')
        self.PseKRAAC_type13.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 13')
        self.PseKRAAC_type13.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type14 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type14.setText(0, 'PseKRAAC type 14')
        self.PseKRAAC_type14.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 14')
        self.PseKRAAC_type14.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type15 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type15.setText(0, 'PseKRAAC type 15')
        self.PseKRAAC_type15.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 15')
        self.PseKRAAC_type15.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')
        self.PseKRAAC_type16 = QTreeWidgetItem(self.Protein)
        self.PseKRAAC_type16.setText(0, 'PseKRAAC type 16')
        self.PseKRAAC_type16.setText(1, 'Pseudo K-tuple Reduced Amino Acids Composition - type 16')
        self.PseKRAAC_type16.setToolTip(1, 'Pseudo K-tuple Reduced Amino Acids Composition.')

        # DNA
        self.DNA = QTreeWidgetItem(self.desc_treeWidget)
        self.DNA.setText(0, 'DNA')
        Kmer = QTreeWidgetItem(self.DNA)
        Kmer.setText(0, 'Kmer')
        Kmer.setText(1, 'The occurrence frequencies of k neighboring nucleic acids')
        Kmer.setToolTip(1, 'For kmer descriptor, the DNA or RNA sequences are represented\n as the occurrence frequencies of k neighboring nucleic acids.')
        RCKmer = QTreeWidgetItem(self.DNA)
        RCKmer.setText(0, 'RCKmer')
        RCKmer.setText(1, 'Reverse Compliment Kmer')
        RCKmer.setToolTip(1, 'The RCKmer descriptor is a variant of kmer descriptor,\n in which the kmers are not expected to be strand-specific. ')
        dnaMismatch = QTreeWidgetItem(self.DNA)
        dnaMismatch.setText(0, 'Mismatch')
        dnaMismatch.setText(1, 'Mismatch profile')
        dnaMismatch.setToolTip(1, 'The mismatch profile also calculates the occurrences of kmers,\n but allows max m inexact matching (m < k).')
        dnaSubsequence = QTreeWidgetItem(self.DNA)
        dnaSubsequence.setText(0, 'Subsequence')
        dnaSubsequence.setText(1, 'Subsequence profile')
        dnaSubsequence.setToolTip(1, 'The subsequence descriptor allows non-contiguous matching.')
        self.NAC = QTreeWidgetItem(self.DNA)
        self.NAC.setText(0, 'NAC')
        self.NAC.setText(1, 'Nucleic Acid Composition')
        self.NAC.setToolTip(1, 'The NAC encoding calculates the frequency of each nucleic acid type in a nucleotide sequence.')
        # DNC = QTreeWidgetItem(self.DNA)
        # DNC.setText(0, 'DNC')
        # DNC.setText(1, 'Di-Nucleotide Composition')
        # TNC = QTreeWidgetItem(self.DNA)
        # TNC.setText(0, 'TNC')
        # TNC.setText(1, 'Tri-Nucleotide Composition')
        ANF = QTreeWidgetItem(self.DNA)
        ANF.setText(0, 'ANF')
        ANF.setText(1, 'Accumulated Nucleotide Frequency')
        ANF.setToolTip(1, 'The ANF encoding include the nucleotide frequency information and the distribution of each nucleotide in the RNA sequence.')
        ENAC = QTreeWidgetItem(self.DNA)
        ENAC.setText(0, 'ENAC')
        ENAC.setText(1, 'Enhanced Nucleic Acid Composition')
        ENAC.setToolTip(1, 'The ENAC descriptor calculates the NAC based on the sequence window\n of fixed length that continuously slides from the 5\' to 3\' terminus\n of each nucleotide sequence and can be usually applied to encode the\n nucleotide sequence with an equal length.')
        DNAbinary = QTreeWidgetItem(self.DNA)
        DNAbinary.setText(0, 'binary')
        DNAbinary.setText(1, 'DNA binary')
        DNAbinary.setToolTip(1, 'In the Binary encoding, each amino acid is represented by a 4-dimensional binary vector.')
        dnaPS2 = QTreeWidgetItem(self.DNA)
        dnaPS2.setText(0, 'PS2')
        dnaPS2.setText(1, 'Position-specific of two nucleotides')
        dnaPS2.setToolTip(1, 'There are 4 x 4 = 16 pairs of adjacent pairwise nucleotides, \nthus a single variable representing one such pair gets one-hot\n (i.e. binary) encoded into 16 binary variables.')
        self.dnaPS3 = QTreeWidgetItem(self.DNA)
        self.dnaPS3.setText(0, 'PS3')
        self.dnaPS3.setText(1, 'Position-specific of three nucleotides')
        self.dnaPS3.setToolTip(1, 'The PS3 descriptor is encoded for three adjacent nucleotides in a similar way with PS2.')
        self.dnaPS4 = QTreeWidgetItem(self.DNA)
        self.dnaPS4.setText(0, 'PS4')
        self.dnaPS4.setText(1, 'Position-specific of four nucleotides')
        self.dnaPS4.setToolTip(1, 'The PS4 descriptor is encoded for four adjacent nucleotides in a similar way with PS2.')
        CKSNAP = QTreeWidgetItem(self.DNA)
        CKSNAP.setText(0, 'CKSNAP')
        CKSNAP.setText(1, 'Composition of k-spaced Nucleic Acid Pairs')
        CKSNAP.setToolTip(1, 'The CKSNAP feature encoding calculates the frequency of nucleic acid pairs separated by any k nucleic acid.')
        NCP = QTreeWidgetItem(self.DNA)
        NCP.setText(0, 'NCP')
        NCP.setText(1, 'Nucleotide Chemical Property')
        NCP.setToolTip(1, 'Based on chemical properties, A can be represented by coordinates (1, 1, 1), \nC can be represented by coordinates (0, 1, 0), G can be represented by coordinates (1, 0, 0), \nU can be represented by coordinates (0, 0, 1). ')
        PSTNPss = QTreeWidgetItem(self.DNA)
        PSTNPss.setText(0, 'PSTNPss')
        PSTNPss.setText(1, 'Position-specific trinucleotide propensity based on single-strand')
        PSTNPss.setToolTip(1, 'The PSTNPss descriptor usie a statistical strategy based on single-stranded characteristics of DNA or RNA.')
        PSTNPds = QTreeWidgetItem(self.DNA)
        PSTNPds.setText(0, 'PSTNPds')
        PSTNPds.setText(1, 'Position-specific trinucleotide propensity based on double-strand')
        PSTNPds.setToolTip(1, 'The PSTNPds descriptor use a statistical strategy based on double-stranded characteristics of DNA according to complementary base pairing.')
        EIIP = QTreeWidgetItem(self.DNA)
        EIIP.setText(0, 'EIIP')
        EIIP.setText(1, 'Electron-ion interaction pseudopotentials')
        EIIP.setToolTip(1, 'The EIIP directly use the EIIP value represent the nucleotide in the DNA sequence.')
        PseEIIP = QTreeWidgetItem(self.DNA)
        PseEIIP.setText(0, 'PseEIIP')
        PseEIIP.setText(1, 'Electron-ion interaction pseudopotentials of trinucleotide')
        PseEIIP.setToolTip(1, 'Electron-ion interaction pseudopotentials of trinucleotide.')
        DNAASDC = QTreeWidgetItem(self.DNA)
        DNAASDC.setText(0, 'ASDC')
        DNAASDC.setText(1, 'Adaptive skip dinucleotide composition')
        DNAASDC.setToolTip(1, 'The adaptive skip dipeptide composition is a modified dinucleotide composition, \nwhich sufficiently considers the correlation information present not only between \nadjacent residues but also between intervening residues.')
        dnaDBE = QTreeWidgetItem(self.DNA)
        dnaDBE.setText(0, 'DBE')
        dnaDBE.setText(1, 'Dinucleotide binary encoding')
        dnaDBE.setToolTip(1, 'The DBE descriptor encapsulates the positional information of the dinucleotide at each position in the sequence.')
        dnaLPDF = QTreeWidgetItem(self.DNA)
        dnaLPDF.setText(0, 'LPDF')
        dnaLPDF.setText(1, 'Local position-specific dinucleotide frequency')
        dnaLPDF.setToolTip(1, 'The LPDF descriptor calculate the local position-specific dinucleotide frequency.')
        dnaDPCP = QTreeWidgetItem(self.DNA)
        dnaDPCP.setText(0, 'DPCP')
        dnaDPCP.setText(1, 'Dinucleotide physicochemical properties')
        dnaDPCP.setToolTip(1, 'The DPCP descriptor calculate the value of frequency of dinucleotide multiplied by dinucleotide physicochemical properties.')
        self.dnaDPCP2 = QTreeWidgetItem(self.DNA)
        self.dnaDPCP2.setText(0, 'DPCP type2')
        self.dnaDPCP2.setText(1, 'Dinucleotide physicochemical properties type 2')
        self.dnaDPCP2.setToolTip(1, 'The DPCP2 descriptor calculate the position specific dinucleotide physicochemical properties.')
        dnaTPCP = QTreeWidgetItem(self.DNA)
        dnaTPCP.setText(0, 'TPCP')
        dnaTPCP.setText(1, 'Trinucleotide physicochemical properties')
        dnaTPCP.setToolTip(1, 'The TPCP descriptor calculate the value of frequency of trinucleotide multiplied by trinucleotide physicochemical properties.')
        self.dnaTPCP2 = QTreeWidgetItem(self.DNA)
        self.dnaTPCP2.setText(0, 'TPCP type2')
        self.dnaTPCP2.setText(1, 'Trinucleotide physicochemical properties type 2')
        self.dnaTPCP2.setToolTip(1, 'The TPCP2 descriptor calculate the position specific trinucleotide physicochemical properties.')
        dnaMMI = QTreeWidgetItem(self.DNA)
        dnaMMI.setText(0, 'MMI')
        dnaMMI.setText(1, 'Multivariate mutual information')
        dnaMMI.setToolTip(1, 'The MMI descriptor calculate multivariate mutual information on a DNA/RNA sequence.')
        dnaKNN = QTreeWidgetItem(self.DNA)
        dnaKNN.setText(0, 'KNN')
        dnaKNN.setText(1, 'K-nearest neighbor')
        dnaKNN.setToolTip(1, 'The K-nearest neighbor descriptor depicts how much one query sample resembles other samples.')
        dnazcurve9bit = QTreeWidgetItem(self.DNA)
        dnazcurve9bit.setText(0, 'Z_curve_9bit')
        dnazcurve9bit.setText(1, 'The Z curve parameters for frequencies of phase-specific mononucleotides')
        dnazcurve9bit.setToolTip(1, 'The Z curve parameters for frequencies of phase-specific mononucleotides.')
        self.dnazcurve12bit = QTreeWidgetItem(self.DNA)
        self.dnazcurve12bit.setText(0, 'Z_curve_12bit')
        self.dnazcurve12bit.setText(1, 'The Z curve parameters for frequencies of phaseindependent di-nucleotides')
        self.dnazcurve12bit.setToolTip(1, 'The Z curve parameters for frequencies of phaseindependent di-nucleotides')
        self.dnazcurve36bit = QTreeWidgetItem(self.DNA)
        self.dnazcurve36bit.setText(0, 'Z_curve_36bit')
        self.dnazcurve36bit.setText(1, 'The Z curve parameters for frequencies of phase-specific di-nucleotides')
        self.dnazcurve36bit.setToolTip(1, 'The Z curve parameters for frequencies of phase-specific di-nucleotides')
        self.dnazcurve48bit = QTreeWidgetItem(self.DNA)
        self.dnazcurve48bit.setText(0, 'Z_curve_48bit')
        self.dnazcurve48bit.setText(1, 'The Z curve parameters for frequencies of phaseindependent tri-nucleotides')
        self.dnazcurve48bit.setToolTip(1, 'The Z curve parameters for frequencies of phaseindependent tri-nucleotides')
        self.dnazcurve144bit = QTreeWidgetItem(self.DNA)
        self.dnazcurve144bit.setText(0, 'Z_curve_144bit')
        self.dnazcurve144bit.setText(1, 'The Z curve parameters for frequencies of phase-specific tri-nucleotides')
        self.dnazcurve144bit.setToolTip(1, 'The Z curve parameters for frequencies of phase-specific tri-nucleotides')
        dnaNMBroto = QTreeWidgetItem(self.DNA)
        dnaNMBroto.setText(0, 'NMBroto')
        dnaNMBroto.setText(1, 'Normalized Moreau-Broto Autocorrelation')
        dnaNMBroto.setToolTip(1, 'The autocorrelation descriptors are defined based on the distribution\n of amino acid properties along the sequence.')
        dnaMoran = QTreeWidgetItem(self.DNA)
        dnaMoran.setText(0, 'Moran')
        dnaMoran.setText(1, 'Moran correlation')
        dnaMoran.setToolTip(1, 'The autocorrelation descriptors are defined based on the distribution\n of amino acid properties along the sequence.')
        dnaGeary = QTreeWidgetItem(self.DNA)
        dnaGeary.setText(0, 'Geary')
        dnaGeary.setText(1, 'Geary correlation')
        dnaGeary.setToolTip(1, 'The autocorrelation descriptors are defined based on the distribution\n of amino acid properties along the sequence.')
        self.DAC = QTreeWidgetItem(self.DNA)
        self.DAC.setText(0, 'DAC')
        self.DAC.setText(1, 'Dinucleotide-based Auto Covariance')
        self.DAC.setToolTip(1, 'The DAC descriptor measures the correlation of the same physicochemical \nindex between two dinucleotides separated by a distance of lag along the sequence.')
        self.DCC = QTreeWidgetItem(self.DNA)
        self.DCC.setText(0, 'DCC')
        self.DCC.setText(1, 'Dinucleotide-based Cross Covariance')
        self.DCC.setToolTip(1, 'The DCC descriptor measures the correlation of two different physicochemical \nindices between two dinucleotides separated by lag nucleic acids along the sequence.')
        DACC = QTreeWidgetItem(self.DNA)
        DACC.setText(0, 'DACC')
        DACC.setText(1, 'Dinucleotide-based Auto-Cross Covariance')
        DACC.setToolTip(1, 'The DACC encoding is a combination of DAC and DCC encoding.')
        self.TAC = QTreeWidgetItem(self.DNA)
        self.TAC.setText(0, 'TAC')
        self.TAC.setText(1, 'Trinucleotide-based Auto Covariance')
        self.TAC.setToolTip(1, 'The TAC descriptor measures the correlation of the same physicochemical \nindex between two trinucleotides separated by a distance of lag along the sequence.')
        self.TCC = QTreeWidgetItem(self.DNA)
        self.TCC.setText(0, 'TCC')
        self.TCC.setText(1, 'Trinucleotide-based Cross Covariance')
        self.TCC.setToolTip(1, 'The DCC descriptor measures the correlation of two different physicochemical \nindices between two trinucleotides separated by lag nucleic acids along the sequence.')
        TACC = QTreeWidgetItem(self.DNA)
        TACC.setText(0, 'TACC')
        TACC.setText(1, 'Trinucleotide-based Auto-Cross Covariance')
        TACC.setToolTip(1, 'The TACC encoding is a combination of TAC and TCC encoding.')
        PseDNC = QTreeWidgetItem(self.DNA)
        PseDNC.setText(0, 'PseDNC')
        PseDNC.setText(1, 'Pseudo Dinucleotide Composition')
        PseDNC.setToolTip(1, 'The PseDNC encodings incorporate contiguous local sequence-order information and the global sequence-order information into the feature vector of the nucleotide sequence.')
        PseKNC = QTreeWidgetItem(self.DNA)
        PseKNC.setText(0, 'PseKNC')
        PseKNC.setText(1, 'Pseudo k-tupler Composition')
        PseKNC.setToolTip(1, 'The PseKNC descriptor incorporate the k-tuple nucleotide composition.')
        PCPseDNC = QTreeWidgetItem(self.DNA)
        PCPseDNC.setText(0, 'PCPseDNC')
        PCPseDNC.setText(1, 'Parallel Correlation Pseudo Dinucleotide Composition')
        PCPseDNC.setToolTip(1, 'The PCPseDNC descriptor consider parallel correlation pseudo trinucleotide composition information.')
        PCPseTNC = QTreeWidgetItem(self.DNA)
        PCPseTNC.setText(0, 'PCPseTNC')
        PCPseTNC.setText(1, 'Parallel Correlation Pseudo Trinucleotide Composition')
        PCPseTNC.setToolTip(1, 'The PCPseTNC descriptor consider parallel correlation pseudo trinucleotide composition information.')
        SCPseDNC = QTreeWidgetItem(self.DNA)
        SCPseDNC.setText(0, 'SCPseDNC')
        SCPseDNC.setText(1, 'Series Correlation Pseudo Dinucleotide Composition')
        SCPseDNC.setToolTip(1, 'The SCPseDNC descriptor consider series correlation pseudo dinucleotide composition information.')
        SCPseTNC = QTreeWidgetItem(self.DNA)
        SCPseTNC.setText(0, 'SCPseTNC')
        SCPseTNC.setText(1, 'Series Correlation Pseudo Trinucleotide Composition')
        SCPseTNC.setToolTip(1, 'The SCPseTNC descriptor consider series correlation pseudo trinucleotide composition.')
        # RNA
        self.RNA = QTreeWidgetItem(self.desc_treeWidget)
        self.RNA.setText(0, 'RNA')
        RNAKmer = QTreeWidgetItem(self.RNA)
        RNAKmer.setText(0, 'Kmer')
        RNAKmer.setText(1, 'The occurrence frequencies of k neighboring nucleic acids')
        RNAKmer.setToolTip(1, 'For kmer descriptor, the DNA or RNA sequences are represented\n as the occurrence frequencies of k neighboring nucleic acids.')
        rnaMismatch = QTreeWidgetItem(self.RNA)
        rnaMismatch.setText(0, 'Mismatch')
        rnaMismatch.setText(1, 'Mismatch profile')
        rnaMismatch.setToolTip(1, 'The mismatch profile also calculates the occurrences of kmers,\n but allows max m inexact matching (m < k).')
        rnaSubsequence = QTreeWidgetItem(self.RNA)
        rnaSubsequence.setText(0, 'Subsequence')
        rnaSubsequence.setText(1, 'Subsequence profile')
        rnaSubsequence.setToolTip(1, 'The subsequence descriptor allows non-contiguous matching.')
        self.RNANAC = QTreeWidgetItem(self.RNA)
        self.RNANAC.setText(0, 'NAC')
        self.RNANAC.setText(1, 'Nucleic Acid Composition')
        self.RNANAC.setToolTip(1, 'The NAC encoding calculates the frequency of each nucleic acid type in a nucleotide sequence.')
        RNAENAC = QTreeWidgetItem(self.RNA)
        RNAENAC.setText(0, 'ENAC')
        RNAENAC.setText(1, 'Enhanced Nucleic Acid Composition')
        RNAENAC.setToolTip(1, 'The ENAC descriptor calculates the NAC based on the sequence window\n of fixed length that continuously slides from the 5\' to 3\' terminus\n of each nucleotide sequence and can be usually applied to encode the\n nucleotide sequence with an equal length.')
        # RNADNC = QTreeWidgetItem(self.RNA)
        # RNADNC.setText(0, 'DNC')
        # RNADNC.setText(1, 'Di-Nucleotide Composition')
        # RNATNC = QTreeWidgetItem(self.RNA)
        # RNATNC.setText(0, 'TNC')
        # RNATNC.setText(1, 'Tri-Nucleotide Composition')
        RNAANF = QTreeWidgetItem(self.RNA)
        RNAANF.setText(0, 'ANF')
        RNAANF.setText(1, 'Accumulated Nucleotide Frequency')
        RNAANF.setToolTip(1, 'The ANF encoding include the nucleotide frequency information and the distribution of each nucleotide in the RNA sequence.')
        RNANCP = QTreeWidgetItem(self.RNA)
        RNANCP.setText(0, 'NCP')
        RNANCP.setText(1, 'Nucleotide Chemical Property')
        RNANCP.setToolTip(1, 'Based on chemical properties, A can be represented by coordinates (1, 1, 1), \nC can be represented by coordinates (0, 1, 0), G can be represented by coordinates (1, 0, 0), \nU can be represented by coordinates (0, 0, 1). ')
        RNAPSTNPss = QTreeWidgetItem(self.RNA)
        RNAPSTNPss.setText(0, 'PSTNPss')
        RNAPSTNPss.setText(1, 'Position-specific trinucleotide propensity based on single-strand')
        RNAPSTNPss.setToolTip(1, 'The PSTNPss descriptor usie a statistical strategy based on single-stranded characteristics of DNA or RNA.')
        RNAbinary = QTreeWidgetItem(self.RNA)
        RNAbinary.setText(0, 'binary')
        RNAbinary.setText(1, 'RNA binary')
        RNAbinary.setToolTip(1, 'In the Binary encoding, each amino acid is represented by a 4-dimensional binary vector.')
        rnaPS2 = QTreeWidgetItem(self.RNA)
        rnaPS2.setText(0, 'PS2')
        rnaPS2.setText(1, 'Position-specific of two nucleotides')
        rnaPS2.setToolTip(1, 'There are 4 x 4 = 16 pairs of adjacent pairwise nucleotides, \nthus a single variable representing one such pair gets one-hot\n (i.e. binary) encoded into 16 binary variables.')
        self.rnaPS3 = QTreeWidgetItem(self.RNA)
        self.rnaPS3.setText(0, 'PS3')
        self.rnaPS3.setText(1, 'Position-specific of three nucleotides')
        self.rnaPS3.setToolTip(1, 'The PS3 descriptor is encoded for three adjacent nucleotides in a similar way with PS2.')
        self.rnaPS4 = QTreeWidgetItem(self.RNA)
        self.rnaPS4.setText(0, 'PS4')
        self.rnaPS4.setText(1, 'Position-specific of four nucleotides')
        self.rnaPS4.setToolTip(1, 'The PS4 descriptor is encoded for four adjacent nucleotides in a similar way with PS2.')
        RNACKSNAP = QTreeWidgetItem(self.RNA)
        RNACKSNAP.setText(0, 'CKSNAP')
        RNACKSNAP.setText(1, 'Composition of k-spaced Nucleic Acid Pairs')
        RNACKSNAP.setToolTip(1, 'The CKSNAP feature encoding calculates the frequency of nucleic acid pairs separated by any k nucleic acid.')
        RNAASDC = QTreeWidgetItem(self.RNA)
        RNAASDC.setText(0, 'ASDC')
        RNAASDC.setText(1, 'Adaptive skip di-nucleotide composition')
        RNAASDC.setToolTip(1, 'The adaptive skip dipeptide composition is a modified dinucleotide composition, \nwhich sufficiently considers the correlation information present not only between \nadjacent residues but also between intervening residues.')
        rnaDBE = QTreeWidgetItem(self.RNA)
        rnaDBE.setText(0, 'DBE')
        rnaDBE.setText(1, 'Dinucleotide binary encoding')
        rnaDBE.setToolTip(1, 'The DBE descriptor encapsulates the positional information of the dinucleotide at each position in the sequence.')
        rnaLPDF = QTreeWidgetItem(self.RNA)
        rnaLPDF.setText(0, 'LPDF')
        rnaLPDF.setText(1, 'Local position-specific dinucleotide frequency')
        rnaLPDF.setToolTip(1, 'The LPDF descriptor calculate the local position-specific dinucleotide frequency.')
        rnaDPCP = QTreeWidgetItem(self.RNA)
        rnaDPCP.setText(0, 'DPCP')
        rnaDPCP.setText(1, 'Dinucleotide physicochemical properties')
        rnaDPCP.setToolTip(1, 'The DPCP descriptor calculate the value of frequency of dinucleotide multiplied by dinucleotide physicochemical properties.')        
        self.rnaDPCP2 = QTreeWidgetItem(self.RNA)
        self.rnaDPCP2.setText(0, 'DPCP type2')
        self.rnaDPCP2.setText(1, 'Dinucleotide physicochemical properties type 2')
        self.rnaDPCP2.setToolTip(1, 'The DPCP2 descriptor calculate the position specific dinucleotide physicochemical properties.')
        rnaMMI = QTreeWidgetItem(self.RNA)
        rnaMMI.setText(0, 'MMI')
        rnaMMI.setText(1, 'Multivariate mutual information')
        rnaMMI.setToolTip(1, 'The MMI descriptor calculate multivariate mutual information on a DNA/RNA sequence.')
        rnaKNN = QTreeWidgetItem(self.RNA)
        rnaKNN.setText(0, 'KNN')
        rnaKNN.setText(1, 'K-nearest neighbor')
        rnaKNN.setToolTip(1, 'The K-nearest neighbor descriptor depicts how much one query sample resembles other samples.')
        rnazcurve9bit = QTreeWidgetItem(self.RNA)
        rnazcurve9bit.setText(0, 'Z_curve_9bit')
        rnazcurve9bit.setText(1, 'The Z curve parameters for frequencies of phase-specific mononucleotides')
        rnazcurve9bit.setToolTip(1, 'The Z curve parameters for frequencies of phase-specific mononucleotides.')
        self.rnazcurve12bit = QTreeWidgetItem(self.RNA)
        self.rnazcurve12bit.setText(0, 'Z_curve_12bit')
        self.rnazcurve12bit.setText(1, 'The Z curve parameters for frequencies of phaseindependent di-nucleotides')
        self.rnazcurve12bit.setToolTip(1, 'The Z curve parameters for frequencies of phaseindependent di-nucleotides')
        self.rnazcurve36bit = QTreeWidgetItem(self.RNA)
        self.rnazcurve36bit.setText(0, 'Z_curve_36bit')
        self.rnazcurve36bit.setText(1, 'The Z curve parameters for frequencies of phase-specific di-nucleotides')
        self.rnazcurve36bit.setToolTip(1, 'The Z curve parameters for frequencies of phase-specific di-nucleotides')
        self.rnazcurve48bit = QTreeWidgetItem(self.RNA)
        self.rnazcurve48bit.setText(0, 'Z_curve_48bit')
        self.rnazcurve48bit.setText(1, 'The Z curve parameters for frequencies of phaseindependent tri-nucleotides')
        self.rnazcurve48bit.setToolTip(1, 'The Z curve parameters for frequencies of phaseindependent tri-nucleotides')
        self.rnazcurve144bit = QTreeWidgetItem(self.RNA)
        self.rnazcurve144bit.setText(0, 'Z_curve_144bit')
        self.rnazcurve144bit.setText(1, 'The Z curve parameters for frequencies of phase-specific tri-nucleotides')
        self.rnazcurve144bit.setToolTip(1, 'The Z curve parameters for frequencies of phase-specific tri-nucleotides')
        rnaNMBroto = QTreeWidgetItem(self.RNA)
        rnaNMBroto.setText(0, 'NMBroto')
        rnaNMBroto.setText(1, 'Normalized Moreau-Broto Autocorrelation')
        rnaNMBroto.setToolTip(1, 'The autocorrelation descriptors are defined based on the distribution\n of amino acid properties along the sequence.')
        rnaMoran = QTreeWidgetItem(self.RNA)
        rnaMoran.setText(0, 'Moran')
        rnaMoran.setText(1, 'Moran correlation')
        rnaMoran.setToolTip(1, 'The autocorrelation descriptors are defined based on the distribution\n of amino acid properties along the sequence.')
        rnaGeary = QTreeWidgetItem(self.RNA)
        rnaGeary.setText(0, 'Geary')
        rnaGeary.setText(1, 'Geary correlation')
        rnaGeary.setToolTip(1, 'The autocorrelation descriptors are defined based on the distribution\n of amino acid properties along the sequence.')
        self.RNADAC = QTreeWidgetItem(self.RNA)
        self.RNADAC.setText(0, 'DAC')
        self.RNADAC.setText(1, 'Dinucleotide-based Auto Covariance')
        self.RNADAC.setToolTip(1, 'The DAC descriptor measures the correlation of the same physicochemical \nindex between two dinucleotides separated by a distance of lag along the sequence.')
        self.RNADCC = QTreeWidgetItem(self.RNA)
        self.RNADCC.setText(0, 'DCC')
        self.RNADCC.setText(1, 'Dinucleotide-based Cross Covariance')
        self.RNADCC.setToolTip(1, 'The DCC descriptor measures the correlation of two different physicochemical \nindices between two dinucleotides separated by lag nucleic acids along the sequence.')
        RNADACC = QTreeWidgetItem(self.RNA)
        RNADACC.setText(0, 'DACC')
        RNADACC.setText(1, 'Dinucleotide-based Auto-Cross Covariance')
        RNADACC.setToolTip(1, 'The DACC encoding is a combination of DAC and DCC encoding.')
        RNAPseDNC = QTreeWidgetItem(self.RNA)
        RNAPseDNC.setText(0, 'PseDNC')
        RNAPseDNC.setText(1, 'Pseudo Nucleic Acid Composition')
        RNAPseDNC.setToolTip(1, 'The PseDNC encodings incorporate contiguous local sequence-order information and the global sequence-order information into the feature vector of the nucleotide sequence.')
        RNAPseKNC = QTreeWidgetItem(self.RNA)
        RNAPseKNC.setText(0, 'PseKNC')
        RNAPseKNC.setText(1, 'Pseudo k-tupler Composition')
        RNAPseKNC.setToolTip(1, 'The PseKNC descriptor incorporate the k-tuple nucleotide composition.')
        RNAPCPseDNC = QTreeWidgetItem(self.RNA)
        RNAPCPseDNC.setText(0, 'PCPseDNC')
        RNAPCPseDNC.setText(1, 'Parallel Correlation Pseudo Dinucleotide Composition')
        RNAPCPseDNC.setToolTip(1, 'The PCPseDNC descriptor consider parallel correlation pseudo trinucleotide composition information.')
        RNASCPseDNC = QTreeWidgetItem(self.RNA)
        RNASCPseDNC.setText(0, 'SCPseDNC')
        RNASCPseDNC.setText(1, 'Series Correlation Pseudo Dinucleotide Composition')
        RNASCPseDNC.setToolTip(1, 'The SCPseDNC descriptor consider series correlation pseudo dinucleotide composition information.')       
        
        treeLayout.addWidget(self.desc_treeWidget)
        treeGroupBox.setLayout(treeLayout)

        self.Protein.setDisabled(True)
        self.DNA.setDisabled(True)
        self.RNA.setDisabled(True)

        ## parameter
        paraGroupBox = QGroupBox('Parameters', self)
        paraGroupBox.setMaximumHeight(150)
        paraGroupBox.setFont(QFont('Arial', 10))
        paraLayout = QFormLayout(paraGroupBox)
        self.desc_sequenceType_lineEdit = QLineEdit()
        self.desc_sequenceType_lineEdit.setFont(QFont('Arial', 8))
        self.desc_sequenceType_lineEdit.setEnabled(False)
        paraLayout.addRow('Sequence type:', self.desc_sequenceType_lineEdit)
        self.desc_currentDescriptor_lineEdit = QLineEdit()
        self.desc_currentDescriptor_lineEdit.setFont(QFont('Arial', 8))
        self.desc_currentDescriptor_lineEdit.setEnabled(False)
        paraLayout.addRow('Descriptor:', self.desc_currentDescriptor_lineEdit)
        self.desc_para_lineEdit = QLineEdit()
        self.desc_para_lineEdit.setFont(QFont('Arial', 8))
        self.desc_para_lineEdit.setEnabled(False)
        paraLayout.addRow('Parameter(s):', self.desc_para_lineEdit)

        ## start button
        startGroupBox = QGroupBox('Operator', self)
        startGroupBox.setFont(QFont('Arial', 10))
        startLayout = QHBoxLayout(startGroupBox)
        self.desc_start_button = QPushButton('Start')
        self.desc_start_button.clicked.connect(self.run_calculate_descriptor)
        self.desc_start_button.setFont(QFont('Arial', 10))
        self.desc_save_button = QPushButton('Save')
        self.desc_save_button.clicked.connect(self.save_descriptor)
        self.desc_save_button.setFont(QFont('Arial', 10))
        self.desc_slim_button = QPushButton('Show descriptor slims')
        self.desc_slim_button.clicked.connect(self.showDescriptorSlims)
        self.desc_slim_button.setFont(QFont('Arial', 10))

        startLayout.addWidget(self.desc_start_button)
        startLayout.addWidget(self.desc_save_button)
        startLayout.addWidget(self.desc_slim_button)

        ### layout
        left_vertical_layout = QVBoxLayout()
        left_vertical_layout.addWidget(topGroupBox)
        left_vertical_layout.addWidget(treeGroupBox)
        left_vertical_layout.addWidget(paraGroupBox)
        left_vertical_layout.addWidget(startGroupBox)

        #### widget
        leftWidget = QWidget()
        leftWidget.setLayout(left_vertical_layout)

        #### QTableWidget
        viewWidget = QTabWidget()
        self.desc_tableWidget = TableWidget.TableWidget()
        self.desc_histWidget = QWidget()
        self.desc_hist_layout = QVBoxLayout(self.desc_histWidget)
        self.desc_histogram = PlotWidgets.HistogramWidget()
        self.desc_hist_layout.addWidget(self.desc_histogram)
        viewWidget.addTab(self.desc_tableWidget, ' Data ')
        viewWidget.addTab(self.desc_histWidget, ' Data distribution ')

        ##### splitter
        splitter_1 = QSplitter(Qt.Horizontal)
        splitter_1.addWidget(leftWidget)
        splitter_1.addWidget(viewWidget)
        splitter_1.setSizes([100, 1000])

        ###### vertical layout
        vLayout = QVBoxLayout()

        ## status bar
        statusGroupBox = QGroupBox('Status', self)
        statusGroupBox.setFont(QFont('Arial', 10))
        statusLayout = QHBoxLayout(statusGroupBox)
        self.desc_status_label = QLabel('Welcome to the iLearnPlus Basic')
        self.desc_progress_bar = QLabel()
        self.desc_progress_bar.setMaximumWidth(230)
        statusLayout.addWidget(self.desc_status_label)
        statusLayout.addWidget(self.desc_progress_bar)

        splitter_2 = QSplitter(Qt.Vertical)
        splitter_2.addWidget(splitter_1)
        splitter_2.addWidget(statusGroupBox)
        splitter_2.setSizes([1000, 100])
        vLayout.addWidget(splitter_2)
        self.tab_descriptor.setLayout(vLayout)

    def setup_tab_cluster(self):
        # file
        topGroupBox = QGroupBox('Load data', self)
        topGroupBox.setFont(QFont('Arial', 10))
        topGroupBox.setMinimumHeight(100)
        topGroupBoxLayout = QGridLayout()

        self.clust_file_lineEdit = QLineEdit()
        self.clust_file_lineEdit.setFont(QFont('Arial', 8))
        self.clust_file_button = QPushButton('Open')
        self.clust_file_button.setFont(QFont('Arial', 10))
        self.clust_file_button.clicked.connect(self.data_from_file)
        self.clust_data_lineEdit = QLineEdit()
        self.clust_data_lineEdit.setFont(QFont('Arial', 8))
        self.clust_data_button = QPushButton('Select')
        self.clust_data_button.clicked.connect(self.data_from_descriptor)
        self.clust_label2 = QLabel('Data shape: ')
        topGroupBoxLayout.addWidget(self.clust_file_lineEdit, 0, 0)
        topGroupBoxLayout.addWidget(self.clust_file_button, 0, 1)
        topGroupBoxLayout.addWidget(self.clust_data_lineEdit, 1, 0)
        topGroupBoxLayout.addWidget(self.clust_data_button, 1, 1)
        topGroupBoxLayout.addWidget(self.clust_label2, 2, 0, 1, 2)
        topGroupBox.setLayout(topGroupBoxLayout)

        # tree
        treeGroupBox = QGroupBox('Analysis algorithms', self)
        treeGroupBox.setFont(QFont('Arial', 10))
        treeLayout = QHBoxLayout()
        self.clust_treeWidget = QTreeWidget()
        self.clust_treeWidget.setColumnCount(2)
        self.clust_treeWidget.setMinimumWidth(300)
        self.clust_treeWidget.setColumnWidth(0, 150)
        self.clust_treeWidget.setFont(QFont('Arial', 8))
        self.clust_treeWidget.setHeaderLabels(['Methods', 'Definition'])
        self.clusterMethods = QTreeWidgetItem(self.clust_treeWidget)
        self.clusterMethods.setExpanded(True)  # set node expanded
        self.clusterMethods.setText(0, 'Cluster algorithms')
        self.clust_treeWidget.clicked.connect(self.clust_tree_clicked)
        kmeans = QTreeWidgetItem(self.clusterMethods)
        kmeans.setText(0, 'kmeans')
        kmeans.setText(1, 'kmeans clustering')
        minikmeans = QTreeWidgetItem(self.clusterMethods)
        minikmeans.setText(0, 'MiniBatchKMeans')
        minikmeans.setText(1, 'MiniBatchKMeans clustering')
        gmm = QTreeWidgetItem(self.clusterMethods)
        gmm.setText(0, 'GM')
        gmm.setText(1, 'Gaussian mixture clustering')
        agg = QTreeWidgetItem(self.clusterMethods)
        agg.setText(0, 'Agglomerative')
        agg.setText(1, 'Agglomerative clustering')
        spectral = QTreeWidgetItem(self.clusterMethods)
        spectral.setText(0, 'Spectral')
        spectral.setText(1, 'Spectral clustering')
        mcl = QTreeWidgetItem(self.clusterMethods)
        mcl.setText(0, 'MCL')
        mcl.setText(1, 'Markov Cluster algorithm')
        hcluster = QTreeWidgetItem(self.clusterMethods)
        hcluster.setText(0, 'hcluster')
        hcluster.setText(1, 'Hierarchical clustering')
        apc = QTreeWidgetItem(self.clusterMethods)
        apc.setText(0, 'APC')
        apc.setText(1, 'Affinity Propagation Clustering')
        meanshift = QTreeWidgetItem(self.clusterMethods)
        meanshift.setText(0, 'meanshift')
        meanshift.setText(1, 'Mean-shift Clustering')
        dbscan = QTreeWidgetItem(self.clusterMethods)
        dbscan.setText(0, 'DBSCAN')
        dbscan.setText(1, 'DBSCAN Clustering')
        self.dimensionReduction = QTreeWidgetItem(self.clust_treeWidget)
        self.dimensionReduction.setExpanded(True)  # set node expanded
        self.dimensionReduction.setText(0, 'Dimensionality reduction algorithms')
        pca = QTreeWidgetItem(self.dimensionReduction)
        pca.setText(0, 'PCA')
        pca.setText(1, 'Principal component analysis')
        tsne = QTreeWidgetItem(self.dimensionReduction)
        tsne.setText(0, 't_SNE')
        tsne.setText(1, 't-distributed Stochastic Neighbor Embedding')
        lda = QTreeWidgetItem(self.dimensionReduction)
        lda.setText(0, 'LDA')
        lda.setText(1, 'Latent Dirichlet Allocation')
        treeLayout.addWidget(self.clust_treeWidget)
        treeGroupBox.setLayout(treeLayout)

        ## parameter
        paraGroupBox = QGroupBox('Parameters', self)
        paraGroupBox.setMaximumHeight(150)
        paraGroupBox.setFont(QFont('Arial', 10))
        paraLayout = QFormLayout(paraGroupBox)
        self.clust_analysisType_lineEdit = QLineEdit()
        self.clust_analysisType_lineEdit.setFont(QFont('Arial', 8))
        self.clust_analysisType_lineEdit.setEnabled(False)
        paraLayout.addRow('Analysis:', self.clust_analysisType_lineEdit)
        self.cluster_algorithm_lineEdit = QLineEdit()
        self.cluster_algorithm_lineEdit.setFont(QFont('Arial', 8))
        self.cluster_algorithm_lineEdit.setEnabled(False)
        paraLayout.addRow('Algorithm:', self.cluster_algorithm_lineEdit)
        self.clust_para_lineEdit = QLineEdit()
        self.clust_para_lineEdit.setFont(QFont('Arial', 8))
        self.clust_para_lineEdit.setEnabled(False)
        paraLayout.addRow('Parameter(s):', self.clust_para_lineEdit)

        ## start button
        startGroupBox = QGroupBox('Operator', self)
        startGroupBox.setFont(QFont('Arial', 10))
        startLayout = QHBoxLayout(startGroupBox)
        self.clust_start_button = QPushButton('Start')
        self.clust_start_button.clicked.connect(self.run_data_analysis)
        self.clust_start_button.setFont(QFont('Arial', 10))
        self.clust_save_button = QPushButton('Save txt')
        self.clust_save_button.setFont(QFont('Arial', 10))
        self.clust_save_button.clicked.connect(self.save_cluster_rd)
        # self.clust_image_button = QPushButton('Save image')
        # self.clust_image_button.setFont(QFont('Arial', 10))
        # self.clust_image_button.clicked.connect(self.save_image)
        startLayout.addWidget(self.clust_start_button)
        startLayout.addWidget(self.clust_save_button)
        # startLayout.addWidget(self.clust_image_button)

        ### layout
        left_vertical_layout = QVBoxLayout()
        left_vertical_layout.addWidget(topGroupBox)
        left_vertical_layout.addWidget(treeGroupBox)
        left_vertical_layout.addWidget(paraGroupBox)
        left_vertical_layout.addWidget(startGroupBox)

        #### widget
        leftWidget = QWidget()
        leftWidget.setLayout(left_vertical_layout)

        #### view region
        self.clust_tableWidget = QTableWidget()
        self.clust_tableWidget.setFont(QFont('Arial', 8))
        self.clust_diagram = PlotWidgets.ClusteringDiagramMatplotlib()
        clust_diagram_widget = QWidget()
        self.clust_diagram_layout = QVBoxLayout(clust_diagram_widget)
        self.clust_diagram_layout.addWidget(self.clust_diagram)
        clust_tabWidget = QTabWidget()
        clust_tabWidget.addTab(self.clust_tableWidget, ' Result ')
        clust_tabWidget.addTab(clust_diagram_widget, ' Scatter plot ')

        # self.clust_graph_panel = pg.PlotWidget()
        # splitter_view = QSplitter(Qt.Horizontal)
        # splitter_view.addWidget(self.clust_tableWidget)
        # splitter_view.addWidget(self.clust_graph_panel)
        # splitter_view.setSizes([100, 200])

        ##### splitter
        splitter_1 = QSplitter(Qt.Horizontal)
        splitter_1.addWidget(leftWidget)
        splitter_1.addWidget(clust_tabWidget)
        splitter_1.setSizes([100, 1200])

        ###### vertical layout
        vLayout = QVBoxLayout()

        ## status bar
        statusGroupBox = QGroupBox('Status', self)
        statusGroupBox.setFont(QFont('Arial', 10))
        statusLayout = QHBoxLayout(statusGroupBox)
        self.clust_status_label = QLabel('Welcome to iLearnPlus Basic')
        self.clust_progress_bar = QLabel()
        self.clust_progress_bar.setMaximumWidth(230)
        statusLayout.addWidget(self.clust_status_label)
        statusLayout.addWidget(self.clust_progress_bar)

        splitter_2 = QSplitter(Qt.Vertical)
        splitter_2.addWidget(splitter_1)
        splitter_2.addWidget(statusGroupBox)
        splitter_2.setSizes([1000, 100])
        vLayout.addWidget(splitter_2)
        self.tab_cluster.setLayout(vLayout)

    def setup_tab_selection(self):
        # file
        topGroupBox = QGroupBox('Load data', self)
        topGroupBox.setFont(QFont('Arial', 10))
        topGroupBox.setMinimumHeight(100)
        topGroupBoxLayout = QGridLayout()
        self.selection_file_lineEdit = QLineEdit()
        self.selection_file_lineEdit.setFont(QFont('Arial', 8))
        self.selection_file_button = QPushButton('Open')
        self.selection_file_button.clicked.connect(self.data_from_file_s)
        self.selection_file_button.setFont(QFont('Arial', 10))
        self.selection_data_lineEdit = QLineEdit()
        self.selection_data_lineEdit.setFont(QFont('Arial', 8))
        self.selection_data_button = QPushButton('Select')
        self.selection_data_button.clicked.connect(self.data_from_panel_s)
        self.selection_label2 = QLabel('Data shape: ')
        topGroupBoxLayout.addWidget(self.selection_file_lineEdit, 0, 0)
        topGroupBoxLayout.addWidget(self.selection_file_button, 0, 1)
        topGroupBoxLayout.addWidget(self.selection_data_lineEdit, 1, 0)
        topGroupBoxLayout.addWidget(self.selection_data_button, 1, 1)
        topGroupBoxLayout.addWidget(self.selection_label2, 2, 0, 1, 2)
        topGroupBox.setLayout(topGroupBoxLayout)

        # tree
        treeGroupBox = QGroupBox('Analysis algorithms', self)
        treeGroupBox.setFont(QFont('Arial', 10))
        treeLayout = QHBoxLayout()
        self.selection_treeWidget = QTreeWidget()
        self.selection_treeWidget.setColumnCount(2)
        self.selection_treeWidget.setMinimumWidth(300)
        self.selection_treeWidget.setColumnWidth(0, 150)
        self.selection_treeWidget.setFont(QFont('Arial', 8))
        self.selection_treeWidget.setHeaderLabels(['Methods', 'Definition'])
        self.selectionMethods = QTreeWidgetItem(self.selection_treeWidget)
        self.selectionMethods.setExpanded(True)  # set node expanded
        self.selectionMethods.setText(0, 'Feature selection algorithms')
        self.selection_treeWidget.clicked.connect(self.selection_tree_clicked)
        CHI2 = QTreeWidgetItem(self.selectionMethods)
        CHI2.setText(0, 'CHI2')
        CHI2.setText(1, 'Chi-Square feature selection')
        IG = QTreeWidgetItem(self.selectionMethods)
        IG.setText(0, 'IG')
        IG.setText(1, 'Information Gain feature selection')
        FScore = QTreeWidgetItem(self.selectionMethods)
        FScore.setText(0, 'FScore')
        FScore.setText(1, 'F-score value')
        MIC = QTreeWidgetItem(self.selectionMethods)
        MIC.setText(0, 'MIC')
        MIC.setText(1, 'Mutual Information feature selection')
        Pearsonr = QTreeWidgetItem(self.selectionMethods)
        Pearsonr.setText(0, 'Pearsonr')
        Pearsonr.setText(1, 'Pearson Correlation coefficient')

        self.normalizationMethods = QTreeWidgetItem(self.selection_treeWidget)
        self.normalizationMethods.setExpanded(True)  # set node expanded
        self.normalizationMethods.setText(0, 'Feature normalization algorithms')
        ZScore = QTreeWidgetItem(self.normalizationMethods)
        ZScore.setText(0, 'ZScore')
        ZScore.setText(1, 'ZScore')
        MinMax = QTreeWidgetItem(self.normalizationMethods)
        MinMax.setText(0, 'MinMax')
        MinMax.setText(1, 'MinMax')
        treeLayout.addWidget(self.selection_treeWidget)
        treeGroupBox.setLayout(treeLayout)

        ## parameter
        paraGroupBox = QGroupBox('Parameters', self)
        paraGroupBox.setMaximumHeight(150)
        paraGroupBox.setFont(QFont('Arial', 10))
        paraLayout = QFormLayout(paraGroupBox)
        self.selection_analysisType_lineEdit = QLineEdit()
        self.selection_analysisType_lineEdit.setFont(QFont('Arial', 8))
        self.selection_analysisType_lineEdit.setEnabled(False)
        paraLayout.addRow('Analysis:', self.selection_analysisType_lineEdit)
        self.selection_algorithm_lineEdit = QLineEdit()
        self.selection_algorithm_lineEdit.setFont(QFont('Arial', 8))
        self.selection_algorithm_lineEdit.setEnabled(False)
        paraLayout.addRow('Algorithm:', self.selection_algorithm_lineEdit)
        self.selection_para_lineEdit = QLineEdit()
        self.selection_para_lineEdit.setFont(QFont('Arial', 8))
        self.selection_para_lineEdit.setEnabled(False)
        paraLayout.addRow('Parameter(s):', self.selection_para_lineEdit)

        ## start button
        startGroupBox = QGroupBox('Operator', self)
        startGroupBox.setFont(QFont('Arial', 10))
        startLayout = QHBoxLayout(startGroupBox)
        self.selection_start_button = QPushButton('Start')
        self.selection_start_button.clicked.connect(self.run_selection)
        self.selection_start_button.setFont(QFont('Arial', 10))
        self.selection_save_button = QPushButton('Save txt')
        self.selection_save_button.setFont(QFont('Arial', 10))
        self.selection_save_button.clicked.connect(self.save_selection_normalization_data)
        startLayout.addWidget(self.selection_start_button)
        startLayout.addWidget(self.selection_save_button)

        ### layout
        left_vertical_layout = QVBoxLayout()
        left_vertical_layout.addWidget(topGroupBox)
        left_vertical_layout.addWidget(treeGroupBox)
        left_vertical_layout.addWidget(paraGroupBox)
        left_vertical_layout.addWidget(startGroupBox)

        #### widget
        leftWidget = QWidget()
        leftWidget.setLayout(left_vertical_layout)

        #### view region
        selection_viewWidget = QTabWidget()
        self.selection_tableWidget = TableWidget.TableWidgetForSelPanel()
        selection_histWidget = QWidget()
        self.selection_histLayout = QVBoxLayout(selection_histWidget)
        self.selection_hist = PlotWidgets.HistogramWidget()
        self.selection_histLayout.addWidget(self.selection_hist)
        selection_viewWidget.addTab(self.selection_tableWidget, ' Data ')
        selection_viewWidget.addTab(selection_histWidget, ' Data distribution ')

        ##### splitter
        splitter_1 = QSplitter(Qt.Horizontal)
        splitter_1.addWidget(leftWidget)
        splitter_1.addWidget(selection_viewWidget)
        splitter_1.setSizes([100, 1000])

        ###### vertical layout
        vLayout = QVBoxLayout()

        ## status bar
        statusGroupBox = QGroupBox('Status', self)
        statusGroupBox.setFont(QFont('Arial', 10))
        statusLayout = QHBoxLayout(statusGroupBox)
        self.selection_status_label = QLabel('Welcome to iLearnPlus Basic')
        self.selection_progress_bar = QLabel()
        self.selection_progress_bar.setMaximumWidth(230)
        statusLayout.addWidget(self.selection_status_label)
        statusLayout.addWidget(self.selection_progress_bar)

        splitter_2 = QSplitter(Qt.Vertical)
        splitter_2.addWidget(splitter_1)
        splitter_2.addWidget(statusGroupBox)
        splitter_2.setSizes([1000, 100])
        vLayout.addWidget(splitter_2)
        self.tab_selection.setLayout(vLayout)

    def setup_tab_machinelearning(self):
        # file
        topGroupBox = QGroupBox('Load data', self)
        topGroupBox.setFont(QFont('Arial', 10))
        topGroupBox.setMinimumHeight(100)
        topGroupBoxLayout = QFormLayout()
        trainFileButton = QPushButton('Open')
        trainFileButton.clicked.connect(lambda: self.data_from_file_ml('Training'))
        testFileButton = QPushButton('Open')
        testFileButton.clicked.connect(lambda: self.data_from_file_ml('Testing'))
        selectButton = QPushButton('Select')
        selectButton.clicked.connect(self.data_from_panel)
        topGroupBoxLayout.addRow('Open training file:', trainFileButton)
        topGroupBoxLayout.addRow('Open testing file:', testFileButton)
        topGroupBoxLayout.addRow('Select data:', selectButton)
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
        # deep learning algorighms
        net1 = QTreeWidgetItem(self.machineLearningAlgorighms)
        net1.setText(0, 'Net_1_CNN')
        net1.setText(1, 'Convolutional Neural Network')
        net2 = QTreeWidgetItem(self.machineLearningAlgorighms)
        net2.setText(0, 'Net_2_RNN')
        net2.setText(1, 'Recurrent Neural Network')
        net3 = QTreeWidgetItem(self.machineLearningAlgorighms)
        net3.setText(0, 'Net_3_BRNN')
        net3.setText(1, 'Bidirectional Recurrent Neural Network')
        net4 = QTreeWidgetItem(self.machineLearningAlgorighms)
        net4.setText(0, 'Net_4_ABCNN')
        net4.setText(1, 'Attention Based Convolutional Neural Network')
        net5 = QTreeWidgetItem(self.machineLearningAlgorighms)
        net5.setText(0, 'Net_5_ResNet')
        net5.setText(1, 'Deep Residual Network')
        net6 = QTreeWidgetItem(self.machineLearningAlgorighms)
        net6.setText(0, 'Net_6_AE')
        net6.setText(1, 'AutoEncoder')

        treeLayout.addWidget(self.ml_treeWidget)
        treeGroupBox.setLayout(treeLayout)

        ## parameter
        paraGroupBox = QGroupBox('Parameters', self)
        paraGroupBox.setMaximumHeight(150)
        paraGroupBox.setFont(QFont('Arial', 10))
        paraLayout = QFormLayout(paraGroupBox)
        self.ml_fold_lineEdit = InputDialog.MyLineEdit('5')
        self.ml_fold_lineEdit.setFont(QFont('Arial', 8))
        self.ml_fold_lineEdit.clicked.connect(self.setFold)
        paraLayout.addRow('Cross-Validation:', self.ml_fold_lineEdit)
        self.ml_algorithm_lineEdit = QLineEdit()
        self.ml_algorithm_lineEdit.setFont(QFont('Arial', 8))
        self.ml_algorithm_lineEdit.setEnabled(False)
        paraLayout.addRow('Algorithm:', self.ml_algorithm_lineEdit)
        self.ml_para_lineEdit = QLineEdit()
        self.ml_para_lineEdit.setFont(QFont('Arial', 8))
        self.ml_para_lineEdit.setEnabled(False)
        paraLayout.addRow('Parameter(s):', self.ml_para_lineEdit)

        ## start button
        startGroupBox = QGroupBox('Operator', self)
        startGroupBox.setFont(QFont('Arial', 10))
        startLayout = QHBoxLayout(startGroupBox)
        self.ml_start_button = QPushButton('Start')
        self.ml_start_button.clicked.connect(self.run_train_model)
        self.ml_start_button.setFont(QFont('Arial', 10))
        self.ml_save_button = QPushButton('Save')
        self.ml_save_button.setFont(QFont('Arial', 10))
        self.ml_save_button.clicked.connect(self.save_ml_files)
        startLayout.addWidget(self.ml_start_button)
        startLayout.addWidget(self.ml_save_button)

        ### layout
        left_vertical_layout = QVBoxLayout()
        left_vertical_layout.addWidget(topGroupBox)
        left_vertical_layout.addWidget(treeGroupBox)
        left_vertical_layout.addWidget(paraGroupBox)
        left_vertical_layout.addWidget(startGroupBox)

        #### widget
        leftWidget = QWidget()
        leftWidget.setLayout(left_vertical_layout)

        #### view region
        scoreTabWidget = QTabWidget()
        trainScoreWidget = QWidget()
        testScoreWidget = QWidget()
        scoreTabWidget.addTab(trainScoreWidget, 'Training data score')
        train_score_layout = QVBoxLayout(trainScoreWidget)
        self.train_score_tableWidget = QTableWidget()
        self.train_score_tableWidget.setFont(QFont('Arial', 8))
        self.train_score_tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.train_score_tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        train_score_layout.addWidget(self.train_score_tableWidget)
        scoreTabWidget.addTab(testScoreWidget, 'Testing data score')
        test_score_layout = QVBoxLayout(testScoreWidget)
        self.test_score_tableWidget = QTableWidget()
        self.test_score_tableWidget.setFont(QFont('Arial', 8))
        self.test_score_tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.test_score_tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        test_score_layout.addWidget(self.test_score_tableWidget)

        self.metricsTableWidget = QTableWidget()
        self.metricsTableWidget.setFont(QFont('Arial', 8))
        self.metricsTableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.metricsTableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.metricsTableWidget.resizeRowsToContents()
        splitter_middle = QSplitter(Qt.Vertical)
        splitter_middle.addWidget(scoreTabWidget)
        splitter_middle.addWidget(self.metricsTableWidget)

        self.dataTableWidget = QTableWidget(6, 4)
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
        self.ml_status_label = QLabel('Welcome to iLearnPlus Basic')
        self.ml_progress_bar = QLabel()
        self.ml_progress_bar.setMaximumWidth(230)
        statusLayout.addWidget(self.ml_status_label)
        statusLayout.addWidget(self.ml_progress_bar)

        splitter_2 = QSplitter(Qt.Vertical)
        splitter_2.addWidget(splitter_1)
        splitter_2.addWidget(statusGroupBox)
        splitter_2.setSizes([1000, 100])
        vLayout.addWidget(splitter_2)
        self.tab_machine.setLayout(vLayout)

    """ event in tab_descriptor """

    def get_fasta_file_name(self):
        self.desc_tableWidget.tableWidget.clear()
        self.desc_fasta_file, ok = QFileDialog.getOpenFileName(self, 'Open', './data', 'Plan text (*.*)')
        self.desc_file_lineEdit.setText(self.desc_fasta_file)
        if ok:
            self.desc_status_label.setText('Open file ' + self.desc_fasta_file)
            sequence = FileProcessing.Sequence(self.desc_fasta_file)
            self.descriptor = sequence
            if sequence.error_msg != '':
                self.desc_status_label.setText('<font color=red>Error: %s </font>' % sequence.error_msg)
                QMessageBox.critical(self, 'Error', str(sequence.error_msg), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
            if sequence.sequence_type == 'Protein':
                self.Protein.setDisabled(False)
                self.DNA.setDisabled(True)
                self.RNA.setDisabled(True)
                self.Protein.setExpanded(True)
                self.DNA.setExpanded(False)
                self.RNA.setExpanded(False)
                self.desc_status_label.setText('Sequence type is Protein.')
            if sequence.sequence_type == 'DNA':
                self.Protein.setDisabled(True)
                self.DNA.setDisabled(False)
                self.RNA.setDisabled(True)
                self.Protein.setExpanded(False)
                self.DNA.setExpanded(True)
                self.RNA.setExpanded(False)
                self.desc_status_label.setText('Sequence type is DNA.')
            if sequence.sequence_type == 'RNA':
                self.DNA.setDisabled(True)
                self.Protein.setDisabled(True)
                self.RNA.setDisabled(False)
                self.Protein.setExpanded(False)
                self.DNA.setExpanded(False)
                self.RNA.setExpanded(True)
                self.desc_status_label.setText('Sequence type is RNA.')

            del sequence
        else:
            self.Protein.setDisabled(True)
            self.DNA.setDisabled(True)
            self.RNA.setDisabled(True)

    def desc_tree_clicked(self, index):
        item = self.desc_treeWidget.currentItem()  # item = None if currentItem() is disabled
        if item and item.text(0) not in ['Protein', 'DNA', 'RNA'] and not item.isDisabled():
            self.desc_seq_type = item.parent().text(0)  # specify sequence type (DNA, RNA or Protein)
            self.desc_sequenceType_lineEdit.setText(self.desc_seq_type)

            # descriptors without parameters
            if item.text(0) in ['AAC', 'DPC', 'DDE', 'TPC', 'binary', 'GAAC', 'GDPC', 'GTPC', 'ZScale', 'BLOSUM62',
                                'CTDC', 'CTDT', 'CTDD', 'CTriad', 'NAC', 'DNC', 'TNC', 'ANF', 'NCP', 'PSTNPss',
                                'PSTNPds', 'EIIP', 'PseEIIP', 'OPF_10bit', 'binary_6bit']:
                self.desc_selected_descriptor = item.text(0)
                self.desc_currentDescriptor_lineEdit.setText(self.desc_selected_descriptor)
                self.desc_para_lineEdit.setText('None')
            if item.text(0) in ['OPF_7bit type 1', 'OPF_7bit type 2', 'OPF_7bit type 3', 'binary_5bit type 1',
                                'binary_5bit type 2', 'binary_3bit type 1', 'binary_3bit type 2', 'binary_3bit type 3',
                                'binary_3bit type 4', 'binary_3bit type 5', 'binary_3bit type 6', 'binary_3bit type 7',
                                'ASDC', 'DBE', 'LPDF', 'MMI', 'KNN', 'PS2', 'PS3', 'PS4', 'Z_curve_9bit', 'Z_curve_12bit',
                                'Z_curve_36bit', 'Z_curve_48bit', 'Z_curve_144bit', 'AESNN3']:
                self.desc_selected_descriptor = re.sub(' ', '_', item.text(0))
                self.desc_currentDescriptor_lineEdit.setText(item.text(0))
                self.desc_para_lineEdit.setText('None')
            # parameters with "sliding_window"
            elif item.text(0) in ['EAAC', 'EGAAC', 'ENAC']:
                self.desc_selected_descriptor = item.text(0)
                self.desc_currentDescriptor_lineEdit.setText(self.desc_selected_descriptor)
                num, ok = QInputDialog.getInt(self, '%s setting' % self.desc_selected_descriptor, 'Sliding window size',
                                              5, 2, 10, 1)
                if ok:
                    self.desc_default_para['sliding_window'] = num
                    self.desc_para_lineEdit.setText(str(num))
            # parameter with "kspace"
            elif item.text(0) in ['CKSAAP', 'CKSAAGP', 'KSCTriad', 'CKSNAP']:
                self.desc_selected_descriptor = item.text(0)
                self.desc_currentDescriptor_lineEdit.setText(self.desc_selected_descriptor)
                num, ok = QInputDialog.getInt(self, '%s setting' % self.desc_selected_descriptor, 'K-space number', 3,
                                              0, 5, 1)
                if ok:
                    self.desc_default_para['kspace'] = num
                    self.desc_para_lineEdit.setText(str(num))
            elif item.text(0) in ['SOCNumber']:
                self.desc_selected_descriptor = item.text(0)
                self.desc_currentDescriptor_lineEdit.setText(self.desc_selected_descriptor)
                num, ok = QInputDialog.getInt(self, '%s setting' % self.desc_selected_descriptor, 'lag value', 3, 1,
                                              self.descriptor.minimum_length_without_minus - 1, 1)
                if ok:
                    self.desc_default_para['nlag'] = num
                    self.desc_para_lineEdit.setText(str(num))
            elif item.text(0) in ['QSOrder']:
                self.desc_selected_descriptor = item.text(0)
                self.desc_currentDescriptor_lineEdit.setText(self.desc_selected_descriptor)
                lag, weight, ok, = InputDialog.QSOrderInput.getValues(self.descriptor.minimum_length_without_minus - 1)
                if ok:
                    self.desc_para_lineEdit.setText('Lag: %s; Weight: %s' % (lag, weight))
                    self.desc_default_para['nlag'] = lag
                    self.desc_default_para['weight'] = weight
            elif item.text(0) in ['AAIndex']:
                self.desc_selected_descriptor = item.text(0)
                self.desc_currentDescriptor_lineEdit.setText(self.desc_selected_descriptor)
                property, ok = InputDialog.QAAindexInput.getValues()
                if ok:
                    self.desc_para_lineEdit.setText('AAIndex: %s' % property)
                    self.desc_default_para['aaindex'] = property
            elif item.text(0) in ['NMBroto', 'Moran', 'Geary', 'AC', 'CC', 'ACC']:
                self.desc_selected_descriptor = item.text(0)
                self.desc_currentDescriptor_lineEdit.setText(self.desc_selected_descriptor)
                if self.desc_seq_type == 'Protein':
                    lag, property, ok = InputDialog.QAAindex2Input.getValues(
                        self.descriptor.minimum_length_without_minus - 1)
                    if ok:
                        self.desc_para_lineEdit.setText('Lag: %s; AAIndex: %s' % (lag, property))
                        self.desc_default_para['aaindex'] = property
                        self.desc_default_para['nlag'] = lag
                if self.desc_seq_type == 'DNA':
                    num, property, ok = InputDialog.QDNAACC2Input.getValues(
                        self.descriptor.minimum_length_without_minus - 2)
                    if ok:
                        self.desc_default_para['nlag'] = num
                        self.desc_default_para['Di-DNA-Phychem'] = property
                        self.desc_para_lineEdit.setText(
                            'Lag: %s Phychem: %s' % (num, self.desc_default_para['Di-DNA-Phychem']))
                if self.desc_seq_type == 'RNA':
                    num, property, ok = InputDialog.QRNAACC2Input.getValues(
                        self.descriptor.minimum_length_without_minus - 2)
                    if ok:
                        self.desc_default_para['nlag'] = num
                        self.desc_default_para['Di-RNA-Phychem'] = property
                        self.desc_para_lineEdit.setText(
                            'Lag: %s Phychem: %s' % (num, self.desc_default_para['Di-RNA-Phychem']))
            elif item.text(0) in ['PAAC', 'APAAC']:
                self.desc_selected_descriptor = item.text(0)
                self.desc_currentDescriptor_lineEdit.setText(self.desc_selected_descriptor)
                lambdaValue, weight, ok, = InputDialog.QPAACInput.getValues(
                    self.descriptor.minimum_length_without_minus - 1)
                if ok:
                    self.desc_para_lineEdit.setText('Lag: %s; Weight: %s' % (lambdaValue, weight))
                    self.desc_default_para['lambdaValue'] = lambdaValue
                    self.desc_default_para['weight'] = weight
            elif item.text(0) in ['PseKRAAC type 1', 'PseKRAAC type 2', 'PseKRAAC type 3A', 'PseKRAAC type 3B',
                                  'PseKRAAC type 5',
                                  'PseKRAAC type 6A', 'PseKRAAC type 6B', 'PseKRAAC type 6C', 'PseKRAAC type 7',
                                  'PseKRAAC type 8',
                                  'PseKRAAC type 9', 'PseKRAAC type 10', 'PseKRAAC type 11', 'PseKRAAC type 12',
                                  'PseKRAAC type 13',
                                  'PseKRAAC type 14', 'PseKRAAC type 15', 'PseKRAAC type 16']:
                self.desc_selected_descriptor = re.sub(' ', '_', item.text(0))
                self.desc_currentDescriptor_lineEdit.setText(item.text(0))
                model, gap, lambdaValue, ktuple, clust, ok = InputDialog.QPseKRAACInput.getValues(item.text(0))
                if ok:
                    text = 'Model: %s ' % model
                    if model == 'g-gap':
                        text += 'g-gap: %s k-tuple: %s' % (gap, ktuple)
                    else:
                        text += 'lambda: %s k-tuple: %s' % (lambdaValue, ktuple)
                    self.desc_para_lineEdit.setText(text)
                    self.desc_default_para['PseKRAAC_model'] = model
                    self.desc_default_para['g-gap'] = int(gap)
                    self.desc_default_para['lambdaValue'] = int(lambdaValue)
                    self.desc_default_para['k-tuple'] = int(ktuple)
                    self.desc_default_para['RAAC_clust'] = int(clust)
            elif item.text(0) in ['Kmer', 'RCKmer']:
                self.desc_selected_descriptor = item.text(0)
                self.desc_currentDescriptor_lineEdit.setText(self.desc_selected_descriptor)
                num, ok = QInputDialog.getInt(self, '%s setting' % self.desc_selected_descriptor, 'Kmer size', 3, 1, 6,
                                              1)
                if ok:
                    self.desc_default_para['kmer'] = num
                    self.desc_para_lineEdit.setText(str(num))
            elif item.text(0) in ['DPCP', 'DPCP type2']:
                self.desc_selected_descriptor = re.sub(' ', '_', item.text(0))
                self.desc_currentDescriptor_lineEdit.setText(item.text(0))
                if self.desc_seq_type == 'DNA':
                    property, ok = InputDialog.QDNADPCPInput.getValues()
                    if ok:
                        self.desc_default_para['Di-DNA-Phychem'] = property
                        self.desc_para_lineEdit.setText(
                            'Phychem: %s' % (self.desc_default_para['Di-DNA-Phychem']))
                else:
                    property, ok = InputDialog.QRNADPCPInput.getValues()
                    if ok:
                        self.desc_default_para['Di-RNA-Phychem'] = property
                        self.desc_para_lineEdit.setText(
                            'Phychem: %s' % (self.desc_default_para['Di-RNA-Phychem']))
            elif item.text(0) in ['TPCP', 'TPCP type2']:
                self.desc_selected_descriptor = re.sub(' ', '_', item.text(0))
                self.desc_currentDescriptor_lineEdit.setText(item.text(0))
                if self.desc_seq_type == 'DNA':
                    property, ok = InputDialog.QDNATPCPInput.getValues()
                    if ok:
                        self.desc_default_para['Tri-DNA-Phychem'] = property
                        self.desc_para_lineEdit.setText(
                            'Phychem: %s' % (self.desc_default_para['Tri-DNA-Phychem']))
            elif item.text(0) in ['DAC', 'DCC', 'DACC']:
                self.desc_selected_descriptor = item.text(0)
                self.desc_currentDescriptor_lineEdit.setText(self.desc_selected_descriptor)
                if self.desc_seq_type == 'DNA':
                    num, property, ok = InputDialog.QDNAACC2Input.getValues(
                        self.descriptor.minimum_length_without_minus - 2)
                    if ok:
                        self.desc_default_para['nlag'] = num
                        self.desc_default_para['Di-DNA-Phychem'] = property
                        self.desc_para_lineEdit.setText(
                            'Lag: %s Phychem: %s' % (num, self.desc_default_para['Di-DNA-Phychem']))
                else:
                    num, property, ok = InputDialog.QRNAACC2Input.getValues(
                        self.descriptor.minimum_length_without_minus - 2)
                    if ok:
                        self.desc_default_para['nlag'] = num
                        self.desc_default_para['Di-RNA-Phychem'] = property
                        self.desc_para_lineEdit.setText(
                            'Lag: %s Phychem: %s' % (num, self.desc_default_para['Di-RNA-Phychem']))
            elif item.text(0) in ['TAC', 'TCC', 'TACC']:
                self.desc_selected_descriptor = item.text(0)
                self.desc_currentDescriptor_lineEdit.setText(self.desc_selected_descriptor)
                if self.desc_seq_type == 'DNA':
                    num, property, ok = InputDialog.QDNAACC3Input.getValues(
                        self.descriptor.minimum_length_without_minus - 3)
                    if ok:
                        self.desc_default_para['nlag'] = num
                        self.desc_default_para['Tri-DNA-Phychem'] = property
                        self.desc_para_lineEdit.setText(
                            'Lag: %s Phychem: %s' % (num, self.desc_default_para['Tri-DNA-Phychem']))
            elif item.text(0) in ['PseDNC', 'PCPseDNC', 'SCPseDNC']:
                self.desc_selected_descriptor = item.text(0)
                self.desc_currentDescriptor_lineEdit.setText(self.desc_selected_descriptor)
                if self.desc_seq_type == 'DNA':
                    num, weight, property, ok = InputDialog.QDNAPse2Input.getValues(
                        self.descriptor.minimum_length_without_minus - 2)
                    if ok:
                        self.desc_default_para['lambdaValue'] = num
                        self.desc_default_para['weight'] = weight
                        self.desc_default_para['Di-DNA-Phychem'] = property
                        self.desc_para_lineEdit.setText('Lambda: %s; Weight: %s; Phychem: %s' % (
                            num, weight, self.desc_default_para['Di-DNA-Phychem']))
                else:
                    num, weight, property, ok = InputDialog.QRNAPse2Input.getValues(
                        self.descriptor.minimum_length_without_minus - 2)
                    if ok:
                        self.desc_default_para['lambdaValue'] = num
                        self.desc_default_para['weight'] = weight
                        self.desc_default_para['Di-RNA-Phychem'] = property
                        self.desc_para_lineEdit.setText('Lambda: %s; Weight: %s; Phychem: %s' % (
                            num, weight, self.desc_default_para['Di-DNA-Phychem']))
            elif item.text(0) in ['PCPseTNC', 'SCPseTNC']:
                self.desc_selected_descriptor = item.text(0)
                self.desc_currentDescriptor_lineEdit.setText(self.desc_selected_descriptor)
                if self.desc_seq_type == 'DNA':
                    num, weight, property, ok = InputDialog.QDNAPse3Input.getValues(
                        self.descriptor.minimum_length_without_minus - 3)
                    if ok:
                        self.desc_default_para['lambdaValue'] = num
                        self.desc_default_para['weight'] = weight
                        self.desc_default_para['Tri-DNA-Phychem'] = property
                        self.desc_para_lineEdit.setText('Lambda: %s; Weight: %s; Phychem: %s' % (
                            num, weight, self.desc_default_para['Tri-DNA-Phychem']))
            elif item.text(0) in ['PseKNC']:
                self.desc_selected_descriptor = item.text(0)
                self.desc_currentDescriptor_lineEdit.setText(self.desc_selected_descriptor)
                if self.desc_seq_type == 'DNA':
                    num, weight, kmer, property, ok = InputDialog.QDNAPseKNCInput.getValues(
                        self.descriptor.minimum_length_without_minus - 2)
                    if ok:
                        self.desc_default_para['lambdaValue'] = num
                        self.desc_default_para['weight'] = weight
                        self.desc_default_para['kmer'] = kmer
                        self.desc_default_para['Di-DNA-Phychem'] = property
                        self.desc_para_lineEdit.setText('Lambda: %s; Weight: %s; Kmer: %s; Phychem: %s' % (
                            num, weight, kmer, self.desc_default_para['Di-DNA-Phychem']))
                else:
                    num, weight, kmer, property, ok = InputDialog.QRNAPseKNCInput.getValues(
                        self.descriptor.minimum_length_without_minus - 2)
                    if ok:
                        self.desc_default_para['lambdaValue'] = num
                        self.desc_default_para['weight'] = weight
                        self.desc_default_para['kmer'] = kmer
                        self.desc_default_para['Di-RNA-Phychem'] = property
                        self.desc_para_lineEdit.setText('Lambda: %s; Weight: %s; Kmer: %s; Phychem: %s' % (
                            num, weight, kmer, self.desc_default_para['Di-DNA-Phychem']))
            elif item.text(0) in ['Mismatch']:
                self.desc_selected_descriptor = item.text(0)
                self.desc_currentDescriptor_lineEdit.setText(self.desc_selected_descriptor)
                num_k, num_m, ok = InputDialog.QMismatchInput.getValues()
                if ok:
                    if num_m >= num_k:
                        num_m = num_k - 1
                    self.desc_default_para['kmer'] = num_k
                    self.desc_default_para['mismatch'] = num_m
                    self.desc_para_lineEdit.setText('Kmer size: %s; Mismatch: %s' %(num_k, num_m))
            elif item.text(0) in ['Subsequence']:
                self.desc_selected_descriptor = item.text(0)
                self.desc_currentDescriptor_lineEdit.setText(self.desc_selected_descriptor)
                num, delta, ok = InputDialog.QSubsequenceInput.getValues()
                if ok:
                    self.desc_default_para['kmer'] = num
                    self.desc_default_para['delta'] = delta
                    self.desc_para_lineEdit.setText('Kmer size: %s; Delta: %s' % (num, delta))
            elif item.text(0) in ['DistancePair']:
                self.desc_selected_descriptor = item.text(0)
                self.desc_currentDescriptor_lineEdit.setText(self.desc_selected_descriptor)
                num, cp, ok = InputDialog.QDistancePairInput.getValues()
                if ok:
                    self.desc_default_para['distance'] = num
                    self.desc_default_para['cp'] = cp
                    self.desc_para_lineEdit.setText('Maximum distance: %s; Cp: %s' % (num, cp))
            else:
                pass
        else:
            pass

    def run_calculate_descriptor(self):
        self.desc_running_status = False
        if self.desc_fasta_file != '' and self.desc_selected_descriptor != '':
            self.desc_progress_bar.setMovie(self.gif)
            self.gif.start()
            t = threading.Thread(target=self.calculate_descriptor)
            t.start()
        else:
            QMessageBox.warning(self, 'Warning', 'Please check your input!', QMessageBox.Ok | QMessageBox.No,
                                QMessageBox.Ok)

    def calculate_descriptor(self):
        try:
            self.descriptor = None
            self.desc_start_button.setDisabled(True)
            self.tab_descriptor.setDisabled(True)
            self.setTabEnabled(1, False)
            self.setTabEnabled(2, False)
            self.setTabEnabled(3, False)

            if self.desc_fasta_file != '' and self.desc_selected_descriptor != '':
                self.descriptor = FileProcessing.Descriptor(self.desc_fasta_file, self.desc_default_para)
                if self.descriptor.error_msg == '' and self.descriptor.sequence_number > 0:
                    self.desc_status_label.setText('Calculating ...')
                    status = False
                    if self.descriptor.sequence_type == 'Protein':
                        cmd = 'self.descriptor.' + self.descriptor.sequence_type + '_' + self.desc_selected_descriptor + '()'
                        status = eval(cmd)
                    else:
                        if self.desc_selected_descriptor in ['DAC', 'TAC']:
                            my_property_name, my_property_value, my_kmer, ok = CheckAccPseParameter.check_acc_arguments(
                                self.desc_selected_descriptor, self.desc_seq_type, self.desc_default_para)
                            status = self.descriptor.make_ac_vector(my_property_name, my_property_value, my_kmer)
                        elif self.desc_selected_descriptor in ['DCC', 'TCC']:
                            my_property_name, my_property_value, my_kmer, ok = CheckAccPseParameter.check_acc_arguments(
                                self.desc_selected_descriptor, self.desc_seq_type, self.desc_default_para)
                            status = self.descriptor.make_cc_vector(my_property_name, my_property_value, my_kmer)
                        elif self.desc_selected_descriptor in ['DACC', 'TACC']:
                            my_property_name, my_property_value, my_kmer, ok = CheckAccPseParameter.check_acc_arguments(
                                self.desc_selected_descriptor, self.desc_seq_type, self.desc_default_para)
                            status = self.descriptor.make_acc_vector(my_property_name, my_property_value, my_kmer)
                        elif self.desc_selected_descriptor in ['PseDNC', 'PseKNC', 'PCPseDNC', 'PCPseTNC', 'SCPseDNC',
                                                            'SCPseTNC']:
                            my_property_name, my_property_value, ok = CheckAccPseParameter.check_Pse_arguments(
                                self.desc_selected_descriptor, self.desc_seq_type, self.desc_default_para)
                            cmd = 'self.descriptor.' + self.desc_selected_descriptor + '(my_property_name, my_property_value)'
                            status = eval(cmd)
                        else:
                            cmd = 'self.descriptor.' + self.desc_selected_descriptor + '()'
                            status = eval(cmd)
                    self.desc_status_label.setText('Calculation complete.')
            else:
                QMessageBox.warning(self, 'Warning', 'Please check your input!', QMessageBox.Ok | QMessageBox.No,
                                    QMessageBox.Ok)
            self.desc_running_status = status
            self.desc_signal.emit()
            self.desc_progress_bar.clear()
            self.desc_start_button.setDisabled(False)
            self.tab_descriptor.setDisabled(False)
            self.setTabEnabled(1, True)
            self.setTabEnabled(2, True)
            self.setTabEnabled(3, True)
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

    def set_table_header(self, row, column, header):
        self.desc_tableWidget.setColumnCount(column)
        self.desc_tableWidget.setRowCount(row)
        self.desc_tableWidget.setHorizontalHeaderLabels(header)

    def set_table_content(self):
        if self.desc_running_status and not self.descriptor is None:
            self.desc_tableWidget.init_data(self.descriptor.encoding_array[0], self.descriptor.encoding_array[1:])
            # Draw histogram
            self.desc_hist_layout.removeWidget(self.desc_histogram)
            sip.delete(self.desc_histogram)
            data = self.descriptor.encoding_array[1:, 1:].astype(float)
            self.desc_histogram = PlotWidgets.HistogramWidget()
            self.desc_histogram.init_data('All data', data)
            self.desc_hist_layout.addWidget(self.desc_histogram)
        else:
            QMessageBox.critical(self, 'Error', str(self.descriptor.error_msg), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

    def save_descriptor(self):
        try:
            if self.descriptor and 'encoding_array' in dir(self.descriptor):
                saved_file, ok = QFileDialog.getSaveFileName(self, 'Save', './data', 'CSV Files (*.csv);;TSV Files (*.tsv);;TSV Files with labels (*.tsv1);;SVM Files(*.svm);;Weka Files (*.arff)')
                if ok:
                    self.descriptor.save_descriptor(saved_file)
            else:
                QMessageBox.critical(self, 'Error', 'Empty data!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

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
    
    """ event in tab_cluster """

    def data_from_file(self):
        self.clust_file, ok = QFileDialog.getOpenFileName(self, 'Open', './data',
                                                          'CSV Files (*.csv);;TSV Files (*.tsv);;SVM Files(*.svm);;Weka Files (*.arff)')
        self.clust_file_lineEdit.setText(self.clust_file)
        if ok:
            self.clust_status_label.setText('Open file ' + self.clust_file)
            self.clust_data = DataAnalysis.ILearnData(self.clust_default_para)
            ok1 = self.clust_data.load_data_from_file(self.clust_file)
            if ok1:
                self.clust_label2.setText('Data shape: (%s, %s)' % (self.clust_data.row, self.clust_data.column))
                self.clust_status_label.setText('Open file %s successfully.' % self.clust_file)
                self.clust_data_lineEdit.setText('')
            else:
                self.clust_status_label.setText(
                    'Open file %s failed. Error: %s' % (self.clust_file, self.clust_data.error_msg))

    def data_from_descriptor(self):
        data_source, ok = InputDialog.QDataSelection.getValues(descriptor=self.descriptor, selection=self.selection_data)
        if ok and data_source == 'Descriptor data':
            self.clust_data = DataAnalysis.ILearnData(self.clust_default_para)
            ok = self.clust_data.load_data_from_descriptor(self.descriptor)
            if ok:
                self.clust_label2.setText('Data shape: (%s, %s)' % (self.clust_data.row, self.clust_data.column))
                self.clust_data_lineEdit.setText('Data From <Descriptor> panel')
                self.clust_file_lineEdit.setText('')
        if ok and data_source == 'Feature selection data':
            self.clust_data = DataAnalysis.ILearnData(self.clust_default_para)
            ok = self.clust_data.load_data_from_selection(self.selection_data)
            if ok:
                self.clust_label2.setText('Data shape: (%s, %s)' % (self.clust_data.row, self.clust_data.column))
                self.clust_data_lineEdit.setText('Data From <Selection> panel')
                self.clust_file_lineEdit.setText('')
        if ok and data_source == 'Feature normalization data':
            self.clust_data = DataAnalysis.ILearnData(self.clust_default_para)
            ok = self.clust_data.load_data_from_normalization(self.selection_data)
            if ok:
                self.clust_label2.setText('Data shape: (%s, %s)' % (self.clust_data.row, self.clust_data.column))
                self.clust_data_lineEdit.setText('Data From <Selection> panel')
                self.clust_file_lineEdit.setText('')

    def clust_tree_clicked(self, index):
        item = self.clust_treeWidget.currentItem()  # item = None if currentItem() is disabled
        if item and item.text(0) not in ['Cluster algorithms', 'Dimensionality reduction algorithms']:
            self.clust_analysis_type = item.parent().text(0)
            self.clust_analysisType_lineEdit.setText(self.clust_analysis_type)
            if item.text(0) in ['kmeans', 'MiniBatchKMeans', 'GM', 'Agglomerative', 'Spectral']:
                self.clust_selected_algorithm = item.text(0)
                self.cluster_algorithm_lineEdit.setText(self.clust_selected_algorithm)
                num, ok = QInputDialog.getInt(self, '%s setting' % self.clust_selected_algorithm, 'Cluster number', 2,
                                              2, 10, 1)
                if ok:
                    self.clust_default_para['nclusters'] = num
                    self.clust_para_lineEdit.setText('Cluster number: %s' % num)
            elif item.text(0) in ['MCL']:
                self.clust_selected_algorithm = item.text(0)
                self.cluster_algorithm_lineEdit.setText(self.clust_selected_algorithm)
                expand, inflate, mult, ok = InputDialog.QMCLInput.getValues()
                if ok:
                    self.clust_default_para['expand_factor'] = expand
                    self.clust_default_para['inflate_factor'] = inflate
                    self.clust_default_para['multiply_factor'] = mult
                    self.clust_para_lineEdit.setText('Expand: %s; Inflate: %s; Multiply: %s' % (expand, inflate, mult))
            elif item.text(0) in ['hcluster', 'APC', 'meanshift', 'DBSCAN']:
                self.clust_selected_algorithm = item.text(0)
                self.cluster_algorithm_lineEdit.setText(self.clust_selected_algorithm)
                self.clust_para_lineEdit.setText('None')
            elif item.text(0) in ['PCA', 't_SNE', 'LDA']:
                self.clust_selected_algorithm = item.text(0)
                self.cluster_algorithm_lineEdit.setText(self.clust_selected_algorithm)
                num, ok = QInputDialog.getInt(self, '%s setting' % self.clust_selected_algorithm,
                                              'Reduced number of dimensions', 2, 2, 10000, 1)
                if ok:
                    self.clust_default_para['n_components'] = num
                    self.clust_para_lineEdit.setText('Reduced number of dimensions: %s' % num)

    def run_data_analysis(self):
        if self.clust_selected_algorithm != '' and not self.clust_data is None:
            self.clust_status = False
            self.clust_progress_bar.setMovie(self.gif)
            self.gif.start()
            t = threading.Thread(target=self.data_analysis)
            t.start()
        else:
            if self.clust_data is None:
                QMessageBox.critical(self, 'Error', 'Empty data!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
            else:
                QMessageBox.critical(self, 'Error', 'Please select an analysis algorithm.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

    def data_analysis(self):
        if self.clust_selected_algorithm != '' and not self.clust_data is None:
            self.clust_start_button.setDisabled(True)
            self.tab_cluster.setDisabled(True)
            self.setTabEnabled(0, False)
            self.setTabEnabled(2, False)
            self.setTabEnabled(3, False)
            self.clust_status_label.setText('Calculating ...')
            if self.clust_analysis_type == 'Cluster algorithms':
                cmd = 'self.clust_data.' + self.clust_selected_algorithm + '()'
                try:
                    status = eval(cmd)
                    self.clust_status = status
                except Exception as e:
                    self.clust_data.error_msg = 'Clustering failed.'
                    status = False
            else:
                if self.clust_selected_algorithm == 't_SNE':
                    algo = 't_sne'
                else:
                    algo = self.clust_selected_algorithm
                # Note: clust_data.dimension_reduction_result used to show RD data in QTableWidget
                cmd = 'self.clust_data.' + algo + '(self.clust_default_para["n_components"])'                
                try:
                    self.clust_data.dimension_reduction_result, status = eval(cmd)
                    # When ploting, the RD data used in n_components = 2, because when RD data with more than 2-D,
                    self.clust_data.cluster_plot_data, _ = self.clust_data.t_sne(2)
                    self.clust_status = status          
                except Exception as e:
                    self.clust_data.error_msg = str(e)
                    self.clust_status = False
            self.clust_start_button.setDisabled(False)
            self.tab_cluster.setDisabled(False)
            self.setTabEnabled(0, True)
            self.setTabEnabled(2, True)
            self.setTabEnabled(3, True)
            self.clust_signal.emit()
            self.clust_progress_bar.clear()
        else:
            if self.clust_data is None:
                QMessageBox.critical(self, 'Error', 'Empty data!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
            elif self.clust_selected_algorithm == '':
                QMessageBox.critical(self, 'Error', 'Please select an analysis algorithm.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
            else:
                QMessageBox.critical(self, 'Error', str(self.clust_data.error_msg), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

    def display_data_analysis(self):
        if self.clust_analysis_type == 'Cluster algorithms':
            if self.clust_status:
                self.clust_status_label.setText('%s calculation complete.' % self.clust_selected_algorithm)
                self.clust_tableWidget.setColumnCount(2)
                self.clust_tableWidget.setRowCount(self.clust_data.row)
                self.clust_tableWidget.setHorizontalHeaderLabels(['SampleName', 'Cluster'])
                for i in range(self.clust_data.row):
                    cell = QTableWidgetItem(self.clust_data.dataframe.index[i])
                    self.clust_tableWidget.setItem(i, 0, cell)
                    cell1 = QTableWidgetItem(str(self.clust_data.cluster_result[i]))
                    self.clust_tableWidget.setItem(i, 1, cell1)
                """ plot with Matplotlib """
                self.clust_diagram_layout.removeWidget(self.clust_diagram)
                sip.delete(self.clust_diagram)
                plot_data = self.clust_data.generate_plot_data(self.clust_data.cluster_result,
                                                               self.clust_data.cluster_plot_data)
                self.clust_diagram = PlotWidgets.ClusteringDiagramMatplotlib()
                self.clust_diagram.init_data('Clustering', plot_data)
                self.clust_diagram_layout.addWidget(self.clust_diagram)
            else:
                self.clust_status_label.setText(str(self.clust_data.error_msg))
                QMessageBox.critical(self, 'Calculate failed', '%s' % self.clust_data.error_msg,
                                     QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        else:
            if self.clust_status:
                self.clust_status_label.setText('%s calculation complete.' % self.clust_selected_algorithm)
                self.clust_tableWidget.setColumnCount(self.clust_default_para['n_components'] + 1)
                self.clust_tableWidget.setRowCount(self.clust_data.row)
                self.clust_tableWidget.setHorizontalHeaderLabels(
                    ['SampleName'] + ['PC%s' % i for i in range(1, self.clust_default_para['n_components'] + 1)])
                for i in range(self.clust_data.row):
                    cell = QTableWidgetItem(self.clust_data.dataframe.index[i])
                    self.clust_tableWidget.setItem(i, 0, cell)
                    for j in range(self.clust_default_para['n_components']):
                        cell = QTableWidgetItem(str(self.clust_data.dimension_reduction_result[i][j]))
                        self.clust_tableWidget.setItem(i, j + 1, cell)
                
                """ plot with Matplotlib """
                self.clust_diagram_layout.removeWidget(self.clust_diagram)
                sip.delete(self.clust_diagram)
                plot_data = self.clust_data.generate_plot_data(self.clust_data.datalabel,
                                                               self.clust_data.cluster_plot_data)
                self.clust_diagram = PlotWidgets.ClusteringDiagramMatplotlib()
                self.clust_diagram.init_data('Dimension reduction', plot_data)
                self.clust_diagram_layout.addWidget(self.clust_diagram)
            else:
                self.clust_status_label.setText(self.clust_data.error_msg)
                QMessageBox.critical(self, 'Calculate failed', '%s' % self.clust_data.error_msg,
                                     QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

    def save_cluster_rd(self):
        try:
            if self.clust_analysis_type != '' and (
                    not self.clust_data.cluster_result is None or not self.clust_data.dimension_reduction_result is None):
                saved_file, ok = QFileDialog.getSaveFileName(self, 'Save', './data', 'TXT Files (*.txt)')
                if ok:
                    self.clust_data.save_data(saved_file, self.clust_analysis_type)
                
            else:
                QMessageBox.critical(self, 'Error', 'Empty data!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

    """ event in tab_selection """

    def data_from_file_s(self):
        self.selection_file, ok = QFileDialog.getOpenFileName(self, 'Open', './data',
                                                              'CSV Files (*.csv);;TSV Files (*.tsv);;SVM Files(*.svm);;Weka Files (*.arff)')
        self.selection_file_lineEdit.setText(self.selection_file)
        if ok:
            self.selection_status_label.setText('Open file ' + self.selection_file)
            self.selection_data = DataAnalysis.ILearnData(self.selection_default_para)
            ok1 = self.selection_data.load_data_from_file(self.selection_file)
            if ok1:
                self.selection_label2.setText(
                    'Data shape: (%s, %s)' % (self.selection_data.row, self.selection_data.column))
                self.selection_status_label.setText('Open file %s successfully.' % self.selection_file)
            else:
                self.selection_status_label.setText(
                    'Open file %s failed. Error: %s' % (self.selection_file, self.selection_data.error_msg))

    def data_from_panel_s(self):
        data_source, ok = InputDialog.QDataSelection.getValues(descriptor=self.descriptor, reduction=self.clust_data)
        if ok and data_source == 'Descriptor data':
            self.selection_data = DataAnalysis.ILearnData(self.selection_default_para)
            ok = self.selection_data.load_data_from_descriptor(self.descriptor)
            if ok:
                self.selection_label2.setText(
                    'Data shape: (%s, %s)' % (self.selection_data.row, self.selection_data.column))
                self.selection_data_lineEdit.setText('Data From <Descriptor> panel')
                self.selection_file_lineEdit.setText('')
        if ok and data_source == 'Dimensionality reduction data':
            self.selection_data = DataAnalysis.ILearnData(self.selection_default_para)
            ok = self.selection_data.load_data_from_dimension_reduction(self.clust_data)
            if ok:
                self.selection_label2.setText(
                    'Data shape: (%s, %s)' % (self.selection_data.row, self.selection_data.column))
                self.selection_data_lineEdit.setText('Data From <Dimensionality reduction> panel')
                self.selection_file_lineEdit.setText('')

    def selection_tree_clicked(self, index):
        item = self.selection_treeWidget.currentItem()  # item = None if currentItem() is disabled
        if item and item.text(0) not in ['Feature selection algorithms', 'Feature Normalization algorithms']:
            self.selection_analysis_type = item.parent().text(0)
            self.selection_analysisType_lineEdit.setText(self.selection_analysis_type)
            if item.text(0) in ['CHI2', 'IG', 'FScore', 'MIC', 'Pearsonr']:
                self.selection_selected_algorithm = item.text(0)
                self.selection_algorithm_lineEdit.setText(self.selection_selected_algorithm)
                num, ok = QInputDialog.getInt(self, '%s setting' % self.selection_selected_algorithm,
                                              'Selected feature number', 5, 1, 10000, 1)
                if ok:
                    self.selection_default_para['feature_number'] = num
                    self.selection_para_lineEdit.setText('Selected feature number: %s' % num)
            else:
                self.selection_selected_algorithm = item.text(0)
                self.selection_algorithm_lineEdit.setText(self.selection_selected_algorithm)
                self.selection_para_lineEdit.setText('None')

    def run_selection(self):
        if self.selection_selected_algorithm != '' and not self.selection_data is None:
            self.selection_running_status = False
            self.selection_progress_bar.setMovie(self.gif)
            self.gif.start()
            t = threading.Thread(target=self.data_analysis_selTab)
            t.start()
        else:
            if self.selection_data is None:
                QMessageBox.critical(self, 'Error', 'Empty data!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
            else:
                self.selection_status_label.setText(('<font color=red>%s failed. Error: %s </font>' % (
                    self.selection_selected_algorithm, self.selection_data.error_msg)))

    def data_analysis_selTab(self):
        if self.selection_selected_algorithm != '' and not self.selection_data is None:
            self.selection_start_button.setDisabled(True)
            self.tab_selection.setDisabled(True)
            self.setTabEnabled(0, False)
            self.setTabEnabled(1, False)
            self.setTabEnabled(3, False)
            if self.selection_analysis_type == 'Feature selection algorithms':
                cmd = 'self.selection_data.' + self.selection_selected_algorithm + '()'
                status = False
                try:
                    status = eval(cmd)
                    self.selection_signal.emit()
                except Exception as e:
                    QMessageBox.critical(self, 'Error', 'Calculation failed.', QMessageBox.Ok | QMessageBox.No,
                                         QMessageBox.Ok)
                    self.selection_data.error_msg = str(e)
                # if status:
                #     self.selection_status_label.setText('%s calculation complete.' %self.selection_selected_algorithm)
                #     self.selection_tableWidget.setColumnCount(self.selection_data.feature_selection_data.columns.size)
                #     self.selection_tableWidget.setRowCount(self.selection_data.feature_selection_data.index.size)
                #     self.selection_tableWidget.setHorizontalHeaderLabels(list(self.selection_data.feature_selection_data.columns))
                #     for i in range(self.selection_data.feature_selection_data.index.size):
                #         for j in range(self.selection_data.feature_selection_data.columns.size):
                #             cell = QTableWidgetItem(str(self.selection_data.feature_selection_data.iloc[i, j]))
                #             self.selection_tableWidget.setItem(i, j, cell)
            else:
                cmd = 'self.selection_data.' + self.selection_selected_algorithm + '()'
                status = False
                try:
                    status = eval(cmd)
                    self.selection_signal.emit()
                except Exception as e:
                    QMessageBox.critical(self, 'Error', 'Calculate failed.', QMessageBox.Ok | QMessageBox.No,
                                         QMessageBox.Ok)
                    self.selection_data.error_msg = str(e)
            self.selection_running_status = status
            self.selection_start_button.setDisabled(False)
            self.tab_selection.setDisabled(False)
            self.setTabEnabled(0, True)
            self.setTabEnabled(1, True)
            self.setTabEnabled(3, True)
            self.selection_progress_bar.clear()
        else:
            if self.selection_data is None:
                QMessageBox.critical(self, 'Error', 'Empty data!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
            else:
                self.selection_status_label.setText(('<font color=red>%s failed. Error: %s </font>' % (
                    self.selection_selected_algorithm, self.selection_data.error_msg)))

    def display_selection_data(self):
        if self.selection_analysis_type == 'Feature selection algorithms' and self.selection_running_status:
            self.selection_status_label.setText('%s calculation complete.' % self.selection_selected_algorithm)
            self.selection_tableWidget.init_data(self.selection_data.feature_selection_data.columns,
                                                 self.selection_data.feature_selection_data.values)
            # Draw histogram
            self.selection_histLayout.removeWidget(self.selection_hist)
            sip.delete(self.selection_hist)
            data = self.selection_data.feature_selection_data.values
            self.selection_hist = PlotWidgets.HistogramWidget()
            self.selection_hist.init_data('All data', data)
            self.selection_histLayout.addWidget(self.selection_hist)

        if self.selection_analysis_type == 'Feature normalization algorithms' and self.selection_running_status:
            self.selection_status_label.setText('%s calculation complete.' % self.selection_selected_algorithm)
            self.selection_tableWidget.init_data(self.selection_data.feature_normalization_data.columns,
                                                 self.selection_data.feature_normalization_data.values)
            # Draw histogram
            self.selection_histLayout.removeWidget(self.selection_hist)
            sip.delete(self.selection_hist)
            data = self.selection_data.feature_normalization_data.values
            self.selection_hist = PlotWidgets.HistogramWidget()
            self.selection_hist.init_data('All data', data)
            self.selection_histLayout.addWidget(self.selection_hist)

    def save_selection_normalization_data(self):
        try:
            if not self.selection_data is None:
                saved_file, ok = QFileDialog.getSaveFileName(self, 'Save', './data', 'CSV Files (*.csv);;TSV Files (*.tsv);;TSV Files with labels (*.tsv1);;SVM Files(*.svm);;Weka Files (*.arff)')
                if ok:
                    self.selection_data.save_selected_data(saved_file, self.selection_analysis_type)
            else:
                QMessageBox.critical(self, 'Error', 'Empty data!', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

    """ event in tab_machinelearning """
    def ml_panel_clear(self):
        try:
            self.MLData = None
            # self.MLData.training_score = None
            # self.MLData.testing_score = None
            # self.MLData.metrics = None

            self.train_score_tableWidget.clear()
            self.test_score_tableWidget.clear()
            self.metricsTableWidget.clear()
            self.dataTableWidget.clear()            
            self.dataTableWidget.setHorizontalHeaderLabels(['Select', 'Data', 'Shape', 'Source'])
            self.current_data_index = 0
        except Exception as e:
            pass

    def data_from_file_ml(self, target='Training'):
        if target == 'Training':
            self.ml_panel_clear()
        file_name, ok = QFileDialog.getOpenFileName(self, 'Open', './data','CSV Files (*.csv);;TSV Files (*.tsv);;SVM Files(*.svm);;Weka Files (*.arff)')
        if ok:
            if self.MLData is None:
                self.MLData = MachineLearning.ILearnMachineLearning(self.ml_defatult_para)
            ok1 = self.MLData.load_data(file_name, target)
            if ok1:
                index = 0
                if target == 'Training':
                    index = 0
                    self.training_data_radio = QRadioButton()
                    self.dataTableWidget.setCellWidget(index, 0, self.training_data_radio)
                    shape = self.MLData.training_dataframe.values.shape
                else:
                    index = 1
                    self.testing_data_radio = QRadioButton()
                    self.dataTableWidget.setCellWidget(index, 0, self.testing_data_radio)
                    shape = self.MLData.testing_dataframe.values.shape
                self.dataTableWidget.setItem(index, 1, QTableWidgetItem('%s data' % target))
                self.dataTableWidget.setItem(index, 2, QTableWidgetItem(str(shape)))
                self.dataTableWidget.setItem(index, 3, QTableWidgetItem(file_name))
                self.dataTableWidget.resizeRowsToContents()                

                # if not self.data_index['%s_data' % target] is None:
                #     index = self.data_index['%s_data' % target]                    
                #     if target == 'Training':
                #         self.training_data_radio = QRadioButton()
                #         self.dataTableWidget.setCellWidget(index, 0, self.training_data_radio)
                #         shape = self.MLData.training_dataframe.values.shape
                #     else:
                #         self.testing_data_radio = QRadioButton()
                #         self.dataTableWidget.setCellWidget(index, 0, self.testing_data_radio)
                #         shape = self.MLData.testing_dataframe.values.shape
                #     self.dataTableWidget.setItem(index, 1, QTableWidgetItem('%s data' % target))
                #     self.dataTableWidget.setItem(index, 2, QTableWidgetItem(str(shape)))
                #     self.dataTableWidget.setItem(index, 3, QTableWidgetItem(file_name))
                #     self.dataTableWidget.resizeRowsToContents()
                # else:
                #     index = self.current_data_index
                #     self.data_index['%s_data' % target] = index
                #     self.current_data_index += 1
                #     self.dataTableWidget.insertRow(index)
                #     if target == 'Training':
                #         self.training_data_radio = QRadioButton()
                #         self.dataTableWidget.setCellWidget(index, 0, self.training_data_radio)
                #         shape = self.MLData.training_dataframe.values.shape
                #     else:
                #         self.testing_data_radio = QRadioButton()
                #         self.dataTableWidget.setCellWidget(index, 0, self.testing_data_radio)
                #         shape = self.MLData.testing_dataframe.values.shape
                #     self.dataTableWidget.setItem(index, 1, QTableWidgetItem('%s data' % target))
                #     self.dataTableWidget.setItem(index, 2, QTableWidgetItem(str(shape)))
                #     self.dataTableWidget.setItem(index, 3, QTableWidgetItem(file_name))
                #     self.dataTableWidget.resizeRowsToContents()

    def data_from_panel(self):
        data_source, ok = InputDialog.QDataSelection.getValues(descriptor=self.descriptor, selection=self.selection_data, reduction=self.clust_data)
        self.ml_panel_clear()
        training_df, training_label, testing_df, testing_label = None, None, None, None
        source = None
        if ok and data_source == 'Descriptor data':
            data = copy.deepcopy(self.descriptor.encoding_array[1:, 2:].astype(float))
            label = copy.deepcopy(self.descriptor.encoding_array[1:, 1].astype(int))
            index_array = self.descriptor.encoding_array[0, 2:].copy()
            training_sample_index = np.where(self.descriptor.sample_purpose == True)[0]
            testing_sample_index = np.where(self.descriptor.sample_purpose == False)[0]
            if len(training_sample_index) != 0:
                training_df = pd.DataFrame(data[training_sample_index], columns=index_array)
                training_label = label[training_sample_index]
            if len(testing_sample_index) != 0:
                testing_df = pd.DataFrame(data[testing_sample_index], columns=index_array)
                testing_label = label[testing_sample_index]
            source = 'descriptor'
        if ok and data_source == 'Feature selection data':
            training_sample_index = np.where(self.selection_data.data_sample_purpose == True)[0]
            testing_sample_index = np.where(self.selection_data.data_sample_purpose == False)[0]
            if len(training_sample_index) != 0:
                training_df = copy.deepcopy(self.selection_data.feature_selection_data.iloc[training_sample_index, 1:])
                training_label = copy.deepcopy(self.selection_data.datalabel[training_sample_index])
            if len(testing_sample_index) != 0:
                testing_df = copy.deepcopy(self.selection_data.feature_selection_data.iloc[testing_sample_index, 1:])
                testing_label = copy.deepcopy(self.selection_data.datalabel[testing_sample_index])
            source = 'feature selection'
        if ok and data_source == 'Feature normalization data':
            training_sample_index = np.where(self.selection_data.data_sample_purpose == True)[0]
            testing_sample_index = np.where(self.selection_data.data_sample_purpose == False)[0]
            if len(training_sample_index) != 0:
                training_df = copy.deepcopy(self.selection_data.feature_normalization_data.iloc[training_sample_index, 1:])
                training_label = copy.deepcopy(self.selection_data.datalabel[training_sample_index])
            if len(testing_sample_index) != 0:
                testing_df = copy.deepcopy(self.selection_data.feature_normalization_data.iloc[testing_sample_index, 1:])
                testing_label = copy.deepcopy(self.selection_data.datalabel[testing_sample_index])
            source = 'feature selection'
        if ok and data_source == 'Dimensionality reduction data':
            training_sample_index = np.where(self.clust_data.data_sample_purpose == True)[0]
            testing_sample_index = np.where(self.clust_data.data_sample_purpose == False)[0]
            reduction_datafrme = pd.DataFrame(copy.deepcopy(self.clust_data.dimension_reduction_result), index=copy.deepcopy(self.clust_data.datalabel), columns=['PC%s' %(i+1) for i in range(self.clust_data.dimension_reduction_result.shape[1])])
            if len(training_sample_index) != 0:
                training_df = reduction_datafrme.iloc[training_sample_index]
                training_label = self.clust_data.datalabel[training_sample_index].copy()
            if len(testing_sample_index) != 0:
                testing_df = reduction_datafrme.iloc[testing_sample_index]
                testing_label = self.clust_data.datalabel[testing_sample_index].copy()
            source = 'Dimensionality reduction data'
        if self.MLData is None:
            self.MLData = MachineLearning.ILearnMachineLearning(self.ml_defatult_para)
        if not training_df is None:
            self.MLData.import_training_data(training_df, training_label)
            if not self.data_index['Training_data'] is None:
                index = 0
                # index = self.data_index['Training_data']
                shape = self.MLData.training_dataframe.values.shape
                self.dataTableWidget.setItem(index, 1, QTableWidgetItem('Training data'))
                self.dataTableWidget.setItem(index, 2, QTableWidgetItem(str(shape)))
                self.dataTableWidget.setItem(index, 3, QTableWidgetItem('Data from <%s> panel' % source))
            else:
                index = 0
                # index = self.current_data_index
                self.data_index['Training_data'] = index
                self.current_data_index += 1
                self.dataTableWidget.insertRow(index)
                self.training_data_radio = QRadioButton()
                self.dataTableWidget.setCellWidget(index, 0, self.training_data_radio)
                shape = self.MLData.training_dataframe.values.shape
                self.dataTableWidget.setItem(index, 1, QTableWidgetItem('Training data'))
                self.dataTableWidget.setItem(index, 2, QTableWidgetItem(str(shape)))
                self.dataTableWidget.setItem(index, 3, QTableWidgetItem('Data from <%s> panel' % source))
            self.dataTableWidget.resizeRowsToContents()
        if not testing_df is None:
            self.MLData.import_testing_data(testing_df, testing_label)
            if not self.data_index['Testing_data'] is None:
                # index = self.data_index['Testing_data']
                index = 1
                shape = self.MLData.testing_dataframe.values.shape
                self.dataTableWidget.setItem(index, 1, QTableWidgetItem('Testing data'))
                self.dataTableWidget.setItem(index, 2, QTableWidgetItem(str(shape)))
                self.dataTableWidget.setItem(index, 3, QTableWidgetItem('Data from <%s> panel' % source))
            else:
                # index = self.current_data_index
                index = 1
                self.data_index['Testing_data'] = index
                self.current_data_index += 1
                self.dataTableWidget.insertRow(index)
                self.testing_data_radio = QRadioButton()
                self.dataTableWidget.setCellWidget(index, 0, self.testing_data_radio)
                shape = self.MLData.testing_dataframe.values.shape
                self.dataTableWidget.setItem(index, 1, QTableWidgetItem('Testing data'))
                self.dataTableWidget.setItem(index, 2, QTableWidgetItem(str(shape)))
                self.dataTableWidget.setItem(index, 3, QTableWidgetItem('Data from <%s> panel' % source))
            self.dataTableWidget.resizeRowsToContents()

    def setFold(self):
        fold, ok = QInputDialog.getInt(self, 'Fold number', 'Setting K-fold cross-validation', 5, 2, 100, 1)
        if ok:
            self.ml_fold_lineEdit.setText(str(fold))
            self.ml_defatult_para['FOLD'] = fold

    def ml_tree_clicked(self, index):
        item = self.ml_treeWidget.currentItem()
        if item.text(0) in ['RF']:
            self.MLAlgorithm = item.text(0)
            self.ml_algorithm_lineEdit.setText(self.MLAlgorithm)
            num, range, cpu, auto, ok = InputDialog.QRandomForestInput.getValues()
            if ok:
                self.ml_defatult_para['n_trees'] = num
                self.ml_defatult_para['tree_range'] = range
                self.ml_defatult_para['auto'] = auto
                self.ml_defatult_para['cpu'] = cpu
                if auto:
                    self.ml_para_lineEdit.setText('Tree range: %s' % str(range))
                else:
                    self.ml_para_lineEdit.setText('n_trees: %s' % num)
        elif item.text(0) in ['SVM']:
            self.MLAlgorithm = item.text(0)
            self.ml_algorithm_lineEdit.setText(self.MLAlgorithm)
            kernel, penality, gamma, auto, penalityRange, gammaRange, ok = InputDialog.QSupportVectorMachineInput.getValues()            
            if ok:
                self.ml_defatult_para['kernel'] = kernel
                self.ml_defatult_para['penality'] = penality
                self.ml_defatult_para['gamma'] = gamma
                self.ml_defatult_para['auto'] = auto
                self.ml_defatult_para['penalityRange'] = penalityRange
                self.ml_defatult_para['gammaRange'] = gammaRange
                if auto:
                    self.ml_para_lineEdit.setText('kernel: %s; Auto-Optimization' % kernel)
                else:
                    self.ml_para_lineEdit.setText('kernel: %s; Penality=%s, Gamma=%s' % (kernel, penality, gamma))
        elif item.text(0) in ['MLP']:
            self.MLAlgorithm = item.text(0)
            self.ml_algorithm_lineEdit.setText(self.MLAlgorithm)
            layer, epochs, activation, optimizer, ok = InputDialog.QMultiLayerPerceptronInput.getValues()
            if ok:
                self.ml_defatult_para['layer'] = layer
                self.ml_defatult_para['epochs'] = epochs
                self.ml_defatult_para['activation'] = activation
                self.ml_defatult_para['optimizer'] = optimizer
                self.ml_para_lineEdit.setText(
                    'Layer: %s; Epochs: %s; Activation: %s; Optimizer: %s' % (layer, epochs, activation, optimizer))
        elif item.text(0) in ['LR', 'SGD', 'DecisionTree', 'NaiveBayes', 'AdaBoost', 'GBDT', 'LDA', 'QDA']:
            self.MLAlgorithm = item.text(0)
            self.ml_algorithm_lineEdit.setText(self.MLAlgorithm)
            self.ml_para_lineEdit.setText('None')
        elif item.text(0) in ['KNN']:
            self.MLAlgorithm = item.text(0)
            self.ml_algorithm_lineEdit.setText(self.MLAlgorithm)
            topKValue, ok = InputDialog.QKNeighborsInput.getValues()
            if ok:
                self.ml_defatult_para['topKValue'] = topKValue
                self.ml_para_lineEdit.setText('KNN top K value: %s' % topKValue)
        elif item.text(0) in ['LightGBM']:
            self.MLAlgorithm = item.text(0)
            self.ml_algorithm_lineEdit.setText(self.MLAlgorithm)
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
                    self.ml_para_lineEdit.setText('Parameter auto optimization')
                else:
                    self.ml_para_lineEdit.setText(
                        'Boosting type: %s; Leaves number: %s; Max depth: %s; Learning rate: %s' % (
                            type, leaves, depth, rate))
        elif item.text(0) in ['XGBoost']:
            self.MLAlgorithm = item.text(0)
            self.ml_algorithm_lineEdit.setText(self.MLAlgorithm)
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
                self.ml_para_lineEdit.setText('Parameter auto optimization')
            else:
                self.ml_para_lineEdit.setText(
                    'Booster: %s; Maxdepth: %s; Learning rate: %s' % (booster, maxdepth, rate))
        elif item.text(0) in ['Bagging']:
            self.MLAlgorithm = item.text(0)
            self.ml_algorithm_lineEdit.setText(self.MLAlgorithm)
            n_estimators, threads, ok = InputDialog.QBaggingInput.getValues()
            if ok:
                self.ml_defatult_para['n_estimator'] = n_estimators
                self.ml_defatult_para['cpu'] = threads
                self.ml_para_lineEdit.setText('n_estimators: %s' % n_estimators)
        elif item.text(0) in ['Net_1_CNN']:
            self.MLAlgorithm = item.text(0)
            self.ml_algorithm_lineEdit.setText(self.MLAlgorithm)
            if not self.MLData is None:
                input_channel, input_length, output_channel, padding, kernel_size, dropout, learning_rate, epochs, early_stopping, batch_size, fc_size, ok = InputDialog.QNetInput_1.getValues(
                    self.MLData.training_dataframe.values.shape[1])
                if ok:
                    self.ml_defatult_para['input_channel'] = input_channel
                    self.ml_defatult_para['input_length'] = input_length
                    self.ml_defatult_para['output_channel'] = output_channel
                    self.ml_defatult_para['padding'] = padding
                    self.ml_defatult_para['kernel_size'] = kernel_size
                    self.ml_defatult_para['dropout'] = dropout
                    self.ml_defatult_para['learning_rate'] = learning_rate
                    self.ml_defatult_para['epochs'] = epochs
                    self.ml_defatult_para['early_stopping'] = early_stopping
                    self.ml_defatult_para['batch_size'] = batch_size
                    self.ml_defatult_para['fc_size'] = fc_size
                    self.ml_para_lineEdit.setText(
                        'Input channel=%s; Input_length=%s; Output_channel=%s; Padding=%s; Kernel_size=%s; Dropout=%s; lr=%s; Epochs=%s; Early_stopping=%s; Batch_size=%s' % (
                            input_channel, input_length, output_channel, padding, kernel_size, dropout, learning_rate,
                            epochs, early_stopping, batch_size))
            else:
                QMessageBox.warning(self, 'Warning', 'Please input training data at first!',
                                    QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        elif item.text(0) in ['Net_2_RNN']:
            self.MLAlgorithm = item.text(0)
            self.ml_algorithm_lineEdit.setText(self.MLAlgorithm)
            if not self.MLData is None:
                input_channel, input_length, hidden_size, num_layers, dropout, learning_rate, epochs, early_stopping, batch_size, fc_size, ok = InputDialog.QNetInput_2.getValues(
                    self.MLData.training_dataframe.values.shape[1])                
                if ok:
                    self.ml_defatult_para['input_channel'] = input_channel
                    self.ml_defatult_para['input_length'] = input_length
                    self.ml_defatult_para['rnn_hidden_size'] = hidden_size
                    self.ml_defatult_para['rnn_hidden_layers'] = num_layers
                    self.ml_defatult_para['rnn_bidirection'] = False
                    self.ml_defatult_para['dropout'] = dropout
                    self.ml_defatult_para['learning_rate'] = learning_rate
                    self.ml_defatult_para['epochs'] = epochs
                    self.ml_defatult_para['early_stopping'] = early_stopping
                    self.ml_defatult_para['batch_size'] = batch_size
                    self.ml_defatult_para['rnn_bidirectional'] = False
                    self.ml_para_lineEdit.setText(
                        'Input size=%s; Input_length=%s; Hidden_size=%s; Num_hidden_layers=%s; Dropout=%s; lr=%s; Epochs=%s; Early_stopping=%s; Batch_size=%s' % (
                            input_channel, input_length, hidden_size, num_layers, dropout, learning_rate, epochs,
                            early_stopping, batch_size))
            else:
                QMessageBox.warning(self, 'Warning', 'Please input training data at first!',
                                    QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        elif item.text(0) in ['Net_3_BRNN']:
            self.MLAlgorithm = item.text(0)
            self.ml_algorithm_lineEdit.setText(self.MLAlgorithm)
            if not self.MLData is None:
                input_channel, input_length, hidden_size, num_layers, dropout, learning_rate, epochs, early_stopping, batch_size, fc_size, ok = InputDialog.QNetInput_2.getValues(
                    self.MLData.training_dataframe.values.shape[1])
                if ok:
                    self.ml_defatult_para['input_channel'] = input_channel
                    self.ml_defatult_para['input_length'] = input_length
                    self.ml_defatult_para['rnn_hidden_size'] = hidden_size
                    self.ml_defatult_para['rnn_hidden_layers'] = num_layers
                    self.ml_defatult_para['rnn_bidirection'] = False
                    self.ml_defatult_para['dropout'] = dropout
                    self.ml_defatult_para['learning_rate'] = learning_rate
                    self.ml_defatult_para['epochs'] = epochs
                    self.ml_defatult_para['early_stopping'] = early_stopping
                    self.ml_defatult_para['batch_size'] = batch_size
                    self.ml_defatult_para['rnn_bidirectional'] = True
                    self.ml_para_lineEdit.setText(
                        'Input size=%s; Input_length=%s; Hidden_size=%s; Num_hidden_layers=%s; Dropout=%s; lr=%s; Epochs=%s; Early_stopping=%s; Batch_size=%s' % (
                            input_channel, input_length, hidden_size, num_layers, dropout, learning_rate, epochs,
                            early_stopping, batch_size))
            else:
                QMessageBox.warning(self, 'Warning', 'Please input training data at first!',
                                    QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        elif item.text(0) in ['Net_4_ABCNN']:
            self.MLAlgorithm = item.text(0)
            self.ml_algorithm_lineEdit.setText(self.MLAlgorithm)
            if not self.MLData is None:
                input_channel, input_length, dropout, learning_rate, epochs, early_stopping, batch_size, ok = InputDialog.QNetInput_4.getValues(
                    self.MLData.training_dataframe.values.shape[1])                
                if ok:
                    self.ml_defatult_para['input_channel'] = input_channel
                    self.ml_defatult_para['input_length'] = input_length
                    self.ml_defatult_para['dropout'] = dropout
                    self.ml_defatult_para['learning_rate'] = learning_rate
                    self.ml_defatult_para['epochs'] = epochs
                    self.ml_defatult_para['early_stopping'] = early_stopping
                    self.ml_defatult_para['batch_size'] = batch_size
                    self.ml_para_lineEdit.setText(
                        'Input size=%s; Input_length=%s; Dropout=%s; lr=%s; Epochs=%s; Early_stopping=%s; Batch_size=%s' % (
                            input_channel, input_length, dropout, learning_rate, epochs, early_stopping, batch_size))
            else:
                QMessageBox.warning(self, 'Warning', 'Please input training data at first!',
                                    QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        elif item.text(0) in ['Net_5_ResNet']:
            self.MLAlgorithm = item.text(0)
            self.ml_algorithm_lineEdit.setText(self.MLAlgorithm)
            if not self.MLData is None:
                input_channel, input_length, learning_rate, epochs, early_stopping, batch_size, ok = InputDialog.QNetInput_5.getValues(
                    self.MLData.training_dataframe.values.shape[1])                
                if ok:
                    self.ml_defatult_para['input_channel'] = input_channel
                    self.ml_defatult_para['input_length'] = input_length
                    self.ml_defatult_para['learning_rate'] = learning_rate
                    self.ml_defatult_para['epochs'] = epochs
                    self.ml_defatult_para['early_stopping'] = early_stopping
                    self.ml_defatult_para['batch_size'] = batch_size
                    self.ml_para_lineEdit.setText(
                        'Input size=%s; Input_length=%s; lr=%s; Epochs=%s; Early_stopping=%s; Batch_size=%s' % (
                            input_channel, input_length, learning_rate, epochs, early_stopping, batch_size))
            else:
                QMessageBox.warning(self, 'Warning', 'Please input training data at first!',
                                    QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        elif item.text(0) in ['Net_6_AE']:
            self.MLAlgorithm = item.text(0)
            self.ml_algorithm_lineEdit.setText(self.MLAlgorithm)
            if not self.MLData is None:
                input_dim, dropout, learning_rate, epochs, early_stopping, batch_size, ok = InputDialog.QNetInput_6.getValues(
                    self.MLData.training_dataframe.values.shape[1])                
                if ok:
                    self.ml_defatult_para['mlp_input_dim'] = input_dim
                    self.ml_defatult_para['dropout'] = dropout
                    self.ml_defatult_para['learning_rate'] = learning_rate
                    self.ml_defatult_para['epochs'] = epochs
                    self.ml_defatult_para['early_stopping'] = early_stopping
                    self.ml_defatult_para['batch_size'] = batch_size
                    self.ml_para_lineEdit.setText(
                        'Input dimension=%s; Dropout=%s; lr=%s; Epochs=%s; Early_stopping=%s; Batch_size=%s' % (
                            input_dim, dropout, learning_rate, epochs, early_stopping, batch_size))
            else:
                QMessageBox.warning(self, 'Warning', 'Please input training data at first!',
                                    QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

    def run_train_model(self):
        if not self.MLAlgorithm is None and not self.MLData is None:
            self.ml_running_status = False
            self.ml_progress_bar.setMovie(self.gif)
            self.gif.start()
            t = threading.Thread(target=self.train_model)
            t.start()
        else:
            QMessageBox.critical(self, 'Error', 'Please load data or specify an algorithm.',
                                 QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

    def train_model(self):
        try:
            if not self.MLAlgorithm is None and not self.MLData is None:
                self.ml_status_label.setText('Training model ... ')
                self.tab_machine.setDisabled(True)
                self.setTabEnabled(0, False)
                self.setTabEnabled(1, False)
                self.setTabEnabled(2, False)
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
                elif self.MLAlgorithm == 'Net_1_CNN':
                    ok = self.MLData.run_networks(1)
                elif self.MLAlgorithm == 'Net_2_RNN':
                    ok = self.MLData.run_networks(2)
                elif self.MLAlgorithm == 'Net_3_BRNN':
                    ok = self.MLData.run_networks(3)
                elif self.MLAlgorithm == 'Net_4_ABCNN':
                    ok = self.MLData.run_networks(4)
                elif self.MLAlgorithm == 'Net_5_ResNet':
                    ok = self.MLData.run_networks(5)
                elif self.MLAlgorithm == 'Net_6_AE':
                    ok = self.MLData.run_networks(6)

                self.ml_running_status = ok
                self.ml_status_label.setText('Training model complete.')
                self.ml_signal.emit()
                self.tab_machine.setDisabled(False)
                self.setTabEnabled(0, True)
                self.setTabEnabled(1, True)
                self.setTabEnabled(2, True)
            else:
                QMessageBox.critical(self, 'Error', 'Please load data or specify an algorithm.',
                                    QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

    def display_ml_data(self):
        if self.ml_running_status:
            if not self.MLData.message is None:
                self.ml_status_label.setText(self.MLData.message)
            # display predicton score
            if not self.MLData.training_score is None:
                data = self.MLData.training_score.values
                self.train_score_tableWidget.setRowCount(data.shape[0])
                self.train_score_tableWidget.setColumnCount(data.shape[1])
                self.train_score_tableWidget.setHorizontalHeaderLabels(self.MLData.training_score.columns)
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        cell = QTableWidgetItem(str(round(data[i][j], 4)))
                        self.train_score_tableWidget.setItem(i, j, cell)
                if self.data_index['Training_score'] is None:
                    # index = self.current_data_index
                    index = 2
                    self.data_index['Training_score'] = index
                    self.dataTableWidget.insertRow(index)
                    self.current_data_index += 1
                else:
                    # index = self.data_index['Training_score']
                    index = 2
                self.training_score_radio = QRadioButton()
                self.dataTableWidget.setCellWidget(index, 0, self.training_score_radio)
                self.dataTableWidget.setItem(index, 1, QTableWidgetItem('Training score'))
                self.dataTableWidget.setItem(index, 2, QTableWidgetItem(str(data.shape)))
                self.dataTableWidget.setItem(index, 3, QTableWidgetItem('%s model' % self.MLAlgorithm))
            if not self.MLData.testing_score is None:
                data = self.MLData.testing_score.values
                self.test_score_tableWidget.setRowCount(data.shape[0])
                self.test_score_tableWidget.setColumnCount(data.shape[1])
                self.test_score_tableWidget.setHorizontalHeaderLabels(self.MLData.training_score.columns)
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        if j == 0:
                            cellData = data[i][j]
                        else:
                            cellData = str(round(data[i][j], 4))
                        cell = QTableWidgetItem(cellData)
                        self.test_score_tableWidget.setItem(i, j, cell)
                if self.data_index['Testing_score'] is None:
                    # index = self.current_data_index
                    index = 3
                    self.data_index['Testing_score'] = index
                    self.dataTableWidget.insertRow(index)
                    self.current_data_index += 1
                else:
                    # index = self.data_index['Testing_score']
                    index = 3
                self.testing_score_radio = QRadioButton()
                self.dataTableWidget.setCellWidget(index, 0, self.testing_score_radio)
                self.dataTableWidget.setItem(index, 1, QTableWidgetItem('Testing score'))
                self.dataTableWidget.setItem(index, 2, QTableWidgetItem(str(data.shape)))
                self.dataTableWidget.setItem(index, 3, QTableWidgetItem('%s model' % self.MLAlgorithm))

            # display evaluation metrics
            data = self.MLData.metrics.values
            self.metricsTableWidget.setRowCount(data.shape[0])
            self.metricsTableWidget.setColumnCount(data.shape[1])
            self.metricsTableWidget.setHorizontalHeaderLabels(
                ['Sn (%)', 'Sp (%)', 'Pre (%)', 'Acc (%)', 'MCC', 'F1', 'AUROC', 'AUPRC'])
            self.metricsTableWidget.setVerticalHeaderLabels(self.MLData.metrics.index)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    cell = QTableWidgetItem(str(data[i][j]))
                    self.metricsTableWidget.setItem(i, j, cell)
            if self.data_index['Metrics'] is None:
                # index = self.current_data_index
                index = 4
                self.data_index['Metrics'] = index
                self.dataTableWidget.insertRow(index)
                self.current_data_index += 1
            else:
                # index = self.data_index['Metrics']
                index = 4
            self.metrics_radio = QRadioButton()
            self.dataTableWidget.setCellWidget(index, 0, self.metrics_radio)
            self.dataTableWidget.setItem(index, 1, QTableWidgetItem('Evaluation metrics'))
            self.dataTableWidget.setItem(index, 2, QTableWidgetItem(str(data.shape)))
            self.dataTableWidget.setItem(index, 3, QTableWidgetItem('%s model' % self.MLAlgorithm))

            # display model
            if self.data_index['Model'] is None:
                # index = self.current_data_index
                index = 5
                self.data_index['Model'] = index
                self.dataTableWidget.insertRow(index)
                self.current_data_index += 1
            else:
                # index = self.data_index['Model']
                index = 5
            self.model_radio = QRadioButton()
            self.dataTableWidget.setCellWidget(index, 0, self.model_radio)
            self.dataTableWidget.setItem(index, 1, QTableWidgetItem('Models'))
            self.dataTableWidget.setItem(index, 2, QTableWidgetItem(str(self.fold_num)))
            self.dataTableWidget.setItem(index, 3, QTableWidgetItem('%s model' % self.MLAlgorithm))

            # plot ROC
            try:
                # Draw ROC curve
                if not self.MLData.aucData is None:
                    self.rocLayout.removeWidget(self.roc_curve_widget)
                    sip.delete(self.roc_curve_widget)
                    self.roc_curve_widget = PlotWidgets.CurveWidget()
                    self.roc_curve_widget.init_data(0, 'ROC curve', self.MLData.aucData, self.MLData.meanAucData,
                                                    self.MLData.indepAucData)
                    self.rocLayout.addWidget(self.roc_curve_widget)
                # plot PRC
                if not self.MLData.prcData is None:
                    self.prcLayout.removeWidget(self.prc_curve_widget)
                    sip.delete(self.prc_curve_widget)
                    self.prc_curve_widget = PlotWidgets.CurveWidget()
                    self.prc_curve_widget.init_data(1, 'PRC curve', self.MLData.prcData, self.MLData.meanPrcData,
                                                    self.MLData.indepPrcData)
                    self.prcLayout.addWidget(self.prc_curve_widget)
            except Exception as e:
                self.ml_status_label.setText(str(e))
        else:
            QMessageBox.critical(self, 'Error', str(self.MLData.error_msg), QMessageBox.Ok | QMessageBox.No,
                                 QMessageBox.Ok)
        self.ml_progress_bar.clear()

    def save_ml_files_orig_with_bugs(self):
        if 'training_data_radio' in dir(self) and self.training_data_radio.isChecked():
            saved_file, ok = QFileDialog.getSaveFileName(self, 'Save', './data',
                                                         'CSV Files (*.csv);;TSV Files (*.tsv);;SVM Files(*.svm);;Weka Files (*.arff)')
            if ok:
                ok1 = self.MLData.save_coder(saved_file, 'training')
                if not ok1:
                    QMessageBox.critical(self, 'Error', str(self.MLData.error_msg), QMessageBox.Ok | QMessageBox.No,
                                         QMessageBox.Ok)
        elif 'testing_data_radio' in dir(self) and self.testing_data_radio.isChecked():
            saved_file, ok = QFileDialog.getSaveFileName(self, 'Save', './data',
                                                         'CSV Files (*.csv);;TSV Files (*.tsv);;SVM Files(*.svm);;Weka Files (*.arff)')
            if ok:
                ok1 = self.MLData.save_coder(saved_file, 'testing')
                if not ok1:
                    QMessageBox.critical(self, 'Error', str(self.MLData.error_msg), QMessageBox.Ok | QMessageBox.No,
                                         QMessageBox.Ok)
        elif 'training_score_radio' in dir(self) and self.training_score_radio.isChecked():
            save_file, ok = QFileDialog.getSaveFileName(self, 'Save', './data', 'TSV Files (*.tsv)')
            if ok:
                ok1 = self.MLData.save_prediction_score(save_file, 'training')
                if not ok1:
                    QMessageBox.critical(self, 'Error', str(self.MLData.error_msg), QMessageBox.Ok | QMessageBox.No,
                                         QMessageBox.Ok)
        elif 'testing_score_radio' in dir(self) and self.testing_score_radio.isChecked():
            save_file, ok = QFileDialog.getSaveFileName(self, 'Save', './data', 'TSV Files (*.tsv)')
            if ok:
                ok1 = self.MLData.save_prediction_score(save_file, 'testing')
                if not ok1:
                    QMessageBox.critical(self, 'Error', str(self.MLData.error_msg), QMessageBox.Ok | QMessageBox.No,
                                         QMessageBox.Ok)
        elif 'metrics_radio' in dir(self) and self.metrics_radio.isChecked():
            save_file, ok = QFileDialog.getSaveFileName(self, 'Save', './data', 'TSV Files (*.tsv)')
            if ok:
                ok1 = self.MLData.save_metrics(save_file)
                if not ok1:
                    QMessageBox.critical(self, 'Error', str(self.MLData.error_msg), QMessageBox.Ok | QMessageBox.No,
                                         QMessageBox.Ok)
        elif 'model_radio' in dir(self) and self.model_radio.isChecked():
            save_directory = QFileDialog.getExistingDirectory(self, 'Save', './data')
            if self.MLData.best_model is not None:
                for i, model in enumerate(self.MLData.best_model):
                    model_name = '%s/%s_model_%s.pkl' % (save_directory, self.MLData.algorithm, i + 1)
                    if self.MLData.algorithm in ['RF', 'SVM', 'MLP', 'LR', 'KNN', 'LightGBM', 'XGBoost', 'SGD', 'DecisionTree', 'Bayes', 'AdaBoost', 'Bagging', 'GBDT', 'LDA', 'QDA']:
                        joblib.dump(model, model_name)
                    else:
                        torch.save(model, model_name)
                QMessageBox.information(self, 'Model saved', 'The models have been saved to directory %s' %save_directory,  QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        else:
            QMessageBox.critical(self, 'Error', 'Please select which data to save.', QMessageBox.Ok | QMessageBox.No,
                                 QMessageBox.Ok)

    def save_ml_files(self):
        tag = 0
        try:
            if self.training_data_radio.isChecked():
                tag = 1
                saved_file, ok = QFileDialog.getSaveFileName(self, 'Save', './data',
                                                            'CSV Files (*.csv);;TSV Files (*.tsv);;SVM Files(*.svm);;Weka Files (*.arff)')
                if ok:
                    ok1 = self.MLData.save_coder(saved_file, 'training')
                    if not ok1:
                        QMessageBox.critical(self, 'Error', str(self.MLData.error_msg), QMessageBox.Ok | QMessageBox.No,
                                            QMessageBox.Ok)
        except Exception as e:
            pass
        
        try:
            if self.testing_data_radio.isChecked():
                tag = 1
                saved_file, ok = QFileDialog.getSaveFileName(self, 'Save', './data',
                                                            'CSV Files (*.csv);;TSV Files (*.tsv);;SVM Files(*.svm);;Weka Files (*.arff)')
                if ok:
                    ok1 = self.MLData.save_coder(saved_file, 'testing')
                    if not ok1:
                        QMessageBox.critical(self, 'Error', str(self.MLData.error_msg), QMessageBox.Ok | QMessageBox.No,
                                            QMessageBox.Ok)
        except Exception as e:
            pass
        
        try:
            if self.training_score_radio.isChecked():
                tag = 1
                save_file, ok = QFileDialog.getSaveFileName(self, 'Save', './data', 'TSV Files (*.tsv)')
                if ok:
                    ok1 = self.MLData.save_prediction_score(save_file, 'training')
                    if not ok1:
                        QMessageBox.critical(self, 'Error', str(self.MLData.error_msg), QMessageBox.Ok | QMessageBox.No,
                                            QMessageBox.Ok)
        except Exception as e:
            pass
        
        try:
            if self.testing_score_radio.isChecked():
                tag = 1
                save_file, ok = QFileDialog.getSaveFileName(self, 'Save', './data', 'TSV Files (*.tsv)')
                if ok:
                    ok1 = self.MLData.save_prediction_score(save_file, 'testing')
                    if not ok1:
                        QMessageBox.critical(self, 'Error', str(self.MLData.error_msg), QMessageBox.Ok | QMessageBox.No,
                                            QMessageBox.Ok)
        except Exception as e:
            pass

        try:
            if self.metrics_radio.isChecked():
                tag = 1
                save_file, ok = QFileDialog.getSaveFileName(self, 'Save', './data', 'TSV Files (*.tsv)')
                if ok:
                    ok1 = self.MLData.save_metrics(save_file)
                    if not ok1:
                        QMessageBox.critical(self, 'Error', str(self.MLData.error_msg), QMessageBox.Ok | QMessageBox.No,
                                            QMessageBox.Ok)
        except Exception as e:
            pass
        
        try:
            if self.model_radio.isChecked():
                tag = 1
                save_directory = QFileDialog.getExistingDirectory(self, 'Save', './data')
                if os.path.exists(save_directory):
                    if self.MLData.best_model is not None:
                        for i, model in enumerate(self.MLData.best_model):
                            model_name = '%s/%s_model_%s.pkl' % (save_directory, self.MLData.algorithm, i + 1)
                            if self.MLData.algorithm in ['RF', 'SVM', 'MLP', 'LR', 'KNN', 'LightGBM', 'XGBoost', 'SGD', 'DecisionTree', 'Bayes', 'AdaBoost', 'Bagging', 'GBDT', 'LDA', 'QDA']:
                                joblib.dump(model, model_name)
                            else:
                                torch.save(model, model_name)
                        QMessageBox.information(self, 'Model saved', 'The models have been saved to directory %s' %save_directory,  QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
                    else:
                        pass
        except Exception as e:
            pass
            
        if tag == 0:
            QMessageBox.critical(self, 'Error', 'Please select which data to save.', QMessageBox.Ok | QMessageBox.No,
                                 QMessageBox.Ok)

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Confirm Exit', 'Are you sure want to quit iLearnPlus?', QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close_signal.emit('Basic')
            self.close()
        else:
            if event:
                event.ignore()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ILearnPlusBasic()
    app.setFont(QFont('Arial', 10))
    window.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window.show()
    sys.exit(app.exec_())
