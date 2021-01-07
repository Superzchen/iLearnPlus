#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import sys, os
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
from PyQt5.QtWidgets import (QApplication, QTableWidget, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                             QLineEdit, QTableWidgetItem, QMessageBox, QComboBox, QSpacerItem, QSizePolicy, QAbstractItemView)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import pyqtSignal
import pandas as pd
import numpy as np
from PlotWidgets import HistogramWidget
import qdarkstyle

class TableWidget(QWidget):
    control_signal = pyqtSignal(list)

    def __init__(self):
        super(TableWidget, self).__init__()
        self.data = None
        self.row = 50
        self.column = 0
        self.page_dict = {}
        self.final_page = ''
        self.header = []
        self.selected_column = None
        self.initUI()

    def initUI(self):
        self.control_signal.connect(self.page_controller)
        layout = QVBoxLayout(self)
        self.tableWidget = QTableWidget()
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableWidget.horizontalHeader().sectionClicked.connect(self.horizontal_header_clicked)
        self.tableWidget.setFont(QFont('Arial', 8))
        layout.addWidget(self.tableWidget)

        control_layout = QHBoxLayout()
        homePage = QPushButton("Home")
        prePage = QPushButton("< Previous")
        self.curPage = QLabel("1")
        nextPage = QPushButton("Next >")
        finalPage = QPushButton("Final")
        self.totalPage = QLabel("Total")
        skipLable_0 = QLabel("Jump to")
        self.skipPage = QLineEdit()
        self.skipPage.setMaximumWidth(50)
        confirmSkip = QPushButton("OK")
        spacer = QSpacerItem(20, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.label_comboBox = QComboBox()
        # self.label_comboBox.setMaximumWidth(50)
        self.label_comboBox.addItem('All')
        self.label_comboBox.setCurrentIndex(0)
        self.label_comboBox.setToolTip('Select the sample category used to draw the histogram.')
        self.display_hist = QPushButton(' Histogram ')
        self.display_hist.clicked.connect(self.display_histogram)

        control_layout.addStretch(1)
        homePage.clicked.connect(self.__home_page)
        prePage.clicked.connect(self.__pre_page)
        nextPage.clicked.connect(self.__next_page)
        finalPage.clicked.connect(self.__final_page)
        confirmSkip.clicked.connect(self.__confirm_skip)
        control_layout.addWidget(homePage)
        control_layout.addWidget(prePage)
        control_layout.addWidget(self.curPage)
        control_layout.addWidget(nextPage)
        control_layout.addWidget(finalPage)
        control_layout.addWidget(self.totalPage)
        control_layout.addWidget(skipLable_0)
        control_layout.addWidget(self.skipPage)
        control_layout.addWidget(confirmSkip)
        control_layout.addItem(spacer)
        # control_layout.addWidget(self.label_comboBox)
        control_layout.addWidget(self.display_hist)
        control_layout.addStretch(1)
        layout.addLayout(control_layout)

    def setRowAndColumns(self, row, column):
        self.row = row
        self.column = column
        self.tableWidget.setRowCount(row)
        self.tableWidget.setColumnCount(column)

    def init_data(self, header, data):
        self.tableWidget.setHorizontalHeaderLabels(header)
        self.header = header
        self.data = data
        page = 1
        for i in range(0, self.data.shape[0], self.row):
            end = i + self.row if i + self.row < self.data.shape[0] else self.data.shape[0]
            self.page_dict[str(page)] = (i, end)
            self.final_page = str(page)
            page += 1
        self.curPage.setText('1')
        self.display_table('1')
        self.totalPage.setText('Total page: %s ' %self.final_page)

    def __home_page(self):
        self.control_signal.emit(["home", self.curPage.text()])

    def __pre_page(self):
        self.control_signal.emit(["pre", self.curPage.text()])

    def __next_page(self):
        self.control_signal.emit(["next", self.curPage.text()])

    def __final_page(self):
        self.control_signal.emit(["final", self.curPage.text()])

    def __confirm_skip(self):
        self.control_signal.emit(["confirm", self.skipPage.text()])

    def page_controller(self, signal):
        try:
            if not self.data is None:
                if 'home' == signal[0] and signal[1] != '':
                    self.display_table('1')
                    self.curPage.setText('1')
                elif 'final' == signal[0] and signal[1] != '':
                    self.display_table(self.final_page)
                    self.curPage.setText(self.final_page)
                elif 'pre' == signal[0] and signal[1] != '':
                    page = int(signal[1]) - 1 if int(signal[1]) - 1 > 0 else 1
                    self.curPage.setText(str(page))
                    self.display_table(str(page))
                elif 'next' == signal[0] and signal[1] != '':
                    page = int(signal[1]) + 1 if int(signal[1]) + 1 <= int(self.final_page) else int(self.final_page)
                    self.curPage.setText(str(page))
                    self.display_table(str(page))
                elif "confirm" == signal[0] and signal[1] != '':
                    if 1 <= int(signal[1]) <= int(self.final_page):
                        self.curPage.setText(signal[1])
                        self.display_table(signal[1])
        except Exception as e:
            pass

    def display_table(self, page):
        if page in self.page_dict:
            tmp_data = self.data[self.page_dict[page][0]: self.page_dict[page][1]]
            self.tableWidget.setRowCount(tmp_data.shape[0])
            self.tableWidget.setColumnCount(tmp_data.shape[1])
            self.tableWidget.setHorizontalHeaderLabels(self.header)
            for i in range(tmp_data.shape[0]):
                for j in range(tmp_data.shape[1]):
                    if j == 0:
                        cell = QTableWidgetItem(str(tmp_data[i][j]))
                    else:
                        cell = QTableWidgetItem(str(round(float(tmp_data[i][j]), 6)))
                    self.tableWidget.setItem(i, j, cell)
        else:
            QMessageBox.critical(self, 'Error', 'Page number out of index.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        labels = ['All'] + [str(i) for i in set(self.data[1:, 1])]
        self.label_comboBox.clear()
        self.label_comboBox.addItems(labels)
        self.label_comboBox.setCurrentIndex(0)

    def horizontal_header_clicked(self, index):
        if index == 0:
            self.selected_column = None
        else:
            self.selected_column = index

    def display_histogram(self):
        if self.selected_column is None:
            QMessageBox.critical(self, 'Error', 'Please select a column (except col 0)).', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        elif self.selected_column > len(self.header) - 1:
            QMessageBox.critical(self, 'Error', 'Incorrect column index.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        else:
            # labels = np.array(self.data[1:, 1]).astype(int)
            # selected_label = self.label_comboBox.currentText()
            # data = np.array(self.data[1:, self.selected_column])
            # if selected_label == 'All':
            #     plot_data = data
            # else:
            #     plot_data = data[np.where(labels == int(selected_label))]
            data = np.hstack((self.data[1:, 1].reshape((-1, 1)), self.data[1:, self.selected_column].reshape((-1, 1)))).astype(float)
            self.hist = HistogramWidget()
            self.hist.init_data(self.header[self.selected_column], data)
            self.hist.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
            self.hist.show()

class TableWidgetForSelPanel(TableWidget):
    def __init__(self):
        super(TableWidgetForSelPanel, self).__init__()

    def display_table(self, page):
        if page in self.page_dict:
            tmp_data = self.data[self.page_dict[page][0]: self.page_dict[page][1]]
            self.tableWidget.setRowCount(tmp_data.shape[0])
            self.tableWidget.setColumnCount(tmp_data.shape[1])
            self.tableWidget.setHorizontalHeaderLabels(self.header)
            for i in range(tmp_data.shape[0]):
                for j in range(tmp_data.shape[1]):
                    if j == 0:
                        cell = QTableWidgetItem(str(tmp_data[i][j]))
                    else:
                        cell = QTableWidgetItem(str(round(float(tmp_data[i][j]), 6)))
                    self.tableWidget.setItem(i, j, cell)
        else:
            QMessageBox.critical(self, 'Error', 'Page number out of index.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        labels = ['All'] + [str(i) for i in set(self.data[1:, 0])]
        self.label_comboBox.clear()
        self.label_comboBox.addItems(labels)
        self.label_comboBox.setCurrentIndex(0)

    def horizontal_header_clicked(self, index):
        self.selected_column = index

    def display_histogram(self):
        if self.selected_column is None:
            QMessageBox.critical(self, 'Error', 'Please select a column.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        elif self.selected_column > len(self.header) - 1:
            QMessageBox.critical(self, 'Error', 'Incorrect column index.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        else:
            # labels = np.array(self.data[:, 0]).astype(int)
            # selected_label = self.label_comboBox.currentText()
            # data = np.array(self.data[:, self.selected_column])
            # if selected_label == 'All':
            #     plot_data = data
            # else:
            #     plot_data = data[np.where(labels == int(selected_label))]
            data = np.hstack((self.data[:, 0].reshape((-1, 1)), self.data[:, self.selected_column].reshape((-1, 1)))).astype(float)
            self.hist = HistogramWidget()
            self.hist.init_data(self.header[self.selected_column], data)
            self.hist.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
            self.hist.show()



if __name__ == '__main__':
    df = pd.read_csv('mutitask.tsv', delimiter='\t', header=0)
    app = QApplication(sys.argv)
    win = TableWidget()
    win.setRowAndColumns(40, df.values.shape[1])
    win.setTabletHeader(df.columns)
    win.init_data(df.values)

    win.show()
    sys.exit(app.exec_())
