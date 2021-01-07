#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import math
import time


class DealDataset(Dataset):
    def __init__(self, np_data, np_label):
        self.X = torch.from_numpy(np_data)
        self.y = torch.from_numpy(np_label).view(-1, 1)
        self.len = np_data.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


class Attention(nn.Module):
    """Attention layer
    Args:
        in_features_2d: the input shape for a sample. e.g. (sequence_length, dimension)
    """
    def __init__(self, in_features_2d, hidden = 10):
        super(Attention, self).__init__()
        self.input_dim = in_features_2d[1]
        self.input_length = in_features_2d[0]
        self.hidden = hidden
        self.W0 = nn.Parameter(torch.Tensor(self.input_dim, self.hidden))
        self.b0 = nn.Parameter(torch.Tensor(self.hidden, ))
        self.W  = nn.Parameter(torch.Tensor(self.hidden, 1))
        self.b  = nn.Parameter(torch.Tensor(1,))
        self.reset_parameters()

    def reset_parameters(self):
        stdv_W0 = 1./ math.sqrt(self.W0.size(0))
        self.W0.data.uniform_(-stdv_W0, stdv_W0)
        stdv_b0 = 1./ math.sqrt(self.b0.size(0))
        self.b0.data.uniform_(-stdv_b0, stdv_b0)
        stdv_W = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv_W, stdv_W)
        stdv_b = 1. / math.sqrt(self.b.size(0))
        self.b.data.uniform_(-stdv_b, stdv_b)

    def forward(self, x):
        """
        :param x: size: (batch_size, sequence_length, input_dim)
        :return: size: (batch_size, sequence_length + input_dim)
        """
        energy = torch.bmm(x, self.W0.repeat(x.size(0), 1, 1)) + self.b0.repeat(x.size(0), 1, 1)
        energy = torch.bmm(energy, self.W.repeat(x.size(0), 1, 1)) + self.b.repeat(x.size(0), 1, 1)
        energy = energy.view(-1, self.input_length)
        energy = F.softmax(energy, 1)

        xx = torch.bmm(torch.unsqueeze(energy, 1), x)
        output = torch.cat((torch.squeeze(xx, 1), energy), 1)

        return output


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left=nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channel)
        )
        self.right=shortcut

    def forward(self, X):
        out = self.left(X)
        residual = X if self.right is None else self.right(X)
        out += residual
        return F.relu(out)


class Net_CNN_1(nn.Module):
    """
        Simple CNN network.
        Four Conv1d layers
        A fc layer
        A output layer
    """
    def __init__(self, device, category, input_size, sequence_length, out_channel=64, padding=2, conv_kernel_size=5,
                 pool_kernel_size=2, dense_size=64, dropout=0.5):
        super(Net_CNN_1, self).__init__()
        self.device = device
        self.input_size = input_size  # the vector dimension for each AA or Nucleotide
        self.sequence_length = sequence_length  # biological sequence length
        self.conv1 = nn.Sequential(  # (input_size sequence_length)
            nn.Conv1d(in_channels=input_size, out_channels=out_channel, kernel_size=conv_kernel_size, stride=1,
                      padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_kernel_size)  # -> (out_channel, sequence_length//2)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=conv_kernel_size, stride=1,
                      padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_kernel_size)  # -> (out_channel, sequence_length//4)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=conv_kernel_size, stride=1,
                      padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_kernel_size)  # -> (out_channel, sequence_length//8)
        )
        self.dropout3 = nn.Dropout(dropout)
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=conv_kernel_size, stride=1,
                      padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_kernel_size)  # -> (out_channel, sequence_length//16)
        )
        self.dropout4 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(out_channel * self.calculate_dimension(sequence_length, padding, conv_kernel_size),
                             dense_size)
        self.fc2 = nn.Linear(dense_size, category)

        # best model save
        self.best_model_dict = None

    def calculate_dimension(self, sequence_length, padding, kernel_size):
        L_conv1 = (sequence_length + 2 * padding - kernel_size + 1) // 2
        L_conv2 = (L_conv1 + 2 * padding - kernel_size + 1) // 2
        L_conv3 = (L_conv2 + 2 * padding - kernel_size + 1) // 2
        L_conv4 = (L_conv3 + 2 * padding - kernel_size + 1) // 2
        return L_conv4

    def forward(self, X, is_train=False):
        out = X.view(-1, self.sequence_length, self.input_size)
        out = out.permute(0, 2, 1)
        out = self.conv1(out)
        if is_train:
            out = self.dropout1(out)
        out = self.conv2(out)
        if is_train:
            out = self.dropout2(out)
        out = self.conv3(out)
        if is_train:
            out = self.dropout3(out)
        out = self.conv4(out)
        if is_train:
            out = self.dropout4(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))        
        # out = torch.softmax(self.fc2(out), dim=1)
        out = self.fc2(out)
        return out

    def fit(self, train_loader, valid_loader, epochs=1000, early_stopping=100, lr=0.001):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        minmum_loss = np.inf
        last_saving = 0
        for epoch in range(epochs):
            time0 = time.time()
            loss_train = np.inf
            for i, (batch_X, batch_y) in enumerate(train_loader):
                Xb = batch_X.float().to(self.device)
                yb = batch_y.view(-1).to(self.device)

                # Forward + Backward + Optimize
                optimizer.zero_grad()
                y_ = self.forward(Xb, is_train=True)
                loss = criterion(y_, yb.long())
                loss.backward()
                optimizer.step()
                loss_train = loss.item()
            loss_valid = float(self.evaluate(valid_loader))
            if loss_valid < minmum_loss:
                minmum_loss = loss_valid
                # torch.save(self.state_dict(), out)
                self.best_model_dict = self.state_dict()
                last_saving = epoch
            # if last_saving == epoch:
            #     print('Epoch [%d/%d]: val_loss improved to %f, saving model to model file'
            #           % (epoch + 1, epochs, minmum_loss))
            # else:
            #     print('Epoch [%d/%d]: val_loss did not improve' % (epoch + 1, epochs))
            # interval = time.time() - time0
            # print('%.1fs - loss: %f - val_loss: %f\n' % (interval, loss_train, loss_valid))
            if epoch - last_saving > early_stopping: break

        if not self.best_model_dict is None:
            self.load_state_dict(self.best_model_dict)
            # torch.save(self, 'model.pkl')

    def evaluate(self, loader):
        loss = 0
        criterion = nn.CrossEntropyLoss()
        for batch_X, batch_y in loader:
            Xb = batch_X.float().to(self.device)
            yb = batch_y.view(-1).to(self.device)
            y_ = self.forward(Xb)
            loss += criterion(y_, yb.long()).item()
        return loss / len(loader)

    def predict(self, loader):
        score = []
        for batch_X, batch_y in loader:
            Xb = batch_X.float().to(self.device)
            yb = batch_y.view(-1).to(self.device)
            y_ = self.forward(Xb)
            score.append(y_.cpu().data)
        return torch.cat(score).numpy()


class Net_CNN_11(nn.Module):
    """
        Simple CNN network.
        Four Conv1d layers
        A fc layer
        A output layer
    """
    def __init__(self, device, input_size, sequence_length, out_channel=64, padding=2, conv_kernel_size=5,
                 pool_kernel_size=2, dense_size=64, dropout=0.5):
        super(Net_CNN_11, self).__init__()
        self.device = device
        self.input_size = input_size  # the vector dimension for each AA or Nucleotide
        self.sequence_length = sequence_length  # biological sequence length
        self.conv1 = nn.Sequential(  # (input_size sequence_length)
            nn.Conv1d(in_channels=input_size, out_channels=out_channel, kernel_size=conv_kernel_size, stride=1,
                      padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_kernel_size)  # -> (out_channel, sequence_length//2)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=conv_kernel_size, stride=1,
                      padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_kernel_size)  # -> (out_channel, sequence_length//4)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=conv_kernel_size, stride=1,
                      padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_kernel_size)  # -> (out_channel, sequence_length//8)
        )
        self.dropout3 = nn.Dropout(dropout)
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=conv_kernel_size, stride=1,
                      padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_kernel_size)  # -> (out_channel, sequence_length//16)
        )
        self.dropout4 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(out_channel * self.calculate_dimension(sequence_length, padding, conv_kernel_size),
                             dense_size)
        self.fc2 = nn.Linear(dense_size, 1)

        # best model save
        self.best_model_dict = None

    def calculate_dimension(self, sequence_length, padding, kernel_size):
        L_conv1 = (sequence_length + 2 * padding - kernel_size + 1) // 2
        L_conv2 = (L_conv1 + 2 * padding - kernel_size + 1) // 2
        L_conv3 = (L_conv2 + 2 * padding - kernel_size + 1) // 2
        L_conv4 = (L_conv3 + 2 * padding - kernel_size + 1) // 2
        return L_conv4

    def forward(self, X, is_train=False, is_feature=False):
        out = X.view(-1, self.sequence_length, self.input_size)
        out = out.permute(0, 2, 1)
        out = self.conv1(out)
        if is_train:
            out = self.dropout1(out)
        out = self.conv2(out)
        if is_train:
            out = self.dropout2(out)
        out = self.conv3(out)
        if is_train:
            out = self.dropout3(out)
        out = self.conv4(out)
        if is_train:
            out = self.dropout4(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        if is_feature:
            return out
        out = torch.sigmoid(self.fc2(out))
        return out

    def fit(self, train_loader, valid_loader, epochs=1000, early_stopping=100, lr=0.001):
        # Loss and Optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        minmum_loss = np.inf
        last_saving = 0

        for epoch in range(epochs):
            time0 = time.time()
            loss_train = np.inf
            for i, (batch_X, batch_y) in enumerate(train_loader):
                Xb = batch_X.float().to(self.device)
                yb = batch_y.float().to(self.device)

                # Forward + Backward + Optimize
                optimizer.zero_grad()
                y_ = self.forward(Xb, is_train=True)
                loss = criterion(y_, yb)
                loss.backward()
                optimizer.step()
                loss_train = loss.item()
            loss_valid = float(self.evaluate(valid_loader))
            if loss_valid < minmum_loss:
                minmum_loss = loss_valid
                # torch.save(self.state_dict(), out)
                self.best_model_dict = self.state_dict()
                last_saving = epoch
            # if last_saving == epoch:
            #     print('Epoch [%d/%d]: val_loss improved to %f, saving model to model file -- this is net 11'
            #           % (epoch + 1, epochs, minmum_loss))
            # else:
            #     print('Epoch [%d/%d]: val_loss did not improve' % (epoch + 1, epochs))
            # interval = time.time() - time0
            # print('%.1fs - loss: %f - val_loss: %f\n' % (interval, loss_train, loss_valid))
            if epoch - last_saving > early_stopping: break

        if not self.best_model_dict is None:
            self.load_state_dict(self.best_model_dict)

    def evaluate(self, loader):
        loss = 0
        criterion = nn.BCELoss()
        for batch_X, batch_y in loader:
            Xb = batch_X.float().to(self.device)
            yb = batch_y.float().to(self.device)
            y_ = self.forward(Xb)
            loss += criterion(y_, yb).item()
        return loss / len(loader)

    def predict(self, loader):
        score = []
        for batch_X, batch_y in loader:
            Xb = batch_X.float().to(self.device)
            y_ = self.forward(Xb)
            score.append(y_.cpu().data)
        score = torch.cat(score).numpy()
        tmp_score = np.zeros((len(score), 2))
        tmp_score[:, 0] = 1 - score[:, 0]
        tmp_score[:, 1] = score[:, 0]
        return tmp_score


class Net_RNN_2(nn.Module):
    def __init__(self, device, category, input_size, sequence_length, hidden_size=32, num_layers=1, dense_size=64, dropout=0.5, bidirectional=False):
        super(Net_RNN_2, self).__init__()
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.device = device
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        if bidirectional:
            fc_input_size = 2 * hidden_size
        else:
            fc_input_size = hidden_size
        self.fc1 = nn.Linear(fc_input_size, dense_size)
        self.fc2 = nn.Linear(dense_size, category)

        # best model save
        self.best_model_dict = None

    def forward(self, X, is_train=False):
        out, _ = self.lstm(X, None)
        out = self.dropout(out)
        out = out[:, -1, :]
        out = F.relu(self.fc1(out))
        if is_train:
            out = self.dropout(out)
        # out = torch.softmax(self.fc2(out), dim=1)
        out = self.fc2(out)
        return out

    def fit(self, train_loader, valid_loader, epochs=1000, early_stopping=100, lr=0.001):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        minmum_loss = np.inf
        last_saving = 0
        for epoch in range(epochs):
            time0 = time.time()
            loss_train = np.inf
            for i, (batch_X, batch_y) in enumerate(train_loader):
                Xb = batch_X.view(-1, self.sequence_length, self.input_size).float().to(self.device)
                yb = batch_y.view(-1).long().to(self.device)
                optimizer.zero_grad()
                y_ = self.forward(Xb, is_train=True)
                loss = criterion(y_, yb)
                loss.backward()
                optimizer.step()
                loss_train = loss.item()
            loss_valid = float(self.evaluate(valid_loader))
            if loss_valid < minmum_loss:
                minmum_loss = loss_valid
                # torch.save(self.state_dict(), out)
                self.best_model_dict = self.state_dict()
                last_saving = epoch
            # if last_saving == epoch:
            #     print('Epoch [%d/%d]: val_loss improved to %f, saving model to %s'
            #           % (epoch + 1, epochs, minmum_loss, 'out'))
            # else:
            #     print('Epoch [%d/%d]: val_loss did not improve' % (epoch + 1, epochs))
            # interval = time.time() - time0
            # print('%.1fs - loss: %f - val_loss: %f\n' % (interval, loss_train, loss_valid))
            if epoch - last_saving > early_stopping: break
        if not self.best_model_dict is None:
            self.load_state_dict(self.best_model_dict)

    def evaluate(self, loader):
        loss = 0
        criterion = nn.CrossEntropyLoss()
        for batch_X, batch_y in loader:
            Xb = batch_X.view(-1, self.sequence_length, self.input_size).float().to(self.device)
            yb = batch_y.view(-1).long().to(self.device)
            y_ = self.forward(Xb)
            loss += criterion(y_, yb).item()
        return loss / len(loader)

    def predict(self, loader):
        score = []
        for batch_X, batch_y in loader:
            Xb = batch_X.view(-1, self.sequence_length, self.input_size).float().to(self.device)
            yb = batch_y.view(-1).long().to(self.device)
            y_ = self.forward(Xb)
            score.append(y_.cpu().data)
        return torch.cat(score).numpy()


class Net_ABCNN_4(nn.Module):
    def __init__(self, device, category, input_size, sequence_length, dropout=0.75):
        super(Net_ABCNN_4, self).__init__()
        self.device = device
        self.input_size = input_size                                    # the vector dimension for each AA or Nucleotide
        self.sequence_length = sequence_length                          # biological sequence length
        self.conv1 = nn.Sequential(                                     # (200 * sequence_length)
            nn.Conv1d(in_channels=input_size, out_channels=200, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Sequential(                                     # (150 * (sequence_length - 8))
            nn.Conv1d(in_channels=200, out_channels=150, kernel_size=9, stride=1, padding=0),
            nn.ReLU()
        )
        self.dropout2 = nn.Dropout(dropout)
        self.conv3 = nn.Sequential(                                     # (200 * (sequence_length - 17))
            nn.Conv1d(in_channels=150, out_channels=200, kernel_size=10, stride=1, padding=0),
            nn.ReLU()
        )
        self.dropout3 = nn.Dropout(dropout)
        #
        self.attention_1 = Attention((self.calculate_dimension(self.sequence_length, 0, 1, 9, 10), 200), 8)
        self.attention_2 = Attention((200, self.calculate_dimension(self.sequence_length, 0, 1, 9, 10)), 10)
        self.fc1 = nn.Linear(2*(self.calculate_dimension(self.sequence_length, 0, 1, 9, 10) + 200), 150)
        self.fc2 = nn.Linear(150, 8)
        self.fc3 = nn.Linear(8, category)

        #
        self.best_model_dict = None

    def calculate_dimension(self, sequence_length, padding, kernel_size_1, kernel_size_2, kernel_size_3):
        L_conv1 = sequence_length + 2 * padding - kernel_size_1 + 1
        L_conv2 = L_conv1 + 2 * padding - kernel_size_2 + 1
        L_conv3 = L_conv2 + 2 * padding - kernel_size_3 + 1
        return L_conv3

    def forward(self, X, is_train=False):
        out = X.view(-1, self.sequence_length, self.input_size)
        out = out.permute(0, 2, 1)
        out = self.conv1(out)
        if is_train:
            out = self.dropout1(out)
        out = self.conv2(out)
        if is_train:
            out = self.dropout2(out)
        out = self.conv3(out)
        if is_train:
            out = self.dropout3(out)
        out_reshape = out.permute(0, 2, 1)
        out_reshape = self.attention_1(out_reshape)
        out = self.attention_2(out)
        out_merge = torch.cat((out, out_reshape), 1)
        out_merge = F.relu(self.fc1(out_merge))
        out_merge = F.relu(self.fc2(out_merge))
        # out_merge = torch.softmax(self.fc3(out_merge), dim=1)
        out_merge = self.fc3(out_merge)
        return out_merge

    def fit(self, train_loader, valid_loader, epochs=1000, early_stopping=100, lr=0.001):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        minmum_loss = np.inf
        last_saving = 0
        for epoch in range(epochs):
            time0 = time.time()
            loss_train = np.inf
            for i, (batch_X, batch_y) in enumerate(train_loader):
                Xb = batch_X.float().to(self.device)
                yb = batch_y.view(-1).long().to(self.device)
                # Forward + Backward + Optimize
                optimizer.zero_grad()
                y_ = self.forward(Xb, is_train=True)
                loss = criterion(y_, yb)
                loss.backward()
                optimizer.step()
                loss_train = loss.item()
            loss_valid = float(self.evaluate(valid_loader))
            if loss_valid < minmum_loss:
                minmum_loss = loss_valid
                # torch.save(self.state_dict(), out)
                self.best_model_dict = self.state_dict()
                last_saving = epoch
            # if last_saving == epoch:
            #     print('Epoch [%d/%d]: val_loss improved to %f, saving model to %s'
            #           % (epoch + 1, epochs, minmum_loss, 'out'))
            # else:
            #     print('Epoch [%d/%d]: val_loss did not improve' % (epoch + 1, epochs))
            # interval = time.time() - time0
            # print('%.1fs - loss: %f - val_loss: %f\n' % (interval, loss_train, loss_valid))
            if epoch - last_saving > early_stopping: break
        if not self.best_model_dict is None:
            self.load_state_dict(self.best_model_dict)

    def evaluate(self, loader):
        loss = 0
        criterion = nn.CrossEntropyLoss()
        for batch_X, batch_y in loader:
            Xb = batch_X.float().to(self.device)
            yb = batch_y.view(-1).long().to(self.device)
            y_ = self.forward(Xb)
            loss += criterion(y_, yb).item()
        return loss / len(loader)

    def predict(self, loader):
        score = []
        for batch_X, batch_y in loader:
            Xb = batch_X.float().to(self.device)
            yb = batch_y.view(-1).long().to(self.device)
            y_ = self.forward(Xb)
            score.append(y_.cpu().data)
        return torch.cat(score).numpy()


class Net_ResNet_5(nn.Module):
    def __init__(self, device, category, input_size, sequence_length, out_channel=64, dense_size=64):
        super(Net_ResNet_5, self).__init__()
        self.device = device
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.out_channel = out_channel
        self.dense_size = dense_size
        # The first layer
        self.pre = nn.Sequential(
            nn.Conv1d(input_size, out_channel, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )
        # Repeated layers
        self.layer1 = self._make_layer(out_channel, out_channel, 2, 1)
        self.layer2 = self._make_layer(out_channel, 2*out_channel, 2, 2)
        self.layer3 = self._make_layer(2*out_channel, 4*out_channel, 2, 2)
        self.layer4 = self._make_layer(4*out_channel, 8*out_channel, 2, 2)

        # FC layer
        self.fc1 = nn.Linear(self.calculate_dimension(self.sequence_length) * 8 * out_channel, dense_size)
        self.fc2 = nn.Linear(dense_size, category)

        #
        self.best_model_dict = None

    def _make_layer(self, in_channel, out_channel, block_num, stride=1):
        shortcut = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channel)
        )
        layers = []
        layers.append(ResidualBlock(in_channel, out_channel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(out_channel, out_channel))

        return nn.Sequential(*layers)

    def forward(self, X):
        out = X.view(-1, self.sequence_length, self.input_size)
        out = out.permute(0, 2, 1)
        # print('out: ', out.size())
        out = self.pre(out)
        # print('Pre: ', out.size())
        out = self.layer1(out)
        # print('layer 1: ', out.size())
        out = self.layer2(out)
        # print('layer 2: ', out.size())
        out = self.layer3(out)
        # print('layer 3: ', out.size())
        out = self.layer4(out)
        # print('layer 4: ', out.size())
        out = F.avg_pool1d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        # out = torch.softmax(self.fc2(out), dim=1)
        out = self.fc2(out)
        return out

    def fit(self, train_loader, valid_loader, epochs=1000, early_stopping=100, lr=0.001):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        minmum_loss = np.inf
        last_saving = 0

        for epoch in range(epochs):
            time0 = time.time()
            loss_train = np.inf
            for i, (batch_X, batch_y) in enumerate(train_loader):
                Xb = batch_X.float().to(self.device)
                yb = batch_y.view(-1).long().to(self.device)

                # Forward + Backward + Optimize
                optimizer.zero_grad()
                y_ = self.forward(Xb)
                loss = criterion(y_, yb)
                loss.backward()
                optimizer.step()
                loss_train = loss.item()
            loss_valid = float(self.evaluate(valid_loader))
            if loss_valid < minmum_loss:
                minmum_loss = loss_valid
                # torch.save(self.state_dict(), out)
                self.best_model_dict = self.state_dict()
                last_saving = epoch
            # if last_saving == epoch:
            #     print('Epoch [%d/%d]: val_loss improved to %f, saving model to %s'
            #           % (epoch + 1, epochs, minmum_loss, 'out'))
            # else:
            #     print('Epoch [%d/%d]: val_loss did not improve' % (epoch + 1, epochs))
            # interval = time.time() - time0
            # print('%.1fs - loss: %f - val_loss: %f\n' % (interval, loss_train, loss_valid))
            if epoch - last_saving > early_stopping: break
        if not self.best_model_dict is None:
            self.load_state_dict(self.best_model_dict)

    def evaluate(self, loader):
        loss = 0
        criterion = nn.CrossEntropyLoss()
        for batch_X, batch_y in loader:
            Xb = batch_X.float().to(self.device)
            yb = batch_y.view(-1).long().to(self.device)
            y_ = self.forward(Xb)
            loss += criterion(y_, yb).item()
        return loss / len(loader)

    def predict(self, loader):
        score = []
        for batch_X, batch_y in loader:
            Xb = batch_X.float().to(self.device)
            yb = batch_y.view(-1).to(self.device)
            y_ = self.forward(Xb)
            score.append(y_.cpu().data)
        return torch.cat(score).numpy()

    def calculate_dimension(self, sequence_length):
        pre_out = ((sequence_length - 1) // 2 + 1) // 2
        layer1_out = pre_out
        layer2_out = (layer1_out - 1) // 2 + 1
        layer3_out = (layer2_out - 1) // 2 + 1
        layer4_out = (layer3_out - 1) // 2 + 1
        return layer4_out // 2


class Net_AutoEncoder_6(nn.Module):
    def __init__(self, device, category, input_dim):
        super(Net_AutoEncoder_6, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.category = category
        self.best_model_dict = None
        self.best_classification_model_dict = None

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, X, re_train=False):
        encoder = self.encoder(X)
        if re_train:
            decoder = torch.softmax(self.decoder(encoder), dim=1)
        else:
            decoder = self.decoder(encoder)
        return decoder

    def fit(self, train_loader, valid_loader, epochs=1000, early_stopping=100, lr=0.001):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        minmum_loss = np.inf
        last_saving = 0
        for epoch in range(epochs):
            time0 = time.time()
            loss_train = np.inf
            for i, (batch_X, batch_y) in enumerate(train_loader):
                Xb = batch_X.float().to(self.device)
                yb = batch_X.float().to(self.device)
                # Forward + Backward + Optimize
                optimizer.zero_grad()
                y_ = self.forward(Xb)
                loss = criterion(y_, yb)
                loss.backward()
                optimizer.step()
                loss_train = loss.item()
            loss_valid = float(self.evaluate(valid_loader))
            if loss_valid < minmum_loss:
                minmum_loss = loss_valid
                self.best_model_dict = self.state_dict()
                last_saving = epoch
            if epoch - last_saving > early_stopping: break
        self.load_state_dict(self.best_model_dict)

    def evaluate(self, loader):
        loss = 0
        criterion = nn.MSELoss()
        for batch_X, batch_y in loader:
            Xb = batch_X.float().to(self.device)
            yb = batch_X.float().to(self.device)
            y_ = self.forward(Xb)
            loss += criterion(y_, yb).item()
        return loss / len(loader)

    def re_build_net(self):
        self.load_state_dict(self.best_model_dict)
        self.decoder = nn.Linear(128, self.category).to(self.device)

        for name, param in self.named_parameters():
            if 'decoder' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def re_fit(self, train_loader, valid_loader, epochs=1000, early_stopping=100, lr=0.001):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        minmum_loss = np.inf
        last_saving = 0
        for epoch in range(epochs):
            time0 = time.time()
            loss_train = np.inf
            for i, (batch_X, batch_y) in enumerate(train_loader):
                Xb = batch_X.float().to(self.device)
                yb = batch_y.view(-1).long().to(self.device)
                # Forward + Backward + Optimize
                optimizer.zero_grad()
                y_ = self.forward(Xb, re_train=True)
                loss = criterion(y_, yb)
                loss.backward()
                optimizer.step()
                loss_train = loss.item()
            loss_valid = float(self.re_evaluate(valid_loader))
            if loss_valid < minmum_loss:
                minmum_loss = loss_valid
                # torch.save(self.state_dict(), out)
                self.best_classification_model_dict = self.state_dict()
                last_saving = epoch
            if epoch - last_saving > early_stopping: break
        if not self.best_model_dict is None:
            self.load_state_dict(self.best_classification_model_dict)

    def re_evaluate(self, loader):
        loss = 0
        criterion = nn.CrossEntropyLoss()
        for batch_X, batch_y in loader:
            Xb = batch_X.float().to(self.device)
            yb = batch_y.view(-1).long().to(self.device)
            y_ = self.forward(Xb, re_train=True)
            loss += criterion(y_, yb).item()
        return loss / len(loader)

    def predict(self, loader):
        score = []
        for batch_X, batch_y in loader:
            Xb = batch_X.float().to(self.device)
            yb = batch_y.view(-1).long().to(self.device)
            y_ = self.forward(Xb, re_train=True)
            score.append(y_.cpu().data)
        return torch.cat(score).numpy()

