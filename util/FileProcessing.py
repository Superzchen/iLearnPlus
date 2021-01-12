#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import os, sys, re
import platform
from collections import Counter
import itertools
import numpy as np
import math
import random
import pickle


class Sequence(object):
    def __init__(self, file):
        self.file = file  # whole file path
        self.fasta_list = []  # 2-D list [sampleName, fragment, label, training or testing]
        self.sample_purpose = None  # 1-D ndarray, sample used as training dataset (True) or testing dataset(False)
        self.sequence_number = 0  # int: the number of samples
        self.sequence_type = ''  # DNA, RNA or Protein
        self.is_equal = False  # bool: sequence with equal length?
        self.minimum_length = 1  # int
        self.maximum_length = 0  # int
        self.minimum_length_without_minus = 1  # int
        self.maximum_length_without_minus = 0  # int
        self.error_msg = ''  # string

        self.fasta_list, self.sample_purpose, self.error_msg = self.read_fasta(self.file)
        self.sequence_number = len(self.fasta_list)

        if self.sequence_number > 0:
            self.is_equal, self.minimum_length, self.maximum_length, self.minimum_length_without_minus, self.maximum_length_without_minus = self.sequence_with_equal_length()
            self.sequence_type = self.check_sequence_type()


        else:
            self.error_msg = 'File format error.'

    def read_fasta(self, file):
        """
        read fasta sequence
        :param file:
        :return:
        """
        msg = ''
        if not os.path.exists(self.file):
            msg = 'Error: file %s does not exist.' % self.file
            return [], None, msg
        with open(file) as f:
            records = f.read()
        records = records.split('>')[1:]
        fasta_sequences = []
        for fasta in records:
            array = fasta.split('\n')
            header, sequence = array[0].split()[0], re.sub('[^ACDEFGHIKLMNPQRSTUVWY-]', '-', ''.join(array[1:]).upper())
            header_array = header.split('|')
            name = header_array[0]
            label = header_array[1] if len(header_array) >= 2 else '0'
            label_train = header_array[2] if len(header_array) >= 3 else 'training'
            fasta_sequences.append([name, sequence, label, label_train])
        sample_purpose = np.array([item[3] == 'training' for item in fasta_sequences])
        return fasta_sequences, sample_purpose, msg

    def sequence_with_equal_length(self):
        """
        Check if fasta sequence is in equal length
        :return:
        """
        length_set = set()
        length_set_1 = set()
        for item in self.fasta_list:
            length_set.add(len(item[1]))
            length_set_1.add(len(re.sub('-', '', item[1])))

        length_set = sorted(length_set)
        length_set_1 = sorted(length_set_1)
        if len(length_set) == 1:
            return True, length_set[0], length_set[-1], length_set_1[0], length_set_1[-1]
        else:
            return False, length_set[0], length_set[-1], length_set_1[0], length_set_1[-1]

    def check_sequence_type(self):
        """
        Specify sequence type (Protein, DNA or RNA)
        :return:
        """
        tmp_fasta_list = []
        if len(self.fasta_list) < 100:
            tmp_fasta_list = self.fasta_list
        else:
            random_index = random.sample(range(0, len(self.fasta_list)), 100)
            for i in random_index:
                tmp_fasta_list.append(self.fasta_list[i])

        sequence = ''
        for item in tmp_fasta_list:
            sequence += item[1]

        char_set = set(sequence)
        if 5 < len(char_set) <= 21:
            for line in self.fasta_list:
                line[1] = re.sub('[^ACDEFGHIKLMNPQRSTVWY]', '-', line[1])
            return 'Protein'
        elif 0 < len(char_set) <= 5 and 'T' in char_set:
            return 'DNA'
        elif 0 < len(char_set) <= 5 and 'U' in char_set:
            for line in self.fasta_list:
                line[1] = re.sub('U', 'T', line[1])
            return 'RNA'
        else:
            return 'Unknown'


class Descriptor(Sequence):
    def __init__(self, file, kw):
        super(Descriptor, self).__init__(file=file)
        self.kw = kw  # dict
        self.encoding_array = np.array([])  # 2-D ndarray with column name and index name
        self.column = 0  # int
        self.row = 0  # int
        """ variable for ACC descriptors """
        self.myDiIndex = {
            'AA': 0, 'AC': 1, 'AG': 2, 'AT': 3,
            'CA': 4, 'CC': 5, 'CG': 6, 'CT': 7,
            'GA': 8, 'GC': 9, 'GG': 10, 'GT': 11,
            'TA': 12, 'TC': 13, 'TG': 14, 'TT': 15
        }
        self.myTriIndex = {
            'AAA': 0, 'AAC': 1, 'AAG': 2, 'AAT': 3,
            'ACA': 4, 'ACC': 5, 'ACG': 6, 'ACT': 7,
            'AGA': 8, 'AGC': 9, 'AGG': 10, 'AGT': 11,
            'ATA': 12, 'ATC': 13, 'ATG': 14, 'ATT': 15,
            'CAA': 16, 'CAC': 17, 'CAG': 18, 'CAT': 19,
            'CCA': 20, 'CCC': 21, 'CCG': 22, 'CCT': 23,
            'CGA': 24, 'CGC': 25, 'CGG': 26, 'CGT': 27,
            'CTA': 28, 'CTC': 29, 'CTG': 30, 'CTT': 31,
            'GAA': 32, 'GAC': 33, 'GAG': 34, 'GAT': 35,
            'GCA': 36, 'GCC': 37, 'GCG': 38, 'GCT': 39,
            'GGA': 40, 'GGC': 41, 'GGG': 42, 'GGT': 43,
            'GTA': 44, 'GTC': 45, 'GTG': 46, 'GTT': 47,
            'TAA': 48, 'TAC': 49, 'TAG': 50, 'TAT': 51,
            'TCA': 52, 'TCC': 53, 'TCG': 54, 'TCT': 55,
            'TGA': 56, 'TGC': 57, 'TGG': 58, 'TGT': 59,
            'TTA': 60, 'TTC': 61, 'TTG': 62, 'TTT': 63
        }

    """ Protein descriptors """

    def Protein_AAC(self):
        try:
            # clear
            self.encoding_array = np.array([])
            AA = 'ACDEFGHIKLMNPQRSTVWY'
            header = ['SampleName', 'label']
            encodings = []
            for i in AA:
                header.append(i)
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                count = Counter(sequence)
                for key in count:
                    count[key] = count[key] / len(sequence)
                code = [name, label]
                for aa in AA:
                    code.append(count[aa])
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_EAAC(self):
        try:
            self.encoding_array = np.array([])

            if not self.is_equal:
                self.error_msg = 'EAAC descriptor need fasta sequence with equal length.'
                return False

            AA = 'ARNDCQEGHILKMFPSTWYV'
            encodings = []
            header = ['SampleName', 'label']
            for w in range(1, len(self.fasta_list[0][1]) - self.kw['sliding_window'] + 2):
                for aa in AA:
                    header.append('SW.' + str(w) + '.' + aa)
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for j in range(len(sequence)):
                    if j < len(sequence) and j + self.kw['sliding_window'] <= len(sequence):
                        count = Counter(sequence[j:j + self.kw['sliding_window']])
                        for key in count:
                            count[key] = count[key] / len(sequence[j:j + self.kw['sliding_window']])
                        for aa in AA:
                            code.append(count[aa])
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_CKSAAP(self):
        try:
            # clear
            self.encoding_array = np.array([])

            AA = 'ACDEFGHIKLMNPQRSTVWY'
            encodings = []
            aaPairs = []
            for aa1 in AA:
                for aa2 in AA:
                    aaPairs.append(aa1 + aa2)

            header = ['SampleName', 'label']

            gap = self.kw['kspace']
            for g in range(gap + 1):
                for aa in aaPairs:
                    header.append(aa + '.gap' + str(g))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '' , i[1]), i[2]
                code = [name, label]
                for g in range(gap + 1):
                    myDict = {}
                    for pair in aaPairs:
                        myDict[pair] = 0
                    sum = 0
                    for index1 in range(len(sequence)):
                        index2 = index1 + g + 1
                        if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[
                            index2] in AA:
                            myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1
                            sum = sum + 1
                    for pair in aaPairs:
                        code.append(myDict[pair] / sum)
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_DistancePair(self):
        try:
            cp20_dict = {
                'A': 'A',
                'C': 'C',
                'D': 'D',
                'E': 'E',
                'F': 'F',
                'G': 'G',
                'H': 'H',
                'I': 'I',
                'K': 'K',
                'L': 'L',
                'M': 'M',
                'N': 'N',
                'P': 'P',
                'Q': 'Q',
                'R': 'R',
                'S': 'S',
                'T': 'T',
                'V': 'V',
                'W': 'W',
                'Y': 'Y',
            }
            cp19_dict = {
                'A': 'A',
                'C': 'C',
                'D': 'D',
                'E': 'E',
                'F': 'F',
                'G': 'G',
                'H': 'H',
                'I': 'I',
                'K': 'K',
                'L': 'L',
                'M': 'M',
                'N': 'N',
                'P': 'P',
                'Q': 'Q',
                'R': 'R',
                'S': 'S',
                'T': 'T',
                'V': 'V',
                'W': 'W',
                'Y': 'F',  # YF
            }
            cp14_dict = {
                'A': 'A',
                'C': 'C',
                'D': 'D',
                'E': 'E',
                'F': 'F',
                'G': 'G',
                'H': 'H',
                'I': 'I',
                'K': 'H',      # HRKQ
                'L': 'L',
                'M': 'I',      # IMV
                'N': 'N',
                'P': 'P',
                'Q': 'H',      # HRKQ
                'R': 'H',      # HRKQ
                'S': 'S',
                'T': 'T',
                'V': 'I',      # IMV
                'W': 'W',
                'Y': 'W',      # WY
            }
            cp13_dict = {
                'A': 'A',
                'C': 'C',
                'D': 'D',
                'E': 'E',
                'F': 'F',
                'G': 'G',
                'H': 'H',
                'I': 'I',
                'K': 'K',
                'L': 'I',   # IL
                'M': 'F',   # FM
                'N': 'N',
                'P': 'H',   # HPQWY
                'Q': 'H',   # HPQWY
                'R': 'K',   # KR
                'S': 'S',
                'T': 'T',
                'V': 'V',
                'W': 'H',   # HPQWY
                'Y': 'H',   # HPQWY
            }

            cp20_AA = 'ACDEFGHIKLMNPQRSTVWY'
            cp19_AA = 'ACDEFGHIKLMNPQRSTVW'
            cp14_AA = 'ACDEFGHILNPSTW'
            cp13_AA = 'ACDEFGHIKNSTV'

            distance = self.kw['distance']
            cp = self.kw['cp']

            if self.minimum_length_without_minus < distance + 1:
                self.error_msg = 'The distance value is too large.'
                return False

            AA = cp20_AA
            AA_dict = cp20_dict
            if cp == 'cp(19)':
                AA = cp19_AA
                AA_dict = cp19_dict
            if cp == 'cp(14)':
                AA = cp14_AA
                AA_dict = cp14_dict
            if cp == 'cp(13)':
                AA = cp13_AA
                AA_dict = cp13_dict

            # clear
            self.encoding_array = np.array([])

            encodings = []
            pair_dict = {}
            single_dict = {}
            for aa1 in AA:
                single_dict[aa1] = 0
                for aa2 in AA:
                    pair_dict[aa1+aa2] = 0

            header = ['SampleName', 'label']

            for d in range(distance+1):
                if d == 0:
                    for key in sorted(single_dict):
                        header.append(key)
                else:
                    for key in sorted(pair_dict):
                        header.append('%s.distance%s' %(key, d))
            encodings.append(header)

            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                code = [name, label]

                for d in range(distance + 1):
                    if d == 0:
                        tmp_dict = single_dict.copy()
                        for i in range(len(sequence)):
                            tmp_dict[AA_dict[sequence[i]]] += 1
                        for key in sorted(tmp_dict):
                            code.append(tmp_dict[key]/len(sequence))
                    else:
                        tmp_dict = pair_dict.copy()
                        for i in range(len(sequence) - d):
                            tmp_dict[AA_dict[sequence[i]] + AA_dict[sequence[i+d]]] += 1
                        for key in sorted(tmp_dict):
                            code.append(tmp_dict[key]/(len(sequence) -d))
                encodings.append(code)
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_DPC(self):
        try:
            # clear
            self.encoding_array = np.array([])

            AA = 'ACDEFGHIKLMNPQRSTVWY'
            encodings = []
            diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
            header = ['SampleName', 'label'] + diPeptides
            encodings.append(header)

            AADict = {}
            for i in range(len(AA)):
                AADict[AA[i]] = i

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                tmpCode = [0] * 400
                for j in range(len(sequence) - 2 + 1):
                    tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j + 1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[
                        sequence[j + 1]]] + 1
                if sum(tmpCode) != 0:
                    tmpCode = [i / sum(tmpCode) for i in tmpCode]
                code = code + tmpCode
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_DDE(self):
        try:
            # clear
            self.encoding_array = np.array([])
            AA = 'ACDEFGHIKLMNPQRSTVWY'

            myCodons = {'A': 4, 'C': 2, 'D': 2, 'E': 2, 'F': 2, 'G': 4, 'H': 2, 'I': 3, 'K': 2, 'L': 6,
                        'M': 1, 'N': 2, 'P': 4, 'Q': 2, 'R': 6, 'S': 6, 'T': 4, 'V': 4, 'W': 1, 'Y': 2
                        }

            encodings = []
            diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
            header = ['SampleName', 'label'] + diPeptides
            encodings.append(header)

            myTM = []
            for pair in diPeptides:
                myTM.append((myCodons[pair[0]] / 61) * (myCodons[pair[1]] / 61))

            AADict = {}
            for i in range(len(AA)):
                AADict[AA[i]] = i

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                tmpCode = [0] * 400
                for j in range(len(sequence) - 2 + 1):
                    tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j + 1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[
                        sequence[j + 1]]] + 1
                if sum(tmpCode) != 0:
                    tmpCode = [i / sum(tmpCode) for i in tmpCode]

                myTV = []
                for j in range(len(myTM)):
                    myTV.append(myTM[j] * (1 - myTM[j]) / (len(sequence) - 1))

                for j in range(len(tmpCode)):
                    tmpCode[j] = (tmpCode[j] - myTM[j]) / math.sqrt(myTV[j])

                code = code + tmpCode
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_TPC(self):
        try:
            # clear
            self.encoding_array = np.array([])

            AA = 'ACDEFGHIKLMNPQRSTVWY'
            encodings = []
            triPeptides = [aa1 + aa2 + aa3 for aa1 in AA for aa2 in AA for aa3 in AA]
            header = ['SampleName', 'label'] + triPeptides
            encodings.append(header)

            AADict = {}
            for i in range(len(AA)):
                AADict[AA[i]] = i

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                tmpCode = [0] * 8000
                for j in range(len(sequence) - 3 + 1):
                    tmpCode[AADict[sequence[j]] * 400 + AADict[sequence[j + 1]] * 20 + AADict[sequence[j + 2]]] = tmpCode[
                                                                                                                    AADict[
                                                                                                                        sequence[
                                                                                                                            j]] * 400 +
                                                                                                                    AADict[
                                                                                                                        sequence[
                                                                                                                            j + 1]] * 20 +
                                                                                                                    AADict[
                                                                                                                        sequence[
                                                                                                                            j + 2]]] + 1
                if sum(tmpCode) != 0:
                    tmpCode = [i / sum(tmpCode) for i in tmpCode]
                code = code + tmpCode
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_binary(self):
        try:
            # clear
            self.encoding_array = np.array([])

            if not self.is_equal:
                self.error_msg = 'binary descriptor need fasta sequence with equal length.'
                return False

            AA = 'ARNDCQEGHILKMFPSTWYV'
            encodings = []
            header = ['SampleName', 'label']
            for i in range(1, len(self.fasta_list[0][1]) * 20 + 1):
                header.append('BINARY.F' + str(i))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for aa in sequence:
                    if aa == '-':
                        code = code + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        continue
                    for aa1 in AA:
                        tag = 1 if aa == aa1 else 0
                        code.append(tag)
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_binary_6bit(self):
        try:
            # clear
            self.encoding_array = np.array([])

            if not self.is_equal:
                self.error_msg = 'binary descriptor need fasta sequence with equal length.'
                return False

            AA_group_list = [
                'HRK',
                'DENQ',
                'C',
                'STPAG',
                'MILV',
                'FYW',
            ]
            encodings = []
            header = ['SampleName', 'label']
            header += ['Pos%s_group%s' % (i + 1, j + 1) for i in range(len(self.fasta_list[0][1])) for j in
                    range(len(AA_group_list))]
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for aa in sequence:
                    for j in AA_group_list:
                        if aa in j:
                            code.append(1)
                        else:
                            code.append(0)
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_binary_5bit_type_1(self):
        try:
            # clear
            self.encoding_array = np.array([])

            if not self.is_equal:
                self.error_msg = 'binary descriptor need fasta sequence with equal length.'
                return False

            AA_group_list = [
                'GAVLMI',
                'FYW',
                'KRH',
                'DE',
                'STCPNQ',
            ]
            AA_group_index = ['alphatic', 'aromatic', 'postivecharge', 'negativecharge', 'uncharge']

            encodings = []
            header = ['SampleName', 'label']
            header += ['Pos%s_%s' % (i + 1, j) for i in range(len(self.fasta_list[0][1])) for j in AA_group_index]
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for aa in sequence:
                    for j in AA_group_list:
                        if aa in j:
                            code.append(1)
                        else:
                            code.append(0)
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_binary_5bit_type_2(self):
        try:
            # clear
            self.encoding_array = np.array([])

            if not self.is_equal:
                self.error_msg = 'binary descriptor need fasta sequence with equal length.'
                return False

            aa_dict = {
                'A': [0, 0, 0, 1, 1],
                'C': [0, 0, 1, 0, 1],
                'D': [0, 0, 1, 1, 0],
                'E': [0, 0, 1, 1, 1],
                'F': [0, 1, 0, 0, 1],
                'G': [0, 1, 0, 1, 0],
                'H': [0, 1, 0, 1, 1],
                'I': [0, 1, 1, 0, 0],
                'K': [0, 1, 1, 0, 1],
                'L': [0, 1, 1, 1, 0],
                'M': [1, 0, 0, 0, 1],
                'N': [1, 0, 0, 1, 0],
                'P': [1, 0, 0, 1, 1],
                'Q': [1, 0, 1, 0, 0],
                'R': [1, 0, 1, 0, 1],
                'S': [1, 0, 1, 1, 0],
                'T': [1, 1, 0, 0, 0],
                'V': [1, 1, 0, 0, 1],
                'W': [1, 1, 0, 1, 0],
                'Y': [1, 1, 1, 0, 0],
            }

            encodings = []
            header = ['SampleName', 'label']
            for i in range(1, len(self.fasta_list[0][1]) * 5 + 1):
                header.append('BINARY(5bit).F' + str(i))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for aa in sequence:
                    if aa in aa_dict:
                        code += aa_dict[aa]
                    else:
                        code += [0, 0, 0, 0, 0]
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_binary_3bit_type_1(self):
        try:
            # clear
            self.encoding_array = np.array([])

            if not self.is_equal:
                self.error_msg = 'binary descriptor need fasta sequence with equal length.'
                return False

            AA_group_list = [
                'RKEDQN',
                'GASTPHY',
                'CLVIMFW',
            ]
            AA_group_index = ['Polar', 'Neutral', 'Hydrophobicity']

            encodings = []
            header = ['SampleName', 'label']
            header += ['Pos%s_%s' % (i + 1, j) for i in range(len(self.fasta_list[0][1])) for j in AA_group_index]
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for aa in sequence:
                    for j in AA_group_list:
                        if aa in j:
                            code.append(1)
                        else:
                            code.append(0)
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_binary_3bit_type_2(self):
        try:
            # clear
            self.encoding_array = np.array([])

            if not self.is_equal:
                self.error_msg = 'binary descriptor need fasta sequence with equal length.'
                return False

            AA_group_list = [
                'GASTPD',
                'NVEQIL',
                'MHKFRYW',
            ]
            AA_group_index = ['Volume_range(0-2.78)', 'Volumn_range(2.95-4.0)', 'Volumn_range(4.03-8.08)']

            encodings = []
            header = ['SampleName', 'label']
            header += ['Pos%s_%s' % (i + 1, j) for i in range(len(self.fasta_list[0][1])) for j in AA_group_index]
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for aa in sequence:
                    for j in AA_group_list:
                        if aa in j:
                            code.append(1)
                        else:
                            code.append(0)
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_binary_3bit_type_3(self):
        try:
            # clear
            self.encoding_array = np.array([])

            if not self.is_equal:
                self.error_msg = 'binary descriptor need fasta sequence with equal length.'
                return False

            AA_group_list = [
                'RKEDQN',
                'GASTPHY',
                'CLVIMFW',
            ]
            AA_group_index = ['PolarityValue(4.9-6.2)', 'PolarityValue(8.0-9.2)', 'PolarityValue(10.4-13.0)']

            encodings = []
            header = ['SampleName', 'label']
            header += ['Pos%s_%s' % (i + 1, j) for i in range(len(self.fasta_list[0][1])) for j in AA_group_index]
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for aa in sequence:
                    for j in AA_group_list:
                        if aa in j:
                            code.append(1)
                        else:
                            code.append(0)
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_binary_3bit_type_4(self):
        try:
            # clear
            self.encoding_array = np.array([])

            if not self.is_equal:
                self.error_msg = 'binary descriptor need fasta sequence with equal length.'
                return False

            AA_group_list = [
                'GASDT',
                'CPNVEQIL',
                'KMHFRYW',
            ]
            AA_group_index = ['PolarizabilityValue(0-0.108)', 'PolarizabilityValue(0.128-0.186)',
                            'PolarizabilityValue(0.219-0.409)']

            encodings = []
            header = ['SampleName', 'label']
            header += ['Pos%s_%s' % (i + 1, j) for i in range(len(self.fasta_list[0][1])) for j in AA_group_index]
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for aa in sequence:
                    for j in AA_group_list:
                        if aa in j:
                            code.append(1)
                        else:
                            code.append(0)
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_binary_3bit_type_5(self):
        try:
            # clear
            self.encoding_array = np.array([])

            if not self.is_equal:
                self.error_msg = 'binary descriptor need fasta sequence with equal length.'
                return False

            AA_group_list = [
                'KR',
                'ANCQGHILMFPStWYV',
                'DE',
            ]
            AA_group_index = ['Positive', 'Neutral', 'Negative']

            encodings = []
            header = ['SampleName', 'label']
            header += ['Pos%s_%s' % (i + 1, j) for i in range(len(self.fasta_list[0][1])) for j in AA_group_index]
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for aa in sequence:
                    for j in AA_group_list:
                        if aa in j:
                            code.append(1)
                        else:
                            code.append(0)
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_binary_3bit_type_6(self):
        try:
            # clear
            self.encoding_array = np.array([])

            if not self.is_equal:
                self.error_msg = 'binary descriptor need fasta sequence with equal length.'
                return False

            AA_group_list = [
                'EALMQKRH',
                'VIYCWFT',
                'GNPSD',
            ]
            AA_group_index = ['Helix', 'Strand', 'Coil']

            encodings = []
            header = ['SampleName', 'label']
            header += ['Pos%s_%s' % (i + 1, j) for i in range(len(self.fasta_list[0][1])) for j in AA_group_index]
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for aa in sequence:
                    for j in AA_group_list:
                        if aa in j:
                            code.append(1)
                        else:
                            code.append(0)
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_binary_3bit_type_7(self):
        try:
            # clear
            self.encoding_array = np.array([])

            if not self.is_equal:
                self.error_msg = 'binary descriptor need fasta sequence with equal length.'
                return False

            AA_group_list = [
                'ALFCGIVW',
                'PKQEND',
                'MPSTHY',
            ]
            AA_group_index = ['Buried', 'Exposed', 'Intermediate']

            encodings = []
            header = ['SampleName', 'label']
            header += ['Pos%s_%s' % (i + 1, j) for i in range(len(self.fasta_list[0][1])) for j in AA_group_index]
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for aa in sequence:
                    for j in AA_group_list:
                        if aa in j:
                            code.append(1)
                        else:
                            code.append(0)
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_AESNN3(self):
        try:
            self.encoding_array = np.array([])

            if not self.is_equal:
                self.error_msg = 'AESNN3 descriptor need fasta sequence with equal length.'
                return False

            AESNN3_dict = {
                'A': [-0.99, -0.61,  0.00],
                'R': [ 0.28, -0.99, -0.22],
                'N': [ 0.77, -0.24,  0.59],
                'D': [ 0.74, -0.72, -0.35],
                'C': [ 0.34,  0.88,  0.35],
                'Q': [ 0.12, -0.99, -0.99],
                'E': [ 0.59, -0.55, -0.99],
                'G': [-0.79, -0.99,  0.10],
                'H': [ 0.08, -0.71,  0.68],
                'I': [-0.77,  0.67, -0.37],
                'L': [-0.92,  0.31, -0.99],
                'K': [-0.63,  0.25,  0.50],
                'M': [-0.80,  0.44, -0.71],
                'F': [ 0.87,  0.65, -0.53],
                'P': [-0.99, -0.99, -0.99],
                'S': [ 0.99,  0.40,  0.37],
                'T': [ 0.42,  0.21,  0.97],
                'W': [-0.13,  0.77, -0.90],
                'Y': [ 0.59,  0.33, -0.99],
                'V': [-0.99,  0.27, -0.52],
                '-': [    0,     0,     0],
            }
            encodings = []
            header = ['SampleName', 'label']
            for p in range(1, len(self.fasta_list[0][1]) + 1):
                for z in ('1', '2', '3'):
                    header.append('Pos' + str(p) + '.AESNN3.' + z)
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for aa in sequence:
                    code += AESNN3_dict.get(aa, [0, 0, 0])
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_GAAC(self):
        try:
            # clear
            self.encoding_array = np.array([])

            group = {
                'alphatic': 'GAVLMI',
                'aromatic': 'FYW',
                'postivecharge': 'KRH',
                'negativecharge': 'DE',
                'uncharge': 'STCPNQ'
            }

            groupKey = group.keys()

            encodings = []
            header = ['SampleName', 'label']
            for key in groupKey:
                header.append(key)
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                count = Counter(sequence)
                myDict = {}
                for key in groupKey:
                    for aa in group[key]:
                        myDict[key] = myDict.get(key, 0) + count[aa]

                for key in groupKey:
                    code.append(myDict[key] / len(sequence))
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_EGAAC(self):
        try:
            self.encoding_array = np.array([])

            if not self.is_equal:
                self.error_msg = 'EGAAC descriptor need fasta sequence with equal length.'
                return False

            group = {
                'alphaticr': 'GAVLMI',
                'aromatic': 'FYW',
                'postivecharger': 'KRH',
                'negativecharger': 'DE',
                'uncharger': 'STCPNQ'
            }

            groupKey = group.keys()

            encodings = []
            header = ['SampleName', 'label']
            window = self.kw['sliding_window']
            for w in range(1, len(self.fasta_list[0][1]) - window + 2):
                for g in groupKey:
                    header.append('SW.' + str(w) + '.' + g)
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for j in range(len(sequence)):
                    if j + window <= len(sequence):
                        count = Counter(sequence[j:j + window])
                        myDict = {}
                        for key in groupKey:
                            for aa in group[key]:
                                myDict[key] = myDict.get(key, 0) + count[aa]
                        for key in groupKey:
                            code.append(myDict[key] / window)
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def generateGroupPairs(self, groupKey):
        gPair = {}
        for key1 in groupKey:
            for key2 in groupKey:
                gPair[key1 + '.' + key2] = 0
        return gPair

    def Protein_CKSAAGP(self):
        try:
            # clear
            self.encoding_array = np.array([])
            gap = self.kw['kspace']

            group = {
                'alphaticr': 'GAVLMI',
                'aromatic': 'FYW',
                'postivecharger': 'KRH',
                'negativecharger': 'DE',
                'uncharger': 'STCPNQ'
            }

            AA = 'ARNDCQEGHILKMFPSTWYV'

            groupKey = group.keys()

            index = {}
            for key in groupKey:
                for aa in group[key]:
                    index[aa] = key

            gPairIndex = []
            for key1 in groupKey:
                for key2 in groupKey:
                    gPairIndex.append(key1 + '.' + key2)

            encodings = []
            header = ['SampleName', 'label']
            for g in range(gap + 1):
                for p in gPairIndex:
                    header.append(p + '.gap' + str(g))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                for g in range(gap + 1):
                    gPair = self.generateGroupPairs(groupKey)
                    sum = 0
                    for p1 in range(len(sequence)):
                        p2 = p1 + g + 1
                        if p2 < len(sequence) and sequence[p1] in AA and sequence[p2] in AA:
                            gPair[index[sequence[p1]] + '.' + index[sequence[p2]]] = gPair[
                                                                                        index[sequence[p1]] + '.' + index[
                                                                                            sequence[p2]]] + 1
                            sum = sum + 1

                    if sum == 0:
                        for gp in gPairIndex:
                            code.append(0)
                    else:
                        for gp in gPairIndex:
                            code.append(gPair[gp] / sum)
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_GDPC(self):
        try:
            # clear
            self.encoding_array = np.array([])

            group = {
                'alphaticr': 'GAVLMI',
                'aromatic': 'FYW',
                'postivecharger': 'KRH',
                'negativecharger': 'DE',
                'uncharger': 'STCPNQ'
            }

            groupKey = group.keys()
            baseNum = len(groupKey)
            dipeptide = [g1 + '.' + g2 for g1 in groupKey for g2 in groupKey]

            index = {}
            for key in groupKey:
                for aa in group[key]:
                    index[aa] = key

            encodings = []
            header = ['SampleName', 'label'] + dipeptide
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]

                code = [name, label]
                myDict = {}
                for t in dipeptide:
                    myDict[t] = 0

                sum = 0
                for j in range(len(sequence) - 2 + 1):
                    myDict[index[sequence[j]] + '.' + index[sequence[j + 1]]] = myDict[index[sequence[j]] + '.' + index[
                        sequence[j + 1]]] + 1
                    sum = sum + 1

                if sum == 0:
                    for t in dipeptide:
                        code.append(0)
                else:
                    for t in dipeptide:
                        code.append(myDict[t] / sum)
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_GTPC(self):
        try:
            # clear
            self.encoding_array = np.array([])

            group = {
                'alphaticr': 'GAVLMI',
                'aromatic': 'FYW',
                'postivecharger': 'KRH',
                'negativecharger': 'DE',
                'uncharger': 'STCPNQ'
            }

            groupKey = group.keys()
            baseNum = len(groupKey)
            triple = [g1 + '.' + g2 + '.' + g3 for g1 in groupKey for g2 in groupKey for g3 in groupKey]

            index = {}
            for key in groupKey:
                for aa in group[key]:
                    index[aa] = key

            encodings = []
            header = ['SampleName', 'label'] + triple
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]

                code = [name, label]
                myDict = {}
                for t in triple:
                    myDict[t] = 0

                sum = 0
                for j in range(len(sequence) - 3 + 1):
                    myDict[index[sequence[j]] + '.' + index[sequence[j + 1]] + '.' + index[sequence[j + 2]]] = myDict[index[
                                                                                                                        sequence[
                                                                                                                            j]] + '.' +
                                                                                                                    index[
                                                                                                                        sequence[
                                                                                                                            j + 1]] + '.' +
                                                                                                                    index[
                                                                                                                        sequence[
                                                                                                                            j + 2]]] + 1
                    sum = sum + 1

                if sum == 0:
                    for t in triple:
                        code.append(0)
                else:
                    for t in triple:
                        code.append(myDict[t] / sum)
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_AAIndex(self):
        try:
            props = self.kw['aaindex'].split(';')
            self.encoding_array = np.array([])
            if not self.is_equal:
                self.error_msg = 'AAIndex descriptor need fasta sequence with equal length.'
                return False
            AA = 'ARNDCQEGHILKMFPSTWYV'
            fileAAindex = os.path.split(os.path.realpath(__file__))[
                            0] + r'\data\AAindex.txt' if platform.system() == 'Windows' else \
                os.path.split(os.path.realpath(__file__))[0] + r'/data/AAindex.txt'
            with open(fileAAindex) as f:
                records = f.readlines()[1:]

            AAindex = []
            AAindexName = []
            for i in records:
                AAindex.append(i.rstrip().split()[1:] if i.rstrip() != '' else None)
                AAindexName.append(i.rstrip().split()[0] if i.rstrip() != '' else None)

            index = {}
            for i in range(len(AA)):
                index[AA[i]] = i

            #  use the user inputed properties
            if props:
                tmpIndexNames = []
                tmpIndex = []
                for p in props:
                    if AAindexName.index(p) != -1:
                        tmpIndexNames.append(p)
                        tmpIndex.append(AAindex[AAindexName.index(p)])
                if len(tmpIndexNames) != 0:
                    AAindexName = tmpIndexNames
                    AAindex = tmpIndex

            encodings = []
            header = ['SampleName', 'label']
            for pos in range(1, len(self.fasta_list[0][1]) + 1):
                for idName in AAindexName:
                    header.append('SeqPos.' + str(pos) + '.' + idName)
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for aa in sequence:
                    if aa == '-':
                        for j in AAindex:
                            code.append(0)
                        continue
                    for j in AAindex:
                        code.append(j[index[aa]])
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_ZScale(self):
        try:
            self.encoding_array = np.array([])

            if not self.is_equal:
                self.error_msg = 'ZScale descriptor need fasta sequence with equal length.'
                return False

            zscale = {
                'A': [0.24, -2.32, 0.60, -0.14, 1.30],  # A
                'C': [0.84, -1.67, 3.71, 0.18, -2.65],  # C
                'D': [3.98, 0.93, 1.93, -2.46, 0.75],  # D
                'E': [3.11, 0.26, -0.11, -0.34, -0.25],  # E
                'F': [-4.22, 1.94, 1.06, 0.54, -0.62],  # F
                'G': [2.05, -4.06, 0.36, -0.82, -0.38],  # G
                'H': [2.47, 1.95, 0.26, 3.90, 0.09],  # H
                'I': [-3.89, -1.73, -1.71, -0.84, 0.26],  # I
                'K': [2.29, 0.89, -2.49, 1.49, 0.31],  # K
                'L': [-4.28, -1.30, -1.49, -0.72, 0.84],  # L
                'M': [-2.85, -0.22, 0.47, 1.94, -0.98],  # M
                'N': [3.05, 1.62, 1.04, -1.15, 1.61],  # N
                'P': [-1.66, 0.27, 1.84, 0.70, 2.00],  # P
                'Q': [1.75, 0.50, -1.44, -1.34, 0.66],  # Q
                'R': [3.52, 2.50, -3.50, 1.99, -0.17],  # R
                'S': [2.39, -1.07, 1.15, -1.39, 0.67],  # S
                'T': [0.75, -2.18, -1.12, -1.46, -0.40],  # T
                'V': [-2.59, -2.64, -1.54, -0.85, -0.02],  # V
                'W': [-4.36, 3.94, 0.59, 3.44, -1.59],  # W
                'Y': [-2.54, 2.44, 0.43, 0.04, -1.47],  # Y
                '-': [0.00, 0.00, 0.00, 0.00, 0.00],  # -
            }
            encodings = []
            header = ['SampleName', 'label']
            for p in range(1, len(self.fasta_list[0][1]) + 1):
                for z in ('1', '2', '3', '4', '5'):
                    header.append('Pos' + str(p) + '.ZSCALE' + z)
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for aa in sequence:
                    code = code + zscale[aa]
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_BLOSUM62(self):
        try:
            self.encoding_array = np.array([])

            if not self.is_equal:
                self.error_msg = 'BLOSUM62 descriptor need fasta sequence with equal length.'
                return False

            blosum62 = {
                'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],  # A
                'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],  # R
                'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],  # N
                'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],  # D
                'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],  # C
                'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],  # Q
                'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],  # E
                'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],  # G
                'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],  # H
                'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],  # I
                'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],  # L
                'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],  # K
                'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],  # M
                'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],  # F
                'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],  # P
                'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],  # S
                'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],  # T
                'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],  # W
                'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],  # Y
                'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],  # V
                '-': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -
            }
            encodings = []
            header = ['SampleName', 'label']
            for i in range(1, len(self.fasta_list[0][1]) * 20 + 1):
                header.append('blosum62.F' + str(i))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for aa in sequence:
                    code = code + blosum62[aa]
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_NMBroto(self):
        try:
            self.encoding_array = np.array([])

            props = self.kw['aaindex'].split(';')
            nlag = self.kw['nlag']
            AA = 'ARNDCQEGHILKMFPSTWYV'
            fileAAidx = os.path.split(os.path.realpath(__file__))[
                            0] + r'\data\AAidx.txt' if platform.system() == 'Windows' else \
                os.path.split(os.path.realpath(__file__))[0] + r'/data/AAidx.txt'
            with open(fileAAidx) as f:
                records = f.readlines()[1:]
            myDict = {}
            for i in records:
                array = i.rstrip().split('\t')
                myDict[array[0]] = array[1:]

            AAidx = []
            AAidxName = []
            for i in props:
                if i in myDict:
                    AAidx.append(myDict[i])
                    AAidxName.append(i)
                else:
                    print('"' + i + '" properties not exist.')
                    return None

            AAidx1 = np.array([float(j) for i in AAidx for j in i])
            AAidx = AAidx1.reshape((len(AAidx), 20))
            pstd = np.std(AAidx, axis=1)
            pmean = np.average(AAidx, axis=1)

            for i in range(len(AAidx)):
                for j in range(len(AAidx[i])):
                    AAidx[i][j] = (AAidx[i][j] - pmean[i]) / pstd[i]

            index = {}
            for i in range(len(AA)):
                index[AA[i]] = i

            encodings = []
            header = ['SampleName', 'label']
            for p in props:
                for n in range(1, nlag + 1):
                    header.append(p + '.lag' + str(n))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                N = len(sequence)
                for prop in range(len(props)):
                    for n in range(1, nlag + 1):
                        if len(sequence) > nlag:
                            # if key is '-', then the value is 0
                            rn = sum(
                                [AAidx[prop][index.get(sequence[j], 0)] * AAidx[prop][index.get(sequence[j + n], 0)] for j
                                in range(len(sequence) - n)]) / (N - n)
                        else:
                            rn = 'NA'
                        code.append(rn)
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_Moran(self):
        try:
            self.encoding_array = np.array([])

            props = self.kw['aaindex'].split(';')
            nlag = self.kw['nlag']

            AA = 'ARNDCQEGHILKMFPSTWYV'
            fileAAidx = os.path.split(os.path.realpath(__file__))[
                            0] + r'\data\AAidx.txt' if platform.system() == 'Windows' else \
                os.path.split(os.path.realpath(__file__))[0] + r'/data/AAidx.txt'
            with open(fileAAidx) as f:
                records = f.readlines()[1:]
            myDict = {}
            for i in records:
                array = i.rstrip().split('\t')
                myDict[array[0]] = array[1:]

            AAidx = []
            AAidxName = []
            for i in props:
                if i in myDict:
                    AAidx.append(myDict[i])
                    AAidxName.append(i)
                else:
                    print('"' + i + '" properties not exist.')
                    return None

            AAidx1 = np.array([float(j) for i in AAidx for j in i])
            AAidx = AAidx1.reshape((len(AAidx), 20))

            propMean = np.mean(AAidx, axis=1)
            propStd = np.std(AAidx, axis=1)

            for i in range(len(AAidx)):
                for j in range(len(AAidx[i])):
                    AAidx[i][j] = (AAidx[i][j] - propMean[i]) / propStd[i]

            index = {}
            for i in range(len(AA)):
                index[AA[i]] = i

            encodings = []
            header = ['SampleName', 'label']
            for p in props:
                for n in range(1, nlag + 1):
                    header.append(p + '.lag' + str(n))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                N = len(sequence)
                for prop in range(len(props)):
                    xmean = sum([AAidx[prop][index[aa]] for aa in sequence]) / N
                    for n in range(1, nlag + 1):
                        if len(sequence) > nlag:
                            # if key is '-', then the value is 0
                            fenzi = sum([(AAidx[prop][index.get(sequence[j], 0)] - xmean) * (
                                    AAidx[prop][index.get(sequence[j + n], 0)] - xmean) for j in
                                        range(len(sequence) - n)]) / (N - n)
                            fenmu = sum(
                                [(AAidx[prop][index.get(sequence[j], 0)] - xmean) ** 2 for j in range(len(sequence))]) / N
                            rn = fenzi / fenmu
                        else:
                            rn = 'NA'
                        code.append(rn)
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_Geary(self):
        try:
            self.encoding_array = np.array([])

            props = self.kw['aaindex'].split(';')
            nlag = self.kw['nlag']
            fileAAidx = os.path.split(os.path.realpath(__file__))[
                            0] + r'\data\AAidx.txt' if platform.system() == 'Windows' else \
                os.path.split(os.path.realpath(__file__))[0] + r'/data/AAidx.txt'
            AA = 'ARNDCQEGHILKMFPSTWYV'

            with open(fileAAidx) as f:
                records = f.readlines()[1:]
            myDict = {}
            for i in records:
                array = i.rstrip().split('\t')
                myDict[array[0]] = array[1:]

            AAidx = []
            AAidxName = []
            for i in props:
                if i in myDict:
                    AAidx.append(myDict[i])
                    AAidxName.append(i)
                else:
                    print('"' + i + '" properties not exist.')
                    return None

            AAidx1 = np.array([float(j) for i in AAidx for j in i])
            AAidx = AAidx1.reshape((len(AAidx), 20))

            propMean = np.mean(AAidx, axis=1)
            propStd = np.std(AAidx, axis=1)

            for i in range(len(AAidx)):
                for j in range(len(AAidx[i])):
                    AAidx[i][j] = (AAidx[i][j] - propMean[i]) / propStd[i]

            index = {}
            for i in range(len(AA)):
                index[AA[i]] = i

            encodings = []
            header = ['SampleName', 'label']
            for p in props:
                for n in range(1, nlag + 1):
                    header.append(p + '.lag' + str(n))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                N = len(sequence)
                for prop in range(len(props)):
                    xmean = sum([AAidx[prop][index[aa]] for aa in sequence]) / N
                    for n in range(1, nlag + 1):
                        if len(sequence) > nlag:
                            # if key is '-', then the value is 0
                            rn = (N - 1) / (2 * (N - n)) * ((sum(
                                [(AAidx[prop][index.get(sequence[j], 0)] - AAidx[prop][index.get(sequence[j + n], 0)]) ** 2
                                for
                                j in range(len(sequence) - n)])) / (sum(
                                [(AAidx[prop][index.get(sequence[j], 0)] - xmean) ** 2 for j in range(len(sequence))])))
                        else:
                            rn = 'NA'
                        code.append(rn)
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_AC(self):
        try:
            self.encoding_array = np.array([])

            property_name = self.kw['aaindex'].split(';')
            nlag = self.kw['nlag']

            if self.minimum_length_without_minus < nlag + 1:
                self.error_msg = 'The nlag value is too large.'
                return False

            try:
                data_file = os.path.split(os.path.realpath(__file__))[0] + r'\data\AAindex.data' if platform.system() == 'Windows' else os.path.split(os.path.realpath(__file__))[0] + r'/data/AAindex.data'
                with open(data_file, 'rb') as handle:
                    property_dict = pickle.load(handle)
            except Exception as e:
                self.error_msg = 'Could not find the physicochemical properties file.'
                return False

            for p_name in property_name:
                tmp = np.array(property_dict[p_name], dtype=float)
                pmean = np.average(tmp)
                pstd = np.std(tmp)
                property_dict[p_name] = [(elem - pmean) / pstd for elem in tmp]

            AA = 'ARNDCQEGHILKMFPSTWYV'
            AA_order_dict = {}
            for i in range(len(AA)):
                AA_order_dict[AA[i]] = i

            encodings = []
            header = ['SampleName', 'label']
            for p_name in property_name:
                for i in range(nlag):
                    header.append('%s.lag%s' %(p_name, i+1))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                L = len(sequence)
                for p_name in property_name:
                    xmean = sum([property_dict[p_name][AA_order_dict[aa]] for aa in sequence]) / L
                    for lag in range(1, nlag + 1):
                        ac = 0
                        try:
                            ac = sum([(property_dict[p_name][AA_order_dict[sequence[j]]] - xmean) * (property_dict[p_name][AA_order_dict[sequence[j+lag]]] - xmean) for j in range(L - lag)])/(L-lag)
                        except Exception as e:
                            ac = 0
                        code.append(ac)
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_CC(self):
        try:
            self.encoding_array = np.array([])

            property_name = self.kw['aaindex'].split(';')
            if len(property_name) < 2:
                self.error_msg = 'More than two property should be selected for this descriptor.'
                return False

            nlag = self.kw['nlag']

            if self.minimum_length_without_minus < nlag + 1:
                self.error_msg = 'The nlag value is too large.'
                return False

            try:
                data_file = os.path.split(os.path.realpath(__file__))[
                                0] + r'\data\AAindex.data' if platform.system() == 'Windows' else \
                os.path.split(os.path.realpath(__file__))[0] + r'/data/AAindex.data'
                with open(data_file, 'rb') as handle:
                    property_dict = pickle.load(handle)
            except Exception as e:
                self.error_msg = 'Could not find the physicochemical properties file.'
                return False

            for p_name in property_name:
                tmp = np.array(property_dict[p_name], dtype=float)
                pmean = np.average(tmp)
                pstd = np.std(tmp)
                property_dict[p_name] = [(elem - pmean) / pstd for elem in tmp]

            AA = 'ARNDCQEGHILKMFPSTWYV'
            AA_order_dict = {}
            for i in range(len(AA)):
                AA_order_dict[AA[i]] = i

            property_pairs = self.generatePropertyPairs(property_name)

            encodings = []
            header = ['SampleName', 'label']
            header += [p[0] + '_' + p[1] + '_lag.' + str(lag) for p in property_pairs for lag in range(1, nlag + 1)]
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                
                L = len(sequence)
                for pair in property_pairs:
                    mean_p1 = sum([property_dict[pair[0]][AA_order_dict[aa]] for aa in sequence]) / L
                    mean_p2 = sum([property_dict[pair[1]][AA_order_dict[aa]] for aa in sequence]) / L
                    for lag in range(1, nlag + 1):
                        cc = 0
                        try:
                            cc = sum([(property_dict[pair[0]][AA_order_dict[sequence[j]]] - mean_p1) * (property_dict[pair[1]][AA_order_dict[sequence[j+lag]]] - mean_p2) for j in range(L - lag)]) / (L - lag)
                        except Exception as e:
                            cc = 0
                        code.append(cc)
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_ACC(self):
        try:
            self.encoding_array = np.array([])

            property_name = self.kw['aaindex'].split(';')
            if len(property_name) < 2:
                self.error_msg = 'More than two property should be selected for this descriptor.'
                return False

            nlag = self.kw['nlag']

            if self.minimum_length_without_minus < nlag + 1:
                self.error_msg = 'The nlag value is too large.'
                return False

            try:
                data_file = os.path.split(os.path.realpath(__file__))[
                                0] + r'\data\AAindex.data' if platform.system() == 'Windows' else \
                    os.path.split(os.path.realpath(__file__))[0] + r'/data/AAindex.data'
                with open(data_file, 'rb') as handle:
                    property_dict = pickle.load(handle)
            except Exception as e:
                self.error_msg = 'Could not find the physicochemical properties file.'
                return False

            for p_name in property_name:
                tmp = np.array(property_dict[p_name], dtype=float)
                pmean = np.average(tmp)
                pstd = np.std(tmp)
                property_dict[p_name] = [(elem - pmean) / pstd for elem in tmp]

            AA = 'ARNDCQEGHILKMFPSTWYV'
            AA_order_dict = {}
            for i in range(len(AA)):
                AA_order_dict[AA[i]] = i

            property_pairs = self.generatePropertyPairs(property_name)

            encodings = []
            header = ['SampleName', 'label']
            for p_name in property_name:
                for i in range(nlag):
                    header.append('%s.lag%s' % (p_name, i + 1))
            header += [p[0] + '_' + p[1] + '_lag.' + str(lag) for p in property_pairs for lag in range(1, nlag + 1)]
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]

                L = len(sequence)
                for p_name in property_name:
                    xmean = sum([property_dict[p_name][AA_order_dict[aa]] for aa in sequence]) / L
                    for lag in range(1, nlag + 1):
                        ac = 0
                        try:
                            ac = sum([(property_dict[p_name][AA_order_dict[sequence[j]]] - xmean) * (property_dict[p_name][AA_order_dict[sequence[j+lag]]] - xmean) for j in range(L - lag)])/(L-lag)
                        except Exception as e:
                            ac = 0
                        code.append(ac)
                for pair in property_pairs:
                    mean_p1 = sum([property_dict[pair[0]][AA_order_dict[aa]] for aa in sequence]) / L
                    mean_p2 = sum([property_dict[pair[1]][AA_order_dict[aa]] for aa in sequence]) / L
                    for lag in range(1, nlag + 1):
                        cc = 0
                        try:
                            cc = sum([(property_dict[pair[0]][AA_order_dict[sequence[j]]] - mean_p1) * (
                                        property_dict[pair[1]][AA_order_dict[sequence[j + lag]]] - mean_p2) for j in
                                    range(L - lag)]) / (L - lag)
                        except Exception as e:
                            cc = 0
                        code.append(cc)
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Count(self, seq1, seq2):
        sum = 0
        for aa in seq1:
            sum = sum + seq2.count(aa)
        return sum

    def Protein_CTDC(self):
        try:
            group1 = {
                'hydrophobicity_PRAM900101': 'RKEDQN',
                'hydrophobicity_ARGP820101': 'QSTNGDE',
                'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
                'hydrophobicity_PONP930101': 'KPDESNQT',
                'hydrophobicity_CASG920101': 'KDEQPSRNTG',
                'hydrophobicity_ENGD860101': 'RDKENQHYP',
                'hydrophobicity_FASG890101': 'KERSQD',
                'normwaalsvolume': 'GASTPDC',
                'polarity': 'LIFWCMVY',
                'polarizability': 'GASDT',
                'charge': 'KR',
                'secondarystruct': 'EALMQKRH',
                'solventaccess': 'ALFCGIVW'
            }
            group2 = {
                'hydrophobicity_PRAM900101': 'GASTPHY',
                'hydrophobicity_ARGP820101': 'RAHCKMV',
                'hydrophobicity_ZIMJ680101': 'HMCKV',
                'hydrophobicity_PONP930101': 'GRHA',
                'hydrophobicity_CASG920101': 'AHYMLV',
                'hydrophobicity_ENGD860101': 'SGTAW',
                'hydrophobicity_FASG890101': 'NTPG',
                'normwaalsvolume': 'NVEQIL',
                'polarity': 'PATGS',
                'polarizability': 'CPNVEQIL',
                'charge': 'ANCQGHILMFPSTWYV',
                'secondarystruct': 'VIYCWFT',
                'solventaccess': 'RKQEND'
            }
            group3 = {
                'hydrophobicity_PRAM900101': 'CLVIMFW',
                'hydrophobicity_ARGP820101': 'LYPFIW',
                'hydrophobicity_ZIMJ680101': 'LPFYI',
                'hydrophobicity_PONP930101': 'YMFWLCVI',
                'hydrophobicity_CASG920101': 'FIWC',
                'hydrophobicity_ENGD860101': 'CVLIMF',
                'hydrophobicity_FASG890101': 'AYHWVMFLIC',
                'normwaalsvolume': 'MHKFRYW',
                'polarity': 'HQRKNED',
                'polarizability': 'KMHFRYW',
                'charge': 'DE',
                'secondarystruct': 'GNPSD',
                'solventaccess': 'MSPTHY'
            }

            groups = [group1, group2, group3]
            property = (
                'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
                'hydrophobicity_PONP930101',
                'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
                'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

            encodings = []
            header = ['SampleName', 'label']
            for p in property:
                for g in range(1, len(groups) + 1):
                    header.append(p + '.G' + str(g))
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                for p in property:
                    c1 = self.Count(group1[p], sequence) / len(sequence)
                    c2 = self.Count(group2[p], sequence) / len(sequence)
                    c3 = 1 - c1 - c2
                    code = code + [c1, c2, c3]
                encodings.append(code)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_CTDT(self):
        try:
            group1 = {
                'hydrophobicity_PRAM900101': 'RKEDQN',
                'hydrophobicity_ARGP820101': 'QSTNGDE',
                'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
                'hydrophobicity_PONP930101': 'KPDESNQT',
                'hydrophobicity_CASG920101': 'KDEQPSRNTG',
                'hydrophobicity_ENGD860101': 'RDKENQHYP',
                'hydrophobicity_FASG890101': 'KERSQD',
                'normwaalsvolume': 'GASTPDC',
                'polarity': 'LIFWCMVY',
                'polarizability': 'GASDT',
                'charge': 'KR',
                'secondarystruct': 'EALMQKRH',
                'solventaccess': 'ALFCGIVW'
            }
            group2 = {
                'hydrophobicity_PRAM900101': 'GASTPHY',
                'hydrophobicity_ARGP820101': 'RAHCKMV',
                'hydrophobicity_ZIMJ680101': 'HMCKV',
                'hydrophobicity_PONP930101': 'GRHA',
                'hydrophobicity_CASG920101': 'AHYMLV',
                'hydrophobicity_ENGD860101': 'SGTAW',
                'hydrophobicity_FASG890101': 'NTPG',
                'normwaalsvolume': 'NVEQIL',
                'polarity': 'PATGS',
                'polarizability': 'CPNVEQIL',
                'charge': 'ANCQGHILMFPSTWYV',
                'secondarystruct': 'VIYCWFT',
                'solventaccess': 'RKQEND'
            }
            group3 = {
                'hydrophobicity_PRAM900101': 'CLVIMFW',
                'hydrophobicity_ARGP820101': 'LYPFIW',
                'hydrophobicity_ZIMJ680101': 'LPFYI',
                'hydrophobicity_PONP930101': 'YMFWLCVI',
                'hydrophobicity_CASG920101': 'FIWC',
                'hydrophobicity_ENGD860101': 'CVLIMF',
                'hydrophobicity_FASG890101': 'AYHWVMFLIC',
                'normwaalsvolume': 'MHKFRYW',
                'polarity': 'HQRKNED',
                'polarizability': 'KMHFRYW',
                'charge': 'DE',
                'secondarystruct': 'GNPSD',
                'solventaccess': 'MSPTHY'
            }

            groups = [group1, group2, group3]
            property = (
                'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
                'hydrophobicity_PONP930101',
                'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
                'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

            encodings = []
            header = ['SampleName', 'label']
            for p in property:
                for tr in ('Tr1221', 'Tr1331', 'Tr2332'):
                    header.append(p + '.' + tr)
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                aaPair = [sequence[j:j + 2] for j in range(len(sequence) - 1)]
                for p in property:
                    c1221, c1331, c2332 = 0, 0, 0
                    for pair in aaPair:
                        if (pair[0] in group1[p] and pair[1] in group2[p]) or (
                                pair[0] in group2[p] and pair[1] in group1[p]):
                            c1221 = c1221 + 1
                            continue
                        if (pair[0] in group1[p] and pair[1] in group3[p]) or (
                                pair[0] in group3[p] and pair[1] in group1[p]):
                            c1331 = c1331 + 1
                            continue
                        if (pair[0] in group2[p] and pair[1] in group3[p]) or (
                                pair[0] in group3[p] and pair[1] in group2[p]):
                            c2332 = c2332 + 1
                    code = code + [c1221 / len(aaPair), c1331 / len(aaPair), c2332 / len(aaPair)]
                encodings.append(code)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Count1(self, aaSet, sequence):
        number = 0
        for aa in sequence:
            if aa in aaSet:
                number = number + 1
        cutoffNums = [1, math.floor(0.25 * number), math.floor(0.50 * number), math.floor(0.75 * number), number]
        cutoffNums = [i if i >= 1 else 1 for i in cutoffNums]

        code = []
        for cutoff in cutoffNums:
            myCount = 0
            for i in range(len(sequence)):
                if sequence[i] in aaSet:
                    myCount += 1
                    if myCount == cutoff:
                        code.append((i + 1) / len(sequence) * 100)
                        break
            if myCount == 0:
                code.append(0)
        return code

    def Protein_CTDD(self):
        try:
            group1 = {
                'hydrophobicity_PRAM900101': 'RKEDQN',
                'hydrophobicity_ARGP820101': 'QSTNGDE',
                'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
                'hydrophobicity_PONP930101': 'KPDESNQT',
                'hydrophobicity_CASG920101': 'KDEQPSRNTG',
                'hydrophobicity_ENGD860101': 'RDKENQHYP',
                'hydrophobicity_FASG890101': 'KERSQD',
                'normwaalsvolume': 'GASTPDC',
                'polarity': 'LIFWCMVY',
                'polarizability': 'GASDT',
                'charge': 'KR',
                'secondarystruct': 'EALMQKRH',
                'solventaccess': 'ALFCGIVW'
            }
            group2 = {
                'hydrophobicity_PRAM900101': 'GASTPHY',
                'hydrophobicity_ARGP820101': 'RAHCKMV',
                'hydrophobicity_ZIMJ680101': 'HMCKV',
                'hydrophobicity_PONP930101': 'GRHA',
                'hydrophobicity_CASG920101': 'AHYMLV',
                'hydrophobicity_ENGD860101': 'SGTAW',
                'hydrophobicity_FASG890101': 'NTPG',
                'normwaalsvolume': 'NVEQIL',
                'polarity': 'PATGS',
                'polarizability': 'CPNVEQIL',
                'charge': 'ANCQGHILMFPSTWYV',
                'secondarystruct': 'VIYCWFT',
                'solventaccess': 'RKQEND'
            }
            group3 = {
                'hydrophobicity_PRAM900101': 'CLVIMFW',
                'hydrophobicity_ARGP820101': 'LYPFIW',
                'hydrophobicity_ZIMJ680101': 'LPFYI',
                'hydrophobicity_PONP930101': 'YMFWLCVI',
                'hydrophobicity_CASG920101': 'FIWC',
                'hydrophobicity_ENGD860101': 'CVLIMF',
                'hydrophobicity_FASG890101': 'AYHWVMFLIC',
                'normwaalsvolume': 'MHKFRYW',
                'polarity': 'HQRKNED',
                'polarizability': 'KMHFRYW',
                'charge': 'DE',
                'secondarystruct': 'GNPSD',
                'solventaccess': 'MSPTHY'
            }

            groups = [group1, group2, group3]
            property = (
                'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
                'hydrophobicity_PONP930101',
                'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
                'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

            encodings = []
            header = ['SampleName', 'label']
            for p in property:
                for g in ('1', '2', '3'):
                    for d in ['0', '25', '50', '75', '100']:
                        header.append(p + '.' + g + '.residue' + d)
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                for p in property:
                    code = code + self.Count1(group1[p], sequence) + self.Count1(group2[p], sequence) + self.Count1(
                        group3[p], sequence)
                encodings.append(code)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def CalculateKSCTriad(self, sequence, gap, features, AADict):
        res = []
        for g in range(gap + 1):
            myDict = {}
            for f in features:
                myDict[f] = 0

            for i in range(len(sequence)):
                if i + g + 1 < len(sequence) and i + 2 * g + 2 < len(sequence):
                    fea = AADict[sequence[i]] + '.' + AADict[sequence[i + g + 1]] + '.' + AADict[
                        sequence[i + 2 * g + 2]]
                    myDict[fea] = myDict[fea] + 1

            maxValue, minValue = max(myDict.values()), min(myDict.values())
            for f in features:
                res.append((myDict[f] - minValue) / maxValue)
        return res

    def Protein_CTriad(self):
        try:
            if self.minimum_length_without_minus < 3:
                self.error_msg = 'CTriad descriptor need fasta sequence with minimum length > 3.'
                return False

            AAGroup = {
                'g1': 'AGV',
                'g2': 'ILFP',
                'g3': 'YMTS',
                'g4': 'HNQW',
                'g5': 'RK',
                'g6': 'DE',
                'g7': 'C'
            }

            myGroups = sorted(AAGroup.keys())

            AADict = {}
            for g in myGroups:
                for aa in AAGroup[g]:
                    AADict[aa] = g

            features = [f1 + '.' + f2 + '.' + f3 for f1 in myGroups for f2 in myGroups for f3 in myGroups]

            encodings = []
            header = ['SampleName', 'label']
            for f in features:
                header.append(f)
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                code = code + self.CalculateKSCTriad(sequence, 0, features, AADict)
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_KSCTriad(self):
        try:
            gap = self.kw['kspace']
            if self.minimum_length_without_minus < 2 * gap + 3:
                self.error_msg = 'KSCTriad" encoding, the input fasta sequences should be greater than (2*gap+3).'
                return False

            AAGroup = {
                'g1': 'AGV',
                'g2': 'ILFP',
                'g3': 'YMTS',
                'g4': 'HNQW',
                'g5': 'RK',
                'g6': 'DE',
                'g7': 'C'
            }

            myGroups = sorted(AAGroup.keys())

            AADict = {}
            for g in myGroups:
                for aa in AAGroup[g]:
                    AADict[aa] = g

            features = [f1 + '.' + f2 + '.' + f3 for f1 in myGroups for f2 in myGroups for f3 in myGroups]

            encodings = []
            header = ['SampleName', 'label']
            for g in range(gap + 1):
                for f in features:
                    header.append(f + '.gap' + str(g))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                if len(sequence) < 2 * gap + 3:
                    print(
                        'Error: for "KSCTriad" encoding, the input fasta sequences should be greater than (2*gap+3). \n\n')
                    return 0
                code = code + self.CalculateKSCTriad(sequence, gap, features, AADict)
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_SOCNumber(self):
        try:
            nlag = self.kw['nlag']
            dataFile = os.path.split(os.path.realpath(__file__))[
                        0] + r'\data\Schneider-Wrede.txt' if platform.system() == 'Windows' else \
                os.path.split(os.path.realpath(__file__))[0] + r'/data/Schneider-Wrede.txt'
            dataFile1 = os.path.split(os.path.realpath(__file__))[
                            0] + r'\data\Grantham.txt' if platform.system() == 'Windows' else \
                os.path.split(os.path.realpath(__file__))[0] + r'/data/Grantham.txt'
            AA = 'ACDEFGHIKLMNPQRSTVWY'
            AA1 = 'ARNDCQEGHILKMFPSTWYV'

            DictAA = {}
            for i in range(len(AA)):
                DictAA[AA[i]] = i

            DictAA1 = {}
            for i in range(len(AA1)):
                DictAA1[AA1[i]] = i

            with open(dataFile) as f:
                records = f.readlines()[1:]
            AADistance = []
            for i in records:
                array = i.rstrip().split()[1:] if i.rstrip() != '' else None
                AADistance.append(array)
            AADistance = np.array(
                [float(AADistance[i][j]) for i in range(len(AADistance)) for j in range(len(AADistance[i]))]).reshape(
                (20, 20))

            with open(dataFile1) as f:
                records = f.readlines()[1:]
            AADistance1 = []
            for i in records:
                array = i.rstrip().split()[1:] if i.rstrip() != '' else None
                AADistance1.append(array)
            AADistance1 = np.array(
                [float(AADistance1[i][j]) for i in range(len(AADistance1)) for j in range(len(AADistance1[i]))]).reshape(
                (20, 20))

            encodings = []
            header = ['SampleName', 'label']
            for n in range(1, nlag + 1):
                header.append('Schneider.lag' + str(n))
            for n in range(1, nlag + 1):
                header.append('gGrantham.lag' + str(n))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                for n in range(1, nlag + 1):
                    code.append(sum([AADistance[DictAA[sequence[j]]][DictAA[sequence[j + n]]] ** 2 for j in
                                    range(len(sequence) - n)]) / (len(sequence) - n))

                for n in range(1, nlag + 1):
                    code.append(sum([AADistance1[DictAA1[sequence[j]]][DictAA1[sequence[j + n]]] ** 2 for j in
                                    range(len(sequence) - n)]) / (len(sequence) - n))
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_QSOrder(self):
        try:
            nlag = self.kw['nlag']
            w = self.kw['weight']

            if nlag > self.minimum_length_without_minus - 1:
                self.error_msg = 'The lag value is out of range.'
                return False

            dataFile = os.path.split(os.path.realpath(__file__))[
                        0] + r'\data\Schneider-Wrede.txt' if platform.system() == 'Windows' else \
                os.path.split(os.path.realpath(__file__))[0] + r'/data/Schneider-Wrede.txt'
            dataFile1 = os.path.split(os.path.realpath(__file__))[
                            0] + r'\data\Grantham.txt' if platform.system() == 'Windows' else \
                os.path.split(os.path.realpath(__file__))[0] + r'/data/Grantham.txt'
            AA = 'ACDEFGHIKLMNPQRSTVWY'
            AA1 = 'ARNDCQEGHILKMFPSTWYV'

            DictAA = {}
            for i in range(len(AA)):
                DictAA[AA[i]] = i

            DictAA1 = {}
            for i in range(len(AA1)):
                DictAA1[AA1[i]] = i

            with open(dataFile) as f:
                records = f.readlines()[1:]
            AADistance = []
            for i in records:
                array = i.rstrip().split()[1:] if i.rstrip() != '' else None
                AADistance.append(array)
            AADistance = np.array(
                [float(AADistance[i][j]) for i in range(len(AADistance)) for j in range(len(AADistance[i]))]).reshape(
                (20, 20))

            with open(dataFile1) as f:
                records = f.readlines()[1:]
            AADistance1 = []
            for i in records:
                array = i.rstrip().split()[1:] if i.rstrip() != '' else None
                AADistance1.append(array)
            AADistance1 = np.array(
                [float(AADistance1[i][j]) for i in range(len(AADistance1)) for j in range(len(AADistance1[i]))]).reshape(
                (20, 20))

            encodings = []
            header = ['SampleName', 'label']
            for aa in AA1:
                header.append('Schneider.Xr.' + aa)
            for aa in AA1:
                header.append('Grantham.Xr.' + aa)
            for n in range(1, nlag + 1):
                header.append('Schneider.Xd.' + str(n))
            for n in range(1, nlag + 1):
                header.append('Grantham.Xd.' + str(n))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                arraySW = []
                arrayGM = []
                for n in range(1, nlag + 1):
                    arraySW.append(
                        sum([AADistance[DictAA[sequence[j]]][DictAA[sequence[j + n]]] ** 2 for j in
                            range(len(sequence) - n)]))
                    arrayGM.append(sum(
                        [AADistance1[DictAA1[sequence[j]]][DictAA1[sequence[j + n]]] ** 2 for j in
                        range(len(sequence) - n)]))
                myDict = {}
                for aa in AA1:
                    myDict[aa] = sequence.count(aa)
                for aa in AA1:
                    code.append(myDict[aa] / (1 + w * sum(arraySW)))
                for aa in AA1:
                    code.append(myDict[aa] / (1 + w * sum(arrayGM)))
                for num in arraySW:
                    code.append((w * num) / (1 + w * sum(arraySW)))
                for num in arrayGM:
                    code.append((w * num) / (1 + w * sum(arrayGM)))
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Rvalue(self, aa1, aa2, AADict, Matrix):
        return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)

    def Protein_PAAC(self):
        try:
            lambdaValue = self.kw['lambdaValue']
            if lambdaValue > self.minimum_length_without_minus - 1:
                self.error_msg = 'The lambda value is out of range.'
                return False

            w = self.kw['weight']
            dataFile = os.path.split(os.path.realpath(__file__))[
                        0] + r'\data\PAAC.txt' if platform.system() == 'Windows' else \
                os.path.split(os.path.realpath(__file__))[0] + r'/data/PAAC.txt'
            with open(dataFile) as f:
                records = f.readlines()
            AA = ''.join(records[0].rstrip().split()[1:])
            AADict = {}
            for i in range(len(AA)):
                AADict[AA[i]] = i
            AAProperty = []
            AAPropertyNames = []
            for i in range(1, len(records)):
                array = records[i].rstrip().split() if records[i].rstrip() != '' else None
                AAProperty.append([float(j) for j in array[1:]])
                AAPropertyNames.append(array[0])

            AAProperty1 = []
            for i in AAProperty:
                meanI = sum(i) / 20
                fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
                AAProperty1.append([(j - meanI) / fenmu for j in i])

            encodings = []
            header = ['SampleName', 'label']
            for aa in AA:
                header.append('Xc1.' + aa)
            for n in range(1, lambdaValue + 1):
                header.append('Xc2.lambda' + str(n))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                theta = []
                for n in range(1, lambdaValue + 1):
                    theta.append(
                        sum([self.Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1) for j in
                            range(len(sequence) - n)]) / (
                                len(sequence) - n))
                myDict = {}
                for aa in AA:
                    myDict[aa] = sequence.count(aa)
                code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
                code = code + [(w * j) / (1 + w * sum(theta)) for j in theta]
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_APAAC(self):
        try:
            lambdaValue = self.kw['lambdaValue']
            if lambdaValue > self.minimum_length_without_minus - 1:
                self.error_msg = 'The lambda value is out of range.'
                return False
            w = self.kw['weight']
            dataFile = os.path.split(os.path.realpath(__file__))[
                        0] + r'\data\PAAC.txt' if platform.system() == 'Windows' else \
                os.path.split(os.path.realpath(__file__))[0] + r'/data/PAAC.txt'
            with open(dataFile) as f:
                records = f.readlines()
            AA = ''.join(records[0].rstrip().split()[1:])
            AADict = {}
            for i in range(len(AA)):
                AADict[AA[i]] = i
            AAProperty = []
            AAPropertyNames = []
            for i in range(1, len(records) - 1):
                array = records[i].rstrip().split() if records[i].rstrip() != '' else None
                AAProperty.append([float(j) for j in array[1:]])
                AAPropertyNames.append(array[0])

            AAProperty1 = []
            for i in AAProperty:
                meanI = sum(i) / 20
                fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
                AAProperty1.append([(j - meanI) / fenmu for j in i])

            encodings = []
            header = ['SampleName', 'label']
            for i in AA:
                header.append('Pc1.' + i)
            for j in range(1, lambdaValue + 1):
                for i in AAPropertyNames:
                    header.append('Pc2.' + i + '.' + str(j))
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                theta = []
                for n in range(1, lambdaValue + 1):
                    for j in range(len(AAProperty1)):
                        theta.append(
                            sum([AAProperty1[j][AADict[sequence[k]]] * AAProperty1[j][AADict[sequence[k + n]]] for k in
                                range(len(sequence) - n)]) / (len(sequence) - n))
                myDict = {}
                for aa in AA:
                    myDict[aa] = sequence.count(aa)

                code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
                code = code + [w * value / (1 + w * sum(theta)) for value in theta]
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_OPF_10bit(self):
        try:
            # clear
            self.encoding_array = np.array([])

            if not self.is_equal:
                self.error_msg = 'OPF descriptor need fasta sequence with equal length.'
                return False

            physicochemical_properties_list = [
                'FYWH',
                'DE',
                'KHR',
                'NQSDECTKRHYW',
                'AGCTIVLKHFYWM',
                'IVL',
                'ASGC',
                'KHRDE',
                'PNDTCAGSV',
                'P',
            ]

            physicochemical_properties_index = ['Aromatic', 'Negative', 'Positive', 'Polar', 'Hydrophobic', 'Aliphatic',
                                                'Tiny', 'Charged', 'Small', 'Proline']

            header = ['SampleName', 'label']
            encodings = []
            header += ['Pos%s_%s' % (i + 1, j) for i in range(len(self.fasta_list[0][1])) for j in
                    physicochemical_properties_index]
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for aa in i[1]:
                    for j in physicochemical_properties_list:
                        if aa in j:
                            code.append(1)
                        else:
                            code.append(0)
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_OPF_7bit_type_1(self):
        try:
            # clear
            self.encoding_array = np.array([])

            if not self.is_equal:
                self.error_msg = 'OPF descriptor need fasta sequence with equal length.'
                return False

            physicochemical_properties_list = [
                'ACFGHILMNPQSTVWY',
                'CFILMVW',
                'ACDGPST',
                'CFILMVWY',
                'ADGST',
                'DGNPS',
                'ACFGILVW',
            ]

            physicochemical_properties_index = ['Charge', 'Hydrophobicity', 'Normalized vander Waals volume', 'Polarity',
                                                'Polariizability', 'Secondary Structure', 'Solvent Accessibility']

            header = ['SampleName', 'label']
            encodings = []
            header += ['Pos%s_%s' % (i + 1, j) for i in range(len(self.fasta_list[0][1])) for j in
                    physicochemical_properties_index]
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for aa in i[1]:
                    for j in physicochemical_properties_list:
                        if aa in j:
                            code.append(1)
                        else:
                            code.append(0)
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_OPF_7bit_type_2(self):
        try:
            # clear
            self.encoding_array = np.array([])

            if not self.is_equal:
                self.error_msg = 'OPF descriptor need fasta sequence with equal length.'
                return False

            physicochemical_properties_list = [
                'DE',
                'AGHPSTY',
                'EILNQV',
                'AGPST',
                'CEILNPQV',
                'AEHKLMQR',
                'HMPSTY',
            ]

            physicochemical_properties_index = ['Charge', 'Hydrophobicity', 'Normalized vander Waals volume', 'Polarity',
                                                'Polariizability', 'Secondary Structure', 'Solvent Accessibility']

            header = ['SampleName', 'label']
            encodings = []
            header += ['Pos%s_%s' % (i + 1, j) for i in range(len(self.fasta_list[0][1])) for j in
                    physicochemical_properties_index]
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for aa in i[1]:
                    for j in physicochemical_properties_list:
                        if aa in j:
                            code.append(1)
                        else:
                            code.append(0)
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_OPF_7bit_type_3(self):
        try:
            # clear
            self.encoding_array = np.array([])

            if not self.is_equal:
                self.error_msg = 'OPF descriptor need fasta sequence with equal length.'
                return False

            physicochemical_properties_list = [
                'KR',
                'DEKNQR',
                'FHKMRWY',
                'DEHKNQR',
                'FHKMRWY',
                'CFITVWY',
                'DEKNRQ',
            ]

            physicochemical_properties_index = ['Charge', 'Hydrophobicity', 'Normalized vander Waals volume', 'Polarity',
                                                'Polariizability', 'Secondary Structure', 'Solvent Accessibility']

            header = ['SampleName', 'label']
            encodings = []
            header += ['Pos%s_%s' % (i + 1, j) for i in range(len(self.fasta_list[0][1])) for j in
                    physicochemical_properties_index]
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for aa in i[1]:
                    for j in physicochemical_properties_list:
                        if aa in j:
                            code.append(1)
                        else:
                            code.append(0)
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_ASDC(self):
        try:
            # clear
            self.encoding_array = np.array([])

            AA = 'ACDEFGHIKLMNPQRSTVWY'
            encodings = []
            aaPairs = []
            for aa1 in AA:
                for aa2 in AA:
                    aaPairs.append(aa1 + aa2)

            header = ['SampleName', 'label']
            header += [aa1 + aa2 for aa1 in AA for aa2 in AA]
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                sum = 0
                pair_dict = {}
                for pair in aaPairs:
                    pair_dict[pair] = 0
                for j in range(len(sequence)):
                    for k in range(j + 1, len(sequence)):
                        if sequence[j] in AA and sequence[k] in AA:
                            pair_dict[sequence[j] + sequence[k]] += 1
                            sum += 1
                for pair in aaPairs:
                    code.append(pair_dict[pair] / sum)
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    ''' KNN descriptor '''
    def Sim(self, a, b):
        blosum62 = [
            [ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, 0],  # A
            [-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, 0],  # R
            [-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3, 0],  # N
            [-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3, 0],  # D
            [ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, 0],  # C
            [-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2, 0],  # Q
            [-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2, 0],  # E
            [ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, 0],  # G
            [-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3, 0],  # H
            [-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, 0],  # I
            [-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, 0],  # L
            [-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2, 0],  # K
            [-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, 0],  # M
            [-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, 0],  # F
            [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, 0],  # P
            [ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2, 0],  # S
            [ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, 0],  # T
            [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, 0],  # W
            [-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, 0],  # Y
            [ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, 0],  # V
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0],  # -
        ]
        AA = 'ARNDCQEGHILKMFPSTWYV-'
        myDict = {}
        for i in range(len(AA)):
            myDict[AA[i]] = i
        maxValue, minValue = 11, -4
        return (blosum62[myDict[a]][myDict[b]] - minValue) / (maxValue - minValue)

    def CalculateDistance(self, sequence1, sequence2):
        if len(sequence1) != len(sequence2):
            self.error_msg = 'KNN descriptor need fasta sequence with equal length.'
            return 1
        distance = 1 - sum([self.Sim(sequence1[i], sequence2[i]) for i in range(len(sequence1))]) / len(sequence1)
        return distance

    def CalculateContent(self, myDistance, j, myLabelSets):
        content = []
        myDict = {}
        for i in myLabelSets:
            myDict[i] = 0
        for i in range(j):
            myDict[myDistance[i][0]] = myDict[myDistance[i][0]] + 1
        for i in myLabelSets:
            content.append(myDict[myLabelSets[i]] / j)
        return content

    def Protein_KNN(self):
        try:
            # clear
            self.encoding_array = np.array([])

            if not self.is_equal:
                self.error_msg = 'KNN descriptor need fasta sequence with equal length.'
                return False

            topK_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16,
                        0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30]

            training_data = []
            training_label = {}
            for i in self.fasta_list:
                if i[3] == 'training':
                    training_data.append(i)
                    training_label[i[0]] = int(i[2])
            tmp_label_sets = list(set(training_label.values()))

            topK_numbers = []
            for i in topK_values:
                topK_numbers.append(math.ceil(len(training_data) * i))

            # calculate pair distance
            distance_dict = {}
            for i in range(len(self.fasta_list)):
                name_seq1, sequence_1, label_1, usage_1 = self.fasta_list[i][0], self.fasta_list[i][1], self.fasta_list[i][2], self.fasta_list[i][3]
                for j in range(i+1, len(self.fasta_list)):
                    name_seq2, sequence_2, label_2, usage_2 = self.fasta_list[j][0], self.fasta_list[j][1], self.fasta_list[j][2], self.fasta_list[j][3]
                    if usage_1 == 'testing' and usage_2 == 'testing':
                        continue
                    else:
                        distance_dict[':'.join(sorted([name_seq1, name_seq2]))] = self.CalculateDistance(sequence_1, sequence_2)

            encodings = []
            header = ['#', 'label']
            for k in topK_numbers:
                for l in tmp_label_sets:
                    header.append('Top' + str(k) + '.label' + str(l))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                tmp_distance_list = []
                for j in range(len(training_data)):
                    if name != training_data[j][0]:
                        tmp_distance_list.append([int(training_data[j][2]), distance_dict.get(':'.join(sorted([name, training_data[j][0]])), 1)])

                tmp_distance_list = np.array(tmp_distance_list)
                tmp_distance_list = tmp_distance_list[np.lexsort(tmp_distance_list.T)]

                for j in topK_numbers:
                    code += self.CalculateContent(tmp_distance_list, j, tmp_label_sets)
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False
    ''' end Protein KNN descriptor '''

    def gapModel(self, fastas, myDict, gDict, gNames, ktuple, glValue):
        encodings = []
        header = ['SampleName', 'label']
        if ktuple == 1:
            header = header + [g + '_gap' + str(glValue) for g in gNames]
            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                numDict = {}
                for j in range(0, len(sequence), glValue + 1):
                    numDict[gDict[myDict[sequence[j]]]] = numDict.get(gDict[myDict[sequence[j]]], 0) + 1

                for g in gNames:
                    code.append(numDict.get(g, 0))
                encodings.append(code)
        if ktuple == 2:
            header = header + [g1 + '_' + g2 + '_gap' + str(glValue) for g1 in gNames for g2 in gNames]
            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                numDict = {}
                for j in range(0, len(sequence), glValue + 1):
                    if j + 1 < len(sequence):
                        numDict[gDict[myDict[sequence[j]]] + '_' + gDict[myDict[sequence[j + 1]]]] = numDict.get(
                            gDict[myDict[sequence[j]]] + '_' + gDict[myDict[sequence[j + 1]]], 0) + 1

                for g in [g1 + '_' + g2 for g1 in gNames for g2 in gNames]:
                    code.append(numDict.get(g, 0))
                encodings.append(code)
        if ktuple == 3:
            header = header + [g1 + '_' + g2 + '_' + g3 + '_gap' + str(glValue) for g1 in gNames for g2 in gNames for g3
                               in gNames]
            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                numDict = {}
                for j in range(0, len(sequence), glValue + 1):
                    if j + 1 < len(sequence) and j + 2 < len(sequence):
                        numDict[gDict[myDict[sequence[j]]] + '_' + gDict[myDict[sequence[j + 1]]] + '_' + gDict[
                            myDict[sequence[j + 2]]]] = numDict.get(
                            gDict[myDict[sequence[j]]] + '_' + gDict[myDict[sequence[j + 1]]] + '_' + gDict[
                                myDict[sequence[j + 2]]], 0) + 1

                for g in [g1 + '_' + g2 + '_' + g3 for g1 in gNames for g2 in gNames for g3 in gNames]:
                    code.append(numDict.get(g, 0))
                encodings.append(code)
        return encodings

    def lambdaModel(self, fastas, myDict, gDict, gNames, ktuple, glValue):
        encodings = []
        header = ['SampleName', 'label']
        if ktuple == 1:
            header = header + [g + '_LC' + str(glValue) for g in gNames]
            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                numDict = {}
                for j in range(0, len(sequence)):
                    numDict[gDict[myDict[sequence[j]]]] = numDict.get(gDict[myDict[sequence[j]]], 0) + 1

                for g in gNames:
                    code.append(numDict.get(g, 0))
                encodings.append(code)
        if ktuple == 2:
            header = header + [g1 + '_' + g2 + '_LC' + str(glValue) for g1 in gNames for g2 in gNames]
            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                numDict = {}
                for j in range(0, len(sequence)):
                    if j + glValue < len(sequence):
                        numDict[gDict[myDict[sequence[j]]] + '_' + gDict[myDict[sequence[j + glValue]]]] = numDict.get(
                            gDict[myDict[sequence[j]]] + '_' + gDict[myDict[sequence[j + glValue]]], 0) + 1

                for g in [g1 + '_' + g2 for g1 in gNames for g2 in gNames]:
                    code.append(numDict.get(g, 0))
                encodings.append(code)
        if ktuple == 3:
            header = header + [g1 + '_' + g2 + '_' + g3 + '_LC' + str(glValue) for g1 in gNames for g2 in gNames for g3
                               in
                               gNames]
            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                numDict = {}
                for j in range(0, len(sequence)):
                    if j + glValue < len(sequence) and j + 2 * glValue < len(sequence):
                        numDict[gDict[myDict[sequence[j]]] + '_' + gDict[myDict[sequence[j + glValue]]] + '_' + gDict[
                            myDict[sequence[j + 2 * glValue]]]] = numDict.get(
                            gDict[myDict[sequence[j]]] + '_' + gDict[myDict[sequence[j + glValue]]] + '_' + gDict[
                                myDict[sequence[j + 2 * glValue]]], 0) + 1

                for g in [g1 + '_' + g2 + '_' + g3 for g1 in gNames for g2 in gNames for g3 in gNames]:
                    code.append(numDict.get(g, 0))
                encodings.append(code)
        return encodings

    def Protein_PseKRAAC_type_1(self):
        try:
            AAGroup = {
                2: ['CMFILVWY', 'AGTSNQDEHRKP'],
                3: ['CMFILVWY', 'AGTSP', 'NQDEHRK'],
                4: ['CMFWY', 'ILV', 'AGTS', 'NQDEHRKP'],
                5: ['WFYH', 'MILV', 'CATSP', 'G', 'NQDERK'],
                6: ['WFYH', 'MILV', 'CATS', 'P', 'G', 'NQDERK'],
                7: ['WFYH', 'MILV', 'CATS', 'P', 'G', 'NQDE', 'RK'],
                8: ['WFYH', 'MILV', 'CA', 'NTS', 'P', 'G', 'DE', 'QRK'],
                9: ['WFYH', 'MI', 'LV', 'CA', 'NTS', 'P', 'G', 'DE', 'QRK'],
                10: ['WFY', 'ML', 'IV', 'CA', 'TS', 'NH', 'P', 'G', 'DE', 'QRK'],
                11: ['WFY', 'ML', 'IV', 'CA', 'TS', 'NH', 'P', 'G', 'D', 'QE', 'RK'],
                12: ['WFY', 'ML', 'IV', 'C', 'A', 'TS', 'NH', 'P', 'G', 'D', 'QE', 'RK'],
                13: ['WFY', 'ML', 'IV', 'C', 'A', 'T', 'S', 'NH', 'P', 'G', 'D', 'QE', 'RK'],
                14: ['WFY', 'ML', 'IV', 'C', 'A', 'T', 'S', 'NH', 'P', 'G', 'D', 'QE', 'R', 'K'],
                15: ['WFY', 'ML', 'IV', 'C', 'A', 'T', 'S', 'N', 'H', 'P', 'G', 'D', 'QE', 'R', 'K'],
                16: ['W', 'FY', 'ML', 'IV', 'C', 'A', 'T', 'S', 'N', 'H', 'P', 'G', 'D', 'QE', 'R', 'K'],
                17: ['W', 'FY', 'ML', 'IV', 'C', 'A', 'T', 'S', 'N', 'H', 'P', 'G', 'D', 'Q', 'E', 'R', 'K'],
                18: ['W', 'FY', 'M', 'L', 'IV', 'C', 'A', 'T', 'S', 'N', 'H', 'P', 'G', 'D', 'Q', 'E', 'R', 'K'],
                19: ['W', 'F', 'Y', 'M', 'L', 'IV', 'C', 'A', 'T', 'S', 'N', 'H', 'P', 'G', 'D', 'Q', 'E', 'R', 'K'],
                20: ['W', 'F', 'Y', 'M', 'L', 'I', 'V', 'C', 'A', 'T', 'S', 'N', 'H', 'P', 'G', 'D', 'Q', 'E', 'R', 'K'],
            }
            fastas = self.fasta_list
            subtype = self.kw['PseKRAAC_model']
            raactype = self.kw['RAAC_clust']
            ktuple = self.kw['k-tuple']
            glValue = self.kw['g-gap'] if subtype == 'g-gap' else self.kw['lambdaValue']

            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False

            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i

            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))

            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue)
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_PseKRAAC_type_2(self):
        try:
            AAGroup = {
                2: ['LVIMCAGSTPFYW', 'EDNQKRH'],
                3: ['LVIMCAGSTP', 'FYW', 'EDNQKRH'],
                4: ['LVIMC', 'AGSTP', 'FYW', 'EDNQKRH'],
                5: ['LVIMC', 'AGSTP', 'FYW', 'EDNQ', 'KRH'],
                6: ['LVIM', 'AGST', 'PHC', 'FYW', 'EDNQ', 'KR'],
                8: ['LVIMC', 'AG', 'ST', 'P', 'FYW', 'EDNQ', 'KR', 'H'],
                15: ['LVIM', 'C', 'A', 'G', 'S', 'T', 'P', 'FY', 'W', 'E', 'D', 'N', 'Q', 'KR', 'H'],
                20: ['L', 'V', 'I', 'M', 'C', 'A', 'G', 'S', 'T', 'P', 'F', 'Y', 'W', 'E', 'D', 'N', 'Q', 'K', 'R', 'H'],
            }
            fastas = self.fasta_list
            subtype = self.kw['PseKRAAC_model']
            raactype = self.kw['RAAC_clust']
            ktuple = self.kw['k-tuple']
            glValue = self.kw['g-gap'] if subtype == 'g-gap' else self.kw['lambdaValue']

            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False

            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i

            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))

            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue)
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_PseKRAAC_type_3A(self):
        try:
            AAGroup = {
                2: ['AGSPDEQNHTKRMILFYVC', 'W'],
                3: ['AGSPDEQNHTKRMILFYV', 'W', 'C'],
                4: ['AGSPDEQNHTKRMIV', 'W', 'YFL', 'C'],
                5: ['AGSPDEQNHTKR', 'W', 'YF', 'MIVL', 'C'],
                6: ['AGSP', 'DEQNHTKR', 'W', 'YF', 'MIL', 'VC'],
                7: ['AGP', 'DEQNH', 'TKRMIV', 'W', 'YF', 'L', 'CS'],
                8: ['AG', 'DEQN', 'TKRMIV', 'HY', 'W', 'L', 'FP', 'CS'],
                9: ['AG', 'P', 'DEQN', 'TKRMI', 'HY', 'W', 'F', 'L', 'VCS'],
                10: ['AG', 'P', 'DEQN', 'TKRM', 'HY', 'W', 'F', 'I', 'L', 'VCS'],
                11: ['AG', 'P', 'DEQN', 'TK', 'RI', 'H', 'Y', 'W', 'F', 'ML', 'VCS'],
                12: ['FAS', 'P', 'G', 'DEQ', 'NL', 'TK', 'R', 'H', 'W', 'Y', 'IM', 'VC'],
                13: ['FAS', 'P', 'G', 'DEQ', 'NL', 'T', 'K', 'R', 'H', 'W', 'Y', 'IM', 'VC'],
                14: ['FA', 'P', 'G', 'T', 'DE', 'QM', 'NL', 'K', 'R', 'H', 'W', 'Y', 'IV', 'CS'],
                15: ['FAS', 'P', 'G', 'T', 'DE', 'Q', 'NL', 'K', 'R', 'H', 'W', 'Y', 'M', 'I', 'VC'],
                16: ['FA', 'P', 'G', 'ST', 'DE', 'Q', 'N', 'K', 'R', 'H', 'W', 'Y', 'M', 'L', 'I', 'VC'],
                17: ['FA', 'P', 'G', 'S', 'T', 'DE', 'Q', 'N', 'K', 'R', 'H', 'W', 'Y', 'M', 'L', 'I', 'VC'],
                18: ['FA', 'P', 'G', 'S', 'T', 'DE', 'Q', 'N', 'K', 'R', 'H', 'W', 'Y', 'M', 'L', 'I', 'V', 'C'],
                19: ['FA', 'P', 'G', 'S', 'T', 'D', 'E', 'Q', 'N', 'K', 'R', 'H', 'W', 'Y', 'M', 'L', 'I', 'V', 'C'],
                20: ['F', 'A', 'P', 'G', 'S', 'T', 'D', 'E', 'Q', 'N', 'K', 'R', 'H', 'W', 'Y', 'M', 'L', 'I', 'V', 'C'],
            }
            fastas = self.fasta_list
            subtype = self.kw['PseKRAAC_model']
            raactype = self.kw['RAAC_clust']
            ktuple = self.kw['k-tuple']
            glValue = self.kw['g-gap'] if subtype == 'g-gap' else self.kw['lambdaValue']

            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False

            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i

            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))

            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue)
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_PseKRAAC_type_3B(self):
        try:
            AAGroup = {
                2: ['HRKQNEDSTGPACVIM', 'LFYW'],
                3: ['HRKQNEDSTGPACVIM', 'LFY', 'W'],
                4: ['HRKQNEDSTGPA', 'CIV', 'MLFY', 'W'],
                5: ['HRKQNEDSTGPA', 'CV', 'IML', 'FY', 'W'],
                6: ['HRKQNEDSTPA', 'G', 'CV', 'IML', 'FY', 'W'],
                7: ['HRKQNEDSTA', 'G', 'P', 'CV', 'IML', 'FY', 'W'],
                8: ['HRKQSTA', 'NED', 'G', 'P', 'CV', 'IML', 'FY', 'W'],
                9: ['HRKQ', 'NED', 'ASTG', 'P', 'C', 'IV', 'MLF', 'Y', 'W'],
                10: ['RKHSA', 'Q', 'NED', 'G', 'P', 'C', 'TIV', 'MLF', 'Y', 'W'],
                11: ['RKQ', 'NG', 'ED', 'AST', 'P', 'C', 'IV', 'HML', 'F', 'Y', 'W'],
                12: ['RKQ', 'ED', 'NAST', 'G', 'P', 'C', 'IV', 'H', 'ML', 'F', 'Y', 'W'],
                13: ['RK', 'QE', 'D', 'NG', 'HA', 'ST', 'P', 'C', 'IV', 'ML', 'F', 'Y', 'W'],
                14: ['R', 'K', 'QE', 'D', 'NG', 'HA', 'ST', 'P', 'C', 'IV', 'ML', 'F', 'Y', 'W'],
                15: ['R', 'K', 'QE', 'D', 'NG', 'HA', 'ST', 'P', 'C', 'IV', 'M', 'L', 'F', 'Y', 'W'],
                16: ['R', 'K', 'Q', 'E', 'D', 'NG', 'HA', 'ST', 'P', 'C', 'IV', 'M', 'L', 'F', 'Y', 'W'],
                17: ['R', 'K', 'Q', 'E', 'D', 'NG', 'HA', 'S', 'T', 'P', 'C', 'IV', 'M', 'L', 'F', 'Y', 'W'],
                18: ['R', 'K', 'Q', 'E', 'D', 'NG', 'HA', 'S', 'T', 'P', 'C', 'I', 'V', 'M', 'L', 'F', 'Y', 'W'],
                19: ['R', 'K', 'Q', 'E', 'D', 'NG', 'H', 'A', 'S', 'T', 'P', 'C', 'I', 'V', 'M', 'L', 'F', 'Y', 'W'],
                20: ['R', 'K', 'Q', 'E', 'D', 'N', 'G', 'H', 'A', 'S', 'T', 'P', 'C', 'I', 'V', 'M', 'L', 'F', 'Y', 'W'],
            }

            fastas = self.fasta_list
            subtype = self.kw['PseKRAAC_model']
            raactype = self.kw['RAAC_clust']
            ktuple = self.kw['k-tuple']
            glValue = self.kw['g-gap'] if subtype == 'g-gap' else self.kw['lambdaValue']

            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False

            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i

            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))

            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue)
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False   

    def Protein_PseKRAAC_type_4(self):
        try:
            AAGroup = {
                5: ['G', 'IVFYW', 'ALMEQRK', 'P', 'NDHSTC'],
                8: ['G', 'IV', 'FYW', 'ALM', 'EQRK', 'P', 'ND', 'HSTC'],
                9: ['G', 'IV', 'FYW', 'ALM', 'EQRK', 'P', 'ND', 'HS', 'TC'],
                11: ['G', 'IV', 'FYW', 'A', 'LM', 'EQRK', 'P', 'ND', 'HS', 'T', 'C'],
                13: ['G', 'IV', 'FYW', 'A', 'L', 'M', 'E', 'QRK', 'P', 'ND', 'HS', 'T', 'C'],
                20: ['G', 'I', 'V', 'F', 'Y', 'W', 'A', 'L', 'M', 'E', 'Q', 'R', 'K', 'P', 'N', 'D', 'H', 'S', 'T', 'C'],
            }
            fastas = self.fasta_list
            subtype = self.kw['PseKRAAC_model']
            raactype = self.kw['RAAC_clust']
            ktuple = self.kw['k-tuple']
            glValue = self.kw['g-gap'] if subtype == 'g-gap' else self.kw['lambdaValue']

            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False

            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i

            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))

            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue)
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_PseKRAAC_type_5(self):
        try:
            AAGroup = {
                3: ['FWYCILMVAGSTPHNQ', 'DE', 'KR'],
                4: ['FWY', 'CILMV', 'AGSTP', 'EQNDHKR'],
                8: ['FWY', 'CILMV', 'GA', 'ST', 'P', 'EQND', 'H', 'KR'],
                10: ['G', 'FYW', 'A', 'ILMV', 'RK', 'P', 'EQND', 'H', 'ST', 'C'],
                15: ['G', 'FY', 'W', 'A', 'ILMV', 'E', 'Q', 'RK', 'P', 'N', 'D', 'H', 'S', 'T', 'C'],
                20: ['G', 'I', 'V', 'F', 'Y', 'W', 'A', 'L', 'M', 'E', 'Q', 'R', 'K', 'P', 'N', 'D', 'H', 'S', 'T', 'C'],
            }
            fastas = self.fasta_list
            subtype = self.kw['PseKRAAC_model']
            raactype = self.kw['RAAC_clust']
            ktuple = self.kw['k-tuple']
            glValue = self.kw['g-gap'] if subtype == 'g-gap' else self.kw['lambdaValue']

            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False

            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i

            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))

            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue)
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_PseKRAAC_type_6A(self):
        try:
            AAGroup = {
                4: ['AGPST', 'CILMV', 'DEHKNQR', 'FYW'],
                5: ['AHT', 'CFILMVWY', 'DE', 'GP', 'KNQRS'],
                20: ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'],
            }
            fastas = self.fasta_list
            subtype = self.kw['PseKRAAC_model']
            raactype = self.kw['RAAC_clust']
            ktuple = self.kw['k-tuple']
            glValue = self.kw['g-gap'] if subtype == 'g-gap' else self.kw['lambdaValue']

            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False

            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i

            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))

            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue)
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_PseKRAAC_type_6B(self):
        try:
            AAGroup = {
                5: ['AEHKQRST', 'CFILMVWY', 'DN', 'G', 'P'],
            }
            fastas = self.fasta_list
            subtype = self.kw['PseKRAAC_model']
            raactype = self.kw['RAAC_clust']
            ktuple = self.kw['k-tuple']
            glValue = self.kw['g-gap'] if subtype == 'g-gap' else self.kw['lambdaValue']

            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False

            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i

            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))

            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue)
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_PseKRAAC_type_6C(self):
        try:
            AAGroup = {
                5: ['AG', 'C', 'DEKNPQRST', 'FILMVWY', 'H'],
            }
            fastas = self.fasta_list
            subtype = self.kw['PseKRAAC_model']
            raactype = self.kw['RAAC_clust']
            ktuple = self.kw['k-tuple']
            glValue = self.kw['g-gap'] if subtype == 'g-gap' else self.kw['lambdaValue']

            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False

            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i

            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))

            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue)
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_PseKRAAC_type_7(self):
        try:
            AAGroup = {
                2: ['C', 'MFILVWYAGTSNQDEHRKP'],
                3: ['C', 'MFILVWYAKR', 'GTSNQDEHP'],
                4: ['C', 'KR', 'MFILVWYA', 'GTSNQDEHP'],
                5: ['C', 'KR', 'MFILVWYA', 'DE', 'GTSNQHP'],
                6: ['C', 'KR', 'WYA', 'MFILV', 'DE', 'GTSNQHP'],
                7: ['C', 'KR', 'WYA', 'MFILV', 'DE', 'QH', 'GTSNP'],
                8: ['C', 'KR', 'WYA', 'MFILV', 'D', 'E', 'QH', 'GTSNP'],
                9: ['C', 'KR', 'WYA', 'MFILV', 'D', 'E', 'QH', 'TP', 'GSN'],
                10: ['C', 'KR', 'WY', 'A', 'MFILV', 'D', 'E', 'QH', 'TP', 'GSN'],
                11: ['C', 'K', 'R', 'WY', 'A', 'MFILV', 'D', 'E', 'QH', 'TP', 'GSN'],
                12: ['C', 'K', 'R', 'WY', 'A', 'MFILV', 'D', 'E', 'QH', 'TP', 'GS', 'N'],
                13: ['C', 'K', 'R', 'W', 'Y', 'A', 'MFILV', 'D', 'E', 'QH', 'TP', 'GS', 'N'],
                14: ['C', 'K', 'R', 'W', 'Y', 'A', 'FILV', 'M', 'D', 'E', 'QH', 'TP', 'GS', 'N'],
                15: ['C', 'K', 'R', 'W', 'Y', 'A', 'FILV', 'M', 'D', 'E', 'Q', 'H', 'TP', 'GS', 'N'],
                16: ['C', 'K', 'R', 'W', 'Y', 'A', 'FILV', 'M', 'D', 'E', 'Q', 'H', 'TP', 'G', 'S', 'N'],
                17: ['C', 'K', 'R', 'W', 'Y', 'A', 'FI', 'LV', 'M', 'D', 'E', 'Q', 'H', 'TP', 'G', 'S', 'N'],
                18: ['C', 'K', 'R', 'W', 'Y', 'A', 'FI', 'LV', 'M', 'D', 'E', 'Q', 'H', 'T', 'P', 'G', 'S', 'N'],
                19: ['C', 'K', 'R', 'W', 'Y', 'A', 'F', 'I', 'LV', 'M', 'D', 'E', 'Q', 'H', 'T', 'P', 'G', 'S', 'N'],
                20: ['C', 'K', 'R', 'W', 'Y', 'A', 'F', 'I', 'L', 'V', 'M', 'D', 'E', 'Q', 'H', 'T', 'P', 'G', 'S', 'N'],
            }
            fastas = self.fasta_list
            subtype = self.kw['PseKRAAC_model']
            raactype = self.kw['RAAC_clust']
            ktuple = self.kw['k-tuple']
            glValue = self.kw['g-gap'] if subtype == 'g-gap' else self.kw['lambdaValue']

            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False

            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i

            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))

            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue)
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_PseKRAAC_type_8(self):
        try:
            AAGroup = {
                2: ['ADEGKNPQRST', 'CFHILMVWY'],
                3: ['ADEGNPST', 'CHKQRW', 'FILMVY'],
                4: ['AGNPST', 'CHWY', 'DEKQR', 'FILMV'],
                5: ['AGPST', 'CFWY', 'DEN', 'HKQR', 'ILMV'],
                6: ['APST', 'CW', 'DEGN', 'FHY', 'ILMV', 'KQR'],
                7: ['AGST', 'CW', 'DEN', 'FY', 'HP', 'ILMV', 'KQR'],
                8: ['AST', 'CG', 'DEN', 'FY', 'HP', 'ILV', 'KQR', 'MW'],
                9: ['AST', 'CW', 'DE', 'FY', 'GN', 'HQ', 'ILV', 'KR', 'MP'],
                10: ['AST', 'CW', 'DE', 'FY', 'GN', 'HQ', 'IV', 'KR', 'LM', 'P'],
                11: ['AST', 'C', 'DE', 'FY', 'GN', 'HQ', 'IV', 'KR', 'LM', 'P', 'W'],
                12: ['AST', 'C', 'DE', 'FY', 'G', 'HQ', 'IV', 'KR', 'LM', 'N', 'P', 'W'],
                13: ['AST', 'C', 'DE', 'FY', 'G', 'H', 'IV', 'KR', 'LM', 'N', 'P', 'Q', 'W'],
                14: ['AST', 'C', 'DE', 'FL', 'G', 'H', 'IV', 'KR', 'M', 'N', 'P', 'Q', 'W', 'Y'],
                15: ['AST', 'C', 'DE', 'F', 'G', 'H', 'IV', 'KR', 'L', 'M', 'N', 'P', 'Q', 'W', 'Y'],
                16: ['AT', 'C', 'DE', 'F', 'G', 'H', 'IV', 'KR', 'L', 'M', 'N', 'P', 'Q', 'S', 'W', 'Y'],
                17: ['AT', 'C', 'DE', 'F', 'G', 'H', 'IV', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'W', 'Y'],
                18: ['A', 'C', 'DE', 'F', 'G', 'H', 'IV', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'W', 'Y'],
                19: ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'IV', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'W', 'Y'],
                20: ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'V', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'W', 'Y'],
            }
            fastas = self.fasta_list
            subtype = self.kw['PseKRAAC_model']
            raactype = self.kw['RAAC_clust']
            ktuple = self.kw['k-tuple']
            glValue = self.kw['g-gap'] if subtype == 'g-gap' else self.kw['lambdaValue']

            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False

            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i

            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))

            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue)
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_PseKRAAC_type_9(self):
        try:
            AAGroup = {
                2: ['ACDEFGHILMNPQRSTVWY', 'K'],
                3: ['ACDFGMPQRSTW', 'EHILNVY', 'K'],
                4: ['AGPT', 'CDFMQRSW', 'EHILNVY', 'K'],
                5: ['AGPT', 'CDQ', 'EHILNVY', 'FMRSW', 'K'],
                6: ['AG', 'CDQ', 'EHILNVY', 'FMRSW', 'K', 'PT'],
                7: ['AG', 'CDQ', 'EHNY', 'FMRSW', 'ILV', 'K', 'PT'],
                8: ['AG', 'C', 'DQ', 'EHNY', 'FMRSW', 'ILV', 'K', 'PT'],
                9: ['AG', 'C', 'DQ', 'EHNY', 'FMW', 'ILV', 'K', 'PT', 'RS'],
                10: ['A', 'C', 'DQ', 'EHNY', 'FMW', 'G', 'ILV', 'K', 'PT', 'RS'],
                11: ['A', 'C', 'DQ', 'EHNY', 'FM', 'G', 'ILV', 'K', 'PT', 'RS', 'W'],
                12: ['A', 'C', 'DQ', 'EHNY', 'FM', 'G', 'IL', 'K', 'PT', 'RS', 'V', 'W'],
                13: ['A', 'C', 'DQ', 'E', 'FM', 'G', 'HNY', 'IL', 'K', 'PT', 'RS', 'V', 'W'],
                14: ['A', 'C', 'D', 'E', 'FM', 'G', 'HNY', 'IL', 'K', 'PT', 'Q', 'RS', 'V', 'W'],
                15: ['A', 'C', 'D', 'E', 'FM', 'G', 'HNY', 'IL', 'K', 'PT', 'Q', 'R', 'S', 'V', 'W'],
                16: ['A', 'C', 'D', 'E', 'F', 'G', 'HNY', 'IL', 'K', 'M', 'PT', 'Q', 'R', 'S', 'V', 'W'],
                17: ['A', 'C', 'D', 'E', 'F', 'G', 'HNY', 'IL', 'K', 'M', 'P', 'Q', 'R', 'S', 'T', 'V', 'W'],
                18: ['A', 'C', 'D', 'E', 'F', 'G', 'HNY', 'I', 'K', 'L', 'M', 'P', 'Q', 'R', 'S', 'T', 'V', 'W'],
                19: ['A', 'C', 'D', 'E', 'F', 'G', 'HN', 'I', 'K', 'L', 'M', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'],
                20: ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'N', 'I', 'K', 'L', 'M', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'],
            }
            fastas = self.fasta_list
            subtype = self.kw['PseKRAAC_model']
            raactype = self.kw['RAAC_clust']
            ktuple = self.kw['k-tuple']
            glValue = self.kw['g-gap'] if subtype == 'g-gap' else self.kw['lambdaValue']

            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False

            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i

            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))

            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue)
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_PseKRAAC_type_10(self):
        try:
            AAGroup = {
                2: ['CMFILVWY', 'AGTSNQDEHRKP'],
                3: ['CMFILVWY', 'AGTSP', 'NQDEHRK'],
                4: ['CMFWY', 'ILV', 'AGTS', 'NQDEHRKP'],
                5: ['FWYH', 'MILV', 'CATSP', 'G', 'NQDERK'],
                6: ['FWYH', 'MILV', 'CATS', 'P', 'G', 'NQDERK'],
                7: ['FWYH', 'MILV', 'CATS', 'P', 'G', 'NQDE', 'RK'],
                8: ['FWYH', 'MILV', 'CA', 'NTS', 'P', 'G', 'DE', 'QRK'],
                9: ['FWYH', 'ML', 'IV', 'CA', 'NTS', 'P', 'G', 'DE', 'QRK'],
                10: ['FWY', 'ML', 'IV', 'CA', 'TS', 'NH', 'P', 'G', 'DE', 'QRK'],
                11: ['FWY', 'ML', 'IV', 'CA', 'TS', 'NH', 'P', 'G', 'D', 'QE', 'RK'],
                12: ['FWY', 'ML', 'IV', 'C', 'A', 'TS', 'NH', 'P', 'G', 'D', 'QE', 'RK'],
                13: ['FWY', 'ML', 'IV', 'C', 'A', 'T', 'S', 'NH', 'P', 'G', 'D', 'QE', 'RK'],
                14: ['FWY', 'ML', 'IV', 'C', 'A', 'T', 'S', 'NH', 'P', 'G', 'D', 'QE', 'R', 'K'],
                15: ['FWY', 'ML', 'IV', 'C', 'A', 'T', 'S', 'N', 'H', 'P', 'G', 'D', 'QE', 'R', 'K'],
                16: ['W', 'FY', 'ML', 'IV', 'C', 'A', 'T', 'S', 'N', 'H', 'P', 'G', 'D', 'QE', 'R', 'K'],
                17: ['W', 'FY', 'ML', 'IV', 'C', 'A', 'T', 'S', 'N', 'H', 'P', 'G', 'D', 'Q', 'E', 'R', 'K'],
                18: ['W', 'FY', 'M', 'L', 'IV', 'C', 'A', 'T', 'S', 'N', 'H', 'P', 'G', 'D', 'Q', 'E', 'R', 'K'],
                19: ['W', 'F', 'Y', 'M', 'L', 'IV', 'C', 'A', 'T', 'S', 'N', 'H', 'P', 'G', 'D', 'Q', 'E', 'R', 'K'],
                20: ['W', 'F', 'Y', 'M', 'L', 'I', 'V', 'C', 'A', 'T', 'S', 'N', 'H', 'P', 'G', 'D', 'Q', 'E', 'R', 'K'],
            }
            fastas = self.fasta_list
            subtype = self.kw['PseKRAAC_model']
            raactype = self.kw['RAAC_clust']
            ktuple = self.kw['k-tuple']
            glValue = self.kw['g-gap'] if subtype == 'g-gap' else self.kw['lambdaValue']

            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False

            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i

            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))

            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue)
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_PseKRAAC_type_11(self):
        try:
            AAGroup = {
                2: ['CFYWMLIV', 'GPATSNHQEDRK'],
                3: ['CFYWMLIV', 'GPATS', 'NHQEDRK'],
                4: ['CFYW', 'MLIV', 'GPATS', 'NHQEDRK'],
                5: ['CFYW', 'MLIV', 'G', 'PATS', 'NHQEDRK'],
                6: ['CFYW', 'MLIV', 'G', 'P', 'ATS', 'NHQEDRK'],
                7: ['CFYW', 'MLIV', 'G', 'P', 'ATS', 'NHQED', 'RK'],
                8: ['CFYW', 'MLIV', 'G', 'P', 'ATS', 'NH', 'QED', 'RK'],
                9: ['CFYW', 'ML', 'IV', 'G', 'P', 'ATS', 'NH', 'QED', 'RK'],
                10: ['C', 'FYW', 'ML', 'IV', 'G', 'P', 'ATS', 'NH', 'QED', 'RK'],
                11: ['C', 'FYW', 'ML', 'IV', 'G', 'P', 'A', 'TS', 'NH', 'QED', 'RK'],
                12: ['C', 'FYW', 'ML', 'IV', 'G', 'P', 'A', 'TS', 'NH', 'QE', 'D', 'RK'],
                13: ['C', 'FYW', 'ML', 'IV', 'G', 'P', 'A', 'T', 'S', 'NH', 'QE', 'D', 'RK'],
                14: ['C', 'FYW', 'ML', 'IV', 'G', 'P', 'A', 'T', 'S', 'N', 'H', 'QE', 'D', 'RK'],
                15: ['C', 'FYW', 'ML', 'IV', 'G', 'P', 'A', 'T', 'S', 'N', 'H', 'QE', 'D', 'R', 'K'],
                16: ['C', 'FY', 'W', 'ML', 'IV', 'G', 'P', 'A', 'T', 'S', 'N', 'H', 'QE', 'D', 'R', 'K'],
                17: ['C', 'FY', 'W', 'ML', 'IV', 'G', 'P', 'A', 'T', 'S', 'N', 'H', 'Q', 'E', 'D', 'R', 'K'],
                18: ['C', 'FY', 'W', 'M', 'L', 'IV', 'G', 'P', 'A', 'T', 'S', 'N', 'H', 'Q', 'E', 'D', 'R', 'K'],
                19: ['C', 'F', 'Y', 'W', 'M', 'L', 'IV', 'G', 'P', 'A', 'T', 'S', 'N', 'H', 'Q', 'E', 'D', 'R', 'K'],
                20: ['C', 'F', 'Y', 'W', 'M', 'L', 'I', 'V', 'G', 'P', 'A', 'T', 'S', 'N', 'H', 'Q', 'E', 'D', 'R', 'K'],
            }
            fastas = self.fasta_list
            subtype = self.kw['PseKRAAC_model']
            raactype = self.kw['RAAC_clust']
            ktuple = self.kw['k-tuple']
            glValue = self.kw['g-gap'] if subtype == 'g-gap' else self.kw['lambdaValue']

            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False

            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i

            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))

            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue)
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_PseKRAAC_type_12(self):
        try:
            AAGroup = {
                2: ['IVMLFWYC', 'ARNDQEGHKPST'],
                3: ['IVLMFWC', 'YA', 'RNDQEGHKPST'],
                4: ['IVLMFW', 'C', 'YA', 'RNDQEGHKPST'],
                5: ['IVLMFW', 'C', 'YA', 'G', 'RNDQEHKPST'],
                6: ['IVLMF', 'WY', 'C', 'AH', 'G', 'RNDQEKPST'],
                7: ['IVLMF', 'WY', 'C', 'AH', 'GP', 'R', 'NDQEKST'],
                8: ['IVLMF', 'WY', 'C', 'A', 'G', 'R', 'Q', 'NDEHKPST'],
                9: ['IVLMF', 'WY', 'C', 'A', 'G', 'P', 'H', 'K', 'RNDQEST'],
                10: ['IVLM', 'F', 'W', 'Y', 'C', 'A', 'H', 'G', 'RN', 'DQEKPST'],
                11: ['IVLMF', 'W', 'Y', 'C', 'A', 'H', 'G', 'R', 'N', 'Q', 'DEKPST'],
                12: ['IVLM', 'F', 'W', 'Y', 'C', 'A', 'H', 'G', 'N', 'Q', 'T', 'RDEKPS'],
                13: ['IVLM', 'F', 'W', 'Y', 'C', 'A', 'H', 'G', 'N', 'Q', 'P', 'R', 'DEKST'],
                14: ['IVLM', 'F', 'W', 'Y', 'C', 'A', 'H', 'G', 'N', 'Q', 'P', 'R', 'K', 'DEST'],
                15: ['IVLM', 'F', 'W', 'Y', 'C', 'A', 'H', 'G', 'N', 'Q', 'P', 'R', 'K', 'D', 'EST'],
                16: ['IVLM', 'F', 'W', 'Y', 'C', 'A', 'H', 'G', 'N', 'Q', 'P', 'R', 'K', 'S', 'T', 'DE'],
                17: ['IVL', 'M', 'F', 'W', 'Y', 'C', 'A', 'H', 'G', 'N', 'Q', 'P', 'R', 'K', 'S', 'T', 'DE'],
                18: ['IVL', 'M', 'F', 'W', 'Y', 'C', 'A', 'H', 'G', 'N', 'Q', 'P', 'R', 'K', 'S', 'T', 'D', 'E'],
                20: ['I', 'V', 'L', 'M', 'F', 'W', 'Y', 'C', 'A', 'H', 'G', 'N', 'Q', 'P', 'R', 'K', 'S', 'T', 'D', 'E'],
            }
            fastas = self.fasta_list
            subtype = self.kw['PseKRAAC_model']
            raactype = self.kw['RAAC_clust']
            ktuple = self.kw['k-tuple']
            glValue = self.kw['g-gap'] if subtype == 'g-gap' else self.kw['lambdaValue']

            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False

            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i

            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))

            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue)
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_PseKRAAC_type_13(self):
        try:
            AAGroup = {
                4: ['ADKERNTSQ', 'YFLIVMCWH', 'G', 'P'],
                12: ['A', 'D', 'KER', 'N', 'TSQ', 'YF', 'LIVM', 'C', 'W', 'H', 'G', 'P'],
                17: ['A', 'D', 'KE', 'R', 'N', 'T', 'S', 'Q', 'Y', 'F', 'LIV', 'M', 'C', 'W', 'H', 'G', 'P'],
                20: ['A', 'D', 'K', 'E', 'R', 'N', 'T', 'S', 'Q', 'Y', 'F', 'L', 'I', 'V', 'M', 'C', 'W', 'H', 'G', 'P'],
            }
            fastas = self.fasta_list
            subtype = self.kw['PseKRAAC_model']
            raactype = self.kw['RAAC_clust']
            ktuple = self.kw['k-tuple']
            glValue = self.kw['g-gap'] if subtype == 'g-gap' else self.kw['lambdaValue']

            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False

            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i

            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))

            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue)
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_PseKRAAC_type_14(self):
        try:
            AAGroup = {
                2: ['ARNDCQEGHKPST', 'ILMFWYV'],
                3: ['ARNDQEGHKPST', 'C', 'ILMFWYV'],
                4: ['ARNDQEGHKPST', 'C', 'ILMFYV', 'W'],
                5: ['AGPST', 'RNDQEHK', 'C', 'ILMFYV', 'W'],
                6: ['AGPST', 'RNDQEK', 'C', 'H', 'ILMFYV', 'W'],
                7: ['ANDGST', 'RQEK', 'C', 'H', 'ILMFYV', 'P', 'W'],
                8: ['ANDGST', 'RQEK', 'C', 'H', 'ILMV', 'FY', 'P', 'W'],
                9: ['AGST', 'RQEK', 'ND', 'C', 'H', 'ILMV', 'FY', 'P', 'W'],
                10: ['AGST', 'RK', 'ND', 'C', 'QE', 'H', 'ILMV', 'FY', 'P', 'W'],
                11: ['AST', 'RK', 'ND', 'C', 'QE', 'G', 'H', 'ILMV', 'FY', 'P', 'W'],
                12: ['AST', 'RK', 'ND', 'C', 'QE', 'G', 'H', 'IV', 'LM', 'FY', 'P', 'W'],
                13: ['AST', 'RK', 'N', 'D', 'C', 'QE', 'G', 'H', 'IV', 'LM', 'FY', 'P', 'W'],
                14: ['AST', 'RK', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'IV', 'LM', 'FY', 'P', 'W'],
                15: ['A', 'RK', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'IV', 'LM', 'FY', 'P', 'ST', 'W'],
                16: ['A', 'RK', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'IV', 'LM', 'F', 'P', 'ST', 'W', 'Y'],
                17: ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'IV', 'LM', 'K', 'F', 'P', 'ST', 'W', 'Y'],
                18: ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'IV', 'LM', 'K', 'F', 'P', 'S', 'T', 'W', 'Y'],
                19: ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'IV', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y'],
                20: ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'V', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y'],
            }
            fastas = self.fasta_list
            subtype = self.kw['PseKRAAC_model']
            raactype = self.kw['RAAC_clust']
            ktuple = self.kw['k-tuple']
            glValue = self.kw['g-gap'] if subtype == 'g-gap' else self.kw['lambdaValue']

            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False

            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i

            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))

            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue)
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_PseKRAAC_type_15(self):
        try:
            AAGroup = {
                2: ['MFILVAW', 'CYQHPGTSNRKDE'],
                3: ['MFILVAW', 'CYQHPGTSNRK', 'DE'],
                4: ['MFILV', 'ACW', 'YQHPGTSNRK', 'DE'],
                5: ['MFILV', 'ACW', 'YQHPGTSN', 'RK', 'DE'],
                6: ['MFILV', 'A', 'C', 'WYQHPGTSN', 'RK', 'DE'],
                7: ['MFILV', 'A', 'C', 'WYQHP', 'GTSN', 'RK', 'DE'],
                8: ['MFILV', 'A', 'C', 'WYQHP', 'G', 'TSN', 'RK', 'DE'],
                9: ['MF', 'ILV', 'A', 'C', 'WYQHP', 'G', 'TSN', 'RK', 'DE'],
                10: ['MF', 'ILV', 'A', 'C', 'WYQHP', 'G', 'TSN', 'RK', 'D', 'E'],
                11: ['MF', 'IL', 'V', 'A', 'C', 'WYQHP', 'G', 'TSN', 'RK', 'D', 'E'],
                12: ['MF', 'IL', 'V', 'A', 'C', 'WYQHP', 'G', 'TS', 'N', 'RK', 'D', 'E'],
                13: ['MF', 'IL', 'V', 'A', 'C', 'WYQHP', 'G', 'T', 'S', 'N', 'RK', 'D', 'E'],
                14: ['MF', 'I', 'L', 'V', 'A', 'C', 'WYQHP', 'G', 'T', 'S', 'N', 'RK', 'D', 'E'],
                15: ['MF', 'IL', 'V', 'A', 'C', 'WYQ', 'H', 'P', 'G', 'T', 'S', 'N', 'RK', 'D', 'E'],
                16: ['MF', 'I', 'L', 'V', 'A', 'C', 'WYQ', 'H', 'P', 'G', 'T', 'S', 'N', 'RK', 'D', 'E'],
                20: ['M', 'F', 'I', 'L', 'V', 'A', 'C', 'W', 'Y', 'Q', 'H', 'P', 'G', 'T', 'S', 'N', 'R', 'K', 'D', 'E'],
            }
            fastas = self.fasta_list
            subtype = self.kw['PseKRAAC_model']
            raactype = self.kw['RAAC_clust']
            ktuple = self.kw['k-tuple']
            glValue = self.kw['g-gap'] if subtype == 'g-gap' else self.kw['lambdaValue']

            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False

            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i

            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))

            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue)
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Protein_PseKRAAC_type_16(self):
        try:
            AAGroup = {
                2: ['IMVLFWY', 'GPCASTNHQEDRK'],
                3: ['IMVLFWY', 'GPCAST', 'NHQEDRK'],
                4: ['IMVLFWY', 'G', 'PCAST', 'NHQEDRK'],
                5: ['IMVL', 'FWY', 'G', 'PCAST', 'NHQEDRK'],
                6: ['IMVL', 'FWY', 'G', 'P', 'CAST', 'NHQEDRK'],
                7: ['IMVL', 'FWY', 'G', 'P', 'CAST', 'NHQED', 'RK'],
                8: ['IMV', 'L', 'FWY', 'G', 'P', 'CAST', 'NHQED', 'RK'],
                9: ['IMV', 'L', 'FWY', 'G', 'P', 'C', 'AST', 'NHQED', 'RK'],
                10: ['IMV', 'L', 'FWY', 'G', 'P', 'C', 'A', 'STNH', 'RKQE', 'D'],
                11: ['IMV', 'L', 'FWY', 'G', 'P', 'C', 'A', 'STNH', 'RKQ', 'E', 'D'],
                12: ['IMV', 'L', 'FWY', 'G', 'P', 'C', 'A', 'ST', 'N', 'HRKQ', 'E', 'D'],
                13: ['IMV', 'L', 'F', 'WY', 'G', 'P', 'C', 'A', 'ST', 'N', 'HRKQ', 'E', 'D'],
                14: ['IMV', 'L', 'F', 'WY', 'G', 'P', 'C', 'A', 'S', 'T', 'N', 'HRKQ', 'E', 'D'],
                15: ['IMV', 'L', 'F', 'WY', 'G', 'P', 'C', 'A', 'S', 'T', 'N', 'H', 'RKQ', 'E', 'D'],
                16: ['IMV', 'L', 'F', 'W', 'Y', 'G', 'P', 'C', 'A', 'S', 'T', 'N', 'H', 'RKQ', 'E', 'D'],
                20: ['I', 'M', 'V', 'L', 'F', 'W', 'Y', 'G', 'P', 'C', 'A', 'S', 'T', 'N', 'H', 'R', 'K', 'Q', 'E', 'D'],
            }
            fastas = self.fasta_list
            subtype = self.kw['PseKRAAC_model']
            raactype = self.kw['RAAC_clust']
            ktuple = self.kw['k-tuple']
            glValue = self.kw['g-gap'] if subtype == 'g-gap' else self.kw['lambdaValue']

            if raactype not in AAGroup:
                self.error_msg = 'PseKRAAC descriptor value error.'
                return False

            # index each amino acids to their group
            myDict = {}
            for i in range(len(AAGroup[raactype])):
                for aa in AAGroup[raactype][i]:
                    myDict[aa] = i

            gDict = {}
            gNames = []
            for i in range(len(AAGroup[raactype])):
                gDict[i] = 'T1.G.' + str(i + 1)
                gNames.append('T1.G.' + str(i + 1))

            encodings = []
            if subtype == 'g-gap':
                encodings = self.gapModel(fastas, myDict, gDict, gNames, ktuple, glValue)
            else:
                encodings = self.lambdaModel(fastas, myDict, gDict, gNames, ktuple, glValue)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    """ DNA and RNA descriptors """

    def kmerArray(self, sequence, k):
        kmer = []
        for i in range(len(sequence) - k + 1):
            kmer.append(sequence[i:i + k])
        return kmer

    def Kmer(self):
        try:
            fastas = self.fasta_list
            k = self.kw['kmer']
            upto = False
            normalize = True
            type = self.sequence_type

            encoding = []
            header = ['SampleName', 'label']
            NA = 'ACGT'
            if type in ("DNA", 'RNA'):
                NA = 'ACGT'
            else:
                NA = 'ACDEFGHIKLMNPQRSTVWY'

            if upto == True:
                for tmpK in range(1, k + 1):
                    for kmer in itertools.product(NA, repeat=tmpK):
                        header.append(''.join(kmer))
                encoding.append(header)
                for i in fastas:
                    name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                    count = Counter()
                    for tmpK in range(1, k + 1):
                        kmers = self.kmerArray(sequence, tmpK)
                        count.update(kmers)
                        if normalize == True:
                            for key in count:
                                if len(key) == tmpK:
                                    count[key] = count[key] / len(kmers)
                    code = [name, label]
                    for j in range(2, len(header)):
                        if header[j] in count:
                            code.append(count[header[j]])
                        else:
                            code.append(0)
                    encoding.append(code)
            else:
                for kmer in itertools.product(NA, repeat=k):
                    header.append(''.join(kmer))
                encoding.append(header)
                for i in fastas:
                    name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                    kmers = self.kmerArray(sequence, k)
                    count = Counter()
                    count.update(kmers)
                    if normalize == True:
                        for key in count:
                            count[key] = count[key] / len(kmers)
                    code = [name, label]
                    for j in range(2, len(header)):
                        if header[j] in count:
                            code.append(count[header[j]])
                        else:
                            code.append(0)
                    encoding.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encoding, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encoding
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def mismatch_count(self, seq1, seq2):
        mismatch = 0
        for i in range(min([len(seq1), len(seq2)])):
            if seq1[i] != seq2[i]:
                mismatch += 1
        return mismatch

    def Mismatch(self):
        try:
            k = self.kw['kmer']
            m = self.kw['mismatch']
            NN = 'ACGT'

            encoding = []
            header = ['SampleName', 'label']
            template_dict = {}
            for kmer in itertools.product(NN, repeat=k):
                header.append(''.join(kmer))
                template_dict[''.join(kmer)] = 0
            encoding.append(header)

            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                kmers = self.kmerArray(sequence, k)
                tmp_dict = template_dict.copy()
                for kmer in kmers:
                    for key in tmp_dict:
                        if self.mismatch_count(kmer, key) <= m:
                            tmp_dict[key] += 1
                code = [name, label] + [tmp_dict[key] for key in sorted(tmp_dict.keys())]
                encoding.append(code)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encoding, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encoding
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    """ Subsequence """
    # code from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4830532/bin/pone.0153268.s001.zip
    def GetKmerDict(self, piRNAletter, k):
        kmerlst = []
        partkmers = list(itertools.combinations_with_replacement(piRNAletter, k))
        for element in partkmers:
            elelst = set(itertools.permutations(element, k))
            strlst = [''.join(ele) for ele in elelst]
            kmerlst += strlst
        kmerlst = np.sort(kmerlst)
        kmerdict = {kmerlst[i]: i for i in range(len(kmerlst))}
        return kmerdict

    def GetSubsequenceProfile(self, instances, piRNAletter, k, delta):
        kmerdict = self.GetKmerDict(piRNAletter, k)
        X = []
        for sequence in instances:
            vector = self.GetSubsequenceProfileVector(sequence, kmerdict, k, delta)
            X.append(vector)
        X = np.array(X)
        return X

    def GetSubsequenceProfileVector(self, sequence, kmerdict, k, delta):
        vector = np.zeros((1, len(kmerdict)))
        sequence = np.array(list(sequence))
        n = len(sequence)
        index_lst = list(itertools.combinations(range(n), k))
        for subseq_index in index_lst:
            subseq_index = list(subseq_index)
            subsequence = sequence[subseq_index]
            position = kmerdict.get(''.join(subsequence))
            subseq_length = subseq_index[-1] - subseq_index[0] + 1
            subseq_score = 1 if subseq_length == k else delta ** subseq_length
            vector[0, position] += subseq_score
        return list(vector[0])

    def Subsequence(self):
        try:
            k = self.kw['kmer']
            delta = self.kw['delta']
            NN_list = ['A', 'C', 'G', 'T']

            encoding = []
            header = ['SampleName', 'label']
            for kmer in itertools.product(NN_list, repeat=k):
                header.append(''.join(kmer))
            encoding.append(header)

            instances = [elem[1] for elem in self.fasta_list]

            code = self.GetSubsequenceProfile(instances, NN_list, k, delta)
            info_list = np.array([[item[0], item[2]] for item in self.fasta_list])
            code = np.hstack((info_list, code))

            self.encoding_array = np.vstack((header, code))
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    """ end of Subsequence """

    def RC(self, kmer):
        myDict = {
            'A': 'T',
            'C': 'G',
            'G': 'C',
            'T': 'A'
        }
        return ''.join([myDict[nc] for nc in kmer[::-1]])

    def generateRCKmer(self, kmerList):
        rckmerList = set()
        myDict = {
            'A': 'T',
            'C': 'G',
            'G': 'C',
            'T': 'A'
        }
        for kmer in kmerList:
            rckmerList.add(sorted([kmer, ''.join([myDict[nc] for nc in kmer[::-1]])])[0])
        return sorted(rckmerList)

    def RCKmer(self):
        try:
            fastas = self.fasta_list
            k = self.kw['kmer']
            upto = False
            normalize = True

            encoding = []
            header = ['SampleName', 'label']
            NA = 'ACGT'

            if upto == True:
                for tmpK in range(1, k + 1):
                    tmpHeader = []
                    for kmer in itertools.product(NA, repeat=tmpK):
                        tmpHeader.append(''.join(kmer))
                    header = header + self.generateRCKmer(tmpHeader)
                myDict = {}
                for kmer in header[2:]:
                    rckmer = self.RC(kmer)
                    if kmer != rckmer:
                        myDict[rckmer] = kmer
                encoding.append(header)
                for i in fastas:
                    name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                    count = Counter()
                    for tmpK in range(1, k + 1):
                        kmers = self.kmerArray(sequence, tmpK)
                        for j in range(len(kmers)):
                            if kmers[j] in myDict:
                                kmers[j] = myDict[kmers[j]]
                        count.update(kmers)
                        if normalize == True:
                            for key in count:
                                if len(key) == tmpK:
                                    count[key] = count[key] / len(kmers)
                    code = [name, label]
                    for j in range(2, len(header)):
                        if header[j] in count:
                            code.append(count[header[j]])
                        else:
                            code.append(0)
                    encoding.append(code)
            else:
                tmpHeader = []
                for kmer in itertools.product(NA, repeat=k):
                    tmpHeader.append(''.join(kmer))
                header = header + self.generateRCKmer(tmpHeader)
                myDict = {}
                for kmer in header[2:]:
                    rckmer = self.RC(kmer)
                    if kmer != rckmer:
                        myDict[rckmer] = kmer

                encoding.append(header)
                for i in fastas:
                    name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                    kmers = self.kmerArray(sequence, k)
                    for j in range(len(kmers)):
                        if kmers[j] in myDict:
                            kmers[j] = myDict[kmers[j]]
                    count = Counter()
                    count.update(kmers)
                    if normalize == True:
                        for key in count:
                            count[key] = count[key] / len(kmers)
                    code = [name, label]
                    for j in range(2, len(header)):
                        if header[j] in count:
                            code.append(count[header[j]])
                        else:
                            code.append(0)
                    encoding.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encoding, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encoding
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def NAC(self):
        try:
            NA = 'ACGT'
            encodings = []
            header = ['SampleName', 'label']
            for i in NA:
                header.append(i)
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                count = Counter(sequence)
                for key in count:
                    count[key] = count[key] / len(sequence)
                code = [name, label]
                for na in NA:
                    code.append(count[na])
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def DNC(self):
        try:
            base = 'ACGT'
            encodings = []
            dinucleotides = [n1 + n2 for n1 in base for n2 in base]
            header = ['SampleName', 'label'] + dinucleotides
            encodings.append(header)

            AADict = {}
            for i in range(len(base)):
                AADict[base[i]] = i

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                tmpCode = [0] * 16
                for j in range(len(sequence) - 2 + 1):
                    tmpCode[AADict[sequence[j]] * 4 + AADict[sequence[j + 1]]] = tmpCode[AADict[sequence[j]] * 4 + AADict[
                        sequence[j + 1]]] + 1
                if sum(tmpCode) != 0:
                    tmpCode = [i / sum(tmpCode) for i in tmpCode]
                code = code + tmpCode
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def TNC(self):
        try:
            AA = 'ACGT'
            encodings = []
            triPeptides = [aa1 + aa2 + aa3 for aa1 in AA for aa2 in AA for aa3 in AA]
            header = ['SampleName', 'label'] + triPeptides
            encodings.append(header)

            AADict = {}
            for i in range(len(AA)):
                AADict[AA[i]] = i

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                tmpCode = [0] * 64
                for j in range(len(sequence) - 3 + 1):
                    tmpCode[AADict[sequence[j]] * 16 + AADict[sequence[j + 1]] * 4 + AADict[sequence[j + 2]]] = tmpCode[
                                                                                                                    AADict[
                                                                                                                        sequence[
                                                                                                                            j]] * 16 +
                                                                                                                    AADict[
                                                                                                                        sequence[
                                                                                                                            j + 1]] * 4 +
                                                                                                                    AADict[
                                                                                                                        sequence[
                                                                                                                            j + 2]]] + 1
                if sum(tmpCode) != 0:
                    tmpCode = [i / sum(tmpCode) for i in tmpCode]
                code = code + tmpCode
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def ANF(self):
        try:
            if not self.is_equal:
                self.error_msg = 'ANF descriptor need fasta sequence with equal length.'
                return False
            AA = 'ACGT'
            encodings = []
            header = ['SampleName', 'label']
            for i in range(1, len(self.fasta_list[0][1]) + 1):
                header.append('ANF.' + str(i))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for j in range(len(sequence)):
                    code.append(sequence[0: j + 1].count(sequence[j]) / (j + 1))
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def NCP(self):
        try:
            if not self.is_equal:
                self.error_msg = 'NCP descriptor need fasta sequence with equal length.'
                return False
            chemical_property = {
                'A': [1, 1, 1],
                'C': [0, 1, 0],
                'G': [1, 0, 0],
                'T': [0, 0, 1],
                'U': [0, 0, 1],
                '-': [0, 0, 0],
            }
            AA = 'ACGT'
            encodings = []
            header = ['SampleName', 'label']
            for i in range(1, len(self.fasta_list[0][1]) * 3 + 1):
                header.append('NCP.F' + str(i))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for aa in sequence:
                    code = code + chemical_property.get(aa, [0, 0, 0])
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def ENAC(self):
        try:
            if not self.is_equal:
                self.error_msg = 'ENAC descriptor need fasta sequence with equal length.'
                return False
            window = self.kw['sliding_window']
            if self.minimum_length < window:
                self.error_msg = 'ENAC descriptor, all the sequence length should be larger than the sliding window'
                return False
            AA = 'ACGT'
            encodings = []
            header = ['SampleName', 'label']
            for w in range(1, len(self.fasta_list[0][1]) - window + 2):
                for aa in AA:
                    header.append('SW.' + str(w) + '.' + aa)
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for j in range(len(sequence)):
                    if j < len(sequence) and j + window <= len(sequence):
                        count = Counter(sequence[j:j + window])
                        for key in count:
                            count[key] = count[key] / len(sequence[j:j + window])
                        for aa in AA:
                            code.append(count[aa])
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def binary(self):
        try:
            if not self.is_equal:
                self.error_msg = 'binary descriptor need fasta sequence with equal length.'
                return False
            AA = 'ACGT'
            encodings = []
            header = ['SampleName', 'label']
            for i in range(1, len(self.fasta_list[0][1]) * 4 + 1):
                header.append('BINARY.F' + str(i))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for aa in sequence:
                    if aa == '-':
                        code = code + [0, 0, 0, 0]
                        continue
                    for aa1 in AA:
                        tag = 1 if aa == aa1 else 0
                        code.append(tag)
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def CKSNAP(self):
        try:
            gap = self.kw['kspace']
            if self.minimum_length_without_minus < gap + 2:
                self.error_msg = 'CKSNAP - all the sequence length should be larger than the (gap value) + 2 = %s.' % (
                        gap + 2)
                return False

            AA = 'ACGT'
            encodings = []
            aaPairs = []
            for aa1 in AA:
                for aa2 in AA:
                    aaPairs.append(aa1 + aa2)

            header = ['SampleName', 'label']
            for g in range(gap + 1):
                for aa in aaPairs:
                    header.append(aa + '.gap' + str(g))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for g in range(gap + 1):
                    myDict = {}
                    for pair in aaPairs:
                        myDict[pair] = 0
                    sum = 0
                    for index1 in range(len(sequence)):
                        index2 = index1 + g + 1
                        if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[
                            index2] in AA:
                            myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1
                            sum = sum + 1
                    for pair in aaPairs:
                        code.append(myDict[pair] / sum)
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def CalculateMatrix(self, data, order):
        matrix = np.zeros((len(data[0]) - 2, 64))
        for i in range(len(data[0]) - 2):  # position
            for j in range(len(data)):
                if re.search('-', data[j][i:i + 3]):
                    pass
                else:
                    matrix[i][order[data[j][i:i + 3]]] += 1
        return matrix

    def PSTNPss(self):
        try:
            if not self.is_equal:
                self.error_msg = 'PSTNPss descriptor need fasta sequence with equal length.'
                return False

            fastas = []
            for item in self.fasta_list:
                if item[3] == 'training':
                    fastas.append(item)
                    fastas.append([item[0], item[1], item[2], 'testing'])
                else:
                    fastas.append(item)

            for i in fastas:
                if re.search('[^ACGT-]', i[1]):
                    self.error_msg = 'Illegal character included in the fasta sequences, only the "ACGT[U]" are allowed by this encoding scheme.'
                    return False

            encodings = []
            header = ['SampleName', 'label']
            for pos in range(len(fastas[0][1]) - 2):
                header.append('Pos.%d' % (pos + 1))
            encodings.append(header)

            positive = []
            negative = []
            positive_key = []
            negative_key = []
            for i in fastas:
                if i[3] == 'training':
                    if i[2] == '1':
                        positive.append(i[1])
                        positive_key.append(i[0])
                    else:
                        negative.append(i[1])
                        negative_key.append(i[0])

            nucleotides = ['A', 'C', 'G', 'T']
            trinucleotides = [n1 + n2 + n3 for n1 in nucleotides for n2 in nucleotides for n3 in nucleotides]
            order = {}
            for i in range(len(trinucleotides)):
                order[trinucleotides[i]] = i

            matrix_po = self.CalculateMatrix(positive, order)
            matrix_ne = self.CalculateMatrix(negative, order)

            positive_number = len(positive)
            negative_number = len(negative)

            for i in fastas:
                if i[3] == 'testing':
                    name, sequence, label = i[0], i[1], i[2]
                    code = [name, label]
                    for j in range(len(sequence) - 2):
                        if re.search('-', sequence[j: j + 3]):
                            code.append(0)
                        else:
                            p_num, n_num = positive_number, negative_number
                            po_number = matrix_po[j][order[sequence[j: j + 3]]]
                            if i[0] in positive_key and po_number > 0:
                                po_number -= 1
                                p_num -= 1
                            ne_number = matrix_ne[j][order[sequence[j: j + 3]]]
                            if i[0] in negative_key and ne_number > 0:
                                ne_number -= 1
                                n_num -= 1
                            code.append(po_number / p_num - ne_number / n_num)
                            # print(sequence[j: j+3], order[sequence[j: j+3]], po_number, p_num, ne_number, n_num)
                    encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def PSTNPds(self):
        try:
            if not self.is_equal:
                self.error_msg = 'PSTNPds descriptor need fasta sequence with equal length.'
                return False

            fastas = []
            for item in self.fasta_list:
                if item[3] == 'training':
                    fastas.append(item)
                    fastas.append([item[0], item[1], item[2], 'testing'])
                else:
                    fastas.append(item)

            for i in fastas:
                if re.search('[^ACGT-]', i[1]):
                    self.error_msg = 'Illegal character included in the fasta sequences, only the "ACGT[U]" are allowed by this encoding scheme.'
                    return False

            for i in fastas:
                i[1] = re.sub('T', 'A', i[1])
                i[1] = re.sub('G', 'C', i[1])

            encodings = []
            header = ['SampleName', 'label']
            for pos in range(len(fastas[0][1]) - 2):
                header.append('Pos.%d' % (pos + 1))
            encodings.append(header)

            positive = []
            negative = []
            positive_key = []
            negative_key = []
            for i in fastas:
                if i[3] == 'training':
                    if i[2] == '1':
                        positive.append(i[1])
                        positive_key.append(i[0])
                    else:
                        negative.append(i[1])
                        negative_key.append(i[0])

            nucleotides = ['A', 'C']
            trinucleotides = [n1 + n2 + n3 for n1 in nucleotides for n2 in nucleotides for n3 in nucleotides]
            order = {}
            for i in range(len(trinucleotides)):
                order[trinucleotides[i]] = i

            matrix_po = self.CalculateMatrix(positive, order)
            matrix_ne = self.CalculateMatrix(negative, order)

            positive_number = len(positive)
            negative_number = len(negative)

            for i in fastas:
                if i[3] == 'testing':
                    name, sequence, label = i[0], i[1], i[2]
                    code = [name, label]
                    for j in range(len(sequence) - 2):
                        if re.search('-', sequence[j: j + 3]):
                            code.append(0)
                        else:
                            p_num, n_num = positive_number, negative_number
                            po_number = matrix_po[j][order[sequence[j: j + 3]]]
                            if i[0] in positive_key and po_number > 0:
                                po_number -= 1
                                p_num -= 1
                            ne_number = matrix_ne[j][order[sequence[j: j + 3]]]
                            if i[0] in negative_key and ne_number > 0:
                                ne_number -= 1
                                n_num -= 1
                            code.append(po_number / p_num - ne_number / n_num)
                            # print(sequence[j: j+3], order[sequence[j: j+3]], po_number, p_num, ne_number, n_num)
                    encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def EIIP(self):
        try:
            if not self.is_equal:
                self.error_msg = 'EIIP descriptor need fasta sequence with equal length.'
                return False
            fastas = self.fasta_list
            AA = 'ACGT'
            EIIP_dict = {
                'A': 0.1260,
                'C': 0.1340,
                'G': 0.0806,
                'T': 0.1335,
                '-': 0,
            }
            encodings = []
            header = ['SampleName', 'label']
            for i in range(1, len(fastas[0][1]) + 1):
                header.append('F' + str(i))
            encodings.append(header)

            for i in fastas:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for aa in sequence:
                    code.append(EIIP_dict.get(aa, 0))
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def TriNcleotideComposition(self, sequence, base):
        trincleotides = [nn1 + nn2 + nn3 for nn1 in base for nn2 in base for nn3 in base]
        tnc_dict = {}
        for triN in trincleotides:
            tnc_dict[triN] = 0
        for i in range(len(sequence) - 2):
            tnc_dict[sequence[i:i + 3]] += 1
        for key in tnc_dict:
            tnc_dict[key] /= (len(sequence) - 2)
        return tnc_dict

    def PseEIIP(self):
        try:
            fastas = self.fasta_list
            for i in fastas:
                if re.search('[^ACGT-]', i[1]):
                    self.error_msg = 'Illegal character included in the fasta sequences, only the "ACGT-" are allowed by this PseEIIP scheme.'
                    return False
            base = 'ACGT'
            EIIP_dict = {
                'A': 0.1260,
                'C': 0.1340,
                'G': 0.0806,
                'T': 0.1335,
            }
            trincleotides = [nn1 + nn2 + nn3 for nn1 in base for nn2 in base for nn3 in base]
            EIIPxyz = {}
            for triN in trincleotides:
                EIIPxyz[triN] = EIIP_dict[triN[0]] + EIIP_dict[triN[1]] + EIIP_dict[triN[2]]

            encodings = []
            header = ['SampleName', 'label'] + trincleotides
            encodings.append(header)

            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                trincleotide_frequency = self.TriNcleotideComposition(sequence, base)
                code = code + [EIIPxyz[triN] * trincleotide_frequency[triN] for triN in trincleotides]
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def ASDC(self):
        try:
            # clear
            self.encoding_array = np.array([])

            AA = 'ACGT'
            encodings = []
            aaPairs = []
            for aa1 in AA:
                for aa2 in AA:
                    aaPairs.append(aa1 + aa2)

            header = ['SampleName', 'label']
            header += [aa1 + aa2 for aa1 in AA for aa2 in AA]
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                sum = 0
                pair_dict = {}
                for pair in aaPairs:
                    pair_dict[pair] = 0
                for j in range(len(sequence)):
                    for k in range(j + 1, len(sequence)):
                        if sequence[j] in AA and sequence[k] in AA:
                            pair_dict[sequence[j] + sequence[k]] += 1
                            sum += 1
                for pair in aaPairs:
                    code.append(pair_dict[pair] / sum)
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def DBE(self):
        try:
            # clear
            self.encoding_array = np.array([])

            if not self.is_equal:
                self.error_msg = 'DBE descriptor need fasta sequence with equal length.'
                return False

            AA_dict = {
                'AA': [0, 0, 0, 0],
                'AC': [0, 0, 0, 1],
                'AG': [0, 0, 1, 0],
                'AT': [0, 0, 1, 1],
                'CA': [0, 1, 0, 0],
                'CC': [0, 1, 0, 1],
                'CG': [0, 1, 1, 0],
                'CT': [0, 1, 1, 1],
                'GA': [1, 0, 0, 0],
                'GC': [1, 0, 0, 1],
                'GG': [1, 0, 1, 0],
                'GT': [1, 0, 1, 1],
                'TA': [1, 1, 0, 0],
                'TC': [1, 1, 0, 1],
                'TG': [1, 1, 1, 0],
                'TT': [1, 1, 1, 1],
            }

            encodings = []
            header = ['SampleName', 'label']
            for i in range((len(self.fasta_list[0][1]) - 1) * 4):
                header.append('F%s' %(i + 1))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for j in range(len(sequence) - 1):
                    if sequence[j] + sequence[j + 1] in AA_dict:
                        code += AA_dict[sequence[j] + sequence[j + 1]]
                    else:
                        code += [0.5, 0.5, 0.5, 0.5]
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def LPDF(self):
        try:
            # clear
            self.encoding_array = np.array([])

            if not self.is_equal:
                self.error_msg = 'LPDF descriptor need fasta sequence with equal length.'
                return False

            encodings = []
            header = ['SampleName', 'label']
            header += ['LPDF_pos%s' % i for i in range(1, len(self.fasta_list[0][1]))]
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                dinucleotide_dict = {
                    'AA': 0,
                    'AC': 0,
                    'AG': 0,
                    'AT': 0,
                    'A-': 0,
                    'CA': 0,
                    'CC': 0,
                    'CG': 0,
                    'CT': 0,
                    'C-': 0,
                    'GA': 0,
                    'GC': 0,
                    'GG': 0,
                    'GT': 0,
                    'G-': 0,
                    'TA': 0,
                    'TC': 0,
                    'TG': 0,
                    'TT': 0,
                    'T-': 0,
                    '-A': 0,
                    '-C': 0,
                    '-G': 0,
                    '-T': 0,
                    '--': 0,
                }
                for j in range(1, len(sequence)):
                    dinucleotide_dict[sequence[j] + sequence[j - 1]] += 1
                    code.append(dinucleotide_dict[sequence[j] + sequence[j - 1]] / j)
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def DPCP(self):
        try:
            # clear
            self.encoding_array = np.array([])

            if self.sequence_type == 'DNA':
                file_name = 'didnaPhyche.data'
                property_name = self.kw['Di-DNA-Phychem'].split(';')
            else:
                file_name = 'dirnaPhyche.data'
                property_name = self.kw['Di-RNA-Phychem'].split(';')

            try:
                data_file = os.path.split(os.path.realpath(__file__))[0] + r'\data\%s' %file_name if platform.system() == 'Windows' else os.path.split(os.path.realpath(__file__))[0] + r'/data/%s' %file_name
                with open(data_file, 'rb') as handle:
                    property_dict = pickle.load(handle)
            except Exception as e:
                self.error_msg = 'Could not find the physicochemical properties file.'
                return False

            base = 'ACGT'
            encodings = []
            dinucleotides = [n1 + n2 + '_' + p_name for p_name in property_name for n1 in base for n2 in base]
            header = ['SampleName', 'label'] + dinucleotides
            encodings.append(header)

            AADict = {}
            for i in range(len(base)):
                AADict[base[i]] = i

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                tmpCode = [0] * 16
                for j in range(len(sequence) - 2 + 1):
                    tmpCode[AADict[sequence[j]] * 4 + AADict[sequence[j + 1]]] = tmpCode[AADict[sequence[j]] * 4 + AADict[
                        sequence[j + 1]]] + 1
                if sum(tmpCode) != 0:
                    tmpCode = [i / sum(tmpCode) for i in tmpCode]

                for p_name in property_name:
                    normalized_code = []
                    for j in range(len(tmpCode)):
                        normalized_code.append(tmpCode[j] * float(property_dict[p_name][j]))
                    code += normalized_code
                encodings.append(code)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def DPCP_type2(self):
        try:
            # clear
            self.encoding_array = np.array([])

            if not self.is_equal:
                self.error_msg = 'DPCP type2 descriptor need fasta sequence with equal length.'
                return False

            if self.sequence_type == 'DNA':
                file_name = 'didnaPhyche.data'
                property_name = self.kw['Di-DNA-Phychem'].split(';')
            else:
                file_name = 'dirnaPhyche.data'
                property_name = self.kw['Di-RNA-Phychem'].split(';')

            try:
                data_file = os.path.split(os.path.realpath(__file__))[0] + r'\data\%s' %file_name if platform.system() == 'Windows' else os.path.split(os.path.realpath(__file__))[0] + r'/data/%s' %file_name
                with open(data_file, 'rb') as handle:
                    property_dict = pickle.load(handle)
            except Exception as e:
                self.error_msg = 'Could not find the physicochemical properties file.'
                return False

            AA = 'ACGT'
            encodings = []
            header = ['SampleName', 'label']
            for i in range(len(self.fasta_list[0][1]) - 1):
                for p_name in property_name:
                    header.append(p_name + '_pos%s' %(i+1))
            encodings.append(header)

            AADict = {}
            AA_list = [aa1 + aa2 for aa1 in AA for aa2 in AA]
            for i in range(len(AA_list)):
                AADict[AA_list[i]] = i

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for p_name in property_name:
                    for j in range(len(sequence) - 1):
                        if sequence[j: j + 2] in AADict:
                            code.append(property_dict[p_name][AADict[sequence[j: j + 2]]])
                        else:
                            code.append(0)
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def TPCP(self):
        try:
            # clear
            self.encoding_array = np.array([])

            file_name = 'tridnaPhyche.data'
            property_name = self.kw['Tri-DNA-Phychem'].split(';')

            try:
                data_file = os.path.split(os.path.realpath(__file__))[0] + r'\data\%s' %file_name if platform.system() == 'Windows' else os.path.split(os.path.realpath(__file__))[0] + r'/data/%s' %file_name
                with open(data_file, 'rb') as handle:
                    property_dict = pickle.load(handle)
                property_name = property_dict.keys()
            except Exception as e:
                self.error_msg = 'Could not find the physicochemical properties file.'
                return False

            AA = 'ACGT'
            encodings = []
            triPeptides = [aa1 + aa2 + aa3 + '_' + p_name for p_name in property_name for aa1 in AA for aa2 in AA for aa3 in
                        AA]
            header = ['SampleName', 'label'] + triPeptides
            encodings.append(header)

            AADict = {}
            for i in range(len(AA)):
                AADict[AA[i]] = i

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                tmpCode = [0] * 64
                for j in range(len(sequence) - 3 + 1):
                    tmpCode[AADict[sequence[j]] * 16 + AADict[sequence[j + 1]] * 4 + AADict[sequence[j + 2]]] = tmpCode[
                                                                                                                    AADict[
                                                                                                                        sequence[
                                                                                                                            j]] * 16 +
                                                                                                                    AADict[
                                                                                                                        sequence[
                                                                                                                            j + 1]] * 4 +
                                                                                                                    AADict[
                                                                                                                        sequence[
                                                                                                                            j + 2]]] + 1
                if sum(tmpCode) != 0:
                    tmpCode = [i / sum(tmpCode) for i in tmpCode]

                for p_name in property_name:
                    normalized_code = []
                    for j in range(len(tmpCode)):
                        normalized_code.append(tmpCode[j] * float(property_dict[p_name][j]))
                    code += normalized_code
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def TPCP_type2(self):
        try:
            # clear
            self.encoding_array = np.array([])

            if not self.is_equal:
                self.error_msg = 'TPCP type2 descriptor need fasta sequence with equal length.'
                return False

            if self.sequence_type == 'DNA':
                file_name = 'tridnaPhyche.data'
                property_name = self.kw['Tri-DNA-Phychem'].split(';')

            try:
                data_file = os.path.split(os.path.realpath(__file__))[0] + r'\data\%s' %file_name if platform.system() == 'Windows' else os.path.split(os.path.realpath(__file__))[0] + r'/data/%s' %file_name
                with open(data_file, 'rb') as handle:
                    property_dict = pickle.load(handle)
            except Exception as e:
                self.error_msg = 'Could not find the physicochemical properties file.'
                return False

            AA = 'ACGT'
            encodings = []
            header = ['SampleName', 'label']
            for i in range(len(self.fasta_list[0][1]) - 2):
                for p_name in property_name:
                    header.append(p_name + '_pos%s' %(i+1))
            encodings.append(header)

            AADict = {}
            AA_list = [aa1 + aa2 + aa3 for aa1 in AA for aa2 in AA for aa3 in AA]
            for i in range(len(AA_list)):
                AADict[AA_list[i]] = i

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for p_name in property_name:
                    for j in range(len(sequence) - 2):
                        if sequence[j: j + 3] in AADict:
                            code.append(property_dict[p_name][AADict[sequence[j: j + 3]]])
                        else:
                            code.append(0)
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def MMI(self):
        try:
            NA = 'ACGT'
            dinucleotide_list = [a1 + a2 for a1 in NA for a2 in NA]
            trinucleotide_list = [a1 + a2 + a3 for a1 in NA for a2 in NA for a3 in NA]
            dinucleotide_dict = {}
            trinucleotide_dict = {}
            for elem in dinucleotide_list:
                dinucleotide_dict[''.join(sorted(elem))] = 0
            for elem in trinucleotide_list:
                trinucleotide_dict[''.join(sorted(elem))] = 0

            encodings = []
            header = ['SampleName', 'label']
            header += ['MMI_%s' % elem for elem in sorted(dinucleotide_dict.keys())]
            header += ['MMI_%s' % elem for elem in sorted(trinucleotide_dict.keys())]
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                f1_dict = {
                    'A': 0,
                    'C': 0,
                    'G': 0,
                    'T': 0,
                }
                f2_dict = dinucleotide_dict.copy()
                f3_dict = trinucleotide_dict.copy()

                for elem in sequence:
                    if elem in f1_dict:
                        f1_dict[elem] += 1
                for key in f1_dict:
                    f1_dict[key] /= len(sequence)

                for i in range(len(sequence) - 1):
                    if ''.join(sorted(sequence[i: i + 2])) in f2_dict:
                        f2_dict[''.join(sorted(sequence[i: i + 2]))] += 1
                for key in f2_dict:
                    f2_dict[key] /= (len(sequence) - 1)

                for i in range(len(sequence) - 2):
                    if ''.join(sorted(sequence[i: i + 3])) in f3_dict:
                        f3_dict[''.join(sorted(sequence[i: i + 3]))] += 1
                for key in f3_dict:
                    f3_dict[key] /= (len(sequence) - 2)

                for key in sorted(f2_dict.keys()):
                    if f2_dict[key] != 0 and f1_dict[key[0]] * f1_dict[key[1]] != 0:
                        code.append(f2_dict[key] * math.log(f2_dict[key] / (f1_dict[key[0]] * f1_dict[key[1]])))
                    else:
                        code.append(0)
                for key in sorted(f3_dict.keys()):
                    element_1 = 0
                    element_2 = 0
                    element_3 = 0
                    if f2_dict[key[0:2]] != 0 and f1_dict[key[0]] * f1_dict[key[1]] != 0:
                        element_1 = f2_dict[key[0:2]] * math.log(f2_dict[key[0:2]] / (f1_dict[key[0]] * f1_dict[key[1]]))
                    if f2_dict[key[0] + key[2]] != 0 and f1_dict[key[2]] != 0:
                        element_2 = (f2_dict[key[0] + key[2]] / f1_dict[key[2]]) * math.log(
                            f2_dict[key[0] + key[2]] / f1_dict[key[2]])
                    if f2_dict[key[1:3]] != 0 and f3_dict[key] / f2_dict[key[1:3]] != 0:
                        element_3 = (f3_dict[key] / f2_dict[key[1:3]]) * math.log(f3_dict[key] / f2_dict[key[1:3]])
                    code.append(element_1 + element_2 - element_3)
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    ''' DNA/RNA KNN descriptor '''
    def SimN(self, a, b):
        score_matrix = [
            [ 2, -1, -1, -1, -1],  # A
            [-1,  2, -1, -1, -1],  # C
            [-1, -1,  2, -1, -1],  # G
            [-1, -1, -1,  2, -1],  # T
            [-1, -1, -1, -1,  2],  # -
        ]
        AA = 'ACGT-'
        myDict = {}
        for i in range(len(AA)):
            myDict[AA[i]] = i
        maxValue, minValue = 2, -1
        return (score_matrix[myDict[a]][myDict[b]] - minValue) / (maxValue - minValue)

    def CalculateDistanceN(self, sequence1, sequence2):
        if len(sequence1) != len(sequence2):
            self.error_msg = 'KNN descriptor need fasta sequence with equal length.'
            return 1
        distance = 1 - sum([self.SimN(sequence1[i], sequence2[i]) for i in range(len(sequence1))]) / len(sequence1)
        return distance

    def KNN(self):
        try:
            # clear
            self.encoding_array = np.array([])

            if not self.is_equal:
                self.error_msg = 'KNN descriptor need fasta sequence with equal length.'
                return False

            topK_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16,
                        0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30]

            training_data = []
            training_label = {}
            for i in self.fasta_list:
                if i[3] == 'training':
                    training_data.append(i)
                    training_label[i[0]] = int(i[2])
            tmp_label_sets = list(set(training_label.values()))

            topK_numbers = []
            for i in topK_values:
                topK_numbers.append(math.ceil(len(training_data) * i))

            # calculate pair distance
            distance_dict = {}
            for i in range(len(self.fasta_list)):
                name_seq1, sequence_1, label_1, usage_1 = self.fasta_list[i][0], self.fasta_list[i][1], self.fasta_list[i][2], self.fasta_list[i][3]
                for j in range(i+1, len(self.fasta_list)):
                    name_seq2, sequence_2, label_2, usage_2 = self.fasta_list[j][0], self.fasta_list[j][1], self.fasta_list[j][2], self.fasta_list[j][3]
                    if usage_1 == 'testing' and usage_2 == 'testing':
                        continue
                    else:
                        distance_dict[':'.join(sorted([name_seq1, name_seq2]))] = self.CalculateDistanceN(sequence_1, sequence_2)

            encodings = []
            header = ['#', 'label']
            for k in topK_numbers:
                for l in tmp_label_sets:
                    header.append('Top' + str(k) + '.label' + str(l))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                tmp_distance_list = []
                for j in range(len(training_data)):
                    if name != training_data[j][0]:
                        tmp_distance_list.append([int(training_data[j][2]), distance_dict.get(':'.join(sorted([name, training_data[j][0]])), 1)])

                tmp_distance_list = np.array(tmp_distance_list)
                tmp_distance_list = tmp_distance_list[np.lexsort(tmp_distance_list.T)]

                for j in topK_numbers:
                    code += self.CalculateContent(tmp_distance_list, j, tmp_label_sets)
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    ''' ending DNA/RNA KNN descriptor '''

    def PS2(self):
        try:
            if not self.is_equal:
                self.error_msg = 'PS2 descriptor need fasta sequence with equal length.'
                return False
            AA = 'ACGT'
            AA_list = [a1 + a2 for a1 in AA for a2 in AA]
            AA_dict = {}
            for i in range(len(AA_list)):
                AA_dict[AA_list[i]] = [0] * 16
                AA_dict[AA_list[i]][i] = 1

            encodings = []
            header = ['SampleName', 'label']
            for i in range(1, (len(self.fasta_list[0][1]) - 1) * 16 + 1):
                header.append('BINARY.F' + str(i))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for j in range(len(sequence) - 1):
                    code += AA_dict.get(sequence[j: j+2], [0]*16)
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def PS3(self):
        try:
            if not self.is_equal:
                self.error_msg = 'PS3 descriptor need fasta sequence with equal length.'
                return False
            AA = 'ACGT'
            AA_list = [a1 + a2 + a3 for a1 in AA for a2 in AA for a3 in AA]
            AA_dict = {}
            for i in range(len(AA_list)):
                AA_dict[AA_list[i]] = [0] * 64
                AA_dict[AA_list[i]][i] = 1

            encodings = []
            header = ['SampleName', 'label']
            for i in range(1, (len(self.fasta_list[0][1]) - 2) * 64 + 1):
                header.append('BINARY.F' + str(i))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for j in range(len(sequence) - 2):
                    code += AA_dict.get(sequence[j: j+3], [0]*64)
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def PS4(self):
        try:
            if not self.is_equal:
                self.error_msg = 'PS4 descriptor need fasta sequence with equal length.'
                return False
            AA = 'ACGT'
            AA_list = [a1 + a2 + a3 + a4 for a1 in AA for a2 in AA for a3 in AA for a4 in AA]
            AA_dict = {}
            for i in range(len(AA_list)):
                AA_dict[AA_list[i]] = [0] * 256
                AA_dict[AA_list[i]][i] = 1

            encodings = []
            header = ['SampleName', 'label']
            for i in range(1, (len(self.fasta_list[0][1]) - 3) * 256 + 1):
                header.append('BINARY.F' + str(i))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for j in range(len(sequence) - 3):
                    code += AA_dict.get(sequence[j: j+4], [0]*256)
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Z_curve_9bit(self):
        try:
            encodings = []
            header = ['SampleName', 'label']
            for pos in range(1, 4):
                for elem in ['x', 'y', 'z']:
                    header.append('Pos_%s.%s' %(pos, elem))
            encodings.append(header)

            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                code = [name, label]
                pos1_dict = {}
                pos2_dict = {}
                pos3_dict = {}
                for i in range(len(sequence)):
                    if (i+1) % 3 == 1:
                        if sequence[i] in pos1_dict:
                            pos1_dict[sequence[i]] += 1
                        else:
                            pos1_dict[sequence[i]] = 1
                    elif (i+1) % 3 == 2:
                        if sequence[i] in pos2_dict:
                            pos2_dict[sequence[i]] += 1
                        else:
                            pos2_dict[sequence[i]] = 1
                    elif (i+1) % 3 == 0:
                        if sequence[i] in pos3_dict:
                            pos3_dict[sequence[i]] += 1
                        else:
                            pos3_dict[sequence[i]] = 1

                code += [
                    (pos1_dict.get('A', 0) + pos1_dict.get('G', 0) - pos1_dict.get('C', 0) - pos1_dict.get('T', 0)) / len(sequence), # x
                    (pos1_dict.get('A', 0) + pos1_dict.get('C', 0) - pos1_dict.get('G', 0) - pos1_dict.get('T', 0)) / len(sequence), # y
                    (pos1_dict.get('A', 0) + pos1_dict.get('T', 0) - pos1_dict.get('G', 0) - pos1_dict.get('C', 0)) / len(sequence)  # z
                    ]
                code += [
                    (pos2_dict.get('A', 0) + pos2_dict.get('G', 0) - pos2_dict.get('C', 0) - pos2_dict.get('T', 0)) / len(sequence),
                    (pos2_dict.get('A', 0) + pos2_dict.get('C', 0) - pos2_dict.get('G', 0) - pos2_dict.get('T', 0)) / len(sequence),
                    (pos2_dict.get('A', 0) + pos2_dict.get('T', 0) - pos2_dict.get('G', 0) - pos2_dict.get('C', 0)) / len(sequence)
                    ]
                code += [
                    (pos3_dict.get('A', 0) + pos3_dict.get('G', 0) - pos3_dict.get('C', 0) - pos3_dict.get('T', 0)) / len(sequence),
                    (pos3_dict.get('A', 0) + pos3_dict.get('C', 0) - pos3_dict.get('G', 0) - pos3_dict.get('T', 0)) / len(sequence),
                    (pos3_dict.get('A', 0) + pos3_dict.get('T', 0) - pos3_dict.get('G', 0) - pos3_dict.get('C', 0)) / len(sequence)
                    ]
                encodings.append(code)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Z_curve_12bit(self):
        try:
            NN = 'ACGT'
            encodings = []
            header = ['SampleName', 'label']
            for base in NN:
                for elem in ['x', 'y', 'z']:
                    header.append('%s.%s' %(base, elem))
            encodings.append(header)

            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                code = [name, label]
                pos_dict = {}
                for i in range(len(sequence) - 1):
                    if sequence[i: i+2] in pos_dict:
                        pos_dict[sequence[i: i+2]] += 1
                    else:
                        pos_dict[sequence[i: i+2]] = 1

                for base in NN:
                    code += [
                        (pos_dict.get('%sA' %base, 0) + pos_dict.get('%sG' %base, 0) - pos_dict.get('%sC' %base, 0) - pos_dict.get('%sT' %base, 0)) / (len(sequence) - 1), # x
                        (pos_dict.get('%sA' %base, 0) + pos_dict.get('%sC' %base, 0) - pos_dict.get('%sG' %base, 0) - pos_dict.get('%sT' %base, 0)) / (len(sequence) - 1), # y
                        (pos_dict.get('%sA' %base, 0) + pos_dict.get('%sT' %base, 0) - pos_dict.get('%sG' %base, 0) - pos_dict.get('%sC' %base, 0)) / (len(sequence) - 1)  # z
                        ]

                encodings.append(code)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Z_curve_36bit(self):
        try:
            NN = 'ACGT'
            encodings = []
            header = ['SampleName', 'label']

            for base in NN:
                for pos in range(1, 4):
                    for elem in ['x', 'y', 'z']:
                        header.append('Pos_%s_%s.%s' %(pos, base, elem))
            encodings.append(header)

            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                code = [name, label]
                pos1_dict = {}
                pos2_dict = {}
                pos3_dict = {}
                for i in range(len(sequence) - 1):
                    if (i+1) % 3 == 1:
                        if sequence[i: i+2] in pos1_dict:
                            pos1_dict[sequence[i: i+2]] += 1
                        else:
                            pos1_dict[sequence[i: i+2]] = 1
                    elif (i+1) % 3 == 2:
                        if sequence[i: i+2] in pos2_dict:
                            pos2_dict[sequence[i: i+2]] += 1
                        else:
                            pos2_dict[sequence[i: i+2]] = 1
                    elif (i+1) % 3 == 0:
                        if sequence[i: i+2] in pos3_dict:
                            pos3_dict[sequence[i: i+2]] += 1
                        else:
                            pos3_dict[sequence[i: i+2]] = 1

                for base in NN:
                    code += [
                        (pos1_dict.get('%sA' %base, 0) + pos1_dict.get('%sG' %base, 0) - pos1_dict.get('%sC' %base, 0) - pos1_dict.get('%sT' %base, 0)) / (len(sequence) - 1), # x
                        (pos1_dict.get('%sA' %base, 0) + pos1_dict.get('%sC' %base, 0) - pos1_dict.get('%sG' %base, 0) - pos1_dict.get('%sT' %base, 0)) / (len(sequence) - 1), # y
                        (pos1_dict.get('%sA' %base, 0) + pos1_dict.get('%sT' %base, 0) - pos1_dict.get('%sG' %base, 0) - pos1_dict.get('%sC' %base, 0)) / (len(sequence) - 1)  # z
                        ]
                    code += [
                        (pos2_dict.get('%sA' %base, 0) + pos2_dict.get('%sG' %base, 0) - pos2_dict.get('%sC' %base, 0) - pos2_dict.get('%sT' %base, 0)) / (len(sequence) - 1),
                        (pos2_dict.get('%sA' %base, 0) + pos2_dict.get('%sC' %base, 0) - pos2_dict.get('%sG' %base, 0) - pos2_dict.get('%sT' %base, 0)) / (len(sequence) - 1),
                        (pos2_dict.get('%sA' %base, 0) + pos2_dict.get('%sT' %base, 0) - pos2_dict.get('%sG' %base, 0) - pos2_dict.get('%sC' %base, 0)) / (len(sequence) - 1)
                        ]
                    code += [
                        (pos3_dict.get('%sA' %base, 0) + pos3_dict.get('%sG' %base, 0) - pos3_dict.get('%sC' %base, 0) - pos3_dict.get('%sT' %base, 0)) / (len(sequence) - 1),
                        (pos3_dict.get('%sA' %base, 0) + pos3_dict.get('%sC' %base, 0) - pos3_dict.get('%sG' %base, 0) - pos3_dict.get('%sT' %base, 0)) / (len(sequence) - 1),
                        (pos3_dict.get('%sA' %base, 0) + pos3_dict.get('%sT' %base, 0) - pos3_dict.get('%sG' %base, 0) - pos3_dict.get('%sC' %base, 0)) / (len(sequence) - 1)
                        ]
                encodings.append(code)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Z_curve_48bit(self):
        try:
            NN = 'ACGT'
            encodings = []
            header = ['SampleName', 'label']
            for base in NN:
                for base1 in NN:
                    for elem in ['x', 'y', 'z']:
                        header.append('%s%s.%s' %(base, base1, elem))
            encodings.append(header)

            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                code = [name, label]
                pos_dict = {}
                for i in range(len(sequence) - 2):
                    if sequence[i: i+3] in pos_dict:
                        pos_dict[sequence[i: i+3]] += 1
                    else:
                        pos_dict[sequence[i: i+3]] = 1

                for base in NN:
                    for base1 in NN:
                        code += [
                            (pos_dict.get('%s%sA' %(base, base1), 0) + pos_dict.get('%s%sG' %(base, base1), 0) - pos_dict.get('%s%sC' %(base, base1), 0) - pos_dict.get('%s%sT' %(base, base1), 0)) / (len(sequence) - 2), # x
                            (pos_dict.get('%s%sA' %(base, base1), 0) + pos_dict.get('%s%sC' %(base, base1), 0) - pos_dict.get('%s%sG' %(base, base1), 0) - pos_dict.get('%s%sT' %(base, base1), 0)) / (len(sequence) - 2), # y
                            (pos_dict.get('%s%sA' %(base, base1), 0) + pos_dict.get('%s%sT' %(base, base1), 0) - pos_dict.get('%s%sG' %(base, base1), 0) - pos_dict.get('%s%sC' %(base, base1), 0)) / (len(sequence) - 2)  # z
                            ]
                encodings.append(code)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Z_curve_144bit(self):
        try:
            NN = 'ACGT'
            encodings = []
            header = ['SampleName', 'label']

            for base in NN:
                for base1 in NN:
                    for pos in range(1, 4):
                        for elem in ['x', 'y', 'z']:
                            header.append('Pos_%s_%s%s.%s' %(pos, base, base1, elem))
            encodings.append(header)

            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                code = [name, label]
                pos1_dict = {}
                pos2_dict = {}
                pos3_dict = {}
                for i in range(len(sequence) - 2):
                    if (i+1) % 3 == 1:
                        if sequence[i: i+3] in pos1_dict:
                            pos1_dict[sequence[i: i+3]] += 1
                        else:
                            pos1_dict[sequence[i: i+3]] = 1
                    elif (i+1) % 3 == 2:
                        if sequence[i: i+3] in pos2_dict:
                            pos2_dict[sequence[i: i+3]] += 1
                        else:
                            pos2_dict[sequence[i: i+3]] = 1
                    elif (i+1) % 3 == 0:
                        if sequence[i: i+3] in pos3_dict:
                            pos3_dict[sequence[i: i+3]] += 1
                        else:
                            pos3_dict[sequence[i: i+3]] = 1

                for base in NN:
                    for base1 in NN:
                        code += [
                            (pos1_dict.get('%s%sA' %(base, base1), 0) + pos1_dict.get('%s%sG' %(base, base1), 0) - pos1_dict.get('%s%sC' %(base, base1), 0) - pos1_dict.get('%s%sT' %(base, base1), 0)) / (len(sequence) - 2), # x
                            (pos1_dict.get('%s%sA' %(base, base1), 0) + pos1_dict.get('%s%sC' %(base, base1), 0) - pos1_dict.get('%s%sG' %(base, base1), 0) - pos1_dict.get('%s%sT' %(base, base1), 0)) / (len(sequence) - 2), # y
                            (pos1_dict.get('%s%sA' %(base, base1), 0) + pos1_dict.get('%s%sT' %(base, base1), 0) - pos1_dict.get('%s%sG' %(base, base1), 0) - pos1_dict.get('%s%sC' %(base, base1), 0)) / (len(sequence) - 2)  # z
                            ]
                        code += [
                            (pos2_dict.get('%s%sA' %(base, base1), 0) + pos2_dict.get('%s%sG' %(base, base1), 0) - pos2_dict.get('%s%sC' %(base, base1), 0) - pos2_dict.get('%s%sT' %(base, base1), 0)) / (len(sequence) - 2), # x
                            (pos2_dict.get('%s%sA' %(base, base1), 0) + pos2_dict.get('%s%sC' %(base, base1), 0) - pos2_dict.get('%s%sG' %(base, base1), 0) - pos2_dict.get('%s%sT' %(base, base1), 0)) / (len(sequence) - 2), # y
                            (pos2_dict.get('%s%sA' %(base, base1), 0) + pos2_dict.get('%s%sT' %(base, base1), 0) - pos2_dict.get('%s%sG' %(base, base1), 0) - pos2_dict.get('%s%sC' %(base, base1), 0)) / (len(sequence) - 2)  # z
                            ]
                        code += [
                            (pos3_dict.get('%s%sA' %(base, base1), 0) + pos3_dict.get('%s%sG' %(base, base1), 0) - pos3_dict.get('%s%sC' %(base, base1), 0) - pos3_dict.get('%s%sT' %(base, base1), 0)) / (len(sequence) - 2), # x
                            (pos3_dict.get('%s%sA' %(base, base1), 0) + pos3_dict.get('%s%sC' %(base, base1), 0) - pos3_dict.get('%s%sG' %(base, base1), 0) - pos3_dict.get('%s%sT' %(base, base1), 0)) / (len(sequence) - 2), # y
                            (pos3_dict.get('%s%sA' %(base, base1), 0) + pos3_dict.get('%s%sT' %(base, base1), 0) - pos3_dict.get('%s%sG' %(base, base1), 0) - pos3_dict.get('%s%sC' %(base, base1), 0)) / (len(sequence) - 2)  # z
                            ]
                encodings.append(code)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def NMBroto(self):
        try:
            self.encoding_array = np.array([])

            if self.sequence_type == 'DNA':
                file_name = 'didnaPhyche.data'
                property_name = self.kw['Di-DNA-Phychem'].split(';')
            else:
                file_name = 'dirnaPhyche.data'
                property_name = self.kw['Di-RNA-Phychem'].split(';')
            try:
                data_file = os.path.split(os.path.realpath(__file__))[0] + r'\data\%s' %file_name if platform.system() == 'Windows' else os.path.split(os.path.realpath(__file__))[0] + r'/data/%s' %file_name
                with open(data_file, 'rb') as handle:
                    property_dict = pickle.load(handle)
            except Exception as e:
                self.error_msg = 'Could not find the physicochemical properties file.'
                return False
            nlag = self.kw['nlag']

            # value normalization
            for p_name in property_name:
                tmp = np.array(property_dict[p_name], dtype=float)
                pmean = np.average(tmp)
                pstd = np.std(tmp)
                property_dict[p_name] = [(elem - pmean) / pstd for elem in tmp]

            base = 'ACGT'
            encodings = []
            header = ['SampleName', 'label']
            for p_name in property_name:
                for d in range(1, nlag + 1):
                    header.append(p_name + '.lag' + str(d))
            encodings.append(header)

            AADict = {}
            AA_list = [aa1 + aa2 for aa1 in base for aa2 in base]
            for i in range(len(AA_list)):
                AADict[AA_list[i]] = i

            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                code = [name, label]
                N = len(sequence) - 1
                for p_name in property_name:
                    for d in range(1, nlag + 1):
                        try:
                            if N > nlag:
                                atsd = sum([float(property_dict[p_name][AADict[sequence[j: j+2]]]) * float(property_dict[p_name][AADict[sequence[j+d: j+d+2]]]) for j in range(N-d)]) / (N - d)
                            else:
                                atsd = 0
                        except Exception as e:
                            atsd = 0
                        code.append(atsd)
                encodings.append(code)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Moran(self):
        try:
            self.encoding_array = np.array([])

            if self.sequence_type == 'DNA':
                file_name = 'didnaPhyche.data'
                property_name = self.kw['Di-DNA-Phychem'].split(';')
            else:
                file_name = 'dirnaPhyche.data'
                property_name = self.kw['Di-RNA-Phychem'].split(';')
            try:
                data_file = os.path.split(os.path.realpath(__file__))[
                                0] + r'\data\%s' % file_name if platform.system() == 'Windows' else \
                os.path.split(os.path.realpath(__file__))[0] + r'/data/%s' % file_name
                with open(data_file, 'rb') as handle:
                    property_dict = pickle.load(handle)
            except Exception as e:
                self.error_msg = 'Could not find the physicochemical properties file.'
                return False
            nlag = self.kw['nlag']

            # value normalization
            for p_name in property_name:
                tmp = np.array(property_dict[p_name], dtype=float)
                pmean = np.average(tmp)
                pstd = np.std(tmp)
                property_dict[p_name] = [(elem - pmean) / pstd for elem in tmp]

            base = 'ACGT'
            encodings = []
            header = ['SampleName', 'label']
            for p_name in property_name:
                for d in range(1, nlag + 1):
                    header.append(p_name + '.lag' + str(d))
            encodings.append(header)

            AADict = {}
            AA_list = [aa1 + aa2 for aa1 in base for aa2 in base]
            for i in range(len(AA_list)):
                AADict[AA_list[i]] = i

            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                code = [name, label]
                N = len(sequence) - 1
                for p_name in property_name:
                    pmean = sum([property_dict[p_name][AADict[sequence[i: i+2]]] for i in range(N)]) / N
                    for d in range(1, nlag + 1):
                        try:
                            Idup = sum([(property_dict[p_name][AADict[sequence[j: j+2]]] - pmean) * (property_dict[p_name][AADict[sequence[j+d: j+d+2]]] - pmean) for j in range(N-d)]) / (N - d)
                            Iddown = sum([(property_dict[p_name][AADict[sequence[j: j+2]]] - pmean) ** 2 for j in range(N-d)]) / N
                            code.append(Idup/Iddown)
                        except Exception as e:
                            code.append(0)
                encodings.append(code)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Geary(self):
        try:
            self.encoding_array = np.array([])

            if self.sequence_type == 'DNA':
                file_name = 'didnaPhyche.data'
                property_name = self.kw['Di-DNA-Phychem'].split(';')
            else:
                file_name = 'dirnaPhyche.data'
                property_name = self.kw['Di-RNA-Phychem'].split(';')
            try:
                data_file = os.path.split(os.path.realpath(__file__))[
                                0] + r'\data\%s' % file_name if platform.system() == 'Windows' else \
                    os.path.split(os.path.realpath(__file__))[0] + r'/data/%s' % file_name
                with open(data_file, 'rb') as handle:
                    property_dict = pickle.load(handle)
            except Exception as e:
                self.error_msg = 'Could not find the physicochemical properties file.'
                return False
            nlag = self.kw['nlag']

            # value normalization
            for p_name in property_name:
                tmp = np.array(property_dict[p_name], dtype=float)
                pmean = np.average(tmp)
                pstd = np.std(tmp)
                property_dict[p_name] = [(elem - pmean) / pstd for elem in tmp]

            base = 'ACGT'
            encodings = []
            header = ['SampleName', 'label']
            for p_name in property_name:
                for d in range(1, nlag + 1):
                    header.append(p_name + '.lag' + str(d))
            encodings.append(header)

            AADict = {}
            AA_list = [aa1 + aa2 for aa1 in base for aa2 in base]
            for i in range(len(AA_list)):
                AADict[AA_list[i]] = i

            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                code = [name, label]
                N = len(sequence) - 1
                for p_name in property_name:
                    pmean = sum([property_dict[p_name][AADict[sequence[i: i + 2]]] for i in range(N)]) / N
                    for d in range(1, nlag + 1):
                        try:
                            Cdup = sum([(property_dict[p_name][AADict[sequence[j: j+2]]] - property_dict[p_name][AADict[sequence[j+d: j+d+2]]]) ** 2 for j in range(N - d)]) / (2 * (N - d))
                            Cddown = sum([(property_dict[p_name][AADict[sequence[j: j+2]]] - pmean) ** 2 for j in range(N - d)]) / (N - 1)
                            code.append(Cdup / Cddown)
                        except Exception as e:
                            code.append(0)
                encodings.append(code)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def generatePropertyPairs(self, myPropertyName):
        pairs = []
        for i in range(len(myPropertyName)):
            for j in range(i + 1, len(myPropertyName)):
                pairs.append([myPropertyName[i], myPropertyName[j]])
                pairs.append([myPropertyName[j], myPropertyName[i]])
        return pairs

    def make_ac_vector(self, myPropertyName, myPropertyValue, kmer):
        try:
            fastas = self.fasta_list
            lag = self.kw['nlag']

            encodings = []
            myIndex = self.myDiIndex if kmer == 2 else self.myTriIndex
            header = ['SampleName', 'label']
            for p in myPropertyName:
                for l in range(1, lag + 1):
                    header.append('%s.lag%d' % (p, l))
            encodings.append(header)

            for i in fastas:
                # print(i[0])
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]

                for p in myPropertyName:
                    meanValue = 0
                    # for j in range(len(sequence) - kmer):
                    for j in range(len(sequence) - kmer + 1):
                        meanValue = meanValue + float(myPropertyValue[p][myIndex[sequence[j: j + kmer]]])
                    # meanValue = meanValue / (len(sequence) - kmer)
                    meanValue = meanValue / (len(sequence) - kmer + 1)

                    for l in range(1, lag + 1):
                        acValue = 0
                        for j in range(len(sequence) - kmer - l + 1):
                            # acValue = acValue + (float(myPropertyValue[p][myIndex[sequence[j: j+kmer]]]) - meanValue) * (float(myPropertyValue[p][myIndex[sequence[j+l:j+l+kmer]]]))
                            acValue = acValue + (float(myPropertyValue[p][myIndex[sequence[j: j + kmer]]]) - meanValue) * (
                                    float(myPropertyValue[p][myIndex[sequence[j + l:j + l + kmer]]]) - meanValue)
                        acValue = acValue / (len(sequence) - kmer - l + 1)
                        # print(acValue)
                        code.append(acValue)
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def make_cc_vector(self, myPropertyName, myPropertyValue, kmer):
        try:
            fastas = self.fasta_list
            lag = self.kw['nlag']

            encodings = []
            myIndex = self.myDiIndex if kmer == 2 else self.myTriIndex
            if len(myPropertyName) < 2:
                self.error_msg = 'two or more property are needed for cross covariance (i.e. DCC and TCC) descriptors'
                return False
            propertyPairs = self.generatePropertyPairs(myPropertyName)
            header = ['SampleName', 'label'] + [n[0] + '-' + n[1] + '-lag.' + str(l) for n in propertyPairs for l in
                                                range(1, lag + 1)]
            encodings.append(header)

            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]

                for pair in propertyPairs:
                    meanP1 = 0
                    meanP2 = 0
                    # for j in range(len(sequence) - kmer):
                    for j in range(len(sequence) - kmer + 1):
                        meanP1 = meanP1 + float(myPropertyValue[pair[0]][myIndex[sequence[j: j + kmer]]])
                        meanP2 = meanP2 + float(myPropertyValue[pair[1]][myIndex[sequence[j: j + kmer]]])
                    # meanP1 = meanP1 / (len(sequence) - kmer)
                    # meanP2 = meanP2 / (len(sequence) - kmer)
                    meanP1 = meanP1 / (len(sequence) - kmer + 1)
                    meanP2 = meanP2 / (len(sequence) - kmer + 1)

                    for l in range(1, lag + 1):
                        ccValue = 0
                        for j in range(len(sequence) - kmer - l + 1):
                            ccValue = ccValue + (
                                    float(myPropertyValue[pair[0]][myIndex[sequence[j: j + kmer]]]) - meanP1) * (
                                            float(
                                                myPropertyValue[pair[1]][myIndex[sequence[j + l:j + l + kmer]]]) - meanP2)
                        ccValue = ccValue / (len(sequence) - kmer - l + 1)
                        code.append(ccValue)
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def make_acc_vector(self, myPropertyName, myPropertyValue, kmer):
        try:
            fastas = self.fasta_list
            lag = self.kw['nlag']

            encodings = []
            myIndex = self.myDiIndex if kmer == 2 else self.myTriIndex
            if len(myPropertyName) < 2:
                self.error_msg = 'two or more property are needed for cross covariance (i.e. DCC and TCC) descriptors'
                return False

            header = ['SampleName', 'label']
            for p in myPropertyName:
                for l in range(1, lag + 1):
                    header.append('%s.lag%d' % (p, l))
            propertyPairs = self.generatePropertyPairs(myPropertyName)
            header = header + [n[0] + '-' + n[1] + '-lag.' + str(l) for n in propertyPairs for l in range(1, lag + 1)]
            encodings.append(header)

            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                ## Auto covariance
                for p in myPropertyName:
                    meanValue = 0
                    # for j in range(len(sequence) - kmer):
                    for j in range(len(sequence) - kmer + 1):
                        meanValue = meanValue + float(myPropertyValue[p][myIndex[sequence[j: j + kmer]]])
                    # meanValue = meanValue / (len(sequence) - kmer)
                    meanValue = meanValue / (len(sequence) - kmer + 1)

                    for l in range(1, lag + 1):
                        acValue = 0
                        for j in range(len(sequence) - kmer - l + 1):
                            # acValue = acValue + (float(myPropertyValue[p][myIndex[sequence[j: j+kmer]]]) - meanValue) * (float(myPropertyValue[p][myIndex[sequence[j+l:j+l+kmer]]]))
                            acValue = acValue + (float(myPropertyValue[p][myIndex[sequence[j: j + kmer]]]) - meanValue) * (
                                    float(myPropertyValue[p][myIndex[sequence[j + l:j + l + kmer]]]) - meanValue)
                        acValue = acValue / (len(sequence) - kmer - l + 1)
                        # print(acValue)
                        code.append(acValue)

                ## Cross covariance
                for pair in propertyPairs:
                    meanP1 = 0
                    meanP2 = 0
                    # for j in range(len(sequence) - kmer):
                    for j in range(len(sequence) - kmer + 1):
                        meanP1 = meanP1 + float(myPropertyValue[pair[0]][myIndex[sequence[j: j + kmer]]])
                        meanP2 = meanP2 + float(myPropertyValue[pair[1]][myIndex[sequence[j: j + kmer]]])
                    # meanP1 = meanP1 / (len(sequence) - kmer)
                    # meanP2 = meanP2 / (len(sequence) - kmer)
                    meanP1 = meanP1 / (len(sequence) - kmer + 1)
                    meanP2 = meanP2 / (len(sequence) - kmer + 1)

                    for l in range(1, lag + 1):
                        ccValue = 0
                        for j in range(len(sequence) - kmer - l + 1):
                            ccValue = ccValue + (
                                    float(myPropertyValue[pair[0]][myIndex[sequence[j: j + kmer]]]) - meanP1) * (
                                            float(
                                                myPropertyValue[pair[1]][myIndex[sequence[j + l:j + l + kmer]]]) - meanP2)
                        ccValue = ccValue / (len(sequence) - kmer - l + 1)
                        code.append(ccValue)
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def get_kmer_frequency(self, sequence, kmer):
        baseSymbol = 'ACGT'
        myFrequency = {}
        for pep in [''.join(i) for i in list(itertools.product(baseSymbol, repeat=kmer))]:
            myFrequency[pep] = 0
        for i in range(len(sequence) - kmer + 1):
            myFrequency[sequence[i: i + kmer]] = myFrequency[sequence[i: i + kmer]] + 1
        for key in myFrequency:
            myFrequency[key] = myFrequency[key] / (len(sequence) - kmer + 1)
        return myFrequency

    def correlationFunction(self, pepA, pepB, myIndex, myPropertyName, myPropertyValue):
        CC = 0
        for p in myPropertyName:
            CC = CC + (float(myPropertyValue[p][myIndex[pepA]]) - float(myPropertyValue[p][myIndex[pepB]])) ** 2
        return CC / len(myPropertyName)

    def correlationFunction_type2(self, pepA, pepB, myIndex, myPropertyName, myPropertyValue):
        CC = 0
        for p in myPropertyName:
            CC = CC + float(myPropertyValue[p][myIndex[pepA]]) * float(myPropertyValue[p][myIndex[pepB]])
        return CC

    def get_theta_array(self, myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, kmer):
        thetaArray = []
        for tmpLamada in range(lamadaValue):
            theta = 0
            for i in range(len(sequence) - tmpLamada - kmer):
                theta = theta + self.correlationFunction(sequence[i:i + kmer],
                                                         sequence[i + tmpLamada + 1: i + tmpLamada + 1 + kmer], myIndex,
                                                         myPropertyName, myPropertyValue)
            thetaArray.append(theta / (len(sequence) - tmpLamada - kmer))
        return thetaArray

    def get_theta_array_type2(self, myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, kmer):
        thetaArray = []
        for tmpLamada in range(lamadaValue):
            for p in myPropertyName:
                theta = 0
                for i in range(len(sequence) - tmpLamada - kmer):
                    theta = theta + self.correlationFunction_type2(sequence[i:i + kmer],
                                                                   sequence[
                                                                   i + tmpLamada + 1: i + tmpLamada + 1 + kmer],
                                                                   myIndex,
                                                                   [p], myPropertyValue)
                thetaArray.append(theta / (len(sequence) - tmpLamada - kmer))
        return thetaArray

    def PseDNC(self, myPropertyName, myPropertyValue):
        try:
            fastas = self.fasta_list
            lamadaValue = self.kw['lambdaValue']
            weight = self.kw['weight']
            encodings = []
            myIndex = self.myDiIndex
            header = ['SampleName', 'label']
            for pair in sorted(myIndex):
                header.append(pair)
            for k in range(1, lamadaValue + 1):
                header.append('lamada_' + str(k))
            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                dipeptideFrequency = self.get_kmer_frequency(sequence, 2)
                thetaArray = self.get_theta_array(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, 2)
                for pair in sorted(myIndex.keys()):
                    code.append(dipeptideFrequency[pair] / (1 + weight * sum(thetaArray)))
                for k in range(17, 16 + lamadaValue + 1):
                    code.append((weight * thetaArray[k - 17]) / (1 + weight * sum(thetaArray)))
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def PCPseDNC(self, myPropertyName, myPropertyValue):
        try:
            fastas = self.fasta_list
            lamadaValue = self.kw['lambdaValue']
            weight = self.kw['weight']
            encodings = []
            myIndex = self.myDiIndex
            header = ['SampleName', 'label']
            for pair in sorted(myIndex):
                header.append(pair)
            for k in range(1, lamadaValue + 1):
                header.append('lamada_' + str(k))
            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                dipeptideFrequency = self.get_kmer_frequency(sequence, 2)
                thetaArray = self.get_theta_array(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, 2)
                for pair in sorted(myIndex.keys()):
                    code.append(dipeptideFrequency[pair] / (1 + weight * sum(thetaArray)))
                for k in range(17, 16 + lamadaValue + 1):
                    code.append((weight * thetaArray[k - 17]) / (1 + weight * sum(thetaArray)))
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def PCPseTNC(self, myPropertyName, myPropertyValue):
        try:
            fastas = self.fasta_list
            lamadaValue = self.kw['lambdaValue']
            weight = self.kw['weight']
            encodings = []
            myIndex = self.myTriIndex

            header = ['SampleName', 'label']
            for tripeptide in sorted(myIndex):
                header.append(tripeptide)
            for k in range(1, lamadaValue + 1):
                header.append('lamada_' + str(k))
            encodings.append(header)

            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                tripeptideFrequency = self.get_kmer_frequency(sequence, 3)
                thetaArray = self.get_theta_array(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, 3)
                for pep in sorted(myIndex.keys()):
                    code.append(tripeptideFrequency[pep] / (1 + weight * sum(thetaArray)))
                for k in range(65, 64 + lamadaValue + 1):
                    code.append((weight * thetaArray[k - 65]) / (1 + weight * sum(thetaArray)))
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def SCPseDNC(self, myPropertyName, myPropertyValue):
        try:
            fastas = self.fasta_list
            lamadaValue = self.kw['lambdaValue']
            weight = self.kw['weight']
            encodings = []
            myIndex = self.myDiIndex
            header = ['SampleName', 'label']
            for pair in sorted(myIndex):
                header.append(pair)
            for k in range(1, lamadaValue * len(myPropertyName) + 1):
                header.append('lamada_' + str(k))

            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                dipeptideFrequency = self.get_kmer_frequency(sequence, 2)
                thetaArray = self.get_theta_array_type2(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, 2)
                for pair in sorted(myIndex.keys()):
                    code.append(dipeptideFrequency[pair] / (1 + weight * sum(thetaArray)))
                for k in range(17, 16 + lamadaValue * len(myPropertyName) + 1):
                    code.append((weight * thetaArray[k - 17]) / (1 + weight * sum(thetaArray)))
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def SCPseTNC(self, myPropertyName, myPropertyValue):
        try:
            fastas = self.fasta_list
            lamadaValue = self.kw['lambdaValue']
            weight = self.kw['weight']
            encodings = []
            myIndex = self.myTriIndex
            header = ['SampleName', 'label']
            for pep in sorted(myIndex):
                header.append(pep)
            for k in range(1, lamadaValue * len(myPropertyName) + 1):
                header.append('lamada_' + str(k))
            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                tripeptideFrequency = self.get_kmer_frequency(sequence, 3)
                thetaArray = self.get_theta_array_type2(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, 3)
                for pep in sorted(myIndex.keys()):
                    code.append(tripeptideFrequency[pep] / (1 + weight * sum(thetaArray)))
                for k in range(65, 64 + lamadaValue * len(myPropertyName) + 1):
                    code.append((weight * thetaArray[k - 65]) / (1 + weight * sum(thetaArray)))
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def PseKNC(self, myPropertyName, myPropertyValue):
        try:
            baseSymbol = 'ACGT'
            fastas = self.fasta_list
            lamadaValue = self.kw['lambdaValue']
            weight = self.kw['weight']
            kmer = self.kw['kmer']
            encodings = []
            myIndex = self.myDiIndex
            header = ['SampleName', 'label']
            header = header + sorted([''.join(i) for i in list(itertools.product(baseSymbol, repeat=kmer))])
            for k in range(1, lamadaValue + 1):
                header.append('lamada_' + str(k))
            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                kmerFreauency = self.get_kmer_frequency(sequence, kmer)
                thetaArray = self.get_theta_array(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, 2)
                for pep in sorted([''.join(j) for j in list(itertools.product(baseSymbol, repeat=kmer))]):
                    code.append(kmerFreauency[pep] / (1 + weight * sum(thetaArray)))
                for k in range(len(baseSymbol) ** kmer + 1, len(baseSymbol) ** kmer + lamadaValue + 1):
                    code.append((weight * thetaArray[k - (len(baseSymbol) ** kmer + 1)]) / (1 + weight * sum(thetaArray)))
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    """ functions """

    def get_header(self):
        if self.encoding_array.shape[0] > 1:
            return list(self.encoding_array[0])
        else:
            return None

    def get_data(self):
        if self.encoding_array.shape[0] >= 2:
            return self.encoding_array[1:, :]
        else:
            return None

    def save_descriptor(self, file_name):
        try:
            if self.row > 1 and self.column > 2:
                if file_name.endswith(".tsv"):
                    np.savetxt(file_name, self.encoding_array[1:, 1:], fmt='%s', delimiter='\t')
                    return True

                if file_name.endswith(".csv"):
                    np.savetxt(file_name, self.encoding_array[1:, 1:], fmt='%s', delimiter=',')
                    return True
                
                if file_name.endswith(".tsv1"):
                    np.savetxt(file_name, self.encoding_array, fmt='%s', delimiter='\t')
                    return Ture

                if file_name.endswith(".svm"):
                    with open(file_name, 'w') as f:
                        for line in self.encoding_array[1:]:
                            line = line[1:]
                            f.write('%s' % line[0])
                            for i in range(1, len(line)):
                                f.write('  %d:%s' % (i, line[i]))
                            f.write('\n')
                    return True

                if file_name.endswith(".arff"):
                    with open(file_name, 'w') as f:
                        f.write('@relation descriptor\n\n')
                        for i in range(1, len(self.encoding_array[0][2:]) + 1):
                            f.write('@attribute f.%d numeric\n' % i)
                        f.write('@attribute play {yes, no}\n\n')
                        f.write('@data\n')
                        for line in self.encoding_array[1:]:
                            line = line[1:]
                            for fea in line[1:]:
                                f.write('%s,' % fea)
                            if line[0] == '1':
                                f.write('yes\n')
                            else:
                                f.write('no\n')
        except Exception as e:
            return False


if __name__ == '__main__':
    M = {
        'sliding_window': 2
    }
    seq = Descriptor('../peptide_sequences.txt', M)
    seq.Protein_EAAC()
    print(seq.get_header())
    print(seq.get_data())
    print(seq.row, seq.column)
    print(seq.minimum_length, seq.maximum_length)
