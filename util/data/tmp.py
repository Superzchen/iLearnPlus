#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import pickle

with open('AAindex.txt') as f:
    records = f.readlines()[1:]

my_dict = {}

for line in records:
    array = line.strip().split('\t') if line.strip() != '' else None
    my_dict[array[0]] = [float(array[i]) for i in range(1, len(array))]

with open('AAindex.data', 'wb') as f:
    pickle.dump(my_dict, f)

with open('AAindex.data', 'rb') as f:
    tmp = pickle.load(f)
print(tmp)