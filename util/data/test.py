#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import pickle

with open('didnaPhyche.data', 'rb') as f:
    my_dict = pickle.load(f)

print(my_dict)