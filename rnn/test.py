#!/usr/bin/env python

import numpy as np
import sys
import os
import argparse
import math
from chainer import cuda, Function, gradient_check, Variable, \
    optimizers, serializers, utils
from models import *

rootdir = (os.path.dirname(__file__) or ".")

# arg
test_default = os.path.join(rootdir, "../data/ptb.test.txt")
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="model file to evaluate", required=True)
parser.add_argument("-t", "--test", default=test_default, help="test data")
args = parser.parse_args()

algo = os.path.basename(args.model).split("-")[0]


#read dict
model_dir = (os.path.dirname(args.model) or ".")
with open(os.path.join(model_dir,'vocab.txt')) as f:
    w_dict = dict([[v.rstrip(), i] for i,v in enumerate(f)])

EOS_ID = w_dict['<eos>']


# convert to id
test_data = []
with open(args.test) as f:
    for line in f:
        sentence = [w_dict[w] if w in w_dict else None for w in line.rstrip().split()]
        if not all(sentence):
            continue
        sentence.append(EOS_ID)
        test_data.append(sentence)

if not test_data:
    exit("Error : no valid data {}".format(args.test))


# pairwise as [input, output]: each_pair()
x_test = [s[0:-1] for s in test_data]
y_test = [s[1:] for s in test_data]
n_test = len(test_data)

# Initialize model
if algo == "rnn":
    model = MyRNN(len(w_dict))
elif algo == "lstm":
    model = MyLSTM(len(w_dict))
elif algo == "gru":
    model = MyGRU(len(w_dict))

serializers.load_npz(args.model, model)

total = 0.0
wnum = 0
ave = 0.0
for x,y in zip(x_test, y_test):
    res = model.eval(x,y)
    ave += sum(res)
    ps = sum([-math.log(pi, 2) for pi in res])
    total += ps
    wnum += len(x)
print("{}\tave:{:.4f}\tpp:{:.2f}".format(args.model, ave / wnum, math.pow(2, total / wnum)), flush=True)



