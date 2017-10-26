#!/usr/bin/env python

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, \
                        optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import collections

import sys
import os
import argparse
import time
from models import *



rootdir = (os.path.dirname(__file__) or ".")

train_default = os.path.join(rootdir, "../data/ptb.train.txt")
outdir_default = os.path.join(rootdir, "output")
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", default=train_default, help="train data")
parser.add_argument("-g", "--gpu", action="store_true", help="enable gpu")
parser.add_argument("-e", "--epoch", type=int, default=5, help="num epoch")
parser.add_argument("-o", "--outputdir", default=outdir_default, help="output dir")
parser.add_argument("-a", "--algo", choices=["rnn", "lstm", "gru"], default="rnn", help="rnn or lstm")
args = parser.parse_args()

#mkdir
os.makedirs(args.outputdir, exist_ok=True)


#select gpu
if args.gpu:
    import cupy
    xp = cupy
else:
    xp = np


# dict
w_dict = collections.defaultdict(lambda:len(w_dict))
EOS_ID = w_dict['<eos>']

# convert to id
train_data = []
with open(args.train) as f:
    for line in f:
        sentence = [w_dict[w] for w in line.rstrip().split()]
        sentence.append(EOS_ID)
        train_data.append(sentence)

# save vocab
with open(os.path.join(args.outputdir,'vocab.txt'), 'w') as f:
    for k,v in sorted(w_dict.items(), key=lambda x: x[1]):
        print(k, file=f)

# convert to [input, output] pair
x_train = [s[0:-1] for s in train_data]
y_train = [s[1:] for s in train_data]
n_train = len(train_data)

# Initialize model
if args.algo == "rnn":
    model = MyRNN(len(w_dict))
elif args.algo == "lstm":
    model = MyLSTM(len(w_dict))
elif args.algo == "gru":
    model = MyGRU(len(w_dict))

optimizer = optimizers.Adam()
optimizer.setup(model)

if args.gpu:
    cuda.get_device(0).use()
    model.to_gpu()


# training and Save
print("{0} sentences".format(len(train_data)))
for epoch in range(args.epoch):
    start_time = time.time()

    for i,(x,y) in enumerate(zip(x_train, y_train)):
        model.zerograds()
        loss = model(x,y)
        loss.backward()
        optimizer.update()

        # progress
        if i > 0 and i % 10 == 0:
            elapsed_time = time.time() - start_time
            pct = 100. * i / n_train
            speed = elapsed_time / (i+1)
            print(" {:5d} / {:5d} ({:5.2f} percent) done  {:5.2f} sec ({:5.2f} sec/sentence)  loss {}".format(i, n_train, pct, elapsed_time, speed, loss.data), flush=True)

        loss.unchain_backward()

    #save model at epoch n
    outfile = os.path.join(args.outputdir, "{}-{}.model".format(args.algo, epoch))
    print("output to {0}".format(outfile), flush=True)
    serializers.save_npz(outfile, model)

