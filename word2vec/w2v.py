#!/usr/bin/env python

import os
import sys

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, \
                        optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.utils import walker_alias
import collections
import random
import argparse

rootdir = (os.path.dirname(__file__) or ".")

# arg
train_default = os.path.join(rootdir, "../data/ptb.train.txt")
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", default=train_default, help="train_data")
parser.add_argument("-d", "--debug", action="store_true", dest="debug", help="debug mode: lines=20 epoch=1 output=debug.model")
parser.add_argument("-s", "--seed", type=int, dest="seed", help="rand seed")
args = parser.parse_args()

# seed
if args.seed:
    np.random.seed(args.seed)

# doc to index
w_dict = collections.defaultdict(lambda:len(w_dict))
w_dict_rev = []
counts = collections.Counter()
dataset = []
with open(args.train) as f:
    for i,line in enumerate(f):
        for word in line.split():
            idx = w_dict[word]
            if idx >= len(w_dict_rev):
                w_dict_rev.append(word)
            counts[idx] += 1
            dataset.append(idx)
        if args.debug and i > 20:        #only 20 sentences for debug
             break
n_vocab = len(w_dict)
datasize = len(dataset)
print('n_vocab', n_vocab)
print('datasize', datasize)

#for negative sampling
cs = [counts[w] for w in range(len(counts))]
power = np.float32(0.75)
p = np.array(cs, power.dtype)
sampler = walker_alias.WalkerAlias(p)


# Define model
class MyW2V(chainer.Chain):
    def __init__(self, n_vocab, n_units):
        super(MyW2V, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
        )
    
    def fwd(self, x, y):
        x1 = self.embed(x)
        x2 = self.embed(y)
        return F.sum(x1 * x2, axis=1)       #inner product

    def __call__(self, xb, yb, tb):
        xc = Variable(np.array(xb, dtype=np.int32))
        yc = Variable(np.array(yb, dtype=np.int32))
        tc = Variable(np.array(tb, dtype=np.int32))
        return F.sigmoid_cross_entropy(self.fwd(xc,yc), tc)
    



# Initialize model
demb = 100
model = MyW2V(n_vocab, demb)
optimizer = optimizers.Adam()
optimizer.setup(model)




# my functions

WINDOW_SIZE = 3         ### window size
SAMPLING_SIZE = 5        ### negative sample size
WINDOW = [d for w in range(1,WINDOW_SIZE) for d in [-w, w]]         #[-1, 1, -2, 2]

def mkbatset(dataset, batch):
    xb, yb, tb = [], [], []
    for pos in batch:
        xid = dataset[pos]
        context = [i for i in [pos+d for d in WINDOW] if i >= 0 and i < datasize]
        for c_pos in context:
            yid = dataset[c_pos]
            xb.append(xid)
            yb.append(yid)
            tb.append(1)
            for nid in sampler.sample(SAMPLING_SIZE):
                xb.append(yid)
                yb.append(nid)
                tb.append(0)
    return [xb, yb, tb]


# Learn
BATCH_SIZE = 100
NEPOCH = 1 if args.debug else 10

for epoch in range(NEPOCH):
    print('------ epoch {0} ------'.format(epoch))
    shuffle_ary = np.random.permutation(datasize)
    batchs = [shuffle_ary[begin_idx:min(begin_idx + BATCH_SIZE, datasize)] for begin_idx in range(0, datasize, BATCH_SIZE)]
    for i,b in enumerate(batchs):
        if i > 0 and i % 100 == 0:
            print("epoch {0} : batch {1} / {2}".format(epoch, i, len(batchs)))
        xb, yb, tb = mkbatset(dataset, b)
        
        model.zerograds()
        loss = model(xb, yb, tb)
        loss.backward()
        optimizer.update()



# Save model
OUTFILE = 'debug.model' if args.debug else 'w2v.model'
with open(OUTFILE, 'w') as f:
    f.write('%d %d\n' % (len(w_dict_rev), 100))
    for surface,w in zip(w_dict_rev, model.embed.W.data):
        f.write('%s %s\n' % (surface, ' '.join(['%f' % v for v in w])))
print("saved to {0}".format(OUTFILE))

