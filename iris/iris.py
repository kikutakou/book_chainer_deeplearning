#!/usr/bin/env python

import numpy as np
import sys
from chainer import cuda, Function, gradient_check, Variable
from chainer import optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import argparse
import os


# arg
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--files", nargs=2, help="iris files")
parser.add_argument("-g", "--gpu", action="store_true", dest="gpu", help="use gpu")
parser.add_argument("-b", "--batch", action="store_true", dest="batch", help="use all data each epoch")
parser.add_argument("-l", "--lossfunc", choices=["sqm", "sce"], default="sce", help="\"sqm\" for sqare mean, \"sce\" for softmax cross entropy")
args = parser.parse_args()

#gpu
if args.gpu:
    import cupy
    cuda.get_device(0).use()
    xp = cupy
else:
    xp = np


# Set data
if args.files:
    print("load from file...", file=sys.stderr)
    X = np.loadtxt(args.files[0]).astype(np.float32)
    Y = np.loadtxt(args.files[1]).astype(np.int32)
else:
    from sklearn import datasets
    print("downloading from sklearn...", file=sys.stderr)
    iris = datasets.load_iris()
    X = iris.data.astype(np.float32)
    Y = iris.target.astype(np.int32)

# size
N = Y.size

# shuffle
shuffle = np.random.permutation(N)
X, Y = X[shuffle], Y[shuffle]

#gpu
X,Y = xp.array(X), xp.array(Y)




# split data: half for traing, other half for testing
N_TRAIN = int(N * 0.9)
xtrain, xtest = X[:N_TRAIN], X[N_TRAIN:]
ytrain, ytest = Y[:N_TRAIN], Y[N_TRAIN:]


# label(0,1,2) to vector([0,0,1], [0,1,0], [0,0,1])
print("loss func =", args.lossfunc)
if args.lossfunc == "sqm":
    eye = xp.eye(3, dtype=xp.float32)
    ytrain = xp.array([eye[y] for y in ytrain])


# Define model
class IrisChain(Chain):
    def __init__(self):
        # set linear function
        super(IrisChain, self).__init__(
            l1=L.Linear(4, 6),
            l2=L.Linear(6, 3),
        )

    def fwd(self, x):
        # forwarding function use linear
        h1 = F.sigmoid(self.l1(x))
        h2 = self.l2(h1)
        return h2

    if args.lossfunc == "sqm":
        def __call__(self, x, y):
            return F.mean_squared_error(self.fwd(x), y)
    else:
        def __call__(self, x, y):
            return F.softmax_cross_entropy(self.fwd(x), y)

    def test(self, x, y):
        x = Variable(x)
        yy = self.fwd(x)
        predicted = [self.xp.argmax(p) for p in yy.data]
        result = [p == a for p, a in zip(predicted, y)]
        return int(sum(result)), int(len(result))


# Initialize model
model = IrisChain()
if args.gpu:
    model.to_gpu()

optimizer = optimizers.SGD()
optimizer.setup(model)


# setup batch
if args.batch:
    EPOCH = 10000
    x_batchs = [xtrain]
    y_batchs = [ytrain]
else:
    EPOCH = 5000
    BATCH_SIZE = 25
    print("minibatch size = ", BATCH_SIZE)
    x_batchs = [xtrain[i:min(i + BATCH_SIZE, N_TRAIN)] for i in range(0, N_TRAIN, BATCH_SIZE)]
    y_batchs = [ytrain[i:min(i + BATCH_SIZE, N_TRAIN)] for i in range(0, N_TRAIN, BATCH_SIZE)]

# training
print("training...", file=sys.stderr)
for i in range(EPOCH):
    for x,y in zip(x_batchs, y_batchs):  # b for 0-24, 25-49, 50-75
        model.zerograds()
        loss = model(x, y)
        loss.backward()
        optimizer.update()
        del loss
    if i % 100 == 0:
        correct,all = model.test(xtest,ytest)
        print("[{:4d}] {:2d} / {:2d} = {:.2f}".format(i, correct, all, (100. * correct * 1.0) / all))


# Test
correct,all = model.test(xtest,ytest)
print("[done] {:2d} / {:2d} = {:.2f}".format(correct, all, (100. * correct * 1.0) / all))




