#!/usr/bin/env python

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable 
from chainer import optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

# Set data

from sklearn import datasets
iris = datasets.load_iris()
xdata = iris.data.astype(np.float32)
N = len(xdata)

# Define model

class AutoEncoder(Chain):
    def __init__(self):
        super(AutoEncoder, self).__init__(
            l1=L.Linear(4,2),
            l2=L.Linear(2,4),
        )
        
    def __call__(self,x):
        bv = self.fwd(x)
        return F.mean_squared_error(bv, x)
        
    def fwd(self,x):
        fv = F.sigmoid(self.l1(x))
        bv = self.l2(fv)
        return bv

# Initialize model        

model = AutoEncoder()
optimizer = optimizers.SGD()
optimizer.setup(model)

# Learn

BATCH_SIZE = 30
for j in range(3000):
    shuffle = np.random.permutation(range(N))
    batches = [shuffle[i:min(i + BATCH_SIZE, N)] for i in range(0, N, BATCH_SIZE)]
    for b in batches:
        x = Variable(xdata[b])
        print(b)
        model.zerograds()
        loss = model(x)
        loss.backward()
        optimizer.update()
                                                                                                                        
# Result

x = Variable(xdata)
yt = F.sigmoid(model.l1(x))
ans = yt.data
for i in range(N):
    print(ans[i,:])



