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


# Define model
class MyRNN(chainer.Chain):
    demb = 100
    def __init__(self, n_vocab):
        super(MyRNN, self).__init__(
            embed = L.EmbedID(n_vocab, self.demb),
            H  = L.Linear(self.demb, self.demb),
            W = L.Linear(self.demb, n_vocab),
        )

    def fwd(self, h, input):
        x_emb = self.embed(Variable(self.xp.array([input], dtype=np.int32)))
        h = F.tanh(x_emb + self.H(h))
        y_vec = self.W(h)
        return h, y_vec

    def __call__(self, x,y):

        #initial hidden layer input
        h = Variable(self.xp.zeros((1,self.demb), dtype=np.float32))

        #for each word
        accum_loss = 0
        for input,output in zip(x,y):

            #forward
            h, y_vec = self.fwd(h, input)

            tx = Variable(self.xp.array([output], dtype=np.int32))
            loss = F.softmax_cross_entropy(y_vec, tx)
            accum_loss += loss

        return accum_loss

    def eval(self,x,y):

        #hidden layer
        h = Variable(self.xp.zeros((1,self.demb), dtype=np.float32))
        
        out = []
        for input, output in zip(x,y):
            #forward
            h, y_vec = self.fwd(h, input)

            y_pred = F.softmax(y_vec)
            pi = y_pred.data[0][output]
            out.append(pi)

        return out



class MyLSTM(chainer.Chain):
    demb = 100
    def __init__(self, n_vocab):
        super(MyLSTM, self).__init__(
            embed = L.EmbedID(n_vocab, self.demb),
            H  = L.LSTM(self.demb, self.demb),
            W = L.Linear(self.demb, n_vocab),
        )

    def fwd(self, input):
        x_emb = self.embed(Variable(np.array([input], dtype=np.int32)))
        h = self.H(x_emb)
        y_vec = self.W(h)
        return y_vec

    def __call__(self, x, y):
        #instead of initializing hidden input
        self.H.reset_state()

        accum_loss = 0
        for input, output in zip(x,y):
            y_vec = self.fwd(input)
            tx = Variable(np.array([output], dtype=np.int32))
            loss = F.softmax_cross_entropy(y_vec, tx)
            accum_loss += loss

        return accum_loss

    def eval(self,x,y):
        self.H.reset_state()

        out = []
        for input, output in zip(x,y):
            #forward
            y_vec = self.fwd(input)

            y_pred = F.softmax(y_vec)
            pi = y_pred.data[0][output]
            out.append(pi)

        return out



class MyGRU(chainer.Chain):
    demb = 100
    def __init__(self, n_vocab):
        super(MyGRU, self).__init__(
            embed = L.EmbedID(n_vocab, self.demb),
            H  = L.StatefulGRU(self.demb, self.demb),
            W = L.Linear(self.demb, n_vocab),
        )

    def fwd(self, input):
        x_emb = self.embed(Variable(np.array([input], dtype=np.int32)))
        h = self.H(x_emb)
        y_vec = self.W(h)
        return y_vec

    def __call__(self, x, y):
        #instead of initializing hidden input
        self.H.reset_state()

        accum_loss = 0
        for input, output in zip(x,y):
            y_vec = self.fwd(input)
            tx = Variable(np.array([output], dtype=np.int32))
            loss = F.softmax_cross_entropy(y_vec, tx)
            accum_loss += loss

        return accum_loss

    def eval(self,x,y):
        self.H.reset_state()

        out = []
        for input, output in zip(x,y):
            #forward
            y_vec = self.fwd(input)

            y_pred = F.softmax(y_vec)
            pi = y_pred.data[0][output]
            out.append(pi)

        return out

