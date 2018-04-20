#!/usr/bin/env python

import chainer
import chainer.links as L
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import Variable
import chainer.functions as F
import chainer.links as L
import numpy as np


class MyMT(chainer.Chain):

    #constant
    PADDING = True
    SWAP = True

    def __init__(self, n_vocab_ja, n_vocab_en, demb, ignore_label):
        print("ignore", ignore_label)
        super(MyMT, self).__init__(
            embed_x = L.EmbedID(n_vocab_ja, demb, ignore_label=ignore_label),
            embed_y = L.EmbedID(n_vocab_en, demb, ignore_label=ignore_label),
            H = L.LSTM(demb, demb),
            W = L.Linear(demb, n_vocab_en),
        )

    def __call__(self, sentence_ja, sentence_en):
        '''
            sentence_ja: list[(sentence_len, batch)]
            sentence_en: list[(sentence_len, batch)]
            return: loss
        '''

        # encoding
        for w_ja in sentence_ja:
            # transfer
            w_ja = Variable(self.xp.array(w_ja))
            x_k = self.embed_x(w_ja)
            h = self.H(x_k)

        #decoding
        accum_loss = 0
        for i, w_en in enumerate(sentence_en):
            w_en = Variable(self.xp.array(w_en))

            # calc loss
            loss = F.softmax_cross_entropy(self.W(h), w_en)
            accum_loss += loss

            # next h
            x_k = self.embed_y(w_en)
            h = self.H(x_k)

        return accum_loss

    def inference(self, sentence_ja, n):
        '''
            sentence_ja: list[(sentence_len)]
            return: output sequence
        '''

        # encoding
        for w_ja in sentence_ja:
            w_ja = Variable(self.xp.array([w_ja]))
            x_k = self.embed_x(w_ja)
            h = self.H(x_k)

        w_en = F.argmax(F.softmax(self.W(h)).data, axis=1)

        output = [w_en.data[0]]
        for i in range(n-1):
            x_k = self.embed_y(w_en)
            h = self.H(x_k)
            w_en = F.argmax(F.softmax(self.W(h)).data, axis=1)
            output.append(w_en.data[0])

        return output



class MyMT_NStep_Padding(chainer.Chain):

    #constant
    PADDING = True
    SWAP = False

    def __init__(self, n_vocab_ja, n_vocab_en, demb, ignore_label):
        super(MyMT_NStep_Padding, self).__init__(
                                         embed_x = L.EmbedID(n_vocab_ja, demb, ignore_label=ignore_label),
                                         embed_y = L.EmbedID(n_vocab_en, demb, ignore_label=ignore_label),
                                         H = L.NStepLSTM(1, demb, demb, 0),
                                         W = L.Linear(demb, n_vocab_en),
                                         )

    def __call__(self, batch_ja, batch_en):

        # transfer
        batch_ja = self.xp.array(batch_ja)
        batch_en = self.xp.array(batch_en)

        # encoding
        x_k = F.separate(self.embed_x(batch_ja))
        h, c, enc_batch_mh = self.H(None, None, x_k)

        # decoding
        x_k = F.separate(self.embed_y(batch_en[:, :-1]))
        _, _, dec_batch_mh = self.H(h, c, x_k)

        #concat
        print([x.shape for x in enc_batch_mh])
        batch_h = [F.concat((e[-1:],d), axis=0) for e,d in zip(enc_batch_mh, dec_batch_mh)]
        batch_y = [self.W(h) for h in batch_h]

        # loss
        loss = [F.softmax_cross_entropy(y, t) for y, t in zip(batch_y, batch_en)]
        accum_loss = sum(loss)

        return accum_loss


    def inference(self, sentence_ja, n):

        sentence_ja = self.xp.array(sentence_ja)

        # encoding
        x_k = self.embed_x(sentence_ja)
        h, c, _ = self.H(None, None, [x_k])

        #first char
        w_en = F.argmax(F.softmax(self.W(h)).data, axis=1)
        output = [cuda.to_cpu(w_en.data)]


        for i in range(n-1):
            x_k = self.embed_y(w_en)
            h, c, _ = self.H(h, c, [x_k])
            w_en = F.argmax(F.softmax(self.W(h)).data, axis=1)
            output.append(cuda.to_cpu(w_en.data))


        return output







class MyMT_NStep(chainer.Chain):

    #constant
    PADDING = False
    SWAP = False

    def __init__(self, n_vocab_ja, n_vocab_en, demb, ignore_label):
        super(MyMT_NStep, self).__init__(
                                   embed_x = L.EmbedID(n_vocab_ja, demb, ignore_label=ignore_label),
                                   embed_y = L.EmbedID(n_vocab_en, demb, ignore_label=ignore_label),
                                   H = L.NStepLSTM(1, demb, demb, 0),
                                   W = L.Linear(demb, n_vocab_en),
                                   )

    def __call__(self, batch_ja, batch_en):

        # transfer
        batch_ja = [self.xp.array(x) for x in batch_ja]
        batch_en = [self.xp.array(t) for t in batch_en]

        # encoding
        x_k = [self.embed_x(x) for x in batch_ja]
        h, c, enc_batch_mh = self.H(None, None, x_k)

        # decoding
        x_k = [self.embed_y(x[:-1]) for x in batch_en]
        _, _, dec_batch_mh = self.H(h, c, x_k)

        #concat
        batch_h = [F.concat((e[-1:,:],d), axis=0) for e,d in zip(enc_batch_mh, dec_batch_mh)]
        batch_y = [self.W(h) for h in batch_h]

        # loss
        loss = [F.softmax_cross_entropy(y, t) for y, t in zip(batch_y, batch_en)]
        accum_loss = sum(loss)

        return accum_loss


    def inference(self, sentence_ja, n):

        sentence_ja = self.xp.array(sentence_ja)

        # encoding
        x_k = self.embed_x(sentence_ja)
        h, c, _ = self.H(None, None, [x_k])

        #first char
        w_en = F.argmax(F.softmax(self.W(h)).data, axis=1)
        output = [cuda.to_cpu(w_en.data)]


        for i in range(n-1):
            x_k = self.embed_y(w_en)
            h, c, _ = self.H(h, c, [x_k])
            w_en = F.argmax(F.softmax(self.W(h)).data, axis=1)
            output.append(cuda.to_cpu(w_en.data))


        return output



