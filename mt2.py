#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, \
                        optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import os
import argparse
import time
import collections

rootdir = (os.path.dirname(__file__) or ".")

# arg
train_ja_default = os.path.join(rootdir, "../data/train.ja")
train_en_default = os.path.join(rootdir, "../data/train.en")
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", nargs=2, default=[train_ja_default, train_en_default], help="train data")
parser.add_argument("-n", "--num-lines", type=int, default=None, help="num of lines for train")
parser.add_argument("-o", "--outputdir", default="./output", help="output directory")
parser.add_argument("-g", "--gpu", action="store_true", help="enable gpu")
parser.add_argument("-e", "--epoch", type=int, default=100, help="num epoch")
parser.add_argument("-b", "--batchsize", type=int, default=30, help="batch size")
args = parser.parse_args()

#mkdir
os.makedirs(args.outputdir, exist_ok=True)

# gpu
if args.gpu is not None:
    import cupy
    cuda.get_device(0).use()
    xp = cupy
    print("gpu enabled")
else:
    xp = np


#dict
dict_ja = collections.defaultdict(lambda:len(dict_ja))
dict_en = collections.defaultdict(lambda:len(dict_en))
EOS_ID = 0
dict_ja['<eos>'] = dict_en['<eos>'] = EOS_ID

#read file
with open(args.train[0]) as f:
    dataset_ja = [[dict_ja[w] for w in line.rstrip().split()[::-1]] + [EOS_ID] for i,line in enumerate(f) if not args.num_lines or i < args.num_lines]
with open(args.train[1]) as f:
    dataset_en = [[dict_en[w] for w in line.rstrip().split()] + [EOS_ID] for i,line in enumerate(f) if not args.num_lines or i < args.num_lines]

#dict reverse
dict_ja_rev = [k for k,v in sorted(dict_ja.items(), key=lambda x: x[1])]
dict_en_rev = [k for k,v in sorted(dict_en.items(), key=lambda x: x[1])]




class MyMT(chainer.Chain):
    def __init__(self, n_vocab_ja, n_vocab_en, demb):
        super(MyMT, self).__init__(
            embed_x = L.EmbedID(n_vocab_ja, demb, ignore_label=-1),
            embed_y = L.EmbedID(n_vocab_en, demb, ignore_label=-1),
            H = L.LSTM(demb, demb),
            W = L.Linear(demb, n_vocab_en),
        )

    def __call__(self, sentence_ja, sentence_en):

        # encoding
        for w_ja in sentence_ja:
            x_k = self.embed_x(w_ja)
            h = self.H(x_k)

        #decoding
        accum_loss = 0
        for i,w_en in enumerate(sentence_en):

            # calc loss
            loss = F.softmax_cross_entropy(self.W(h), w_en)
            accum_loss += loss

            # next h if not last
            x_k = self.embed_y(w_en)
            h = self.H(x_k)

        # calc loss
        loss = F.softmax_cross_entropy(self.W(h), sentence_en[-1])
        accum_loss += loss

        return accum_loss

    def inference(self, sentence_ja, n):

        # encoding
        for w_ja in sentence_ja:
            x_k = self.embed_x(w_ja)
            h = self.H(x_k)

        w_en = F.argmax(F.softmax(self.W(h)).data, axis=1)
        output = [w_en.data]
        for i in range(n-1):
            x_k = self.embed_y(w_en)
            h = self.H(x_k)
            w_en = F.argmax(F.softmax(self.W(h)).data, axis=1)
            output.append(w_en.data)
        return output


demb = 100
model = MyMT(len(dict_ja), len(dict_en), demb)
optimizer = optimizers.Adam()
optimizer.setup(model)
if args.gpu is not None:
    model.to_gpu()


print(' '.join([dict_ja_rev[w] for w in dataset_ja[0]][::-1]))


# padding
ja_max_len = max([len(s) for s in dataset_ja])
en_max_len = max([len(s) for s in dataset_en])
for s in dataset_ja:
    s.extend([-1] * (ja_max_len - len(s)))
for s in dataset_en:
    s.extend([-1] * (en_max_len - len(s)))



# batch
batchset_ja = [dataset_ja[i:min(i + args.batchsize, len(dataset_ja))] for i in range(0, len(dataset_ja), args.batchsize)]
batchset_en = [dataset_en[i:min(i + args.batchsize, len(dataset_en))] for i in range(0, len(dataset_en), args.batchsize)]

# swaqaxis
batchset_ja = [[Variable(xp.array(b)) for b in np.array(batch, dtype=np.int32).T] for batch in batchset_ja]
batchset_en = [[Variable(xp.array(b)) for b in np.array(batch, dtype=np.int32).T] for batch in batchset_en]




# sample text
sample_text = [Variable(xp.array([w], dtype=np.int32)) for w in dataset_ja[0]]
sample_len = len(dataset_en[0])

def to_text_en(inferenced, dict_en_rev):
    output = [int(o[0]) for o in inferenced]
    idx = output.index(EOS_ID) if EOS_ID in output else len(output)
    return ' '.join([dict_en_rev[w_id] for w_id in output[:idx]])

# start
epoch_start_time = time.time()
for e in range(args.epoch):
    for batch_ja, batch_en in zip(batchset_ja, batchset_en):
        model.H.reset_state()
        model.cleargrads()
        loss = model(batch_ja, batch_en)
        loss.backward()
        optimizer.update()
        loss.unchain_backward()  # truncate
        del loss

    inferenced = model.inference(sample_text, sample_len)
    print(to_text_en(inferenced, dict_en_rev))

    print("epoch {:3} / {:3} done {:5.2f} sec/epoch".format((e+1), args.epoch, (time.time() - epoch_start_time) / (e+1)))

    outfile = os.path.join(args.outputdir, "mt-{}.model".format(e))
    serializers.save_npz(outfile, model)

print("done in {} sec".format(time.time() - epoch_start_time))

