#!/usr/bin/env python

import numpy as np
import chainer
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import Variable
import chainer.functions as F
import chainer.links as L
import os
import argparse
import time
import collections
import models

rootdir = (os.path.dirname(__file__) or ".")

# arg
train_ja_default = os.path.join(rootdir, "../data/train_ja.txt")
train_en_default = os.path.join(rootdir, "../data/train_en.txt")
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", type=int, default=0, help="seed for random")
parser.add_argument("-t", "--train", nargs=2, default=[train_ja_default, train_en_default], help="train data")
parser.add_argument("-n", "--num-lines", type=int, default=None, help="num of lines for train")
parser.add_argument("-o", "--outputdir", default="./output", help="output directory")
parser.add_argument("-g", "--gpu", action="store_true", help="enable gpu")
parser.add_argument("-e", "--epoch", type=int, default=100, help="num epoch")
parser.add_argument("-b", "--batchsize", type=int, default=32, help="batch size")
parser.add_argument("--nstep", action="store_true", help="use nstep lstm")
parser.add_argument("--pad", action="store_true", help="padding for nstep lstm")
args = parser.parse_args()

# model switch

Model = (models.MyMT_NStep_Padding if args.pad else models.MyMT_NStep) if args.nstep else models.MyMT


#random seed
if args.seed > -1:
    np.random.seed(args.seed)
    print("seed set to {}".format(args.seed), flush=True)


#mkdir
os.makedirs(args.outputdir, exist_ok=True)

# gpu
if args.gpu:
    import cupy
    cuda.get_device(args.gpu).use()
    xp = cupy
else:
    xp = np


###### CONSTRUCT DATA ######

#dict
dict_ja = collections.defaultdict(lambda:len(dict_ja))
dict_en = collections.defaultdict(lambda:len(dict_en))
PAD_ID = -1
dict_en['<eos>'] = dict_ja['<eos>'] = EOS_ID = 0


#read file
with open(args.train[0]) as f:
    dataset_ja = [[dict_ja[w] for w in line.rstrip().split()[::-1]] for i,line in enumerate(f) if not args.num_lines or i < args.num_lines]
with open(args.train[1]) as f:
    dataset_en = [[dict_en[w] for w in line.rstrip().split()] for i,line in enumerate(f) if not args.num_lines or i < args.num_lines]
print("data jp({}) en({})".format(len(dataset_ja), len(dataset_en)), flush=True)

#dict
dict_ja_rev = [k for k,v in sorted(dict_ja.items(), key=lambda x: x[1])]
dict_en_rev = [k for k,v in sorted(dict_en.items(), key=lambda x: x[1])]
print("vocab jp({}) en({})".format(len(dict_ja), len(dict_en)), flush=True)

# add eos
for s in dataset_ja:
    s.append(EOS_ID)
for s in dataset_en:
    s.append(EOS_ID)

# padding
if Model.PADDING:
    ja_max_len = max([len(s) for s in dataset_ja])
    en_max_len = max([len(s) for s in dataset_en])
    print("padding: len to ja({}) en({})".format(ja_max_len, en_max_len), flush=True)
    for s in dataset_ja:
        s.extend([PAD_ID] * (ja_max_len - len(s)))
    for s in dataset_en:
        s.extend([PAD_ID] * (en_max_len - len(s)))

# to np array(to np.int32)
dataset_ja = [np.array(d, dtype=np.int32) for d in dataset_ja]
dataset_en = [np.array(d, dtype=np.int32) for d in dataset_en]


# print
#for sentence_ja, sentence_en in zip(dataset_ja, dataset_en):
#    print([dict_ja_rev[w] for w in sentence_ja])
#    print([dict_en_rev[w] for w in sentence_en])
#exit()


###### BATCH ######

# batch [ [(batch_size, sentence_len)], [(batch_size, sentence_len)], ... ]
batchset_ja = [dataset_ja[i:min(i + args.batchsize, len(dataset_ja))] for i in range(0, len(dataset_ja), args.batchsize)]
batchset_en = [dataset_en[i:min(i + args.batchsize, len(dataset_en))] for i in range(0, len(dataset_en), args.batchsize)]
print("batchsize {} (last {}) x {} batches".format(args.batchsize, len(batchset_ja[-1]), len(batchset_ja)), flush=True)


# swapaxis [ [(sentence_len, batch_size)], [(sentence_len, batch_size)], ... ]
if Model.SWAP:
    print("swap data", flush=True)
    batchset_ja = [np.array(batch).T for batch in batchset_ja]
    batchset_en = [np.array(batch).T for batch in batchset_en]


# print
#for batch_ja, batch_en in zip(batchset_ja, batchset_en):
#    for b in batch_ja:
#        print([dict_ja_rev[w] for w in b])
#    for b in batch_en:
#        print([dict_en_rev[w] for w in b])
#exit()


###### SETUP NETWORK ######

demb = 100
model = Model(len(dict_ja), len(dict_en), demb, PAD_ID)
optimizer = optimizers.Adam()
optimizer.setup(model)
if args.gpu:
    print("gpu enabled", flush=True)
    model.to_gpu()


###### DEBUG ######

def remove_after_pad_eos(output):
    output = list(output)
    idx = min(output.index(PAD_ID) if PAD_ID in output else len(output),  output.index(EOS_ID) if EOS_ID in output else len(output))
    return output[:idx]


# sample text
sample_ja = dataset_ja[0]
sample_en = dataset_en[0]
print("sample text: ", ' '.join([dict_ja_rev[int(w)] for w in remove_after_pad_eos(sample_ja)][::-1]), flush=True)


###### TRAINING ######


# start
epoch_start_time = time.time()
for e in range(args.epoch):
    for batch_ja, batch_en in zip(batchset_ja, batchset_en):
        if not args.nstep:
            model.H.reset_state()
        model.cleargrads()
        loss = model(batch_ja, batch_en)
        loss.backward()
        optimizer.update()
        loss.unchain_backward()  # truncate
        del loss

    inferenced = model.inference(sample_ja, len(sample_en))
    transralted = [dict_en_rev[int(w)] for w in remove_after_pad_eos(inferenced)]

    print("epoch {:3} / {:3} done {:5.2f} sec/epoch : {}".format((e+1), args.epoch, (time.time() - epoch_start_time) / (e+1), ' '.join(transralted)), flush=True)

    outfile = os.path.join(args.outputdir, "mt-{}.model".format(e))
    serializers.save_npz(outfile, model)

print("done in {} sec".format(time.time() - epoch_start_time), flush=True)

