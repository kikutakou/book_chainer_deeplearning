#!/usr/bin/env python
import os, sys
import numpy as np
from sklearn import datasets
import urllib
import shutil

# change dir to this dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))


#download iris
print("  downloading from sklearn...", file=sys.stderr)
iris = datasets.load_iris()
X = iris.data.astype(np.float32)
Y = iris.target.astype(np.int32)
np.savetxt("iris-x.txt", X, fmt="%.18e", delimiter=" ")
np.savetxt("iris-y.txt", Y, fmt="%.18e", delimiter=" ")


#download ptb
for f in ['ptb.train.txt', 'ptb.test.txt']:
    url = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/' + f
    print("  downloading {0}".format(url))
    urllib.request.urlretrieve(url, f)


for f in ['train.en', 'train.ja', 'test.en', 'test.ja']:
    url = 'https://raw.githubusercontent.com/odashi/small_parallel_enja/master/' + f
    print("  downloading {0}".format(url))
    urllib.request.urlretrieve(url, f)
    shutil.move(f, f.replace(".", "_") + ".txt")

