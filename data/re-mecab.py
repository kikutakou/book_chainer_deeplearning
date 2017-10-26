#!/usr/bin/env python

import MeCab
import argparse
import neologdn


#parser
parser = argparse.ArgumentParser()
parser.add_argument("file", help="input file")
parser.add_argument("-c", "--check-only", action="store_true", help="just check")
parser.add_argument("--in-place", action="store_true", help="modify in-place")
args = parser.parse_args()





tagger = MeCab.Tagger ("-Owakati")


text = []
with open(args.file) as f:
    for i,line in enumerate(f):
        line = line.rstrip()
        combilned = "".join(line.split(" "))

        if max([ord(c) for c in combilned]) < 128:
            exit("Error : {}:{} \"{}\" seems not Japanese sentence".format(args.file, i+1, combilned))

        wakatied = tagger.parse(neologdn.normalize(combilned)).rstrip()

        if args.in_place:
            text.append(wakatied)
        elif args.check_only:
            if line != wakatied:
                print('before : ', line)
                print('after  : ', parsed)
        else:
            print(wakatied)

if args.in_place:
    with open(args.file, 'w') as f:
        for line in text:
            print(line, file=f)

