#!/bin/bash


yes "this is a pen" | head -50 > sample.txt
FILE=(-t sample.txt)

# train
for ALGO in rnn lstm gru ; do
    python train.py -a "$ALGO" "${FILE[@]}" "$@"
done

# eval
for MODEL in output/*.model ; do
    python test.py -m "$MODEL" "${FILE[@]}" "$@"
done




