#!/bin/bash

POS=99

# cat dataset/iwslt14/dev/valid.1k.en | head -${POS} | tail -1
# cat outs/valid.1k.out_0.0 | head -${POS} | tail -1
# cat outs/val_e0_9m/valid.1k.out_888 | head -${POS} | tail -1

echo "Original:"
cat dataset/iwslt14/train/train.1k.en | head -${POS} | tail -1
echo "Translated:"
cat outs/train.1k.out_0.0 | head -${POS} | tail -1
echo "888:"
cat outs/e0_9m/train.1k.out_888 | head -${POS} | tail -1
echo "0.1:"
cat outs/e0_9m/train.1k.out_0.1 | head -${POS} | tail -1
echo "0.5:"
cat outs/e0_9m/train.1k.out_0.5 | head -${POS} | tail -1
echo "1.0:"
cat outs/e0_9m/train.1k.out_1.0 | head -${POS} | tail -1