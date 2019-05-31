#!/bin/bash

# POS=10
# POS=12
POS=14


DATASET="dataset/iwslt14/train"
LOG_PATH="outs/e0/m19"

LOG_PATH="outs/e2/m3"

echo ""
echo "Original:"
cat ${DATASET}/train.1k.en | head -${POS} | tail -1
echo ""
echo "Translated:"
cat outs/train.1k.out_0.0 | head -${POS} | tail -1
# echo "Logbest:"
# cat ${DATASET}/train.1k.out_logbest_30b | head -${POS} | tail -1
echo ""
echo -e "\e[35mBLEUbest:"
cat ${DATASET}/train.1k.out_bleubest_30b | head -${POS} | tail -1
echo -e "\e[39m"
echo "out_norm_al_conf_0.1:"
cat ${LOG_PATH}/train.1k.out_norm_al_conf_0.1 | head -${POS} | tail -1
echo ""
echo "out_norm_al_conf_0.5:"
cat ${LOG_PATH}/train.1k.out_norm_al_conf_0.5 | head -${POS} | tail -1
echo ""
echo "out_norm_al_0.5:"
cat ${LOG_PATH}/train.1k.out_norm_al_0.5 | head -${POS} | tail -1
echo ""