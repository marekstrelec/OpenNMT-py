#!/bin/bash

# POS=10
# POS=12
POS=14


DATA_MODE="train"
DATASET="dataset/iwslt14/train"

# DATA_MODE="valid"
# DATASET="dataset/iwslt14/dev"

LOG_PATH="outs/e1/m5"

echo ""
echo "Original:"
cat ${DATASET}/${DATA_MODE}.1k.en | head -${POS} | tail -1
echo ""
# echo "Translated:"
# cat outs/${DATA_MODE}.1k.out_0.0 | head -${POS} | tail -1
echo "Logbest:"
cat ${DATASET}/${DATA_MODE}.1k.out_logbest_30b | head -${POS} | tail -1
echo ""
echo -e "\e[35mBLEUbest:"
cat ${DATASET}/${DATA_MODE}.1k.out_bleubest_30b | head -${POS} | tail -1
echo -e "\e[39m"
echo "out_norm_al_conf_0.1:"
cat ${LOG_PATH}/${DATA_MODE}.1k.out_norm_al_conf_0.1 | head -${POS} | tail -1
echo ""
echo "out_norm_al_conf_0.5:"
cat ${LOG_PATH}/${DATA_MODE}.1k.out_norm_al_conf_0.5 | head -${POS} | tail -1
echo ""
echo "out_norm_al_conf_0.75:"
cat ${LOG_PATH}/${DATA_MODE}.1k.out_norm_al_conf_0.75 | head -${POS} | tail -1
echo ""
echo "out_norm_al_conf_1.0:"
cat ${LOG_PATH}/${DATA_MODE}.1k.out_norm_al_conf_1.0 | head -${POS} | tail -1
echo ""