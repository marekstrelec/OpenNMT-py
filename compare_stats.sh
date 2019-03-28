#!/bin/bash

POS=10
# POS=12
# POS=14

DATASET="dataset/iwslt14/train"
MODEL_PATH="policy_models/run0/10.1559374593.model"
AL=0.5
MODE="norm_al_conf"
BEAMSIZE=1

cat ${DATASET}/train.1k.de | head -${POS} | tail -1 > ${DATASET}/compare.de
cat ${DATASET}/train.1k.en | head -${POS} | tail -1 > ${DATASET}/compare.en

python3 translate.py \
    -model models/iwslt-brnn2.s131_acc_62.71_ppl_7.74_e20.pt \
    -src ${DATASET}/compare.de \
    -tgt ${DATASET}/compare.en \
    -output ${DATASET}/compare.out \
    -replace_unk \
    -beam_size ${BEAMSIZE} \
    -gpu 0 \
    --il_shardsize 100 \
    --il_beamsize ${BEAMSIZE} \
    --il_model ${MODEL_PATH} \
    --il_alpha ${AL} \
    --il_mode ${MODE}