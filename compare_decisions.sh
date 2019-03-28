#!/bin/bash

# POS=10
# POS=12
# POS=14
# POS=223

POS=${1}

# MODEL_PATH="policy_models_nb5/run1/5.1559522864.model"
MODEL_PATH="policy_models/run2/5.1559667590.model"
AL=1.0
MODE="norm_al_conf"
BEAMSIZE=30
NBEST=1

DATASET="dataset/iwslt14/train"
cat ${DATASET}/train.1k.de | head -${POS} | tail -1 > ${DATASET}/compare.de
cat ${DATASET}/train.1k.en | head -${POS} | tail -1 > ${DATASET}/compare.en

# DATASET="dataset/iwslt14/dev"
# cat ${DATASET}/valid.1k.de | head -${POS} | tail -1 > ${DATASET}/compare.de
# cat ${DATASET}/valid.1k.en | head -${POS} | tail -1 > ${DATASET}/compare.en

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
    --il_alpha ${AL} \
    --il_mode ${MODE} \
    --explore \
    --explore_nbest ${NBEST} \
    --il_model ${MODEL_PATH} \
    --test_mode