#!/bin/bash

set -e
set -u


RUN_IDX=${1}
MODEL_NAME=${2}

FOLDER=dataset/iwslt14/dev
SRC_NAME=valid.1k.de
TGT_NAME=valid.1k.en
OUT_NAME=valid.1k.out

# FOLDER=dataset/iwslt14/train
# SRC_NAME=train.1k.de
# TGT_NAME=train.1k.en
# OUT_NAME=train.1k.out

for AL in 0.1 0.5 1.0; do
    python3 translate.py \
        -model models/iwslt-brnn2.s131_acc_62.71_ppl_7.74_e20.pt \
        -src ${FOLDER}/${SRC_NAME} \
        -tgt ${FOLDER}/${TGT_NAME} \
        -output ${FOLDER}/${OUT_NAME}_${AL} \
        -replace_unk \
        -beam_size 25 \
        -gpu 0 \
        --il_shardsize 50 \
        --il_beamsize 25 \
        --il_model policy_models/run${RUN_IDX}/${MODEL_NAME}.model \
        --il_alpha ${AL}
done

for AL in 0.1 0.5 1.0; do
    echo ">> " ${AL}
    perl multi-bleu.perl ${FOLDER}/${TGT_NAME} < ${FOLDER}/${OUT_NAME}_${AL}
done