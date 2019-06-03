#!/bin/bash

set -e
set -u


RUN_IDX=${1}
RUN_NEXT_IDX=$((${1} + 1))

FOLDER=dataset/iwslt14/train
SRC_NAME=train.160k.de
TGT_NAME=train.160k.en
OUT_NAME=train.160k.out


# E=0
python3 translate.py \
    -model models/iwslt-brnn2.s131_acc_62.71_ppl_7.74_e20.pt \
    -src ${FOLDER}/${SRC_NAME} \
    -tgt ${FOLDER}/${TGT_NAME} \
    -output ${FOLDER}/${OUT_NAME} \
    -replace_unk \
    -beam_size 30 \
    -gpu 0 \
    --il_shardsize 100 \
    --il_beamsize 30 \
    --explore \
    --explore_nbest 5 \
    --explore_dirout /local/scratch/ms2518/collected/run${RUN_NEXT_IDX}
