#!/bin/bash

set -e
set -u


RUN_IDX=${1}
RUN_NEXT_IDX=$((${1} + 1))
MODEL_NAME=${2}

FOLDER=dataset/iwslt14/train
SRC_NAME=train.160k.de
TGT_NAME=train.160k.en
OUT_NAME=train.160k.out

# FOLDER=dataset/euro7/dev
# SRC_NAME=dev.5.de-en.de
# TGT_NAME=dev.5.de-en.en
# OUT_NAME=dev.5.out

# FOLDER=dataset/iwslt14/dev
# SRC_NAME=valid.1k.de
# TGT_NAME=valid.1k.en
# OUT_NAME=valid.1k.out

# FOLDER=dataset/iwslt14/train
# SRC_NAME=train.50k.de
# TGT_NAME=train.50k.en
# OUT_NAME=train.50k.out

# FOLDER=dataset/iwslt14/train
# SRC_NAME=train.1k.de
# TGT_NAME=train.1k.en
# OUT_NAME=train.1k.out


python3 translate.py \
    -model models/iwslt-brnn2.s131_acc_62.71_ppl_7.74_e20.pt \
    -src ${FOLDER}/${SRC_NAME} \
    -tgt ${FOLDER}/${TGT_NAME} \
    -output ${FOLDER}/${OUT_NAME} \
    -replace_unk \
    -beam_size 25 \
    -gpu 0 \
    --il_shardsize 50 \
    --il_beamsize 25 \
    --il_model policy_models/run${RUN_IDX}/${MODEL_NAME}.model \
    --il_alpha 0.5 \
    --explore_dirout /local/scratch/ms2518/collected/run${RUN_NEXT_IDX}/ \
    --explore_nbest 25 \
    