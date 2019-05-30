#!/bin/bash

set -e
set -u


RUN_IDX=${1}
MODEL_NAME=${2}
MODEL_PREFIX=`echo ${2} | cut -d '.' -f 1`

DATASET="VALID"
FOLDER=dataset/iwslt14/dev
SRC_NAME=valid.1k.de
TGT_NAME=valid.1k.en
OUT_NAME=valid.1k.out

# DATASET="TRAIN"
# FOLDER=dataset/iwslt14/train
# SRC_NAME=train.1k.de
# TGT_NAME=train.1k.en
# OUT_NAME=train.1k.out

modes=("norm_al" "norm_al_binconf" "norm_al_conf" "sum_al")  # sum_conf
alphas=(0.1 0.5)

# modes=("sum_al")
# alphas=(1.0)


for MODE in ${modes[@]}; do
    for AL in ${alphas[@]}; do
        python3 translate.py \
            -model models/iwslt-brnn2.s131_acc_62.71_ppl_7.74_e20.pt \
            -src ${FOLDER}/${SRC_NAME} \
            -tgt ${FOLDER}/${TGT_NAME} \
            -output ${FOLDER}/${OUT_NAME}_${MODE}_${AL} \
            -replace_unk \
            -beam_size 30 \
            -gpu 0 \
            --il_shardsize 100 \
            --il_beamsize 30 \
            --il_model policy_models/run${RUN_IDX}/${MODEL_NAME}.model \
            --il_alpha ${AL} \
            --il_mode ${MODE}
    done
done

for MODE in ${modes[@]}; do
    for AL in ${alphas[@]}; do
        echo ">> (${DATASET}) ${MODE} ${AL}"
        mkdir -p outs/e${RUN_IDX}/m${MODEL_PREFIX}/
        cp ${FOLDER}/${OUT_NAME}_${MODE}_${AL} outs/e${RUN_IDX}/m${MODEL_PREFIX}/
        perl multi-bleu.perl ${FOLDER}/${TGT_NAME} < outs/e${RUN_IDX}/m${MODEL_PREFIX}/${OUT_NAME}_${MODE}_${AL}
    done
done
