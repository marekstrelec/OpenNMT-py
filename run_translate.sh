#!/bin/bash

set -e
set -u


RUN_IDX=${1}
# MODEL_NAME=${2}
# MODEL_PREFIX=`echo ${2} | cut -d '.' -f 1`

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

modes=("norm_al2")  # "norm_al"  "sum_conf" "sum_al" "norm_al_binconf" "norm_al_conf2" "norm_al2"
alphas=(0.1 0.25 0.5 1.0)

# modes=("norm_al_conf")
# alphas=(1.0)

if [ -f "bleu.log" ]; then
    rm bleu.log
fi

MODEL_FLDR="policy_models"
model_ids=("2 4 6 8 10")
for MODEL_ID in ${model_ids[@]}; do
    MODEL_PATH=`ls ${MODEL_FLDR}/run${RUN_IDX}/${MODEL_ID}.*.model | head -1`
    ls ${MODEL_PATH}
done

for MODEL_ID in ${model_ids[@]}; do
    MODEL_NAME=`ls ${MODEL_FLDR}/run${RUN_IDX}/${MODEL_ID}.*.model | xargs -n 1 basename | head -1`
    MODEL_PREFIX=`echo ${MODEL_NAME} | cut -d '.' -f 1`

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
                --il_model ${MODEL_FLDR}/run${RUN_IDX}/${MODEL_NAME} \
                --il_alpha ${AL} \
                --il_mode ${MODE}
        done
    done

    for MODE in ${modes[@]}; do
        for AL in ${alphas[@]}; do
            echo ">> (${DATASET}) ${MODEL_NAME} ${MODE} ${AL}" >> bleu.log
            mkdir -p outs/e${RUN_IDX}/m${MODEL_PREFIX}/
            cp ${FOLDER}/${OUT_NAME}_${MODE}_${AL} outs/e${RUN_IDX}/m${MODEL_PREFIX}/
            perl multi-bleu.perl ${FOLDER}/${TGT_NAME} < outs/e${RUN_IDX}/m${MODEL_PREFIX}/${OUT_NAME}_${MODE}_${AL} 2> /dev/null >> bleu.log
            echo "" >> bleu.log
        done
    done
done


