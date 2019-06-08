#!/bin/bash

set -e
set -u


RUN_IDX=${1}
MODE=${2}

# MODEL_NAME=${2}
# MODEL_PREFIX=`echo ${2} | cut -d '.' -f 1`

# AMOUNT="1k"
AMOUNT="6500"

if [ ${MODE} = "valid" ]; then
    DATASET="VALID"
    FOLDER=dataset/iwslt14/dev
    SRC_NAME=valid.${AMOUNT}.de
    TGT_NAME=valid.${AMOUNT}.en
    OUT_NAME=valid.${AMOUNT}.out
elif [ ${MODE} = "train" ]; then
    DATASET="TRAIN"
    FOLDER=dataset/iwslt14/train
    SRC_NAME=train.${AMOUNT}.de
    TGT_NAME=train.${AMOUNT}.en
    OUT_NAME=train.${AMOUNT}.out
elif [ ${MODE} = "test" ]; then
    DATASET="TEST"
    FOLDER=dataset/iwslt14/test
    SRC_NAME=test.6500.de
    TGT_NAME=test.6500.en
    OUT_NAME=test.6500.out
elif [ ${MODE} = "news" ]; then
    DATASET="NEWS"
    FOLDER=dataset/news
    SRC_NAME=newstest2014.de
    TGT_NAME=newstest2014.en
    OUT_NAME=newstest2014.out
else
    echo "Invalid mode (use valid or train)"
    exit 1
fi

OUTPUT_FILE="bleu_${MODE}_${RUN_IDX}.log"

MODEL_FLDR="policy_models"

# modes=("norm_al2 norm_al_conf2")  # "norm_al"  "sum_conf" "sum_al" "norm_al_binconf" "norm_al_conf2" "norm_al2"
# alphas=(0.1 0.25 0.5 1.0)
# model_ids=("1 2 3 4")

modes=("norm_al2")
alphas=(0.2)
model_ids=("3")

BEAM_SIZE=30

if [ -f ${OUTPUT_FILE} ]; then
    rm ${OUTPUT_FILE}
fi

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
                -beam_size ${BEAM_SIZE} \
                -gpu 0 \
                --il_shardsize 100 \
                --il_beamsize ${BEAM_SIZE} \
                --il_model ${MODEL_FLDR}/run${RUN_IDX}/${MODEL_NAME} \
                --il_alpha ${AL} \
                --il_mode ${MODE} 2> /dev/null > /dev/null
                
        done
    done

    for MODE in ${modes[@]}; do
        # for AL in ${alphas[@]}; do
        #     echo ">> (${DATASET}) ${MODEL_NAME} ${MODE} ${AL}" >> ${OUTPUT_FILE}
        #     mkdir -p outs/e${RUN_IDX}/m${MODEL_PREFIX}/
        #     cp ${FOLDER}/${OUT_NAME}_${MODE}_${AL} outs/e${RUN_IDX}/m${MODEL_PREFIX}/
        #     perl multi-bleu.perl ${FOLDER}/${TGT_NAME} < outs/e${RUN_IDX}/m${MODEL_PREFIX}/${OUT_NAME}_${MODE}_${AL} 2> /dev/null >> ${OUTPUT_FILE}
        #     echo "" >> ${OUTPUT_FILE}
        # done

        for AL in ${alphas[@]}; do
            echo ">> (${DATASET}) ${MODEL_NAME} ${MODE} ${AL}"
            perl multi-bleu.perl ${FOLDER}/${TGT_NAME} < ${FOLDER}/${OUT_NAME}_${MODE}_${AL} 2> /dev/null
        done
    done
done


