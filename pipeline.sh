#!/bin/bash

set -e
set -u

RUN_IDX=${1}
RUN_NEXT_IDX=$((${1} + 1))
MODEL_NAME_EXP=${2}
MODEL_NAME_TRAIN=${3}

# E=0
# time ./run_e0.sh ${RUN_IDX} > out.log 2> out.err
# time python3 policy/train_policy.py --working_dir policy_models/run${RUN_NEXT_IDX} --start_epoch 1 --epochs 10 > out_p.log 2> out_p.err

# # E>0
mkdir -p policy_models/run${RUN_NEXT_IDX}
cp policy_models/run${RUN_IDX}/${MODEL_NAME_TRAIN}.model policy_models/run${RUN_NEXT_IDX}/start.model
cp policy_models/run${RUN_IDX}/${MODEL_NAME_TRAIN}.opt policy_models/run${RUN_NEXT_IDX}/start.opt
time ./run.sh ${RUN_IDX} ${MODEL_NAME_EXP} > out.log 2> out.err
time python3 policy/train_policy.py --working_dir policy_models/run${RUN_NEXT_IDX} --load policy_models/run${RUN_NEXT_IDX}/start --start_epoch 1 --epochs 4 > out_p.log 2> out_p.err

./run_translate.sh ${RUN_NEXT_IDX} train
./run_translate.sh ${RUN_NEXT_IDX} valid