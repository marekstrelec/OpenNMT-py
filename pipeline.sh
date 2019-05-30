#!/bin/bash

set -e
set -u

RUN_IDX=${1}
RUN_NEXT_IDX=$((${1} + 1))
MODEL_NAME=${2}

mkdir -p policy_models/run${RUN_NEXT_IDX}
cp policy_models/run${RUN_IDX}/${MODEL_NAME}.model policy_models/run${RUN_NEXT_IDX}/start.model

time ./run.sh ${RUN_IDX} ${MODEL_NAME} > out.log 2> out.err
time python3 policy/train_policy.py --working_dir policy_models/run${RUN_NEXT_IDX} --load policy_models/run${RUN_NEXT_IDX}/start --nooptload --start_epoch 1 --epochs 10 > out_p.log 2> out_p.err
