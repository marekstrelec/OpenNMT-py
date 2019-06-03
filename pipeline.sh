#!/bin/bash

set -e
set -u

RUN_IDX=${1}
RUN_NEXT_IDX=$((${1} + 1))
MODEL_NAME=${2}


# E=0
# time ./run_e0.sh ${RUN_IDX} ${MODEL_NAME} > out.log 2> out.err
# time python3 policy/train_policy.py --working_dir policy_models/run${RUN_NEXT_IDX} --start_epoch 1 --epochs 20 > out_p.log 2> out_p.err

# E>0
# mkdir -p policy_models/run${RUN_NEXT_IDX}
# cp policy_models/run${RUN_IDX}/${MODEL_NAME}.model policy_models/run${RUN_NEXT_IDX}/start.model
# time ./run.sh ${RUN_IDX} ${MODEL_NAME} > out.log 2> out.err
time python3 policy/train_policy.py --working_dir policy_models/run${RUN_NEXT_IDX} --load policy_models/run${RUN_NEXT_IDX}/start --nooptload --start_epoch 1 --epochs 10 > out_p.log 2> out_p.err




# time python3 policy/train_policy.py --working_dir policy_models/run${RUN_NEXT_IDX} --load policy_models/run${RUN_NEXT_IDX}/start --nooptload --start_epoch 1 --epochs 10 > out_p.log 2> out_p.err
# time python3 policy/train_policy.py --working_dir policy_models/run${RUN_NEXT_IDX} --load policy_models/run${RUN_NEXT_IDX}/20.1559471183 --start_epoch 21 --epochs 22
