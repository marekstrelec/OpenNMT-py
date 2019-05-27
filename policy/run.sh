#!/bin/bash

# time python3 policy/train_policy.py --working_dir policy_models/run0 --load 12.1558762217 --start_epoch 13 > out.log 2> out.err
# time python3 policy/train_policy.py --working_dir policy_models/run1 > out.log 2> out.err

# time python3 policy/train_policy.py --working_dir policy_models/run1 --load policy_models/run0/21.1558821372 --start_epoch 1 --nooptload --epochs 10 > out.log 2> out.err
time python3 policy/train_policy.py --working_dir policy_models/run1 --load policy_models/run1/3.1558888448 --start_epoch 4 --epochs 10 > out.log 2> out.err