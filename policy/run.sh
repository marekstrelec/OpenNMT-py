#!/bin/bash

# time python3 policy/train_policy.py --working_dir policy_models/run1
time python3 policy/train_policy.py --working_dir policy_models/run0 --load 12.1558762217 --start_epoch 13 > out.log 2> out.err