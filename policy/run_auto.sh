#!/bin/bash

rm -f auto_models/auto.model

time python3 policy/train_auto.py --working_dir auto_models --model_name auto.model --epochs 100