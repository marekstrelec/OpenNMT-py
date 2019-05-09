#!/bin/bash

# FOLDER=dataset/euro7/dev
# SRC_NAME=dev.5.de-en.de
# TGT_NAME=dev.5.de-en.en
# OUT_NAME=dev.5.out

FOLDER=dataset/iwslt14/dev
SRC_NAME=zz_valid.5.de
TGT_NAME=zz_valid.5.en
OUT_NAME=zz_valid.5.out

# FOLDER=dataset/iwslt14/dev
# SRC_NAME=valid.1k.de
# TGT_NAME=valid.1k.en
# OUT_NAME=valid.1k.out


python translate.py \
    -model models/iwslt-brnn2.s131_acc_62.71_ppl_7.74_e20.pt \
    -src ${FOLDER}/${SRC_NAME} \
    -tgt ${FOLDER}/${TGT_NAME} \
    -output ${FOLDER}/${OUT_NAME} \
    -replace_unk \
    -beam_size 5