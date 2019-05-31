#!/usr/bin/env python

from __future__ import unicode_literals
from itertools import repeat

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator

import argparse
import sys
import copy
import time
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser

from pathlib import Path
from IPython import embed
from onmt.imitation.utils import Explorer
from onmt.imitation.collector import Collector
from onmt.imitation.guide import Guide

import torch


logger = None


def explore(opt, shard_pairs):
    # opt_large
    opt.beam_size = opt.il_beamsize

    if opt.explore:
        opt.n_best = opt.il_beamsize
    else:
        opt.n_best = 1

    # translators
    translator_large = build_translator(opt, report_score=False)

    # guide
    guide = None
    if opt.il_model:
        guide = Guide(
            model_path=opt.il_model,
            mode=opt.il_mode,
            alpha=opt.il_alpha,
            fields=translator_large.fields
        )
    
    # explorer
    explorer_large = None
    if opt.explore:
        # output dir for collected data
        explore_dirout = Path(opt.explore_dirout)
        explore_dirout.mkdir(exist_ok=True, parents=True)

        explorer_large = Explorer('large', translator_large.fields, translator_large.raw_src, explore_dirout, collect_n_best=opt.explore_nbest)

    for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
        # if i < 128:
        #     continue

        if logger:
            logger.info("Translating shard {0}".format(i))

        s_time = time.time()
        logger.info("Explore")
        translator_large.translate(
            src=src_shard,
            tgt=tgt_shard,
            explorer=explorer_large,
            guide=guide,
            src_dir=opt.src_dir,
            batch_size=opt.batch_size,
            attn_debug=opt.attn_debug,
        )
        logger.info("Done. t={0:.2f}s".format(time.time() - s_time))

        if opt.explore:
            explorer_large.dump_data_and_iterate_if(size=100)

    if opt.explore:
        # dump the rest
        explorer_large.dump_data_and_iterate_if(size=1)


def normal(opt, shard_pairs):
    translator = build_translator(opt, report_score=True)
    for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
        if logger:
            logger.info("Translating shard %d." % i)
        translator.translate(
            src=src_shard,
            tgt=tgt_shard,
            src_dir=opt.src_dir,
            batch_size=opt.batch_size,
            attn_debug=opt.attn_debug
        )


def main(opt):
    global logger
    logger = init_logger(opt.log_file)

    ArgumentParser.validate_translate_opts(opt)
    src_shards = split_corpus(opt.src, opt.shard_size)
    tgt_shards = split_corpus(opt.tgt, opt.shard_size) \
        if opt.tgt is not None else repeat(None)
    shard_pairs = zip(src_shards, tgt_shards)

    explore(opt, shard_pairs)


def _get_parser():
    parser = ArgumentParser(description='translate.py')

    opts.imitation_opts(parser)
    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    
    opt.batch_size = opt.il_shardsize
    opt.shard_size = opt.il_shardsize

    main(opt)
