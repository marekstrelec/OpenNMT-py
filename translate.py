#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from itertools import repeat

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator

import sys
import copy
import time
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser

from pathlib import Path
from IPython import embed
from onmt.imitation.utils import Explorer
from onmt.imitation.collector import Collector


logger = None


def explore(opt, shard_pairs):
    # paths
    working_dir = Path("/local/scratch/ms2518/collected")
    # working_dir_small = working_dir.joinpath("small")
    working_dir_large = working_dir.joinpath("explore")
    # working_dir_small.mkdir(exist_ok=True, parents=True)
    working_dir_large.mkdir(exist_ok=True, parents=True)

    # opt_large
    opt_large = copy.deepcopy(opt)
    opt_large.beam_size = 25
    opt_large.n_best = opt_large.beam_size

    # opt
    opt.beam_size = 5
    opt.n_best = 5

    # print(opt)

    # translators
    translator_small = build_translator(opt, report_score=False)
    translator_large = build_translator(opt_large, report_score=False)

    # explorer
    explorer_large = Explorer('large', translator_large.fields, translator_large.raw_src, working_dir_large, collect_n_best=5)
    # explorer_small = Explorer('small', translator_small.fields, translator_small.raw_src, working_dir_small, collect_n_best=None)


    for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
        # if i < 128:
        #     continue

        # if i < 8:
        #     continue

        if logger:
            logger.info("Translating shard {0}".format(i))

        s_time = time.time()
        logger.info("(step 1) explore)")
        translator_large.translate(
            src=src_shard,
            tgt=tgt_shard,
            explorer=explorer_large,
            collector=None,
            src_dir=opt.src_dir,
            batch_size=opt.batch_size,
            attn_debug=opt.attn_debug,
        )

        explorer_large.dump_data_and_iterate_if(size=100)

        # sys.exit(0)

        # print(time.time() - s_time)

        # collector = Collector(translator_large.fields)
        # collector.process_collection(Path("collected/large/e0.pickle"), Path("collected/data/e0.pickle"))

        # logger.info("step 2) translate")
        # translator_small.translate(
        #     src=src_shard,
        #     tgt=tgt_shard,
        #     explorer=None,
        #     collector=collector,
        #     src_dir=opt.src_dir,
        #     batch_size=opt.batch_size,
        #     attn_debug=opt.attn_debug,
        # )

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
    # for k,v in opt.__dict__.items():
    #     print(k, v)
    # sys.exit(0)

    src_shards = split_corpus(opt.src, opt.shard_size)
    tgt_shards = split_corpus(opt.tgt, opt.shard_size) \
        if opt.tgt is not None else repeat(None)
    shard_pairs = zip(src_shards, tgt_shards)

    explore(opt, shard_pairs)


def _get_parser():
    parser = ArgumentParser(description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    
    opt.batch_size = 50
    opt.shard_size = 50

    main(opt)
