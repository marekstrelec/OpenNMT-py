#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from itertools import repeat

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator

import sys
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser

from IPython import embed
from onmt.imitation.utils import Explore


logger = None


def explore(opt, shard_pairs):
    opt.batch_size = 50
    opt.beam_size = 100
    opt.n_best = opt.beam_size
    print(opt)
    translator = build_translator(opt, report_score=True)
    explore = Explore(translator.fields, translator.raw_src)

    for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
        if logger:
            logger.info("Translating shard %d." % i)
        translator.translate(
            src=src_shard,
            tgt=tgt_shard,
            explore=explore,
            src_dir=opt.src_dir,
            batch_size=opt.batch_size,
            attn_debug=opt.attn_debug,
        )



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
    main(opt)
