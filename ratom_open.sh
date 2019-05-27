#!/bin/bash

~/ratom translate.py
~/ratom onmt/translate/translator.py
~/ratom onmt/imitation/utils.py
~/ratom onmt/imitation/collector.py
~/ratom onmt/decoders/decoder.py
~/ratom onmt/translate/beam_search.py
~/ratom run.sh

~/ratom policy/train_policy.py
~/ratom policy/train_policy_test_fit.py
~/ratom policy/train_auto.py
~/ratom policy/model.py
~/ratom policy/auto.py
~/ratom policy/dataset.py
