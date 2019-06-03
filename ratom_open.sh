#!/bin/bash

~/ratom translate.py
~/ratom onmt/translate/translator.py
~/ratom onmt/imitation/utils.py
~/ratom onmt/imitation/guide.py
~/ratom onmt/decoders/decoder.py
~/ratom onmt/translate/beam_search.py
~/ratom run.sh
~/ratom run_translate.sh

~/ratom policy/train_policy.py
~/ratom policy/model.py
~/ratom policy/dataset.py
