
import sys
import copy
import numpy as np
import onmt.imitation.bleu as bleu
import pickle

import torch

from nltk.translate.bleu_score import sentence_bleu
from pathlib import Path
from collections import defaultdict, Counter

from IPython import embed


class Explorer(object):

    def __init__(self, mode, fields, raw_src, working_dirpath, collect_n_best):
        assert mode in ['small', 'large']

        self.mode = mode
        self.fields = fields
        self.raw_src = raw_src
        self.working_dirpath = working_dirpath
        self.collect_n_best = collect_n_best

        if not self.working_dirpath.exists():
            raise Exception("{0} does not exist!".format(str(self.working_dirpath)))

        self.collect_iter = 0
        self.collected_data = []

        self.acc_bleu = []

    def idtoword(self, id, field='tgt'):
        return self.fields[field].base_field.vocab.itos[id]

    def seqtowords(self, seq, field, ignore=None, stop_at=None):
        if ignore is None:
            ignore = []

        if stop_at is None:
            stop_at = []

        all_words = []
        words = []
        ids = []
        all_ids = []
        stopped = False
        for n in seq:
            w = self.idtoword(n, field=field)
            all_words.append(w)
            all_ids.append(n)

            if n in ignore:
                continue

            if n in stop_at:
                stopped = True

            if not stopped:
                words.append(w)
                ids.append(n)

        return {
            'words': words,
            'ids': ids,
            'all_words': all_words,
            'all_ids': all_ids,
            'orig_len': len(seq)
        }

    def process_beam(self, beam, batch):
        predictions = []
        targets = []

        # predictions
        for p in beam.predictions:
            beam_predictions = []
            for pb in p:
                pb = pb.cpu().numpy()
                r = self.seqtowords(
                    pb,
                    field='tgt',
                    ignore=[2, 3],  # <s> and </s>
                    stop_at=[1],  # <blank>
                )
                beam_predictions.append(r)

            predictions.append(beam_predictions)

        # targets
        tgts = np.squeeze(batch.tgt.cpu().numpy().transpose(1, 0, 2), axis=2)
        for t in tgts:
            r = self.seqtowords(
                t,
                field='tgt',
                ignore=[2, 3],  # <s> and </s>
                stop_at=[1],  # <blank>
            )
            targets.append(r)

        # attention
        attentions = copy.deepcopy(beam.attention)

        return predictions, targets, attentions

    def replaced_unks(self, seq, attention, batch_index):
        new_seq = []
        for i_s, s in enumerate(seq):
            if s == '<unk>':
                _, idx = attention[i_s].max(0)
                replaced_token = self.raw_src['data'][batch_index].src[0][idx]
                new_seq.append(replaced_token)
            else:
                new_seq.append(s)

        return new_seq

    def collect_beam_data(self, beam, batch, dec_out_memory, ngrams=2):
        hypots, refs, attns = self.process_beam(beam, batch)

        data = []
        for idx, (r, hs, atts) in enumerate(zip(refs, hypots, attns)):
            assert len(hs) == len(atts)

            # collect all beams
            cooked_refs = bleu.cook_refs([r['words']], ngrams)
            beam_data = []
            for ii in range(len(hs)):
                batch_index = batch.indices.cpu().numpy()[idx]
                hypot_replaced = self.replaced_unks(hs[ii]['words'], atts[ii], batch_index)

                cooked_hypot = bleu.cook_test(hypot_replaced, cooked_refs, ngrams)
                bleu_score = bleu.score_cooked([cooked_hypot], ngrams)

                beam_data.append({
                    "bleu": bleu_score,
                    "stats": cooked_hypot,
                    "ref": r['words'],
                    "hyp": hs[ii]['ids'],
                    "all_hyp": hs[ii]['all_ids'],
                    "hyp_replaced": hypot_replaced,
                    "orig_len": hs[ii]['orig_len'],
                    "all_words": hs[ii]['all_words']
                })

            # sort beams with respect to BLEU score and return n_best while keeping indexes
            beam_data = list(sorted(enumerate(beam_data), key=lambda x:x[1]['bleu'], reverse=True))
            if self.collect_n_best:
                beam_data = beam_data[:self.collect_n_best]

            data.append(beam_data)

        return data


    def collect_data(self, beam, batch, dec_data, model, ngrams=2):
        if 'data' not in self.raw_src:
            raise Exception("Data missing in raw_src!")

        new_batch_beam_data = self.collect_beam_data(beam, batch, ngrams)
        
        # self.beam_data.append(new_batch_beam_data)
        # self.dec_data.append(dec_data)

        collected_data_batches = []
        for bth in range(beam.batch_size):
            origins = []  # [B, T]
            # collect a lists of indexes that represent beam origins
            for beam_beam_idx in range(len(new_batch_beam_data[bth])):
                hyp_length = len(new_batch_beam_data[bth][beam_beam_idx][1]['all_words'])
                t_end = hyp_length - 1  # -1 for index

                # eos_tokenid = 5
                # +1 if the last token is </s>
                # if new_batch_beam_data[bth][beam_beam_idx][1]['hyp'][-1] == eos_tokenid:
                #    t_end += 1

                if t_end >= len(dec_data['index'][bth]):
                    raise Exception("Invalid timestep! (dec_data['index'], bth:{0}, t:{1})".format(bth, t_end))

                b_idx = new_batch_beam_data[bth][beam_beam_idx][0]
                res = [b_idx]
                for t in range(t_end, -1, -1):
                    b_idx = dec_data['index'][bth][t][b_idx]
                    res.append(b_idx)
                res = res[::-1]
                origins.append(res)

            # check that decoders on t0 are all the same
            for n in range(1, len(dec_data['dec_out'][bth][0])):
                assert np.all(dec_data['dec_out'][bth][0][0] == dec_data['dec_out'][bth][0][n])

            # collect decoder states and labels
            collected_data = []
            lengths = [len(n[1]['all_hyp']) for n in new_batch_beam_data[bth]]
            for t in range(max(lengths)):
                states = defaultdict(list)
                for b in range(len(origins)):
                    if t >= lengths[b]:
                        continue

                    # add to states dict
                    key = origins[b][t]
                    val = (new_batch_beam_data[bth][b][1]['all_hyp'][t], new_batch_beam_data[bth][b][1]['all_words'][t])
                    states[key].append(val)

                # add to collected data
                qq = []
                for k, v in states.items():
                    dec_state = dec_data['dec_out'][bth][t][k]
                    qq.append({
                        'dec': dec_state,
                        'vals': v
                    })
                collected_data.append(qq)

            collected_data_batches.append(collected_data)


            # log functions
            def foo():
                for n in range(self.collect_n_best):
                    print(" ".join(new_batch_beam_data[bth][n][1]['all_words']))

            def bar():
                for n in collected_data:
                    for m in n:
                        print(list(Counter(m['vals']).items()), end=" --- ")
                    print()

            def trydec():
                t = len(collected_data) - 1
                qq = collected_data[t][0]['dec'].reshape((1, 500))
                qqt = torch.from_numpy(qq).float().to('cuda')
                score = model.generator(qqt)

                print(np.argmax(score.cpu().numpy()[0]))


            # embed()
            # sys.exit(0)


        # add to the global storage
        self.collected_data.append(collected_data_batches)

    def dump_data_and_iterate_if(self, size):
        if len(self.collected_data) < size:
            return

        dump_path = self.working_dirpath.joinpath("e{0}.pickle".format(self.collect_iter))
        with open(str(dump_path), "wb") as f:
            pickle.dump(self.collected_data, f)

        self.collected_data = []
        self.collect_iter += 1

        return dump_path

    def collect_best_bleu(self, beam, batch, ngrams=2):
        new_batch_beam_data = self.collect_beam_data(beam, batch, ngrams)

        for ch in new_batch_beam_data:
            best_bleu = -1
            for b in ch:
                best_bleu = max(best_bleu, b[1]['bleu'])

            self.acc_bleu.append(best_bleu)



    
