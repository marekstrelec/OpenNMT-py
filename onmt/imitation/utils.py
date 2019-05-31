
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
        batch_indexes = batch.indices.cpu().numpy()
        for idx, (r, hs, atts) in enumerate(zip(refs, hypots, attns)):
            assert len(hs) == len(atts)

            # collect all beams
            cooked_refs = bleu.cook_refs([r['words']], ngrams)
            beam_data = []
            for ii in range(len(hs)):
                batch_index = batch_indexes[idx]
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


    def collect_data(self, beam, batch, expl_data, model, ngrams=2):
        if 'data' not in self.raw_src:
            raise Exception("Data missing in raw_src!")

        batch_indexes = batch.indices.cpu().numpy()
        new_batch_beam_data = self.collect_beam_data(beam, batch, ngrams)

        # assert len(batch.indices.data) == len(new_batch_beam_data)
        # with open("qqq.hyp", "a") as f_hyp:
        #     with open("qqq.loghyp", "a") as f_loghyp:
        #         for b_idx, _ in sorted(enumerate(batch.indices.data.cpu().numpy()), key=lambda x:x[1]):

        #             b = new_batch_beam_data[b_idx]
        #             logbest_hyp = sorted(b, key=lambda x:x[0])[0][1]['hyp_replaced']

        #             best_hyp = b[0][1]['hyp_replaced']

        #             f_hyp.write("{0}\n".format(" ".join(best_hyp)))
        #             f_loghyp.write("{0}\n".format(" ".join(logbest_hyp)))
        

        collected_data_batches = []
        for bth in range(beam.batch_size):
            origins = []  # [B, T]
            # collect a lists of indexes that represent beam origins
            for beam_beam_idx in range(len(new_batch_beam_data[bth])):
                hyp_length = len(new_batch_beam_data[bth][beam_beam_idx][1]['all_words'])
                t_end = hyp_length - 1  # -1 for index

                if t_end >= len(expl_data['index'][bth]):
                    raise Exception("Invalid timestep! (expl_data['index'], bth:{0}, t:{1})".format(bth, t_end))

                b_idx = new_batch_beam_data[bth][beam_beam_idx][0]
                res = [b_idx]
                for t in range(t_end, -1, -1):
                    b_idx = expl_data['index'][bth][t][b_idx]
                    res.append(b_idx)
                res = res[::-1]
                origins.append(res)

            # check that decoders on t0 are all the same
            if False:
                for n in range(1, len(expl_data['h_out'][bth][0])):
                    assert np.all(expl_data['h_out'][bth][0][0] == expl_data['h_out'][bth][0][n])

            # collect decoder states and labels
            collected_data = []
            lengths = [len(n[1]['all_hyp']) for n in new_batch_beam_data[bth]]
            for t in range(max(lengths)):

                # adding beamdata based on their origin
                states = defaultdict(list)  # beam origin => data
                for b in range(len(origins)):
                    if t >= lengths[b]:
                        continue

                    # attn = beam.attention[bth][k]

                    # add to states dict
                    key = origins[b][t]
                    val = (new_batch_beam_data[bth][b][1]['all_hyp'][t], new_batch_beam_data[bth][b][1]['all_words'][t])
                    states[key].append(val)

                # add to collected data
                qq = []
                for k, v in states.items():
                    h_out_state = expl_data['h_out'][bth][t][k]
                    d_out_state = expl_data['d_out'][bth][t][k]
                    attn = expl_data['attn'][bth][t][k]

                    # _, idx = attn.max(0)
                    # src_attn_word = self.raw_src['data'][batch_indexes[bth]].src[0][idx]
                    # src_attn_word_id = self.fields['src'].base_field.vocab.stoi[src_attn_word]

                    qq.append({
                        'h_out': h_out_state,
                        'd_out': d_out_state,
                        'attn': attn,
                        # 'attn_wid': src_attn_word_id,
                        # 'attn_w': src_attn_word,
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
                        # if 'conf' in m:
                        print(m['conf'], end=" ")
                        print(list(Counter(m['vals']).items()), end=" --- ")
                    print()

            def trydec():
                t = len(collected_data) - 1
                qq = collected_data[t][0]['dec'].reshape((1, 500))
                qqt = torch.from_numpy(qq).float().to('cuda')
                score = model.generator(qqt)

                print(np.argmax(score.cpu().numpy()[0]))


            # add confidence
            best_prediction_logscore = beam.predictions[bth][0].cpu().numpy()
            min_range = min(len(best_prediction_logscore), len(collected_data_batches[bth]))
            t = 0
            while t < min_range:
                stop = False
                for b in collected_data_batches[bth][t]:
                    for v in b['vals']:
                        if v[0] != best_prediction_logscore[t]:
                            stop = True
                            break
                    if stop:
                        break
                if stop:
                    break

                for b in collected_data_batches[bth][t]:
                    b['conf'] = 0.
                t += 1

            # add confidence - set therest to 1.
            for t2 in range(t, len(collected_data_batches[bth])):
                for b in collected_data_batches[bth][t2]:
                    b['conf'] = 1.

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



    