
import sys
import copy
import numpy as np
import onmt.imitation.bleu as bleu
import pickle

import torch

from nltk.translate.bleu_score import sentence_bleu
from pathlib import Path
from collections import defaultdict, Counter, OrderedDict

from IPython import embed


class Explorer(object):

    def __init__(self, mode, fields, raw_src, working_dirpath, collect_n_best):
        assert mode in ['small', 'large']

        self.mode = mode
        self.fields = fields
        self.raw_src = raw_src
        self.working_dirpath = working_dirpath
        self.collect_n_best = collect_n_best

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
            if not self.collect_n_best:
                data.append(beam_data)
            else:
                best_beam_data = []
                for n in range(self.collect_n_best):
                    if beam_data[n][0] == 0:  # logbest
                        break

                    best_beam_data.append(beam_data[n])

                data.append(best_beam_data)

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
        

        collected_data_batches = OrderedDict()
        for bth in range(beam.batch_size):
            if not len(new_batch_beam_data[bth]):
                continue

            origins = []  # [B, T]
            # iterate over best beams and collect a lists of indexes that represent beam origins
            for beam_beam_idx in range(len(new_batch_beam_data[bth])):
                hyp_length = len(new_batch_beam_data[bth][beam_beam_idx][1]['all_words'])
                t_end = hyp_length - 1  # -1 for index

                if t_end >= len(expl_data['index'][bth]):
                    raise Exception("Invalid timestep! (expl_data['index'], bth:{0}, t:{1})".format(bth, t_end))

                prediction_bestlog_pos = new_batch_beam_data[bth][beam_beam_idx][0]
                b_idx = beam.orig_positions_logsorted[bth][prediction_bestlog_pos]
                res = [b_idx]
                for t in range(t_end, -1, -1):
                    b_idx = expl_data['index'][bth][t][b_idx]
                    res.append(b_idx)
                res = res[::-1]

                for n_idx, n in enumerate(res[1:]):
                    expected = new_batch_beam_data[bth][beam_beam_idx][1]['all_words']
                    assert self.idtoword(expl_data['seq'][bth][n_idx][n], 'tgt') == expected[n_idx]

                origins.append(res)

            # check that decoders on t0 are all the same
            if True:
                for n in range(1, len(expl_data['h_out'][bth][0])):
                    assert np.all(expl_data['h_out'][bth][0][0] == expl_data['h_out'][bth][0][n])

            # collect decoder states and labels
            collected_data = []
            best_prediction_logscore = beam.predictions[bth][0].cpu().numpy()
            used_conf = set()
            lengths = [len(n[1]['all_hyp']) for n in new_batch_beam_data[bth]]
            for t in range(max(lengths)):

                # adding beamdata based on their origin
                states = defaultdict(list)  # beam origin => data
                for b in range(len(origins)):
                    if t >= lengths[b]:
                        continue

                    # add to states dict
                    key = origins[b][t]
                    val = (new_batch_beam_data[bth][b][1]['all_hyp'][t], new_batch_beam_data[bth][b][1]['all_words'][t], b)
                    states[key].append(val)

                # add to collected data
                qq = []
                for k, vs in states.items():
                    h_out_state = expl_data['h_out'][bth][t][k]
                    d_out_state = expl_data['d_out'][bth][t][k]
                    attn = expl_data['attn'][bth][t][k]

                    conf = 0.
                    if t < len(best_prediction_logscore):
                        most_common_wordid = Counter([x[0] for x in vs]).most_common()[0][0]

                        # check for used
                        any_in_used = any([1 for x in vs if x[2] in used_conf])
                        if not any_in_used:

                            # if most common is the one chosen by log
                            if most_common_wordid == best_prediction_logscore[t]:
                                conf = 0.
                            else:
                                conf = 1.

                            # disable all but logbest path
                            for v in vs:
                                if v[0] != best_prediction_logscore[t]:
                                    used_conf.add(v[2])


                    qq.append({
                        'h_out': h_out_state,
                        'd_out': d_out_state,
                        'attn': attn,
                        'vals': vs,
                        'conf': conf
                    })
                collected_data.append(qq)

            collected_data_batches[bth] = collected_data


            # log functions
            def foo():
                for n in new_batch_beam_data[bth]:
                    print(" ".join(n[1]['all_words']))

            def bar():
                for n_idx, n in enumerate(collected_data):
                    print(n_idx)
                    for m in n:
                        print("\t", end="")
                        print(m['conf'], end=" ")
                        print(m['vals'])
                        # print(list(Counter(m['vals']).items()))

            def logbest():
                print(" ".join(self.seqtowords(beam.predictions[bth][0].cpu().numpy(), 'tgt')['all_words']))

            def trydec():
                t = len(collected_data) - 1
                qq = collected_data[t][0]['dec'].reshape((1, 500))
                qqt = torch.from_numpy(qq).float().to('cuda')
                score = model.generator(qqt)

                print(np.argmax(score.cpu().numpy()[0]))

        # # Compara_decisions
        # embed()
        # sys.exit(0)

        # add to the global storage
        if len(collected_data_batches):
            self.collected_data.append(list(collected_data_batches.values()))

    def dump_data_and_iterate_if(self, size):
        if not self.working_dirpath.exists():
            raise Exception("{0} does not exist!".format(str(self.working_dirpath)))

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



    
