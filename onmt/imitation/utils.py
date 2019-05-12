
import sys
import copy
import numpy as np
import onmt.imitation.bleu as bleu
import pickle

from nltk.translate.bleu_score import sentence_bleu
from pathlib import Path

from IPython import embed


class Explore(object):

    def __init__(self, fields, raw_src):
        self.fields = fields
        self.raw_src = raw_src

        self.collect_iter = -1
        self.beam_data = []
        self.dec_data = []

    def idtoword(self, id, field='tgt'):
        return self.fields[field].base_field.vocab.itos[id]

    def seqtowords(self, seq, field, ignore=None, stop_at=None):
        if ignore is None:
            ignore = []

        if stop_at is None:
            stop_at = []

        res = []
        for n in seq:
            if n in ignore:
                continue

            if n in stop_at:
                break

            w = self.idtoword(n, field=field)
            res.append(w)

        return res

    def process_beam(self, beam, batch):
        predictions = []
        targets = []

        tgts = np.squeeze(batch.tgt.cpu().numpy().transpose(1, 0, 2), axis=2)
        for t in tgts:
            r = self.seqtowords(
                t,
                field='tgt',
                ignore=[2, 3],  # <s> and </s>
                stop_at=[1],  # <blank>
            )
            targets.append(r)

        # predictions
        for p in beam.predictions:
            beam_predictions = []
            for pb in p:
                r = self.seqtowords(
                    pb.cpu().numpy(),
                    field='tgt',
                    ignore=[2, 3],  # <s> and </s>
                    stop_at=[1],  # <blank>
                )
                beam_predictions.append(r)

            predictions.append(beam_predictions)

        # attention
        attentions = copy.deepcopy(beam.attention)

        return predictions, targets, attentions

    def compute_beam_nltk_bleu(self, beam, batch):
        hypots, refs, attns = self.process_beam(beam, batch)

        scores = []
        for r, h in zip(refs, hypots):
            score = sentence_bleu([r], h)
            scores.append(score)

        return scores

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
        # embed()
        # sys.exit(0)
        hypots, refs, attns = self.process_beam(beam, batch)

        data = []
        for idx, (r, hs, atts) in enumerate(zip(refs, hypots, attns)):
            assert len(hs) == len(atts)

            cooked_refs = bleu.cook_refs([r], ngrams)
            beam_data = []
            for ii in range(len(hs)):
                # embed()
                # sys.exit(0)
                batch_index = batch.indices.cpu().numpy()[idx]
                hypot_replaced = self.replaced_unks(hs[ii], atts[ii], batch_index)

                cooked_hypot = bleu.cook_test(hypot_replaced, cooked_refs, ngrams)
                bleu_score = bleu.score_cooked([cooked_hypot], ngrams)

                beam_data.append({
                    "bleu": bleu_score,
                    "stats": cooked_hypot,
                    "ref": r,
                    "hyp": hs[ii],
                    "hyp_replaced": hypot_replaced,
                })

            data.append(beam_data)

        return data


    def collect_data(self, beam, batch, dec_out_memory, ngrams=2):
        if 'data' not in self.raw_src:
            raise Exception("Data missing in raw_src!")

        working_dir = Path("collected")
        working_dir.mkdir(exist_ok=True)

        if self.collect_iter == -1 or len(self.beam_data) >= 200:
            self.beam_data = []
            self.dec_data = []
            self.collect_iter += 1

        with open(str(working_dir.joinpath("e{0}.pickle".format(self.collect_iter))), "wb") as f:
            new_beam_data = self.collect_beam_data(beam, batch, ngrams)
            self.beam_data.extend(new_beam_data)
            self.dec_data.extend(dec_out_memory)


            pickle.dump((self.beam_data, self.dec_data), f)






    
