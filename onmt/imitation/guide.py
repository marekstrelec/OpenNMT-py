
import sys
import numpy as np
import torch

from policy.model import Net
from IPython import embed


INPUT_SIZE = 500
OUTPUT_SIZE = 24725
device = torch.device("cuda")


class Guide(object):

    def __init__(self, model_path, mode, alpha, fields):
        assert mode in ['norm_al', 'norm_al_binconf', 'norm_al_conf', 'sum_al', 'sum_conf']
        assert alpha >= 0. and alpha <= 1.

        self.model_path = model_path
        self.mode = mode
        self.alpha = alpha
        self.fields = fields

        self.model = Net(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE).to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def idtoword(self, id, field='tgt'):
        return self.fields[field].base_field.vocab.itos[id]

    def predict(self, inp):
        with torch.no_grad():
            res = self.model(inp)
            return res

            pred_dist, pred_conf = res[0][0], res[1][0]

            best_preds = sorted(enumerate(pred_dist.cpu().numpy()), key=lambda x:x[1], reverse=True)[:5]
            best_preds = list(map(lambda x: self.idtoword(x[0]), best_preds))

            print("{0}\t{1}".format(
                pred_conf.cpu().numpy()[0],
                best_preds
            ))

            return res

    def get_alpha(self):
        return self.alpha

    def apply(self, inp, log_probs, idx_from, idx_to):
        if self.mode == 'norm_al':
            return self.apply_norm_alpha(inp, log_probs, idx_from, idx_to)
        elif self.mode == 'norm_al_binconf':
            return self.apply_norm_alpha_binconf(inp, log_probs, idx_from, idx_to)
        elif self.mode == 'norm_al_conf':
            return self.apply_norm_alpha_conf(inp, log_probs, idx_from, idx_to)
        elif self.mode == 'sum_al':
            return self.apply_sum_al(inp, log_probs, idx_from, idx_to)
        elif self.mode == 'sum_conf':
            return self.apply_sum_conf(inp, log_probs, idx_from, idx_to)

    def apply_norm_alpha(self, inp, log_probs, idx_from, idx_to):
        pred_dist, pred_conf = self.predict(inp[idx_from:idx_to])

        combined_probs = torch.exp(log_probs[idx_from:idx_to]) * torch.exp(self.alpha * pred_dist)
        denom = torch.sum(combined_probs, dim=1)
        combined_norm = torch.log(combined_probs / denom.view(-1, 1))

        log_probs[idx_from:idx_to] = combined_norm

    def apply_norm_alpha_binconf(self, inp, log_probs, idx_from, idx_to):
        pred_dist, pred_conf = self.predict(inp[idx_from:idx_to])

        pred_conf_binary = pred_conf > 0.5
        pred_conf_binary = pred_conf_binary.float() * self.alpha

        combined_probs = torch.exp(log_probs[idx_from:idx_to]) * torch.exp(pred_conf_binary * pred_dist)
        denom = torch.sum(combined_probs, dim=1)
        combined_norm = torch.log(combined_probs / denom.view(-1, 1))

        log_probs[idx_from:idx_to] = combined_norm

    def apply_norm_alpha_conf(self, inp, log_probs, idx_from, idx_to):
        pred_dist, pred_conf = self.predict(inp[idx_from:idx_to])

        combined_probs = torch.exp(log_probs[idx_from:idx_to]) * torch.exp(self.alpha * pred_conf * pred_dist)
        denom = torch.sum(combined_probs, dim=1)
        combined_norm = torch.log(combined_probs / denom.view(-1, 1))

        log_probs[idx_from:idx_to] = combined_norm

    def apply_sum_al(self, inp, log_probs, idx_from, idx_to):
        pred_dist, _ = self.predict(inp[idx_from:idx_to])

        log_probs[idx_from:idx_to] += self.alpha * pred_dist

    def apply_sum_conf(self, inp, log_probs, idx_from, idx_to):
        pred_dist, pred_conf = self.predict(inp[idx_from:idx_to])

        log_probs[idx_from:idx_to] += pred_conf * pred_dist





