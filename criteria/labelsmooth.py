from torch.nn.utils.rnn import pack_padded_sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothedCE(nn.Module):
    def __init__(self, args, device, eps=0.1):
        super(LabelSmoothedCE, self).__init__()

        self.args = args
        self.device = device
        self.eps = eps

    def forward(self, inputs, targets, lengths):
        # Remove pad-positions and flatten

        inputs, _, _, _ = pack_padded_sequence(
            input=inputs,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        ) # (sum(lengths), vocab_size)
        targets, _, _, _ = pack_padded_sequence(
            input=targets,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        ) # (sum(lengths))

        if self.eps == 0.:
            # Compute cross-entropy loss
            loss = F.cross_entropy(inputs, targets)
            return loss
        else:
            # "Smoothed" one-hot vectors for the gold sequences
            target_vector = torch.zeros_like(inputs).scatter(dim=1, index=targets.unsqueeze(1), value=1.).to(self.device) # (sum(lengths), n_classes), one-hot
            target_vector = target_vector * (1. - self.eps) + self.eps / target_vector.size(1) # (sum(lengths), n_classes), "smoothed" one-hot

            # Compute smoothed cross-entropy loss
            loss = (-1 * target_vector * F.log_softmax(inputs, dim=1)).sum(dim=1) # (sum(lengths))

            # Compute mean loss
            loss = torch.mean(loss)

            return loss
