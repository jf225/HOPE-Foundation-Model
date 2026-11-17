# Modular implementation for the masked time series modelling loss function
# Hides a percentage of patches and then loss is computed based on the reconstruction of the hidden patches

import torch
import torch.nn as nn
from timeseries_embed import patch, max_patches, embed_dim


def pick_hidden_patches(valid_length, ratio=0.4):
    # Randomly hide (base 40%) of each window's real patches, never padding
    hidden = torch.zeros(len(valid_length), max_patches, dtype=torch.bool, device=valid_length.device)

    for i, length in enumerate(valid_length.tolist()):
        real_patches = length // patch
        chosen = torch.randperm(real_patches)[:round(real_patches * ratio)]
        hidden[i, max_patches - real_patches + chosen] = True

    return hidden


class MaskedRebuildLoss(nn.Module):
    # Pretraining loss called as loss(encoder, x, valid_length) and returns one number
    def __init__(self, ratio=0.4):
        super().__init__()
        self.ratio = ratio

        # Turns each output token back into 8 readings
        self.rebuild_head = nn.Linear(embed_dim, patch)

    def forward(self, encoder, x, valid_length):
        # Hide some real patches and score the guesses on reconstruction of those patches
        hidden = pick_hidden_patches(valid_length, self.ratio)
        output, targets = encoder(x, valid_length, hidden)
        guesses = self.rebuild_head(output[:, :-1])
        return ((guesses - targets) ** 2)[hidden].mean()
