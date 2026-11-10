# Builds the embedding, hope, and optimizer combination for timeseries pretraining
# Built modularly so any number of loss functions can be used through a dict


import torch
import torch.nn as nn
from timeseries_embed import Embedding, patch
from hope_block import make_optimizers


class TimeSeriesEncoder(nn.Module):
    def __init__(self, hope):
        super().__init__()
        self.embed = Embedding()
        self.hope = hope

    def forward(self, x, valid_length=None, hidden=None):
        tokens, targets = self.embed(x, valid_length, hidden)

        # padding should not be stored in the memory
        store_mask = None
        if valid_length is not None:
            store_mask = padding_mask(valid_length, tokens.shape[1])

        output = self.hope(tokens, store_mask=store_mask)
        return output, targets


def padding_mask(valid_length, num_tokens):
    # True for real patches and CLS, False for padding
    num_patches = num_tokens - 1
    token_ids = torch.arange(num_tokens, device=valid_length.device)
    return token_ids >= num_patches - valid_length[:, None] // patch


def make_pretraining_optimizers(encoder, losses, lr=1e-4):
    # The embedding and every loss's heads train alongside the HOPE block
    return make_optimizers(encoder.hope, [encoder.embed] + list(losses.values()), lr)


def pretrain_step(encoder, losses, x, valid_length, main_opt, cms_opt, weights=None):
    # losses is a dict of loss modules
    main_opt.zero_grad(set_to_none=True)   # only clears non-CMS grads; CMS grads keep accumulating

    values = {name: loss(encoder, x, valid_length) for name, loss in losses.items()}
    weights = weights or {}
    total = sum(weights.get(name, 1.0) * value for name, value in values.items())

    total.backward()
    main_opt.step()
    cms_opt.step()   
    return {name: value.item() for name, value in values.items()}
