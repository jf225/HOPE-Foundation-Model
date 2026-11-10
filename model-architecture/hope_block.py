# Fully customized HOPE block, can use any input/output, just needs configured embedding and loss functions

import torch
import torch.nn as nn
import torch.nn.functional as F
from titans_pytorch import NeuralMemory
from CMS_module_v2 import CMS, CMSOptimizer, non_cms_parameters


class HopeBlock(nn.Module):
    # Input and output tokens are of the same shape (batch, length, dim) 
    def __init__(self, dim=128, chunk_sizes=(1, 2, 4, 8, 16), hidden_dims=(640,), memory_chunk_size=8):
        super().__init__()

        # Titans updates its memory once per chunk of tokens
        self.memory_chunk_size = memory_chunk_size
        self.titans = NeuralMemory(dim=dim, chunk_size=memory_chunk_size)
        self.cms = CMS(dim, list(hidden_dims), list(chunk_sizes))

    def forward(self, tokens, store_mask=None):
        # store_mask (batch, length) bool false for tokens Titans should not remember (padding)
        batch, length, dim = tokens.shape
        if store_mask is None:
            store_mask = torch.ones(batch, length, dtype=torch.bool, device=tokens.device)

        # A token only reads memory written by earlier chunks so pad the front until
        # the last token starts a chunk of its own and sees every token before it
        extra = (1 - length) % self.memory_chunk_size
        padded_tokens = F.pad(tokens, (0, 0, extra, 0))
        padded_mask = F.pad(store_mask, (extra, 0), value=False)

        memory_output, _ = self.titans(padded_tokens, store_mask=padded_mask)
        memory_output = memory_output[:, extra:]

        return self.cms(tokens + memory_output)


def make_optimizers(hope, task_modules, lr=1e-4):
    # Everything outside CMS
    params = non_cms_parameters(hope, hope.cms)
    for module in task_modules:
        params += list(module.parameters())
    main_opt = torch.optim.AdamW(params, lr=lr)

    # CMS levels update on their own schedule
    cms_opt = CMSOptimizer(hope.cms, lrs=lr)
    return main_opt, cms_opt
