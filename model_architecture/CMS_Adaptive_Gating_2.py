"""
    Implements version 2 of the Adaptive Gating change to the CMS Module, where the update rate paradigm for how the
    MLP Block layer parameters are updated is completely removed. Instead, the update rates are replaced with adaptive gate
    scalars at the end of each layer that determine how much that layer's outputs are multiplied by before they
    are passed forward into the next layer.

    This version of the CMS Module can just be passed into the same loss function and use the same optimizer step as the rest of the HOPE Block.

    Could act as a valid improvement or a baseline to compare the HOPE block models with when thay're using the actual CMS Module.
"""

import itertools
import warnings
from typing import List, Optional, Sequence, Union
 
import torch
import torch.nn as nn
 
from CMS_module_v2 import MLPBlock
 
class CMS_Adaptive(nn.Module):
    """Chain of MLP blocks, one per entry of `update_rates` (fastest first)."""
    def __init__(self, dim: int, hidden_dims: list[int], n_levels: int, use_norm: bool = True):
        super().__init__()
        self.dim = dim # Specify input dimensionality
        self.update_rates = nn.ParameterList(torch.tensor(1.) for _ in n_levels)
        self.blocks = nn.ModuleList(MLPBlock(dim, hidden_dims, use_norm) for _ in n_levels) # Module List stores the list of MLP blocks in order.
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # y = MLP_fk(... MLP_f2(MLP_f1(x)))
        for block, level in self.blocks:
            x = self.update_rates[level] * block(x) # Pass the input through each of the blocks in order.
        return x