"""
    MLP Inversion of the CMS Module.

    In the original version of the Continuum Time Series Module, the update rate C_l
    (aka the rate at which each layer updates) must not increase from layer i to i+1.
    It is meant to decrease instead. This file provides a version of the module where the
    chunk size pattern for the MLP chain is inverted, meaning C_l cannot decrease (and should increase)
    between layers.
"""
import itertools
import warnings
from typing import List, Sequence, Union
 
import torch
import torch.nn as nn
 
 
class MLPBlock(nn.Module):
    """One CMS level: MLP(LayerNorm(x)).  The final layer is linear."""
 
    def __init__(self, dim: int, hidden_dims: list[int], use_norm: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if use_norm else nn.Identity() # Add normalization to the input when it's needed.
        layers: List[nn.Module] = []
        input_dim = dim # Specify the input dimensionality at the beginning.
        for h in hidden_dims:
            layers += [nn.Linear(input_dim, h), nn.GELU()] # Add each Feed-Forward Layer to the list.
            input_dim = h
        layers.append(nn.Linear(input_dim, dim)) # Final layer output should have the same dimensionality as the input
        self.net = nn.Sequential(*layers) # nn.Sequential has Pytorch combine all of the forward passes.
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x)) # Run the forward pass.
 
 
class CMS_Inverse(nn.Module):
    """Chain of MLP blocks, one per entry of `update_rates` (slowest first)."""
 
    def __init__(self, dim: int, hidden_dims: list[int], update_rates: list[int], use_norm: bool = True):
        super().__init__()
        if not update_rates: # update_rates must contain the chunk size for at least one level.
            raise ValueError("update_rates must contain at least one level")
        if not all(isinstance(c, int) and c >= 1 for c in update_rates): # C_l cannot be zero or negative.
            raise ValueError(f"update_rates must be positive integers.")
        if any(b > a for a, b in zip(update_rates, update_rates[1:])): # Revised CMS module must have non-increasing C_l orders.
            raise ValueError(f"update_rates must be non-decreasing (level 0 is the slowest)")
        if any(a % b != 0 for a, b in zip(update_rates, update_rates[1:])): # C_l for an eariler layer l should be a multiple of C_l for a later layer.
            warnings.warn(
                f"update_rates {update_rates}: each size is not a multiple of the previous one, so slow updates will not line up with fast updates.",
                UserWarning,
            )
        self.dim = dim # Specify input dimensionality
        self.update_rates = tuple(update_rates) # Specify the chunk sizes in a class variable.
        self.blocks = nn.ModuleList(MLPBlock(dim, hidden_dims, use_norm) for _ in update_rates) # Module List stores the list of MLP blocks in order.
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # y = MLP_fk(... MLP_f2(MLP_f1(x)))
        for block in self.blocks:
            x = block(x) # Pass the input through each of the blocks in order.
        return x
 
CMSLike = Union[CMS_Inverse, Sequence[CMS_Inverse]]
 
 
def _as_list(cms: CMSLike) -> List[CMS_Inverse]:
    """ Convert the CMSLike class into a list of cms modules. """
    return [cms] if isinstance(cms, CMS_Inverse) else list(cms)
 
 
def non_cms_parameters(model: nn.Module, cms: CMSLike) -> List[nn.Parameter]:
    """Parameters of `model` that are not owned by the given CMS module(s)."""
    cms_ids = {id(p) for m in _as_list(cms) for p in m.parameters()}
    return [p for p in model.parameters() if id(p) not in cms_ids]
 
 
class CMSOptimizer_Inverse:
    def __init__(self, cms: CMSLike, lrs: Union[float, Sequence[float]], optimizer_cls=torch.optim.AdamW, average: bool = False, **optimizer_kwargs):
        self.modules = _as_list(cms)
        if not self.modules:
            raise ValueError("need at least one CMS module")
        self.update_rates = self.modules[0].update_rates # All of the chunk sizes should be the same, so self.modules[0].update_rates can count for all of them.
        if any(m.update_rates != self.update_rates for m in self.modules):
            raise ValueError("all CMS modules must share the same update_rates")
        n_levels = len(self.update_rates) # Number of MLP blocks per CMS module.
        if isinstance(lrs, (int, float)):
            lrs = [float(lrs)] * n_levels # If there is only one learning rate, convert it into a list of length n_levels with every element containing the learning rate.
        lrs = list(lrs)
        if len(lrs) != n_levels:
            raise ValueError(f"expected {n_levels} learning rates, got {len(lrs)}")
        self.average = average # When summing gradients during a single step, self.average averages them out.
        self.step_count = 0
        self.optimizers = []
        for level, lr in enumerate(lrs):
            params = list(itertools.chain.from_iterable(m.blocks[level].parameters() for m in self.modules))
            self.optimizers.append(optimizer_cls(params, lr=lr, **optimizer_kwargs)) # Create an optimizer for each CMS module layer in the list.
 
    def step(self) -> List[bool]:
        """Advance one step and update the levels that are due. Returns a fired-flag per level."""
        self.step_count += 1
        fired = []
        for C, opt in zip(self.update_rates, self.optimizers):
            due = self.step_count % C == 0 # If step_count is a multiple of C_l, update the parameters for the layer with the sum of the gradients.
            if due:
                if self.average and C > 1: # If we are averaging the sum, average them using the grad_div function.
                    for group in opt.param_groups:
                        for p in group["params"]:
                            if p.grad is not None:
                                p.grad.div_(C)
                opt.step() # Pytorch backpropagation will accumulate the gradients every time opt.step() is not done, so the gradient summing is automatic.
                opt.zero_grad(set_to_none=True)
            fired.append(due)
        return fired
 
    def zero_grad(self) -> None:
        """Discard all pending accumulated gradients (every level)."""
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)
 
    def state_dict(self) -> dict: # Save the current optimizer state after a number of iterations.
        return {
            "step_count": self.step_count,
            "optimizers": [opt.state_dict() for opt in self.optimizers],
        }
 
    def load_state_dict(self, state: dict) -> None:# Load the current optimizer state after a number of iterations.
        if len(state["optimizers"]) != len(self.optimizers):
            raise ValueError("state has a different number of levels")
        self.step_count = state["step_count"]
        for opt, s in zip(self.optimizers, state["optimizers"]):
            opt.load_state_dict(s)
