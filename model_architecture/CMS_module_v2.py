"""
Continuum Memory System (CMS)
 
A CMS is a chain of MLP blocks, where Block l has a chunk size C_l and its parameters
are updated once every C_l steps, using the sum of the gradients accumulated over
those C_l steps.  Level 0 is the fastest (smallest chunk), the last level is the
slowest. (At least in the original version.)

Typical loop
------------
    cms_opt  = CMSOptimizer(model_cms_modules, lrs=[1e-3, 5e-4, 1e-4])
    main_opt = torch.optim.AdamW(non_cms_parameters(model, model_cms_modules), lr=1e-3)
    for batch in loader:
        main_opt.zero_grad(set_to_none=True)   # never zero the CMS grads here
        loss = criterion(model(batch.x), batch.y)
        loss.backward()
        main_opt.step()
        cms_opt.step()                         # Updates only the levels that are due.
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
 
 
class CMS(nn.Module):
    """Chain of MLP blocks, one per entry of `chunk_sizes` (fastest first)."""
 
    def __init__(self, dim: int, hidden_dims: list[int], chunk_sizes: list[int], use_norm: bool = True):
        super().__init__()
        if not chunk_sizes: # chunk_sizes must contain the chunk size for at least one level.
            raise ValueError("chunk_sizes must contain at least one level")
        if not all(isinstance(c, int) and c >= 1 for c in chunk_sizes): # C_l cannot be zero or negative.
            raise ValueError(f"chunk_sizes must be positive integers.")
        if any(b < a for a, b in zip(chunk_sizes, chunk_sizes[1:])): # Ordinary CMS module must have non-decreasing C_l orders.
            raise ValueError(f"chunk_sizes must be non-decreasing (level 0 is the fastest)")
        if any(b % a != 0 for a, b in zip(chunk_sizes, chunk_sizes[1:])): # C_l for an eariler layer l should be a multiple of C_l for a later layer.
            warnings.warn(
                f"Chunk_sizes {chunk_sizes}: each size is not a multiple of the previous one, so slow updates will not line up with fast updates.",
                UserWarning,
            )
        self.dim = dim # Specify input dimensionality
        self.chunk_sizes = tuple(chunk_sizes) # Specify the chunk sizes in a class variable.
        self.blocks = nn.ModuleList(MLPBlock(dim, hidden_dims, use_norm) for _ in chunk_sizes) # Module List stores the list of MLP blocks in order.
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # y = MLP_fk(... MLP_f2(MLP_f1(x)))
        for block in self.blocks:
            x = block(x) # Pass the input through each of the blocks in order.
        return x
 
CMSLike = Union[CMS, Sequence[CMS]]
 
 
def _as_list(cms: CMSLike) -> List[CMS]:
    return [cms] if isinstance(cms, CMS) else list(cms)
 
 
def non_cms_parameters(model: nn.Module, cms: CMSLike) -> List[nn.Parameter]:
    """Parameters of `model` that are not owned by the given CMS module(s)."""
    cms_ids = {id(p) for m in _as_list(cms) for p in m.parameters()}
    return [p for p in model.parameters() if id(p) not in cms_ids]
 
 
class CMSOptimizer:
    """
    Per-level optimizer implementing: level l is updated every C_l steps with
    the gradient accumulated since its last update.
 
    The unique thing about the CMSOptimizer is that if the entire model consists of more than HOPE block,
    the CMSOptimizer can act as the cumulative optimizer for all of the CMS modules in all of the HOPE blocks.
    
    If that doesn't really work, we can just use a separate CMSOptimizer for each CMS module in each block.
 
    average=False applies the summed gradient, as in the paper;
    average=True divides it by C_l first.
    """
 
    def __init__(self, cms: CMSLike, lrs: Union[float, Sequence[float]], optimizer_cls=torch.optim.AdamW, average: bool = False, **optimizer_kwargs):
        self.modules = _as_list(cms)
        if not self.modules:
            raise ValueError("need at least one CMS module")
        self.chunk_sizes = self.modules[0].chunk_sizes # All of the chunk sizes should be the same, so self.modules[0].chunk_sizes can count for all of them.
        if any(m.chunk_sizes != self.chunk_sizes for m in self.modules):
            raise ValueError("all CMS modules must share the same chunk_sizes")
        n_levels = len(self.chunk_sizes) # Number of MLP blocks per CMS module.
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
        for C, opt in zip(self.chunk_sizes, self.optimizers):
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

