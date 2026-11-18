"""
    Implements version 1 of the Adaptive Gating change to the CMS Module, where the update rate for each layer
    is optimized along with the other parameters that were already part of the CMS Module. We cannot use gradient
    descent or backpropagation here, because the update rate dictates how often the parameters are updated with the gradients.
    It does not become directly involved in the gradient computation in any way, so there is no way to optimize it with the gradient directly
    without causing other changes/effects we're not looking for.

    Instead, this implementation instead uses a testing system, where each layer's update rate has a potential
    change tested individually before the update is either confirmed or rejected. It does this by incrementing the update rate
    1 or more steps in a certain direction (either increasing or decreasing), and then summing over the average loss for all of the
    steps of the trial run that the model ends up with, and checking if the total loss improves or gets worse.
    If the loss improves, then the new rate is confirmed. Otherwise, it is rejected, the rate resets, and the direction of the next change for
    that level is reversed.

    Under this architecture, each layer has its update rate be tested and changed one at a time. It is recommended that you start training the gates
    after training the other parameters initially, or swap between training the other parameters and the gates, so that
    improving the loss because the model's other parameters were optimized does end up confusing the method we have here.
    
    When to update the adaptive gate:
        1.) Update the adaptive gates for all of the layers right after every layer is finished updating at least once.
            (i.e. if the update rates are currently : 1, 2, 4, 8, 16; update after 16 steps)
        2.) When the user instructs it to, usually after a training epoch or some other timescale.
"""
 
import itertools
import warnings
from typing import List, Optional, Sequence, Union
 
import torch
import torch.nn as nn
 
from CMS_module_v2 import MLPBlock
 
class CMS_Adaptive(nn.Module):
    """Chain of MLP blocks, one per entry of `update_rates` (fastest first)."""
 
    def __init__(self, dim: int, hidden_dims: list[int], update_rates: list[int], use_norm: bool = True):
        super().__init__()
        self.dim = dim  # Specify input dimensionality
        # Store the rates in a buffer so they can be saved/loaded with the model's state_dict and follow .to(device).
        self.register_buffer("update_rates", torch.tensor(list(update_rates), dtype=torch.long))
        self.blocks = nn.ModuleList(MLPBlock(dim, hidden_dims, use_norm) for _ in update_rates)  # Module List stores the list of MLP blocks in order.
 
    def set_update_rates(self, rates: Sequence[int]) -> None:
        if len(rates) != len(self.blocks):
            raise ValueError(f"expected {len(self.blocks)} rates, got {len(rates)}")
        self.update_rates.copy_(torch.tensor(list(rates), dtype=torch.long))
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # y = MLP_fk(... MLP_f2(MLP_f1(x)))
        for block in self.blocks:
            x = block(x)  # Pass the input through each of the blocks in order.
        return x
 
 
CMSLike = Union[CMS_Adaptive, Sequence[CMS_Adaptive]]
 
 
def _as_list(cms: CMSLike) -> List[CMS_Adaptive]:
    return [cms] if isinstance(cms, CMS_Adaptive) else list(cms)
 
 
def non_cms_parameters(model: nn.Module, cms: CMSLike) -> List[nn.Parameter]:
    """Parameters of `model` that are not owned by the given CMS module(s)."""
    cms_ids = {id(p) for m in _as_list(cms) for p in m.parameters()}
    return [p for p in model.parameters() if id(p) not in cms_ids]
 
 
class AdaptiveCMSOptimizer:
    """
        Same backpropagation rules as the original CMSOptimizer class, plus the hill-climbing rule that adapts each level's
        update rate.

        Update Paradigms:
            Paradigm 0: Update it once every time every layer has been updated at least once.
            Paradigm 1: Have the user set them to update at a time they specify manually using the gates_step() function.
        
        step_frac: The fraction/ratio of the current update rate for a layer that it will be altered by. 
        min_rate / max_rate: The upper and optional lower bounds on the update rate for every layer.
        accept_tol: a new trial value for the update rate is kept only if loss_metric <= previous_loss_metric - accept_tol * |previous| (0 keeps ties).
    """
 
    def __init__(self, cms: CMSLike, lrs: Union[float, Sequence[float]], optimizer_cls=torch.optim.AdamW, average: bool = False, paradigm: int = 1, *, step_frac: float = 0.25, min_rate: int = 1, max_rate: Optional[int] = None, accept_tol: float = 0.0, **optimizer_kwargs):
        if paradigm not in (0, 1):
            raise ValueError("paradigm must be 0 or 1")
        if step_frac <= 0:
            raise ValueError("step_frac must be positive")
        if min_rate < 1 or (max_rate is not None and max_rate < min_rate):
            raise ValueError("need 1 <= min_rate <= max_rate")
        self.modules = _as_list(cms)
        if not self.modules:
            raise ValueError("need at least one CMS module")
        self.update_rates = self.modules[0].update_rates
        if any(m.update_rates != self.update_rates for m in self.modules):
            raise ValueError("all CMS modules must share the same update_rates")
        n_levels = len(self.update_rates)
        if isinstance(lrs, (int, float)): # If there is only one learning rate, convert it into a list of length n_levels with every element containing the learning rate.
            lrs = [float(lrs)] * n_levels 
        lrs = list(lrs)
        if len(lrs) != n_levels:
            raise ValueError(f"expected {n_levels} learning rates, got {len(lrs)}")
 
        self.paradigm = paradigm # Sets the rate alteration paradigm that we are going to be using for one instance of a model.
        self.average = average  # When summing gradients during a single step, self.average averages them out.
        self.step_frac = step_frac
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.accept_tol = accept_tol
 
        self.step_count = 0
        self.since_update = [0] * n_levels  # The number of steps accumulated by each level since it last fired.
        self.optimizers = []
        for level, lr in enumerate(lrs):
            params = list(itertools.chain.from_iterable(m.blocks[level].parameters() for m in self.modules))
            self.optimizers.append(optimizer_cls(params, lr=lr, **optimizer_kwargs))
 
        # Gate-search state.
        self.direction = [1] * n_levels  # +1 means try a slower rate next. -1 means try a faster one.
        self.level_ptr = 0               # Next level whose rate is being changed.
        self.trial = None                # (level, old_rate) while a perturbation is being evaluated.
        self.prev_metric = None          # Loss metric of the previous gate window. (should be None to begin with since there is no loss to start with)
        self.fired_since_gate = [False] * n_levels # List of booleans that tell whether or not the layer has updated since the last alteration.
        self.loss_sum = 0.0 # The sum of the calculated losses for the steps since the update rates have last been changed.
        self.loss_n = 0 # Number of times a step's loss has been added to the sum.

    def step(self, loss: Optional[float] = None) -> List[bool]:
        """Advance one step and update the levels that are due. Returns a fired-flag per level.
 
        Passing in the step's loss will let the optimizer compute the gate-window metric itself by accumulating it
        through loss_sum.
        """
        self.step_count += 1
        if loss is not None: # If we have the loss at the current step:
            self.loss_sum += float(loss) # Add it to the sum we are calculating.
            self.loss_n += 1
        fired = []
        for level, opt in enumerate(self.optimizers):
            self.since_update[level] += 1
            due = self.since_update[level] >= self.update_rates[level]
            if due:
                n_acc = self.since_update[level]
                if self.average and n_acc > 1:
                    for group in opt.param_groups:
                        for p in group["params"]:
                            if p.grad is not None:
                                p.grad.div_(n_acc)
                opt.step()  # PyTorch accumulates gradients until opt.step() runs, so the summing is automatic.
                opt.zero_grad(set_to_none=True)
                self.since_update[level] = 0 # Reset the step number for the layer to zero.
                self.fired_since_gate[level] = True # Mark the layer as having been fired at least once.
            fired.append(due)
 
        if self.paradigm == 0 and all(self.fired_since_gate): # If we are using paradigm 0 and all of the layers have been updated since the last time.
            self.gates_step()
        return fired
    
    def gates_step(self, metric: Optional[float] = None) -> None:
        """Run one round of the gate hill-climbing rule. Lower metric = better."""
        if metric is None: # If there is no set loss metric we're trying to measure by, simply use the average loss for all of the recent steps.
            if self.loss_n == 0:
                raise ValueError("gates_step needs a metric: pass one explicitly or give losses to step(loss=...)")
            metric = self.loss_sum / self.loss_n
        metric = float(metric)
        self.loss_sum, self.loss_n = 0.0, 0 # reset the loss_sum, loss_n, and fired_since_gate values for next time.
        self.fired_since_gate = [False] * len(self.optimizers)
 
        if self.trial is not None: # If we are currently running a trial to see if an improved update rate is viable.
            level, old_rate = self.trial
            self.trial = None # End the trial run.
            kept = metric <= self.prev_metric - self.accept_tol * abs(self.prev_metric) # Check if the metric (loss sum) is sufficiently less than the previous metric.
            if not kept: # If it is not:
                self.set_rate(level, old_rate)  # Revert that level's update rate to its original value.
                self.direction[level] *= -1     # Flip that level's direction of improvement.
                self.prev_metric = metric       # Next window is a baseline under the reverted rates.
                return
        self.prev_metric = metric # Have the current metric be the new previous metric for the next check.
        self.propose()
 
    # Set the new update rate for the proper layer in every CMS Module.
    def set_rate(self, level: int, value: int) -> None:
        self.update_rates[level] = value
        for m in self.modules:
            m.set_update_rates(self.update_rates)
 
    def propose(self) -> bool:
        """Perturb one level's rate within [min_rate, max_rate] and its neighbours. False if nothing can move."""
        n = len(self.update_rates)
        for _ in range(n): # For every update rate
            level = self.level_ptr # Determine the current level we are going to check.
            self.level_ptr = (self.level_ptr + 1) % n # Set the new level to be the next one down (If nothing is changed, the loop will continue).
            old = self.update_rates[level]
            # Determine the maximum and minimum possible rate change.
            lo = max(self.min_rate, self.update_rates[level - 1]) if level > 0 else self.min_rate 
            hi = self.update_rates[level + 1] if level < n - 1 else self.max_rate
            if self.max_rate is not None:
                hi = self.max_rate if hi is None else min(hi, self.max_rate)
            for direction in (self.direction[level], -self.direction[level]): # For both possible directions (Does this work?)
                new = old + direction * max(1, round(old * self.step_frac)) # Calculate the proposed new update rate.
                new = max(lo, new)
                if hi is not None:
                    new = min(hi, new)
                if new != old: # If the new rate is different from the old one, set the new direction, trial variable, and rate for the level.
                    self.direction[level] = direction
                    self.trial = (level, old)
                    self.set_rate(level, new)
                    return True
        return False
 
    def zero_grad(self) -> None:
        """Discard all pending accumulated gradients (every level)."""
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)
        self.since_update = [0] * len(self.optimizers)
 
    def state_dict(self) -> dict:
        return {
            "step_count": self.step_count,
            "since_update": list(self.since_update),
            "update_rates": list(self.update_rates),
            "optimizers": [opt.state_dict() for opt in self.optimizers],
            "gate_state": {
                "direction": list(self.direction),
                "level_ptr": self.level_ptr,
                "trial": list(self.trial) if self.trial is not None else None,
                "prev_metric": self.prev_metric,
                "fired_since_gate": list(self.fired_since_gate),
                "loss_sum": self.loss_sum,
                "loss_n": self.loss_n,
            },
        }
 
    def load_state_dict(self, state: dict) -> None:
        if len(state["optimizers"]) != len(self.optimizers):
            raise ValueError("state has a different number of levels")
        self.step_count = state["step_count"]
        self.since_update = list(state["since_update"])
        self.update_rates = list(state["update_rates"])
        for m in self.modules:
            m.set_update_rates(self.update_rates)
        for opt, s in zip(self.optimizers, state["optimizers"]):
            opt.load_state_dict(s)
        g = state["gate_state"]
        self.direction = list(g["direction"])
        self.level_ptr = g["level_ptr"]
        self.trial = tuple(g["trial"]) if g["trial"] is not None else None
        self.prev_metric = g["prev_metric"]
        self.fired_since_gate = list(g["fired_since_gate"])
        self.loss_sum = g["loss_sum"]
        self.loss_n = g["loss_n"]
