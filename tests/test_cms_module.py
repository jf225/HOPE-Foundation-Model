"""
Unit tests for the CMS Module architecture (specifically the one
implemented in CMS_module_v2.py).  Run with:  pytest -q test_cms_module.py
 
Tests are grouped as: A) structure, B) construction/validation, C) update schedule,
D) gradient accumulation, E) optimizer API, F) regressions for the v1 failure modes,
G) a small end-to-end learning check.
"""
import sys
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import copy
import warnings
 
import pytest
import torch
import torch.nn as nn
 
from model_achitecture.CMS_module_v2 import CMS, CMSOptimizer, MLPBlock, non_cms_parameters
 
DIM = 8
SGD = torch.optim.SGD  # plain SGD keeps parameter-change checks exact (no momentum / weight decay)
 
 
# Create a CMS module from scratch to use for the testing function.
def make_cms(chunk_sizes=(1, 2, 4), hidden=(16, 16), seed=0, **kw):
    torch.manual_seed(seed)
    return CMS(DIM, hidden, chunk_sizes, **kw).double()
 
 
# Generate a batch of random tensors with a set architecture.
def rand(*shape, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float64)
 
 
# Sample loss function here is the mean-squared error.
def loss_fn(model, x):
    return model(x).pow(2).mean()
 
 
def snapshot(block):
    return [p.detach().clone() for p in block.parameters()]
 
 
# Has a block's parameters been changed?
def changed(block, snap):
    return any(not torch.equal(p.detach(), s) for p, s in zip(block.parameters(), snap))
 
 
# --------------------------------------------------------------------------- A) structure
@pytest.mark.parametrize("shape", [(3, DIM), (3, 5, DIM)])
# CMS class initialization creates a model with the correct shape.
def test_output_shape_matches_input(shape):
    cms = make_cms()
    assert cms(rand(*shape)).shape == shape
 
 
# Passing the input through the forward function generates the same results as passing it through each block sequentially.
def test_chain_matches_manual_composition():
    cms = make_cms((1, 2, 4))
    x = rand(4, DIM)
    expected = x
    for block in cms.blocks:
        expected = block(expected)
    torch.testing.assert_close(cms(x), expected)
 
 
def test_one_level_is_a_single_block():  # k = 1 is the ordinary FFN case
    cms = make_cms((1,))
    assert len(cms.blocks) == 1
    x = rand(4, DIM)
    torch.testing.assert_close(cms(x), cms.blocks[0](x))
 
 
def test_forward_has_no_side_effects():
    cms = make_cms()
    before = copy.deepcopy(cms.state_dict())
    cms(rand(4, DIM))
    cms(rand(4, DIM))
    after = cms.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert all(p.requires_grad for p in cms.parameters())
 
 
def test_gradients_reach_every_parameter_and_the_input():
    cms = make_cms()
    x = rand(4, DIM).requires_grad_(True)
    loss_fn(cms, x).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for p in cms.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
 
 
# --------------------------------------------------------------------------- B) validation
@pytest.mark.parametrize("bad", [[], [0], [-1], [1.5], [4, 2], [1, 4, 2]])
def test_invalid_chunk_sizes_raise(bad):
    with pytest.raises(ValueError):
        CMS(DIM, (8,), bad)
 
 
def test_non_multiple_chunk_sizes_warn():
    with pytest.warns(UserWarning):
        CMS(DIM, (8,), (2, 3))
 
 
def test_nested_chunk_sizes_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        CMS(DIM, (8,), (1, 2, 4, 8))
 
 
def test_equal_chunk_sizes_allowed():
    CMS(DIM, (8,), (2, 2))
 
 
# --------------------------------------------------------------------------- C) schedule
def test_level_updates_only_on_its_boundaries():
    chunk_sizes = (1, 2, 4, 8)
    cms = make_cms(chunk_sizes)
    opt = CMSOptimizer(cms, 0.05, optimizer_cls=SGD)
    for t in range(1, 17):
        snaps = [snapshot(b) for b in cms.blocks]
        loss_fn(cms, rand(4, DIM, seed=100 + t)).backward()
        fired = opt.step()
        for level, C in enumerate(chunk_sizes):
            due = t % C == 0
            assert fired[level] == due, f"step {t} level {level}: wrong fired flag"
            assert changed(cms.blocks[level], snaps[level]) == due, f"step {t} level {level}"
 
 
def test_step_count_increments():
    cms = make_cms((1, 2))
    opt = CMSOptimizer(cms, 0.01, optimizer_cls=SGD)
    for t in range(1, 6):
        loss_fn(cms, rand(4, DIM)).backward()
        opt.step()
        assert opt.step_count == t
 
 
def test_per_level_learning_rates():
    cms = make_cms((1, 1))
    opt = CMSOptimizer(cms, [0.0, 0.1], optimizer_cls=SGD)
    snaps = [snapshot(b) for b in cms.blocks]
    loss_fn(cms, rand(4, DIM)).backward()
    opt.step()
    assert not changed(cms.blocks[0], snaps[0])  # lr = 0
    assert changed(cms.blocks[1], snaps[1])
 
 
# --------------------------------------------------------------------------- D) accumulation
def test_accumulated_gradient_is_the_sum_over_the_chunk():
    cms = make_cms((2, 4))
    ref = copy.deepcopy(cms)
    opt = CMSOptimizer(cms, [0.0, 1.0], optimizer_cls=SGD)  # level 0 frozen by lr=0, so
    xs = [rand(4, DIM, seed=10 + i) for i in range(4)]       # level 1 sees identical inputs
    for x in xs:
        loss_fn(cms, x).backward()
        opt.step()
    for x in xs:
        loss_fn(ref, x).backward()                           # ref accumulates the 4 grads
    for p_new, p_ref in zip(cms.blocks[1].parameters(), ref.blocks[1].parameters()):
        torch.testing.assert_close(p_new.detach(), p_ref.detach() - 1.0 * p_ref.grad)
 
 
def test_slow_level_accumulates_then_clears_fast_level_always_clears():
    cms = make_cms((1, 4))
    opt = CMSOptimizer(cms, 0.01, optimizer_cls=SGD)
    for t in range(1, 9):
        loss_fn(cms, rand(4, DIM, seed=t)).backward()
        opt.step()
        fast_has_grad = any(p.grad is not None for p in cms.blocks[0].parameters())
        slow_has_grad = any(p.grad is not None for p in cms.blocks[1].parameters())
        assert not fast_has_grad
        assert slow_has_grad == (t % 4 != 0)
 
 
def test_slow_level_is_not_updated_from_a_single_batch():  # v1 used one batch in n
    cms = make_cms((1, 4))
    opt = CMSOptimizer(cms, [0.0, 1.0], optimizer_cls=SGD)
    ref = copy.deepcopy(cms)
    xs = [rand(4, DIM, seed=30 + i) for i in range(4)]
    for x in xs:
        loss_fn(cms, x).backward()
        opt.step()
    ref_last = copy.deepcopy(ref)
    loss_fn(ref_last, xs[-1]).backward()  # what v1 would have applied: last batch only
    last_only = [p.detach() - p.grad for p in ref_last.blocks[1].parameters()]
    assert any(not torch.allclose(p.detach(), q) for p, q in zip(cms.blocks[1].parameters(), last_only))
 
 
# --------------------------------------------------------------------------- E) optimizer API
def test_optimizer_rejects_mismatched_chunk_sizes():
    with pytest.raises(ValueError):
        CMSOptimizer([make_cms((1, 2)), make_cms((1, 4))], 0.01)
 
 
def test_optimizer_rejects_wrong_number_of_lrs():
    with pytest.raises(ValueError):
        CMSOptimizer(make_cms((1, 2, 4)), [0.1, 0.1])
 
 
def test_zero_grad_clears_every_level(): # CMSOptimizer zero_grad function must clear the gradients of every level.
    cms = make_cms((1, 4))
    opt = CMSOptimizer(cms, 0.01, optimizer_cls=SGD)
    loss_fn(cms, rand(4, DIM)).backward()
    opt.step()  # slow level now holds pending grads
    opt.zero_grad()
    assert all(p.grad is None for p in cms.parameters())
 
 
def test_state_dict_roundtrip_preserves_schedule(): # Test to make sure state_dict() and load_state_dict() work properly.
    cms = make_cms((1, 4))
    opt = CMSOptimizer(cms, 0.01, optimizer_cls=SGD)
    for t in range(3):
        loss_fn(cms, rand(4, DIM, seed=t)).backward()
        opt.step()
    state = opt.state_dict()
    opt2 = CMSOptimizer(cms, 0.01, optimizer_cls=SGD)
    opt2.load_state_dict(state)
    assert opt2.step_count == 3
    loss_fn(cms, rand(4, DIM, seed=99)).backward()
    assert opt2.step() == [True, True]  # step 4 is the slow level's boundary
 
 
def test_non_cms_parameters_partitions_the_model():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(DIM, DIM)
            self.cms = CMS(DIM, (8,), (1, 2))
            self.head = nn.Linear(DIM, 2)
 
    toy = Toy()
    others = non_cms_parameters(toy, toy.cms)
    cms_ids = {id(p) for p in toy.cms.parameters()}
    assert all(id(p) not in cms_ids for p in others)
    assert len(others) + len(list(toy.cms.parameters())) == len(list(toy.parameters()))
 
 
# --------------------------------------------------------------------------- F) v1 regressions
def test_every_level_trains():  # v1: only the n=1 block ever received gradients
    chunk_sizes = (1, 2, 4, 8)
    cms = make_cms(chunk_sizes)
    opt = CMSOptimizer(cms, 0.05, optimizer_cls=SGD)
    init = [snapshot(b) for b in cms.blocks]
    for t in range(8):
        loss_fn(cms, rand(4, DIM, seed=t)).backward()
        opt.step()
    for level in range(len(chunk_sizes)):
        assert changed(cms.blocks[level], init[level]), f"level {level} never trained"
 
 
def test_backward_never_fails_on_any_step():  # v1 could produce an output with no grad_fn
    cms = make_cms((2, 4))  # first level slower than every step
    opt = CMSOptimizer(cms, 0.05, optimizer_cls=SGD)
    for t in range(8):
        loss_fn(cms, rand(4, DIM, seed=t)).backward()
        opt.step()
 
 
# --------------------------------------------------------------------------- G) end to end
def test_toy_regression_loss_decreases():
    # Statistical check (seeded); loosen the 0.5 factor if it proves flaky on your setup.
    torch.manual_seed(0)
    cms = CMS(DIM, (32,), (1, 2, 4), use_norm=False)  # no norm: target is exactly linear in x
    W = torch.randn(DIM, DIM) * 0.5
    opt = CMSOptimizer(cms, 1e-2)  # AdamW
 
    def batch():
        x = torch.randn(64, DIM)
        return x, x @ W
 
    x, y = batch()
    initial = (cms(x) - y).pow(2).mean().item()
    for _ in range(400):
        x, y = batch()
        (cms(x) - y).pow(2).mean().backward()
        opt.step()
    x, y = batch()
    final = (cms(x) - y).pow(2).mean().item()
    assert final < 0.5 * initial
