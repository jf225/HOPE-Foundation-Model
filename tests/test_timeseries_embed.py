import pytest
import torch

from model_architecture.timeseries_embed import Embedding, patch, max_patches, embed_dim


@pytest.fixture
def embedding():
    return Embedding()


@pytest.fixture
def sample_input():
    # Two time series, each with 32 readings (4 patches).
    return torch.randn(2, 32)


def test_output_shapes(embedding, sample_input):
    """Output should contain one token per patch plus the CLS token."""
    tokens, targets = embedding(sample_input)

    assert tokens.shape == (2, 5, embed_dim)
    assert targets.shape == (2, 4, patch)


def test_output_shapes_with_padding(embedding):
    """Variable-length series should still produce the same number of tokens."""
    x = torch.randn(2, 32)
    valid_length = torch.tensor([16, 24])

    tokens, targets = embedding(x, valid_length)

    assert tokens.shape == (2, 5, embed_dim)
    assert targets.shape == (2, 4, patch)


def test_revin_normalizes_real_readings(embedding):
    """Real readings should have approximately zero mean and unit variance."""
    x = torch.randn(2, 32)
    valid_length = torch.tensor([16, 24])

    _, targets = embedding(x, valid_length)

    for i, length in enumerate(valid_length):
        num_real_patches = length.item() // patch
        real_targets = targets[i, -num_real_patches:, :].reshape(-1)

        assert real_targets.mean().item() == pytest.approx(0.0, abs=1e-5)
        assert real_targets.std(unbiased=False).item() == pytest.approx(1.0, abs=1e-3)


def test_padding_targets_are_zero(embedding):
    """Front-padded readings should be zero after RevIN normalization."""
    x = torch.randn(1, 32)
    valid_length = torch.tensor([16])

    _, targets = embedding(x, valid_length)

    # First two patches are padding.
    assert torch.allclose(targets[0, :2], torch.zeros(2, patch))


def test_padding_uses_pad_token(embedding):
    """Padded patches should use the learned pad token."""
    x = torch.randn(1, 32)
    valid_length = torch.tensor([16])

    tokens, _ = embedding(x, valid_length)

    # There are four patches, with the first two padded.
    # Positions are added after replacing tokens with pad_token.
    expected_first = embedding.pad_token + embedding.positions[-5]
    expected_second = embedding.pad_token + embedding.positions[-4]

    torch.testing.assert_close(tokens[0, 0], expected_first)
    torch.testing.assert_close(tokens[0, 1], expected_second)


def test_hidden_patches_use_mask_token(embedding):
    """Hidden patches should be replaced with the learned mask token."""
    x = torch.randn(1, 32)
    hidden = torch.zeros(1, 4, dtype=torch.bool)
    hidden[0, 1] = True

    tokens, _ = embedding(x, hidden=hidden)

    # The second patch is hidden.
    expected = embedding.mask_token + embedding.positions[-4]

    torch.testing.assert_close(tokens[0, 1], expected)


def test_unhidden_patches_are_not_masked(embedding):
    """Patches not selected for masking should retain their projected values."""
    x = torch.randn(1, 32)
    hidden = torch.zeros(1, 4, dtype=torch.bool)
    hidden[0, 1] = True

    tokens, targets = embedding(x, hidden=hidden)

    expected = (
        embedding.patch_to_token(targets * embedding.scale + embedding.shift)[0, 0]
        + embedding.positions[-5]
    )

    torch.testing.assert_close(tokens[0, 0], expected)


def test_cls_token_is_at_end(embedding, sample_input):
    """The CLS token should be appended after all patch tokens."""
    tokens, _ = embedding(sample_input)

    expected = embedding.cls_token + embedding.positions[-1]

    for i in range(sample_input.shape[0]):
        torch.testing.assert_close(tokens[i, -1], expected)


def test_cls_token_position_is_consistent_across_lengths(embedding):
    """CLS should always use the final positional embedding."""
    x1 = torch.randn(1, 32)
    x2 = torch.randn(1, 64)

    tokens1, _ = embedding(x1)
    tokens2, _ = embedding(x2)

    expected = embedding.cls_token + embedding.positions[-1]

    torch.testing.assert_close(tokens1[0, -1], expected)
    torch.testing.assert_close(tokens2[0, -1], expected)


def test_learnable_parameters_receive_gradients(embedding):
    """Learnable embedding parameters should receive gradients."""
    x = torch.randn(2, 32)
    valid_length = torch.tensor([16, 32])

    hidden = torch.zeros(2, 4, dtype=torch.bool)
    hidden[0, 2] = True

    tokens, _ = embedding(x, valid_length, hidden)
    loss = tokens.square().mean()
    loss.backward()

    parameters = [
        embedding.scale,
        embedding.shift,
        embedding.patch_to_token.weight,
        embedding.patch_to_token.bias,
        embedding.cls_token,
        embedding.pad_token,
        embedding.mask_token,
        embedding.positions,
    ]

    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_batch_size_one(embedding):
    """The embedding layer should support a batch containing one series."""
    x = torch.randn(1, 32)

    tokens, targets = embedding(x)

    assert tokens.shape == (1, 5, embed_dim)
    assert targets.shape == (1, 4, patch)


def test_output_is_finite(embedding, sample_input):
    """Embedding outputs should not contain NaN or infinite values."""
    tokens, targets = embedding(sample_input)

    assert torch.isfinite(tokens).all()
    assert torch.isfinite(targets).all()


def test_default_valid_length(embedding, sample_input):
    """Without valid_length, every reading should be treated as real."""
    tokens_default, targets_default = embedding(sample_input)

    valid_length = torch.full((2,), 32)
    tokens_explicit, targets_explicit = embedding(sample_input, valid_length)

    torch.testing.assert_close(tokens_default, tokens_explicit)
    torch.testing.assert_close(targets_default, targets_explicit)
