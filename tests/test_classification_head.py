import pytest
import torch
import torch.nn as nn

from model_architecture.classification_head import HybridClassificationHead


@pytest.fixture
def head():
    return HybridClassificationHead(
        embedding_dim=128, num_stat_features=10, num_classes=5, dropout=0.1
    )


@pytest.fixture
def inputs():
    cls_embedding = torch.randn(16, 128)
    stat_features = torch.randn(16, 10)

    return cls_embedding, stat_features


def test_output_shape(head, inputs):
    """Output should have shape (batch_size, num_classes)."""
    cls_embedding, stat_features = inputs

    logits = head(cls_embedding, stat_features)

    assert logits.shape == (16, 5)


def test_output_is_finite(head, inputs):
    """The head should not produce NaN or infinite values."""
    cls_embedding, stat_features = inputs

    logits = head(cls_embedding, stat_features)

    assert torch.isfinite(logits).all()


def test_different_batch_sizes(head):
    """The head should support different batch sizes."""
    for batch_size in [1, 4, 32, 64]:
        cls_embedding = torch.randn(batch_size, 128)
        stat_features = torch.randn(batch_size, 10)

        logits = head(cls_embedding, stat_features)

        assert logits.shape == (batch_size, 5)


def test_gradient_flow(head, inputs):
    """Gradients should flow through both input branches."""
    cls_embedding, stat_features = inputs

    cls_embedding.requires_grad_()
    stat_features.requires_grad_()

    logits = head(cls_embedding, stat_features)
    loss = logits.sum()
    loss.backward()

    assert cls_embedding.grad is not None
    assert stat_features.grad is not None

    assert torch.isfinite(cls_embedding.grad).all()
    assert torch.isfinite(stat_features.grad).all()


def test_parameter_gradients(head, inputs):
    """All trainable head parameters should receive gradients."""
    cls_embedding, stat_features = inputs

    logits = head(cls_embedding, stat_features)
    loss = logits.square().mean()
    loss.backward()

    for name, parameter in head.named_parameters():
        assert parameter.grad is not None, f"Missing gradient for {name}"
        assert torch.isfinite(parameter.grad).all()


def test_invalid_batch_size(head):
    """Inputs with different batch sizes should raise an error."""
    cls_embedding = torch.randn(16, 128)
    stat_features = torch.randn(8, 10)

    with pytest.raises(ValueError):
        head(cls_embedding, stat_features)


def test_invalid_embedding_dimension(head):
    """Incorrect embedding dimensions should raise an error."""
    cls_embedding = torch.randn(16, 64)
    stat_features = torch.randn(16, 10)

    with pytest.raises(ValueError):
        head(cls_embedding, stat_features)


def test_invalid_input_dimensions(head):
    """Inputs must be two-dimensional."""
    cls_embedding = torch.randn(16, 128, 1)
    stat_features = torch.randn(16, 10)

    with pytest.raises(ValueError):
        head(cls_embedding, stat_features)


def test_eval_mode(head, inputs):
    """Evaluation mode should produce deterministic outputs."""
    cls_embedding, stat_features = inputs

    head.eval()

    with torch.no_grad():
        output1 = head(cls_embedding, stat_features)
        output2 = head(cls_embedding, stat_features)

    assert torch.allclose(output1, output2)


def test_output_works_with_cross_entropy(head, inputs):
    """Output logits should work with CrossEntropyLoss."""
    cls_embedding, stat_features = inputs
    labels = torch.randint(0, 5, (16,))

    logits = head(cls_embedding, stat_features)
    loss = nn.CrossEntropyLoss()(logits, labels)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_stat_features_dtype_conversion(head):
    """Statistical features should match the embedding dtype."""
    cls_embedding = torch.randn(4, 128, dtype=torch.float32)
    stat_features = torch.randn(4, 10, dtype=torch.float64)

    logits = head(cls_embedding, stat_features)

    assert logits.dtype == torch.float32
    assert logits.shape == (4, 5)
