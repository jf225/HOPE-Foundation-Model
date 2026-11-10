# Modular implementation for the contrastive learning loss function.
# Creates two augmented views of each time series and encourages
# representations of the same series to be similar.

import torch
import torch.nn as nn
import torch.nn.functional as F

from timeseries_embed import patch, max_patches, embed_dim


class ContrastiveLoss(nn.Module):
    """
    Contrastive pretraining loss.

    Called as:
        loss = contrastive_loss(encoder, x, valid_length)

    Expects encoder(x, valid_length, hidden=None) to return:
        output: contextualized tokens, shape (B, num_tokens, embed_dim)
        targets: patch targets (unused here)

    Returns one scalar InfoNCE loss.
    """

    def __init__(self, temperature=0.1, noise_ratio=0.02, scale_range=(0.9, 1.1)):
        super().__init__()

        self.temperature = temperature
        self.noise_ratio = noise_ratio
        self.scale_range = scale_range

    def augment(self, x, valid_length):
        """
        Create one augmented view of the input time series.

        x: (batch_size, sequence_length)
        valid_length: (batch_size,)

        Padding is assumed to be at the beginning of each series.
        """

        batch_size, length = x.shape

        # Identify real readings, excluding front padding.
        positions = torch.arange(length, device=x.device)
        is_real = positions.unsqueeze(0) >= (length - valid_length.unsqueeze(1))

        # Calculate standard deviation using only real readings.
        mask = is_real.to(x.dtype)
        count = mask.sum(dim=1, keepdim=True).clamp_min(1)

        mean = (x * mask).sum(dim=1, keepdim=True) / count

        variance = (((x - mean) ** 2) * mask).sum(dim=1, keepdim=True) / count

        std = torch.sqrt(variance).clamp_min(1e-6)

        # Add Gaussian noise to real readings only.
        noise = torch.randn_like(x) * (self.noise_ratio * std)
        noise = noise * mask

        # Random scaling factor for each time series.
        low, high = self.scale_range
        scale = torch.empty(batch_size, 1, device=x.device, dtype=x.dtype).uniform_(
            low, high
        )

        augmented = x * scale + noise

        # Keep padded readings unchanged.
        augmented = torch.where(is_real, augmented, x)

        return augmented

    def forward(self, encoder, x, valid_length):
        batch_size = x.size(0)

        if batch_size < 2:
            raise ValueError(
                "Contrastive loss requires at least 2 time series "
                "in a batch to provide negative examples."
            )

        # Generate two independent augmented views.
        x1 = self.augment(x, valid_length)
        x2 = self.augment(x, valid_length)

        # Encode both views using the same encoder.
        output1, _ = encoder(x1, valid_length, hidden=None)
        output2, _ = encoder(x2, valid_length, hidden=None)

        # CLS token is appended at the end by the embedding layer.
        z1 = output1[:, -1, :]
        z2 = output2[:, -1, :]

        # Normalize representations for cosine similarity.
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Combine both views into one batch.
        representations = torch.cat([z1, z2], dim=0)

        # Compute pairwise similarities.
        similarities = (
            torch.matmul(representations, representations.T) / self.temperature
        )

        # Exclude self-similarity.
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=x.device)

        similarities = similarities.masked_fill(mask, float("-inf"))

        # Positive pair indices:
        # z1[i] matches z2[i], and z2[i] matches z1[i].
        targets = torch.cat(
            [
                torch.arange(batch_size, 2 * batch_size, device=x.device),
                torch.arange(batch_size, device=x.device),
            ]
        )

        # Symmetric InfoNCE loss.
        return F.cross_entropy(similarities, targets)
