import torch
import torch.nn as nn


class HybridClassificationHead(nn.Module):
    def __init__(self, embedding_dim, num_stat_features, num_classes, dropout=0.1):
        super().__init__()

        self.input_dim = embedding_dim + num_stat_features

        self.norm = nn.LayerNorm(self.input_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.input_dim, num_classes)

    def forward(self, cls_embedding, stat_features):
        """
        cls_embedding: (batch_size, embedding_dim)
        stat_features: (batch_size, num_stat_features)

        Returns:
            logits: (batch_size, num_classes)
        """

        stat_features = stat_features.to(
            device=cls_embedding.device, dtype=cls_embedding.dtype
        )

        # Combine learned and statistical features
        x = torch.cat([cls_embedding, stat_features], dim=-1)

        x = self.norm(x)
        x = self.dropout(x)
        logits = self.classifier(x)

        return logits
