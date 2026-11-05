import torch
import torch.nn as nn


patch = 8
max_patches = 32
embed_dim = 128


class Embedding(nn.Module):
    def __init__(self):
        super().__init__()

        # RevIN's learnable scale and shift
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))

        # Turns each patch of 8 readings into a token of 128 numbers
        self.patch_to_token = nn.Linear(patch, embed_dim)

        # Learned tokens and positions, trained by the loss like any other weight
        self.cls_token = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.pad_token = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.positions = nn.Parameter(torch.randn(max_patches + 1, embed_dim) * 0.02)

    def revin(self, x, is_real):
        # Mean and spread come from the real readings only
        count = is_real.sum(dim=1, keepdim=True)
        mean = (x * is_real).sum(dim=1, keepdim=True) / count
        var = ((x - mean) ** 2 * is_real).sum(dim=1, keepdim=True) / count
        return (x - mean) / torch.sqrt(var + 1e-5) * is_real

    def forward(self, x, valid_length=None, hidden=None):
        batch, length = x.shape
        num_patches = length // patch

        # UCR has no padding, so every reading is real
        if valid_length is None:
            valid_length = torch.full((batch,), length, device=x.device)

        # Padding is at the front, so the real readings are the last valid_length ones
        is_real = (torch.arange(length, device=x.device) >= length - valid_length[:, None]).float()

        normalized = self.revin(x, is_real)
        targets = normalized.view(batch, num_patches, patch)
        tokens = self.patch_to_token(targets * self.scale + self.shift)

        # Padding fills whole patches, so checking each patch's first reading is enough
        patch_is_real = is_real[:, ::patch] > 0
        tokens = torch.where(patch_is_real[:, :, None], tokens, self.pad_token)

        # Patches hidden for pretraining become the mask token
        if hidden is not None:
            tokens = torch.where(hidden[:, :, None], self.mask_token, tokens)

        # CLS goes on the end so the memory reads it last
        cls = self.cls_token.expand(batch, 1, embed_dim)
        tokens = torch.cat([tokens, cls], dim=1)

        # Use the last num_patches + 1 positions so CLS always gets the same one
        tokens = tokens + self.positions[-(num_patches + 1):]

        return tokens, targets


def pick_hidden_patches(valid_length, ratio=0.4):
    # Randomly hide 40% of each window's real patches, never padding
    hidden = torch.zeros(len(valid_length), max_patches, dtype=torch.bool, device=valid_length.device)

    for i, length in enumerate(valid_length.tolist()):
        real_patches = length // patch
        chosen = torch.randperm(real_patches)[:round(real_patches * ratio)]
        hidden[i, max_patches - real_patches + chosen] = True

    return hidden