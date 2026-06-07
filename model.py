"""
FedGenome Local Models
======================
Two variants — ablate both in the federated experiment:
  Option A: CNN-1D (fast, 64k params)
  Option B: Transformer (stronger, ~2M params)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

NUC_TO_IDX: dict = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 3}
CONTEXT_LEN = 41   # ±20 bp flanking context around variant


def one_hot_encode(seq: str, max_len: int = CONTEXT_LEN) -> torch.Tensor:
    """Encode DNA string as (4, L) one-hot tensor."""
    s = seq.upper().replace("U", "T")[:max_len]
    arr = torch.zeros(4, max_len)
    for i, ch in enumerate(s):
        arr[NUC_TO_IDX.get(ch, 3), i] = 1.0
    return arr


# ─── Option A: CNN-1D ────────────────────────────────────────────────────────

class LocalGenomeNet(nn.Module):
    """
    1D-CNN variant pathogenicity classifier.

    Input:  (B, 4, L)  one-hot nucleotide context
    Output: (B, 2)     logits [benign, pathogenic]
    """

    def __init__(self, in_channels: int = 4, num_classes: int = 2) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x).squeeze(-1))


# ─── Option B: Transformer ───────────────────────────────────────────────────

class NucleotideTransformerNet(nn.Module):
    """
    Lightweight Transformer encoder for variant classification.

    Input:  (B, 4, L) one-hot → transpose to (B, L, 4)
    Output: (B, 2) logits

    Uses fixed sinusoidal PE, 2-layer Transformer encoder.
    For the full model, swap with frozen Nucleotide Transformer:
      InstaDeepAI/nucleotide-transformer-v2-250m-multi-species
    """

    def __init__(
        self,
        in_channels: int = 4,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        num_classes: int = 2,
        max_len: int = CONTEXT_LEN,
    ) -> None:
        super().__init__()
        self.proj   = nn.Linear(in_channels, d_model)
        self.pe     = self._build_pe(max_len, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=0.1, batch_first=True, norm_first=True,
            ),
            num_layers=n_layers,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    @staticmethod
    def _build_pe(max_len: int, d_model: int) -> nn.Parameter:
        pos   = torch.arange(max_len).unsqueeze(1).float()
        omega = torch.exp(
            -math.log(10_000) * torch.arange(0, d_model, 2).float() / d_model
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * omega)
        pe[:, 1::2] = torch.cos(pos * omega)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, L)
        x = x.permute(0, 2, 1)      # → (B, L, 4)
        x = self.proj(x)             # → (B, L, d)
        x = x + self.pe[:, : x.size(1)]
        x = self.encoder(x)          # → (B, L, d)
        cls = x.mean(dim=1)          # mean pooling over sequence
        return self.head(cls)
