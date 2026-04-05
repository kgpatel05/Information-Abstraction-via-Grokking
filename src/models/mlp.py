from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


@dataclass
class MLPConfig:
    input_dim: int
    hidden_dim: int
    output_dim: int
    activation: str = "relu"
    use_bias: bool = True


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class ModularMLP(nn.Module):
    """
    Minimal MLP for modular addition.

    Architecture:
        x -> linear_1 -> activation -> linear_2 -> logits

    We return a dictionary so later analysis code can easily access:
        - hidden pre-activation
        - hidden post-activation
        - logits
    """

    def __init__(self, cfg: MLPConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.linear_1 = nn.Linear(cfg.input_dim, cfg.hidden_dim, bias=cfg.use_bias)
        self.activation = get_activation(cfg.activation)
        self.linear_2 = nn.Linear(cfg.hidden_dim, cfg.output_dim, bias=cfg.use_bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch, input_dim]

        Returns:
            dict with:
                hidden_pre:  [batch, hidden_dim]
                hidden_post: [batch, hidden_dim]
                logits:      [batch, output_dim]
        """
        hidden_pre = self.linear_1(x)
        hidden_post = self.activation(hidden_pre)
        logits = self.linear_2(hidden_post)

        return {
            "hidden_pre": hidden_pre,
            "hidden_post": hidden_post,
            "logits": logits,
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        out = self.forward(x)
        return out["logits"].argmax(dim=-1)