from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


@dataclass
class TransformerConfig:
    vocab_size: int
    seq_len: int
    d_model: int
    n_heads: int
    d_mlp: int
    n_layers: int
    output_dim: int
    use_bias: bool = True


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_mlp: int, use_bias: bool = True) -> None:
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
            bias=use_bias,
        )
        self.ln2 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp, bias=use_bias),
            nn.GELU(),
            nn.Linear(d_mlp, d_model, bias=use_bias),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: [B, T, d_model]
        x_ln = self.ln1(x)
        attn_out, attn_weights = self.attn(x_ln, x_ln, x_ln, need_weights=True)
        x = x + attn_out

        x_ln2 = self.ln2(x)
        mlp_out = self.mlp(x_ln2)
        x = x + mlp_out

        return {
            "x": x,
            "attn_out": attn_out,
            "attn_weights": attn_weights,
            "mlp_out": mlp_out,
        }


class ModularTransformer(nn.Module):
    """
    Tiny transformer for modular addition.

    Input:
        token ids of shape [B, T]

    Output:
        logits over modular classes, using the final token position.
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.seq_len, cfg.d_model)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    d_mlp=cfg.d_mlp,
                    use_bias=cfg.use_bias,
                )
                for _ in range(cfg.n_layers)
            ]
        )

        self.final_ln = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.output_dim, bias=cfg.use_bias)

    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        tokens: [B, T] of dtype long
        """
        bsz, seq_len = tokens.shape
        assert seq_len == self.cfg.seq_len, (
            f"Expected seq_len={self.cfg.seq_len}, got {seq_len}"
        )

        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)  # [1, T]

        tok_emb = self.token_embed(tokens)              # [B, T, d_model]
        pos_emb = self.pos_embed(positions)             # [1, T, d_model]
        x = tok_emb + pos_emb                           # [B, T, d_model]

        block_outputs = []
        for block in self.blocks:
            out = block(x)
            x = out["x"]
            block_outputs.append(out)

        x_final = self.final_ln(x)                      # [B, T, d_model]
        final_token_rep = x_final[:, -1, :]            # [B, d_model]
        logits = self.unembed(final_token_rep)         # [B, output_dim]

        return {
            "token_embeddings": tok_emb,
            "x_final": x_final,
            "final_token_rep": final_token_rep,
            "logits": logits,
            "block_outputs": block_outputs,
        }

    @torch.no_grad()
    def predict(self, tokens: torch.Tensor) -> torch.Tensor:
        out = self.forward(tokens)
        return out["logits"].argmax(dim=-1)