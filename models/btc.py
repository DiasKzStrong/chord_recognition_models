from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .htv2 import CustomLayerNorm, HyperParameters, get_absolute_position_encoding


class BTCDirectionalSelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout_rate: float, direction: str):
        super().__init__()
        if direction not in {"forward", "backward"}:
            raise ValueError("direction must be 'forward' or 'backward'")

        self.direction = direction
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.attn_norm = CustomLayerNorm(d_model)
        self.ffn_norm = CustomLayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout_rate),
        )

    def _attn_mask(self, length: int, device: torch.device) -> torch.Tensor:
        if self.direction == "forward":
            return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)
        return torch.tril(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.attn_norm(x)
        attn_output, _ = self.attn(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=self._attn_mask(x.shape[1], x.device),
            key_padding_mask=~mask.bool(),
            need_weights=False,
        )
        x = residual + self.dropout(attn_output)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class BTCBidirectionalLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout_rate: float):
        super().__init__()
        self.forward_block = BTCDirectionalSelfAttentionBlock(
            d_model=d_model,
            n_heads=n_heads,
            dropout_rate=dropout_rate,
            direction="forward",
        )
        self.backward_block = BTCDirectionalSelfAttentionBlock(
            d_model=d_model,
            n_heads=n_heads,
            dropout_rate=dropout_rate,
            direction="backward",
        )
        self.merge = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        forward_out = self.forward_block(x, mask)
        backward_out = self.backward_block(x, mask)
        return self.merge(torch.cat([forward_out, backward_out], dim=-1))


class BTCBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hyperparameters: HyperParameters,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.d_model = hyperparameters.input_embed_size
        self.max_steps = hyperparameters.n_steps

        self.input_dropout = nn.Dropout(dropout_rate)
        self.input_proj = nn.Linear(input_dim, self.d_model, bias=False)
        self.layers = nn.ModuleList([
            BTCBidirectionalLayer(
                d_model=self.d_model,
                n_heads=hyperparameters.n_heads,
                dropout_rate=dropout_rate,
            )
            for _ in range(hyperparameters.n_layers)
        ])
        self.output_norm = CustomLayerNorm(self.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = self.input_dropout(x)
        y = self.input_proj(y)
        y = y + get_absolute_position_encoding(
            length=y.shape[1],
            hidden_size=self.d_model,
            device=y.device,
            dtype=y.dtype,
        )
        for layer in self.layers:
            y = layer(y, mask)
        return self.output_norm(y)


class BTCChordModel(nn.Module):
    """
    Pure BTC backbone with the same output interface used by HTv2 training:
    framewise chord logits + boundary logits.
    """
    def __init__(
        self,
        input_dim: int,
        n_chords: int,
        hyperparameters: HyperParameters,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.backbone = BTCBackbone(
            input_dim=input_dim,
            hyperparameters=hyperparameters,
            dropout_rate=dropout_rate,
        )
        self.chord_classifier = nn.Linear(hyperparameters.input_embed_size, n_chords)
        self.chord_change_head = nn.Linear(hyperparameters.input_embed_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
        slope: float,
        chord_change_targets: Optional[torch.Tensor] = None,
        boundary_teacher_forcing_prob: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        del target_mask, slope, chord_change_targets, boundary_teacher_forcing_prob

        enc_output = self.backbone(x=x, mask=source_mask)
        return {
            "enc_output": enc_output,
            "chord_logits": self.chord_classifier(enc_output),
            "chord_change_logits": self.chord_change_head(enc_output).squeeze(-1),
        }


class StructuredBTCChordModel(nn.Module):
    """
    Pure BTC backbone with the same structured full-chord objective used by HTv2.
    """
    def __init__(
        self,
        input_dim: int,
        component_sizes: Dict[str, int],
        chord_component_ids,
        hyperparameters: HyperParameters,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.backbone = BTCBackbone(
            input_dim=input_dim,
            hyperparameters=hyperparameters,
            dropout_rate=dropout_rate,
        )
        self.component_names = list(component_sizes.keys())
        self.component_heads = nn.ModuleDict({
            name: nn.Linear(hyperparameters.input_embed_size, size)
            for name, size in component_sizes.items()
        })
        self.chord_change_head = nn.Linear(hyperparameters.input_embed_size, 1)

        chord_component_ids = torch.as_tensor(chord_component_ids, dtype=torch.long)
        self.register_buffer("chord_component_ids", chord_component_ids, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
        slope: float,
        chord_change_targets: Optional[torch.Tensor] = None,
        boundary_teacher_forcing_prob: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        del target_mask, slope, chord_change_targets, boundary_teacher_forcing_prob

        enc_output = self.backbone(x=x, mask=source_mask)
        component_logits = {
            name: self.component_heads[name](enc_output)
            for name in self.component_names
        }

        chord_scores = None
        for component_idx, name in enumerate(self.component_names):
            log_probs = F.log_softmax(component_logits[name], dim=-1)
            component_ids = self.chord_component_ids[:, component_idx]
            component_scores = log_probs.index_select(dim=-1, index=component_ids)
            chord_scores = component_scores if chord_scores is None else chord_scores + component_scores

        return {
            "enc_output": enc_output,
            "component_logits": component_logits,
            "chord_logits": chord_scores,
            "chord_change_logits": self.chord_change_head(enc_output).squeeze(-1),
        }
