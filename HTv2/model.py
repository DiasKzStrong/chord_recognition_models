import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# Hyperparameters
# =========================================================

@dataclass
class HyperParameters:
    n_steps: int
    input_embed_size: int
    n_layers: int
    n_heads: int


# =========================================================
# Positional encodings
# =========================================================

def get_absolute_position_encoding(
    length: int,
    hidden_size: int,
    min_timescale: float = 1.0,
    max_timescale: float = 1e4,
    start_index: int = 0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=dtype) + start_index
    num_timescales = hidden_size // 2

    if num_timescales > 1:
        log_timescale_increment = math.log(max_timescale / min_timescale) / (num_timescales - 1)
    else:
        log_timescale_increment = 0.0

    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, device=device, dtype=dtype) * (-log_timescale_increment)
    )
    scaled_time = position[:, None] * inv_timescales[None, :]
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

    if hidden_size % 2 == 1:
        signal = F.pad(signal, (0, 1))

    return signal.unsqueeze(0)  # [1, L, C]


def get_relative_position_encoding(
    length_q: int,
    length_k: int,
    n_units: int,
    max_dist: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    range_vec_k = torch.arange(length_k, device=device)
    if length_q == length_k:
        range_vec_q = range_vec_k
    else:
        range_vec_q = range_vec_k[-length_q:]

    distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
    distance_mat_clipped = torch.clamp(distance_mat, -max_dist, max_dist)
    final_mat = distance_mat_clipped + max_dist

    vocab_size = max_dist * 2 + 1
    embeddings_table = get_absolute_position_encoding(
        vocab_size,
        n_units,
        device=device,
        dtype=dtype,
    ).squeeze(0)

    embeddings = embeddings_table[final_mat]  # [T_q, T_k, C]
    return embeddings


# =========================================================
# LayerNorm
# =========================================================

class CustomLayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x_norm = (x - mean) * torch.rsqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


# =========================================================
# Straight-through binary round
# =========================================================

class BinaryRoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def binary_round(x: torch.Tensor, cast_to_int: bool = False) -> torch.Tensor:
    y = BinaryRoundSTE.apply(x)
    if cast_to_int:
        return y.long()
    return y


# =========================================================
# Feed-forward blocks
# =========================================================

class FFN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout_rate: float):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_dim, out_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = CustomLayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = x.transpose(1, 2)
        y = F.relu(self.conv1(y))
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.dropout(y)
        y = y.transpose(1, 2)
        y = y + residual
        y = self.norm(y)
        return y


class ConvFFN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout_rate: float):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, out_dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = CustomLayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = x.transpose(1, 2)
        y = F.relu(self.conv1(y))
        y = self.dropout(y)
        y = F.relu(self.conv2(y))
        y = self.dropout(y)
        y = y.transpose(1, 2)
        y = y + residual
        y = self.norm(y)
        return y


# =========================================================
# Multi-head attention
# =========================================================

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_units: int,
        n_heads: int,
        dropout_rate: float,
        relative_position: bool = False,
        max_dist: int = 4,
        positional_attention: bool = False,
    ):
        super().__init__()
        assert n_units % n_heads == 0, "n_units must be divisible by n_heads"

        self.n_units = n_units
        self.n_heads = n_heads
        self.head_dim = n_units // n_heads
        self.relative_position = relative_position
        self.max_dist = max_dist
        self.positional_attention = positional_attention

        self.q_proj = nn.Linear(n_units, n_units)
        self.k_proj = nn.Linear(n_units, n_units)
        self.v_proj = nn.Linear(n_units, n_units)
        self.o_proj = nn.Linear(n_units, n_units)

        if relative_position:
            self.pe_u = nn.Parameter(torch.zeros(n_units))
            self.pe_v = nn.Parameter(torch.zeros(n_units))
            self.rel_pe_proj = nn.Linear(n_units, n_units)

        self.dropout = nn.Dropout(dropout_rate)
        self.norm = CustomLayerNorm(n_units)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        N, T, C = x.shape
        x = x.view(N, T, self.n_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # [N, H, T, D]

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        N, H, T, D = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(N, T, H * D)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        key_mask: Optional[torch.Tensor] = None,
        attention_map: bool = False,
    ):
        if values is None:
            values = keys

        residual = values if self.positional_attention else queries

        Q = self.dropout(self.q_proj(queries))
        K = self.dropout(self.k_proj(keys))
        V = self.dropout(self.v_proj(values))

        N, T_q, _ = Q.shape
        _, T_k, _ = K.shape

        Qh = self._split_heads(Q)
        Kh = self._split_heads(K)
        Vh = self._split_heads(V)

        if not self.relative_position:
            scores = torch.matmul(Qh, Kh.transpose(-1, -2))
        else:
            q_u = Q + self.pe_u.view(1, 1, -1)
            q_u = self._split_heads(q_u)
            ac = torch.matmul(q_u, Kh.transpose(-1, -2))

            rel_pe = get_relative_position_encoding(
                length_q=T_q,
                length_k=T_k,
                n_units=self.n_units,
                max_dist=self.max_dist,
                device=Q.device,
                dtype=Q.dtype,
            )
            rel_pe = self.dropout(self.rel_pe_proj(rel_pe))
            rel_pe = rel_pe.view(T_q, T_k, self.n_heads, self.head_dim).permute(2, 0, 1, 3)

            q_v = Q + self.pe_v.view(1, 1, -1)
            q_v = self._split_heads(q_v)

            bd = torch.einsum("nhtd,htkd->nhtk", q_v, rel_pe)
            scores = ac + bd

        scores = scores / math.sqrt(self.head_dim)

        if key_mask is not None:
            key_mask = key_mask.bool()
            mask = key_mask[:, None, None, :]
            scores = scores.masked_fill(~mask, -1e9)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, Vh)

        out = self._merge_heads(out)
        out = self.dropout(self.o_proj(out))
        out = out + residual
        out = self.norm(out)

        if attention_map:
            attn_map = attn.permute(0, 2, 1, 3).contiguous().view(N, T_q, self.n_heads * T_k)
            return out, attn_map

        return out


# =========================================================
# Chord block compression
# =========================================================

def chord_block_compression(
    hidden_states: torch.Tensor,
    chord_changes: torch.Tensor,
    compression: str = "mean",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if compression not in {"mean", "sum"}:
        raise ValueError("compression must be 'mean' or 'sum'")

    N, T, C = hidden_states.shape
    device = hidden_states.device
    dtype = hidden_states.dtype

    block_ids = torch.cumsum(chord_changes, dim=1)
    change_at_start = (chord_changes[:, 0] == 1).long()
    block_ids = block_ids - change_at_start[:, None]

    num_blocks = block_ids.max(dim=1).values + 1
    max_steps = int(num_blocks.max().item())

    chord_blocks = []
    for b in range(N):
        h = hidden_states[b]
        ids = block_ids[b]
        nb = int(num_blocks[b].item())

        blocks = []
        for seg_id in range(nb):
            seg = h[ids == seg_id]
            if seg.numel() == 0:
                blk = torch.zeros(C, device=device, dtype=dtype)
            else:
                blk = seg.mean(dim=0) if compression == "mean" else seg.sum(dim=0)
            blocks.append(blk)

        blocks = torch.stack(blocks, dim=0)
        if nb < max_steps:
            pad = torch.zeros(max_steps - nb, C, device=device, dtype=dtype)
            blocks = torch.cat([blocks, pad], dim=0)

        chord_blocks.append(blocks)

    chord_blocks = torch.stack(chord_blocks, dim=0)
    return chord_blocks, block_ids, num_blocks


def decode_compressed_sequences(
    compressed_sequences: torch.Tensor,
    block_ids: torch.Tensor,
) -> torch.Tensor:
    outputs = []
    for b in range(block_ids.shape[0]):
        outputs.append(compressed_sequences[b][block_ids[b]])
    return torch.stack(outputs, dim=0)


# =========================================================
# Intra-block attention
# =========================================================

class IntraBlockMHA(nn.Module):
    def __init__(self, n_units: int, n_heads: int, dropout_rate: float):
        super().__init__()
        self.mha = MultiHeadAttention(
            n_units=n_units,
            n_heads=n_heads,
            dropout_rate=dropout_rate,
            relative_position=True,
            max_dist=3,
        )
        self.ffn = ConvFFN(
            in_dim=n_units,
            hidden_dim=n_units,
            out_dim=n_units,
            dropout_rate=dropout_rate,
        )

    def forward(self, inputs: torch.Tensor, n_blocks: int, mask: torch.Tensor) -> torch.Tensor:
        N, T, C = inputs.shape
        assert T % n_blocks == 0, "T must be divisible by n_blocks"

        block_len = T // n_blocks
        blocks = inputs.view(N, n_blocks, block_len, C).reshape(N * n_blocks, block_len, C)
        block_mask = mask.view(N, n_blocks, block_len).reshape(N * n_blocks, block_len)

        blocks = self.mha(blocks, blocks, key_mask=block_mask)
        blocks = self.ffn(blocks)

        blocks = blocks.view(N, n_blocks, block_len, C).reshape(N, T, C)
        return blocks


# =========================================================
# Encoder / Decoder layers
# =========================================================

class HTv2EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_steps: int, dropout_rate: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            n_units=d_model,
            n_heads=n_heads,
            dropout_rate=dropout_rate,
            relative_position=True,
            max_dist=n_steps - 1,
        )
        self.ffn = ConvFFN(d_model, d_model, d_model, dropout_rate)

    def forward(self, x: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        x = self.self_attn(x, x, key_mask=source_mask)
        x = self.ffn(x)
        return x


class HTv2DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_steps: int, dropout_rate: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            n_units=d_model,
            n_heads=n_heads,
            dropout_rate=dropout_rate,
            relative_position=True,
            max_dist=n_steps - 1,
        )
        self.position_attn = MultiHeadAttention(
            n_units=d_model,
            n_heads=n_heads,
            dropout_rate=dropout_rate,
            relative_position=True,
            max_dist=n_steps - 1,
            positional_attention=True,
        )
        self.enc_dec_attn = MultiHeadAttention(
            n_units=d_model,
            n_heads=n_heads,
            dropout_rate=dropout_rate,
            relative_position=False,
        )
        self.ffn = ConvFFN(d_model, d_model, d_model, dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        enc: torch.Tensor,
        dec_pe_batch: torch.Tensor,
        target_mask: torch.Tensor,
        source_mask: torch.Tensor,
    ):
        x, self_attn_map = self.self_attn(
            queries=x,
            keys=x,
            key_mask=target_mask,
            attention_map=True,
        )

        x = self.position_attn(
            queries=dec_pe_batch,
            keys=dec_pe_batch,
            values=x,
            key_mask=target_mask,
        )

        x, attn_map = self.enc_dec_attn(
            queries=x,
            keys=enc,
            key_mask=source_mask,
            attention_map=True,
        )

        x = self.ffn(x)
        return x, self_attn_map, attn_map


# =========================================================
# HTv2 backbone
# =========================================================

class HTv2(nn.Module):
    def __init__(self, input_dim: int, hyperparameters: HyperParameters, dropout_rate: float):
        super().__init__()
        hp = hyperparameters
        self.hp = hp
        self.dropout = nn.Dropout(dropout_rate)

        if hp.n_steps % 4 != 0:
            raise ValueError("hyperparameters.n_steps must be divisible by 4")

        self.enc_input_proj = nn.Linear(input_dim, hp.input_embed_size)
        self.enc_intra_block = IntraBlockMHA(hp.input_embed_size, hp.n_heads, dropout_rate)

        self.enc_layer_weights = nn.Parameter(torch.zeros(hp.n_layers + 1))
        self.encoder_layers = nn.ModuleList([
            HTv2EncoderLayer(hp.input_embed_size, hp.n_heads, hp.n_steps, dropout_rate)
            for _ in range(hp.n_layers)
        ])

        self.chord_change_head = nn.Linear(hp.input_embed_size, 1)

        self.dec_input_proj = nn.Linear(input_dim, hp.input_embed_size)
        self.dec_intra_block = IntraBlockMHA(hp.input_embed_size, hp.n_heads, dropout_rate)

        self.dec_layer_weights = nn.Parameter(torch.zeros(hp.n_layers + 1))
        self.decoder_layers = nn.ModuleList([
            HTv2DecoderLayer(hp.input_embed_size, hp.n_heads, hp.n_steps, dropout_rate)
            for _ in range(hp.n_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,             # [B, T, F]
        source_mask: torch.Tensor,   # [B, T]
        target_mask: torch.Tensor,   # [B, T]
        slope: float,
        chord_change_targets: Optional[torch.Tensor] = None,
        boundary_teacher_forcing_prob: float = 0.0,
    ):
        hp = self.hp
        B, T, _ = x.shape

        if T != hp.n_steps:
            raise ValueError(f"Expected T={hp.n_steps}, got T={T}")

        source_mask = source_mask.bool()
        target_mask = target_mask.bool()

        # Encoder input
        enc_input_embed = self.enc_input_proj(x)
        enc_input_embed = self.dropout(enc_input_embed)
        enc_input_embed = self.enc_intra_block(
            enc_input_embed,
            n_blocks=hp.n_steps // 4,
            mask=source_mask,
        )

        enc_input_embed = enc_input_embed + get_absolute_position_encoding(
            T, hp.input_embed_size, device=x.device, dtype=x.dtype
        )
        enc_input_embed = self.dropout(enc_input_embed)

        # Encoder stack with weighted hidden sum
        enc_weights = F.softmax(self.enc_layer_weights, dim=0)
        enc_weighted_hidden = enc_weights[0] * enc_input_embed

        h_enc = enc_input_embed
        for i, layer in enumerate(self.encoder_layers, start=1):
            h_enc = layer(h_enc, source_mask)
            enc_weighted_hidden = enc_weighted_hidden + enc_weights[i] * h_enc

        enc_output = enc_weighted_hidden

        # Boundary prediction
        chord_change_logits = self.chord_change_head(enc_output).squeeze(-1)   # [B, T]
        chord_change_prob = torch.sigmoid(slope * chord_change_logits)
        chord_change_prediction = binary_round(chord_change_prob, cast_to_int=True)
        chord_change_prediction = chord_change_prediction.clone()
        chord_change_prediction[:, 0] = 0

        regionalization_changes = chord_change_prediction
        if (
            self.training
            and chord_change_targets is not None
            and boundary_teacher_forcing_prob > 0.0
        ):
            use_teacher = torch.rand((), device=x.device) < boundary_teacher_forcing_prob
            if bool(use_teacher.item()):
                regionalization_changes = chord_change_targets.long().clone()
                regionalization_changes[:, 0] = 0

        # Decoder input
        dec_input_embed = self.dec_input_proj(x)
        dec_input_embed = self.dropout(dec_input_embed)
        dec_input_embed = self.dec_intra_block(
            dec_input_embed,
            n_blocks=hp.n_steps // 4,
            mask=target_mask,
        )

        # Regionalization
        dec_input_embed_reg, block_ids, num_blocks = chord_block_compression(
            dec_input_embed,
            regionalization_changes,
            compression="mean",
        )
        dec_input_embed_reg = decode_compressed_sequences(dec_input_embed_reg, block_ids)
        dec_input_embed = dec_input_embed + dec_input_embed_reg + enc_output

        # Decoder PE
        dec_pe = get_absolute_position_encoding(
            hp.n_steps, hp.input_embed_size, device=x.device, dtype=x.dtype
        )
        dec_pe_batch = dec_pe.expand(B, -1, -1)

        dec_input_embed = dec_input_embed + dec_pe_batch
        dec_input_embed = self.dropout(dec_input_embed)

        # Decoder stack with weighted hidden sum
        dec_weights = F.softmax(self.dec_layer_weights, dim=0)
        dec_weighted_hidden = dec_weights[0] * dec_input_embed

        self_attn_map_list = []
        attn_map_list = []

        h_dec = dec_input_embed
        for i, layer in enumerate(self.decoder_layers, start=1):
            h_dec, self_attn_map, attn_map = layer(
                x=h_dec,
                enc=enc_output,
                dec_pe_batch=dec_pe_batch,
                target_mask=target_mask,
                source_mask=source_mask,
            )
            self_attn_map_list.append(self_attn_map)
            attn_map_list.append(attn_map)
            dec_weighted_hidden = dec_weighted_hidden + dec_weights[i] * h_dec

        dec_output = dec_weighted_hidden

        return {
            "chord_change_logits": chord_change_logits,   # [B, T]
            "dec_output": dec_output,                     # [B, T, C]
            "enc_weights": enc_weights,
            "dec_weights": dec_weights,
            "self_attn_maps": self_attn_map_list,
            "attn_maps": attn_map_list,
            "chord_change_prediction": chord_change_prediction,
            "regionalization_changes": regionalization_changes,
        }


# =========================================================
# Classifier model
# =========================================================

class HTv2ChordModel(nn.Module):
    """
    Full model:
      backbone HTv2
      + chord classification head
    """
    def __init__(
        self,
        input_dim: int,
        n_chords: int,
        hyperparameters: HyperParameters,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.backbone = HTv2(
            input_dim=input_dim,
            hyperparameters=hyperparameters,
            dropout_rate=dropout_rate,
        )
        self.chord_classifier = nn.Linear(hyperparameters.input_embed_size, n_chords)

    def forward(
        self,
        x: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
        slope: float,
        chord_change_targets: Optional[torch.Tensor] = None,
        boundary_teacher_forcing_prob: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        out = self.backbone(
            x=x,
            source_mask=source_mask,
            target_mask=target_mask,
            slope=slope,
            chord_change_targets=chord_change_targets,
            boundary_teacher_forcing_prob=boundary_teacher_forcing_prob,
        )

        chord_logits = self.chord_classifier(out["dec_output"])  # [B, T, n_chords]
        out["chord_logits"] = chord_logits
        return out


class StructuredHTv2ChordModel(nn.Module):
    """
    HTv2 backbone with six structured chord-component heads.

    The model predicts root+triad, bass, seventh, ninth, eleventh, and
    thirteenth components. Frame-level full-chord scores are decoded by
    summing component log-probabilities over the fixed legal chord vocabulary.
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
        self.backbone = HTv2(
            input_dim=input_dim,
            hyperparameters=hyperparameters,
            dropout_rate=dropout_rate,
        )
        self.component_names = list(component_sizes.keys())
        self.component_heads = nn.ModuleDict({
            name: nn.Linear(hyperparameters.input_embed_size, size)
            for name, size in component_sizes.items()
        })

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
        out = self.backbone(
            x=x,
            source_mask=source_mask,
            target_mask=target_mask,
            slope=slope,
            chord_change_targets=chord_change_targets,
            boundary_teacher_forcing_prob=boundary_teacher_forcing_prob,
        )

        component_logits = {
            name: self.component_heads[name](out["dec_output"])
            for name in self.component_names
        }

        chord_scores = None
        for component_idx, name in enumerate(self.component_names):
            log_probs = F.log_softmax(component_logits[name], dim=-1)
            chord_component_ids = self.chord_component_ids[:, component_idx]
            component_scores = log_probs.index_select(dim=-1, index=chord_component_ids)
            chord_scores = component_scores if chord_scores is None else chord_scores + component_scores

        out["component_logits"] = component_logits
        out["chord_logits"] = chord_scores
        return out
