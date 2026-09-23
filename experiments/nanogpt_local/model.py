"""A compact decoder-only GPT for the local contextual-feature pilot.

The module follows the architectural organization popularized by Andrej
Karpathy's `nanoGPT` project (MIT License), while providing a fresh local
implementation tailored to this repository.  In particular, it exposes the
final normalized token representations needed by the frozen-feature stage.

Upstream design reference: https://github.com/karpathy/nanoGPT
"""

from __future__ import annotations

import inspect
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GPTConfig:
    """Architecture settings for a decoder-only GPT.

    The local approximately 30M-parameter setting is obtained with six
    layers, six heads, embedding width 384, and a GPT-2 tokenizer-sized
    vocabulary.  ``block_size`` can be reduced independently for an MPS
    memory/throughput tradeoff.
    """

    block_size: int = 256
    vocab_size: int = 50_304
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = True

    def __post_init__(self) -> None:
        integer_fields = {
            "block_size": self.block_size,
            "vocab_size": self.vocab_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
        }
        for name, value in integer_fields.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")


class LayerNorm(nn.Module):
    """LayerNorm with an optional learnable bias."""

    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention implemented with PyTorch SDPA."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_size = channels // self.n_head
        q = q.view(batch_size, sequence_length, self.n_head, head_size).transpose(1, 2)
        k = k.view(batch_size, sequence_length, self.n_head, head_size).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.n_head, head_size).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, sequence_length, channels)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    """Decoder-only GPT that can return final normalized hidden states."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": LayerNorm(config.n_embd, bias=config.bias),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying reduces the 6x384 model to approximately 30M parameters
        # with a GPT-2-sized vocabulary and matches the canonical GPT design.
        self.lm_head.weight = self.transformer.wte.weight

        self.apply(self._init_weights)
        residual_std = 0.02 / math.sqrt(2 * config.n_layer)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(parameter, mean=0.0, std=residual_std)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return trainable parameter count, optionally excluding positions."""

        count = sum(parameter.numel() for parameter in self.parameters())
        if non_embedding:
            count -= self.transformer.wpe.weight.numel()
        return count

    def hidden_states(self, idx: torch.Tensor) -> torch.Tensor:
        """Return final normalized states without materializing vocabulary logits."""
        if idx.ndim != 2:
            raise ValueError(f"idx must have shape (batch, time), got {tuple(idx.shape)}")
        batch_size, sequence_length = idx.shape
        del batch_size
        if sequence_length > self.config.block_size:
            raise ValueError(
                f"sequence length {sequence_length} exceeds block_size "
                f"{self.config.block_size}"
            )
        if idx.dtype != torch.long:
            idx = idx.long()

        positions = torch.arange(0, sequence_length, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(positions))
        for block in self.transformer.h:
            x = block(x)
        return self.transformer.ln_f(x)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Evaluate tokens and optionally return final normalized states.

        Returns ``(logits, loss, hidden)``.  ``hidden`` has shape
        ``(batch, time, n_embd)`` when requested and is otherwise ``None``.
        Feature extraction should call :meth:`hidden_states` directly so it
        does not allocate a full ``vocab_size`` logit tensor.
        """

        hidden = self.hidden_states(idx)
        logits = self.lm_head(hidden)

        loss = None
        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError(
                    f"targets must match idx shape {tuple(idx.shape)}, "
                    f"got {tuple(targets.shape)}"
                )
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.long().reshape(-1),
                ignore_index=-1,
            )
        return logits, loss, hidden if return_hidden else None

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.AdamW:
        """Build AdamW with decay restricted to matrix-like parameters."""

        parameters = {name: value for name, value in self.named_parameters() if value.requires_grad}
        decay = [value for value in parameters.values() if value.dim() >= 2]
        no_decay = [value for value in parameters.values() if value.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        adamw_kwargs: dict[str, Any] = {
            "lr": learning_rate,
            "betas": betas,
        }
        if device_type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters:
            adamw_kwargs["fused"] = True
        return torch.optim.AdamW(groups, **adamw_kwargs)


def _strip_compile_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "_orig_mod."
    return {
        key[len(prefix) :] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[GPT, dict[str, Any]]:
    """Reconstruct a :class:`GPT` and return it with its checkpoint payload."""

    checkpoint_path = Path(path).expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint must be a mapping: {checkpoint_path}")
    if "model_config" not in payload or "model_state" not in payload:
        raise ValueError("checkpoint is missing model_config or model_state")
    config = GPTConfig(**payload["model_config"])
    model = GPT(config)
    model.load_state_dict(_strip_compile_prefix(payload["model_state"]))
    model.to(device)
    return model, payload


def model_config_dict(model: GPT) -> dict[str, Any]:
    """Return a JSON/checkpoint-friendly architecture dictionary."""

    return asdict(model.config)
