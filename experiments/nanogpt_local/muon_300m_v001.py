"""The repository's hybrid Muon optimizer for the 303M-parameter GPT."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import torch

from . import muon_124m_v001 as prior
from .model import GPT


STATE_SCHEMA = "nanogpt300m_hybrid_muon_state_v001"
EXPECTED_MUON_TENSORS = 80
EXPECTED_MUON_PARAMETERS = 251_658_240
EXPECTED_AUX_TENSORS = 43
EXPECTED_AUX_PARAMETERS = 51_815_424


def _sha256(value: Any) -> str:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parameter_partition(model: GPT) -> dict[str, Any]:
    muon = []
    auxiliary = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (muon if name.startswith("transformer.h.") and parameter.ndim == 2 else auxiliary).append(
            (name, parameter)
        )
    report = {
        "rule": "Muon for 2D transformer block weights; AdamW for embeddings, norms, and tied output weight",
        "muon_tensor_count": len(muon),
        "muon_parameter_count": sum(parameter.numel() for _, parameter in muon),
        "aux_adamw_tensor_count": len(auxiliary),
        "aux_adamw_parameter_count": sum(parameter.numel() for _, parameter in auxiliary),
        "muon_parameter_names": [name for name, _ in muon],
        "aux_adamw_parameter_names": [name for name, _ in auxiliary],
    }
    if (
        report["muon_tensor_count"],
        report["muon_parameter_count"],
        report["aux_adamw_tensor_count"],
        report["aux_adamw_parameter_count"],
    ) != (
        EXPECTED_MUON_TENSORS,
        EXPECTED_MUON_PARAMETERS,
        EXPECTED_AUX_TENSORS,
        EXPECTED_AUX_PARAMETERS,
    ):
        raise RuntimeError("300M hybrid Muon parameter partition changed")
    report["sha256"] = _sha256(report)
    return report


class HybridMuon:
    """Native Muon on hidden matrices, fused AdamW on the remaining weights."""

    def __init__(self, model: GPT, learning_rate: float) -> None:
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("this PyTorch build does not provide torch.optim.Muon")
        self.partition = parameter_partition(model)
        muon_names = set(self.partition["muon_parameter_names"])
        named = list(model.named_parameters())
        matrices = [parameter for name, parameter in named if name in muon_names]
        other = [parameter for name, parameter in named if name not in muon_names]
        self.muon = torch.optim.Muon(
            matrices,
            lr=float(learning_rate),
            weight_decay=prior.WEIGHT_DECAY,
            momentum=prior.MUON_MOMENTUM,
            nesterov=prior.MUON_NESTEROV,
            ns_steps=prior.MUON_NS_STEPS,
            eps=prior.MUON_EPS,
            adjust_lr_fn=prior.MUON_ADJUST_LR_FN,
        )
        self.aux_adamw = torch.optim.AdamW(
            [
                {"params": [parameter for parameter in other if parameter.ndim >= 2], "weight_decay": prior.WEIGHT_DECAY},
                {"params": [parameter for parameter in other if parameter.ndim < 2], "weight_decay": 0.0},
            ],
            lr=float(learning_rate),
            betas=prior.AUX_ADAMW_BETAS,
            eps=prior.AUX_ADAMW_EPS,
            fused=True,
        )
        if self.aux_adamw.defaults.get("fused") is not True:
            raise RuntimeError("fused auxiliary AdamW is unavailable")

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return [*self.muon.param_groups, *self.aux_adamw.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.aux_adamw.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.muon.step()
        self.aux_adamw.step()

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STATE_SCHEMA,
            "partition_sha256": self.partition["sha256"],
            "muon": self.muon.state_dict(),
            "aux_adamw": self.aux_adamw.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (
            type(state) is not dict
            or state.get("schema_version") != STATE_SCHEMA
            or state.get("partition_sha256") != self.partition["sha256"]
        ):
            raise RuntimeError("300M hybrid Muon optimizer-state identity changed")
        self.muon.load_state_dict(state["muon"])
        self.aux_adamw.load_state_dict(state["aux_adamw"])


def optimizer_state_fp32_and_finite(optimizer: HybridMuon) -> bool:
    if not optimizer.muon.state or not optimizer.aux_adamw.state:
        return False
    for state in optimizer.muon.state.values():
        value = state.get("momentum_buffer")
        if not torch.is_tensor(value) or value.dtype != torch.float32 or not bool(torch.all(torch.isfinite(value)).detach().cpu()):
            return False
    for state in optimizer.aux_adamw.state.values():
        for name in ("exp_avg", "exp_avg_sq"):
            value = state.get(name)
            if not torch.is_tensor(value) or value.dtype != torch.float32 or not bool(torch.all(torch.isfinite(value)).detach().cpu()):
                return False
    return True


def optimizer_manifest(learning_rate: float | None, gradient_clip: float | None) -> dict[str, Any]:
    return {
        "name": "hybrid_muon",
        "nominal_learning_rate": learning_rate,
        "muon": {
            "implementation": "torch.optim.Muon",
            "momentum": prior.MUON_MOMENTUM,
            "nesterov": prior.MUON_NESTEROV,
            "ns_steps": prior.MUON_NS_STEPS,
            "eps": prior.MUON_EPS,
            "adjust_lr_fn": prior.MUON_ADJUST_LR_FN,
            "weight_decay": prior.WEIGHT_DECAY,
        },
        "aux_adamw": {
            "betas": list(prior.AUX_ADAMW_BETAS),
            "eps": prior.AUX_ADAMW_EPS,
            "weight_decay": prior.WEIGHT_DECAY,
            "fused": True,
            "decay_matrix_parameters_only": True,
        },
        "gradient_clip": gradient_clip,
        "shared_nominal_lr_schedule": True,
        "start_from_initialization": True,
    }
