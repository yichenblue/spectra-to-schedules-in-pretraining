"""Minimal native-PyTorch hybrid Muon optimizer for the 124M GPT model."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import torch

from .model import GPT


STATE_SCHEMA = "nanogpt124m_hybrid_muon_state_v001"
MUON_MOMENTUM = 0.95
MUON_NESTEROV = True
MUON_NS_STEPS = 5
MUON_EPS = 1e-7
MUON_ADJUST_LR_FN = "match_rms_adamw"
AUX_ADAMW_BETAS = (0.9, 0.999)
AUX_ADAMW_EPS = 1e-8
WEIGHT_DECAY = 0.1
EXPECTED_MUON_TENSORS = 48
EXPECTED_MUON_PARAMETERS = 84_934_656
EXPECTED_AUX_TENSORS = 27
EXPECTED_AUX_PARAMETERS = 38_849_280


def _canonical_sha256(value: Any) -> str:
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parameter_partition(model: GPT) -> dict[str, Any]:
    muon: list[tuple[str, torch.nn.Parameter]] = []
    auxiliary: list[tuple[str, torch.nn.Parameter]] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = (
            muon
            if name.startswith("transformer.h.") and parameter.ndim == 2
            else auxiliary
        )
        target.append((name, parameter))
    names = [name for name, _ in muon] + [name for name, _ in auxiliary]
    if len(names) != len(set(names)):
        raise RuntimeError("hybrid Muon parameter partition contains duplicates")
    report = {
        "rule": "Muon for 2D transformer block weights; AdamW for embeddings, norms, and tied output weight",
        "muon_tensor_count": len(muon),
        "muon_parameter_count": sum(parameter.numel() for _, parameter in muon),
        "aux_adamw_tensor_count": len(auxiliary),
        "aux_adamw_parameter_count": sum(
            parameter.numel() for _, parameter in auxiliary
        ),
        "muon_parameter_names": [name for name, _ in muon],
        "aux_adamw_parameter_names": [name for name, _ in auxiliary],
    }
    report["sha256"] = _canonical_sha256(report)
    expected = (
        EXPECTED_MUON_TENSORS,
        EXPECTED_MUON_PARAMETERS,
        EXPECTED_AUX_TENSORS,
        EXPECTED_AUX_PARAMETERS,
    )
    observed = (
        report["muon_tensor_count"],
        report["muon_parameter_count"],
        report["aux_adamw_tensor_count"],
        report["aux_adamw_parameter_count"],
    )
    if observed != expected:
        raise RuntimeError(f"hybrid Muon parameter partition changed: {observed}")
    return report


class HybridMuon:
    """Muon for hidden matrices plus fused AdamW for all remaining parameters."""

    def __init__(self, model: GPT, learning_rate: float) -> None:
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("this PyTorch build does not provide torch.optim.Muon")
        partition = parameter_partition(model)
        named = list(model.named_parameters())
        muon_params = [
            parameter
            for name, parameter in named
            if name in set(partition["muon_parameter_names"])
        ]
        auxiliary_named = [
            (name, parameter)
            for name, parameter in named
            if name in set(partition["aux_adamw_parameter_names"])
        ]
        decay = [parameter for _, parameter in auxiliary_named if parameter.ndim >= 2]
        no_decay = [parameter for _, parameter in auxiliary_named if parameter.ndim < 2]
        self.partition = partition
        self.muon = torch.optim.Muon(
            muon_params,
            lr=float(learning_rate),
            weight_decay=WEIGHT_DECAY,
            momentum=MUON_MOMENTUM,
            nesterov=MUON_NESTEROV,
            ns_steps=MUON_NS_STEPS,
            eps=MUON_EPS,
            adjust_lr_fn=MUON_ADJUST_LR_FN,
        )
        self.aux_adamw = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": WEIGHT_DECAY},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=float(learning_rate),
            betas=AUX_ADAMW_BETAS,
            eps=AUX_ADAMW_EPS,
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
            raise RuntimeError("hybrid Muon optimizer-state identity changed")
        self.muon.load_state_dict(state["muon"])
        self.aux_adamw.load_state_dict(state["aux_adamw"])


def optimizer_state_fp32_and_finite(optimizer: HybridMuon) -> bool:
    if not optimizer.muon.state or not optimizer.aux_adamw.state:
        return False
    for state in optimizer.muon.state.values():
        buffer = state.get("momentum_buffer")
        if (
            not torch.is_tensor(buffer)
            or buffer.dtype != torch.float32
            or not bool(torch.all(torch.isfinite(buffer)).detach().cpu())
        ):
            return False
    for state in optimizer.aux_adamw.state.values():
        for name in ("exp_avg", "exp_avg_sq"):
            value = state.get(name)
            if (
                not torch.is_tensor(value)
                or value.dtype != torch.float32
                or not bool(torch.all(torch.isfinite(value)).detach().cpu())
            ):
                return False
    return True


def optimizer_manifest(learning_rate: float, gradient_clip: float) -> dict[str, Any]:
    return {
        "name": "hybrid_muon",
        "nominal_learning_rate": float(learning_rate),
        "muon": {
            "implementation": "torch.optim.Muon",
            "momentum": MUON_MOMENTUM,
            "nesterov": MUON_NESTEROV,
            "ns_steps": MUON_NS_STEPS,
            "eps": MUON_EPS,
            "adjust_lr_fn": MUON_ADJUST_LR_FN,
            "weight_decay": WEIGHT_DECAY,
        },
        "aux_adamw": {
            "betas": list(AUX_ADAMW_BETAS),
            "eps": AUX_ADAMW_EPS,
            "weight_decay": WEIGHT_DECAY,
            "fused": True,
            "decay_matrix_parameters_only": True,
        },
        "gradient_clip": float(gradient_clip),
        "shared_nominal_lr_schedule": True,
    }
