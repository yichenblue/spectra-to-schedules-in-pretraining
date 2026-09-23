"""Combine the 124M SGD and 300M Muon validation-risk figures."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
from matplotlib.ticker import FuncFormatter

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ARMS = (
    (
        "wsd_exp_80_20__fixed_batch_lr",
        "WSD · fixed batch / LR schedule",
        "#1764ab",
    ),
    (
        "wsd_exp_80_20__fixed_lr_batch",
        "WSD · fixed LR / batch schedule",
        "#1764ab",
    ),
    (
        "eight_one_one__fixed_batch_lr",
        "8-1-1 · fixed batch / LR schedule",
        "#e07a1f",
    ),
    (
        "eight_one_one__fixed_lr_batch",
        "8-1-1 · fixed LR / batch schedule",
        "#e07a1f",
    ),
)

SGD_FOLDERS = {
    "wsd_exp_80_20__fixed_batch_lr": "tail_output_wsd_fblr",
    "wsd_exp_80_20__fixed_lr_batch": "tail_output_wsd_flrbs",
    "eight_one_one__fixed_batch_lr": "tail_output_811_fblr",
    "eight_one_one__fixed_lr_batch": "tail_output_811_flrbs",
}

MUON_BASE_LR = 1.0e-4
MUON_WARMUP_UPDATES = 256


def _compact_large_tick(value: float, _position: int) -> str:
    magnitude = abs(value)
    if magnitude >= 1.0e6:
        return f"{value / 1.0e6:.3g}M"
    if magnitude >= 1.0e3:
        return f"{value / 1.0e3:.3g}k"
    return f"{value:.3g}"


def _compact_scientific_tick(value: float, _position: int) -> str:
    return f"{value:.1e}".replace("e-0", "e-").replace("e+0", "e+")


def _load_sgd(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        values = {
            "step": np.asarray(payload["global_update"], dtype=np.float64),
            "clock": np.asarray(payload["intrinsic_time"], dtype=np.float64),
            "risk": np.asarray(
                payload["validation_cross_entropy"], dtype=np.float64
            ),
        }
    _validate(values, path)
    return values


def _load_muon_tail(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        values = {
            "step": np.asarray(payload["optimizer_update"], dtype=np.float64),
            "clock": np.asarray(payload["sum_eta_squared"], dtype=np.float64),
            "risk": np.asarray(
                payload["validation_cross_entropy"], dtype=np.float64
            ),
        }
    _validate(values, path)
    return values


def _load_muon_prefix(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        step = np.asarray(payload["optimizer_update"], dtype=np.float64)
        risk = np.asarray(payload["validation_cross_entropy"], dtype=np.float64)
    integer_step = step.astype(np.int64)
    warmup_square_sum = np.cumsum(
        np.square(
            np.arange(1, MUON_WARMUP_UPDATES + 1, dtype=np.float64)
            / MUON_WARMUP_UPDATES
        )
    )
    clock_units = np.empty(integer_step.shape, dtype=np.float64)
    within_warmup = integer_step <= MUON_WARMUP_UPDATES
    positive_warmup = within_warmup & (integer_step > 0)
    clock_units[integer_step == 0] = 0.0
    clock_units[positive_warmup] = warmup_square_sum[
        integer_step[positive_warmup] - 1
    ]
    clock_units[~within_warmup] = (
        warmup_square_sum[-1]
        + integer_step[~within_warmup]
        - MUON_WARMUP_UPDATES
    )
    values = {
        "step": step,
        "clock": clock_units * MUON_BASE_LR * MUON_BASE_LR,
        "risk": risk,
    }
    _validate(values, path)
    return values


def _validate(values: dict[str, np.ndarray], path: Path) -> None:
    shape = values["step"].shape
    if (
        values["step"].ndim != 1
        or any(value.shape != shape for value in values.values())
        or not all(np.all(np.isfinite(value)) for value in values.values())
        or not np.all(np.diff(values["step"]) > 0)
        or not np.all(np.diff(values["clock"]) > 0)
    ):
        raise RuntimeError(f"invalid validation trace: {path}")


def _plot_pair(
    axes: tuple[plt.Axes, plt.Axes],
    curves: dict[str, dict[str, np.ndarray]],
    prefix: dict[str, np.ndarray],
    checkpoint: int,
    clock_label: str,
    step_limits: tuple[float, float],
    step_ticks: tuple[float, float, float],
    clock_limits: tuple[float, float],
    clock_ticks: tuple[float, float, float],
    risk_limits: tuple[float, float],
    risk_ticks: tuple[float, float, float],
    show_legend: bool,
) -> None:
    underlays = tuple(arm for arm in ARMS if arm[0].endswith("fixed_lr_batch"))
    overlays = tuple(arm for arm in ARMS if arm[0].endswith("fixed_batch_lr"))
    plot_order = underlays + overlays
    legend_handles: dict[str, plt.Line2D] = {}

    for panel_index, (axis, x_key, x_label, prefix_window) in enumerate(
        (
            (axes[0], "step", "Step", 4096),
            (axes[1], "clock", clock_label, 1024),
        )
    ):
        keep = prefix["step"] >= checkpoint - prefix_window
        axis.plot(
            prefix[x_key][keep],
            prefix["risk"][keep],
            color="#6b7280",
            linewidth=2.0,
            alpha=0.82,
            label="_nolegend_",
            zorder=1,
        )
        for arm, label, color in plot_order:
            is_underlay = arm.endswith("fixed_lr_batch")
            line, = axis.plot(
                curves[arm][x_key],
                curves[arm]["risk"],
                color=color,
                alpha=0.48 if is_underlay else 1.0,
                linewidth=5.4 if is_underlay else 2.1,
                label=label,
                zorder=2 if is_underlay else 3,
            )
            if panel_index == 0:
                legend_handles[arm] = line
        axis.set_box_aspect(1)
        axis.set_xlabel(x_label, fontsize=36)
        if panel_index == 0:
            axis.set_ylabel("Risk", fontsize=36)
        else:
            axis.set_ylabel("")
            axis.tick_params(axis="y", labelleft=False)
        axis.set_ylim(*risk_limits)
        axis.set_yticks(risk_ticks)
        axis.grid(alpha=0.18, linewidth=0.7)
        axis.tick_params(axis="x", labelsize=31.5, pad=14)
        axis.tick_params(axis="y", labelsize=31.5, pad=14)
        axis.xaxis.get_offset_text().set_fontsize(31.5)
        axis.yaxis.get_offset_text().set_fontsize(31.5)
        axis.margins(x=0.012, y=0.08)

    axes[0].set_xlim(*step_limits)
    axes[0].set_xticks(step_ticks)
    axes[0].xaxis.set_major_formatter(FuncFormatter(_compact_large_tick))
    axes[1].set_xlim(*clock_limits)
    axes[1].set_xticks(clock_ticks)
    if max(abs(value) for value in clock_ticks) >= 1.0e3:
        axes[1].xaxis.set_major_formatter(FuncFormatter(_compact_large_tick))
    else:
        axes[1].xaxis.set_major_formatter(
            FuncFormatter(_compact_scientific_tick)
        )
    if show_legend:
        axes[0].legend(
            [
                legend_handles["wsd_exp_80_20__fixed_batch_lr"],
                legend_handles["eight_one_one__fixed_batch_lr"],
            ],
            ["WSD schedule", "8-1-1 schedule"],
            frameon=False,
            fontsize=23.25,
            loc="upper right",
            handlelength=2.4,
        )


def plot(
    sgd_root: Path,
    sgd_prefix_path: Path,
    muon_root: Path,
    muon_prefix_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    sgd_curves = {
        arm: _load_sgd(
            sgd_root.expanduser().resolve()
            / SGD_FOLDERS[arm]
            / "validation_evaluations.npz"
        )
        for arm, *_ in ARMS
    }
    sgd_prefix = _load_sgd(sgd_prefix_path.expanduser().resolve())
    muon_curves = {
        arm: _load_muon_tail(
            muon_root.expanduser().resolve() / arm / "validation_evaluations.npz"
        )
        for arm, *_ in ARMS
    }
    muon_prefix = _load_muon_prefix(muon_prefix_path.expanduser().resolve())

    sgd_checkpoint = 1_953_120
    muon_checkpoint = 79_346
    for curves, checkpoint, name in (
        (sgd_curves, sgd_checkpoint, "124M SGD"),
        (muon_curves, muon_checkpoint, "300M Muon"),
    ):
        if any(int(curve["step"][0]) != checkpoint for curve in curves.values()):
            raise RuntimeError(f"{name} tails do not share the expected checkpoint")

    plt.rcParams.update({"font.size": 16.5, "axes.linewidth": 0.9})
    figure, axes = plt.subplots(1, 4, figsize=(29.5, 8.15))
    figure.subplots_adjust(
        left=0.055,
        right=0.992,
        bottom=0.16,
        top=0.84,
        wspace=0.44,
    )

    _plot_pair(
        (axes[0], axes[1]),
        sgd_curves,
        sgd_prefix,
        sgd_checkpoint,
        r"Intrinsic time $\sum \eta$",
        (1_945_000, 2_445_000),
        (1_945_000, 2_195_000, 2_445_000),
        (9_760.0, 10_720.0),
        (9_760.0, 10_240.0, 10_720.0),
        (3.74, 3.85),
        (3.74, 3.795, 3.85),
        True,
    )
    _plot_pair(
        (axes[2], axes[3]),
        muon_curves,
        muon_prefix,
        muon_checkpoint,
        r"Intrinsic time $\sum \eta^2$",
        (75_000.0, 100_000.0),
        (75_000.0, 87_500.0, 100_000.0),
        (0.0007808, 0.0008350),
        (0.00079, 0.00081, 0.00083),
        (2.98, 3.12),
        (2.98, 3.05, 3.12),
        False,
    )

    figure.text(
        0.275,
        0.95,
        "124M nanoGPT SGD, 2.5B tokens",
        ha="center",
        va="center",
        fontsize=48,
    )
    figure.text(
        0.755,
        0.95,
        "300M nanoGPT Muon, 6.5B tokens",
        ha="center",
        va="center",
        fontsize=48,
    )

    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "124m-sgd-300m-muon-validation-risk-four-panel"
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    figure.savefig(png, dpi=240, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sgd-root", type=Path, required=True)
    parser.add_argument("--sgd-prefix", type=Path, required=True)
    parser.add_argument("--muon-root", type=Path, required=True)
    parser.add_argument("--muon-prefix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    for path in plot(
        args.sgd_root,
        args.sgd_prefix,
        args.muon_root,
        args.muon_prefix,
        args.output_dir,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
