"""Plot the frozen 124M SGD v002 fit with the full observed prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.ticker import FormatStrFormatter


EIGHT_FIXED_BATCH = "eight_one_one__fixed_batch_lr"
EIGHT_FIXED_LR = "eight_one_one__fixed_lr_batch"
WSD_FIXED_BATCH = "wsd_exp_80_20__fixed_batch_lr"
WSD_FIXED_LR = "wsd_exp_80_20__fixed_lr_batch"


def _arm(frame: pd.DataFrame, arm_id: str) -> pd.DataFrame:
    result = frame.loc[frame["arm_id"] == arm_id].sort_values("intrinsic_time")
    if result.empty or not np.all(np.diff(result["intrinsic_time"].to_numpy()) > 0.0):
        raise ValueError(f"invalid prediction rows for {arm_id}")
    return result


def render(directory: Path) -> None:
    fit_summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    wsd_summary = json.loads(
        (directory / "wsd_zero_refit_summary.json").read_text(encoding="utf-8")
    )
    fit_start = float(fit_summary["fit_protocol"]["fit_start_intrinsic_time"])
    fork = float(wsd_summary["windows"]["prefix_fork_intrinsic_time"])

    eight_rows = pd.read_csv(directory / "predictions.csv")
    wsd_rows = pd.read_csv(directory / "wsd_zero_refit_predictions.csv")
    curves = {
        "eight_one_one": (
            _arm(eight_rows, EIGHT_FIXED_BATCH),
            _arm(eight_rows, EIGHT_FIXED_LR),
        ),
        "wsd": (
            _arm(wsd_rows, WSD_FIXED_BATCH),
            _arm(wsd_rows, WSD_FIXED_LR),
        ),
    }
    colors = {
        "eight_one_one": ("#e67e22", "#f2c08f", "#8f3f00"),
        "wsd": ("#2878b5", "#a9cae4", "#0f4f82"),
    }

    # Match the apparent typography of nanogpt_factorization_collapse_sgd_muon.pdf
    # when both figures are included at \linewidth.  The reference PDF is
    # 2168 pt wide while this figure is about 1419 pt wide, so its source font
    # sizes are scaled by 1419 / 2168 rather than copied verbatim.
    figure, axes = plt.subplots(1, 4, figsize=(20.0, 5.5))
    panels = (
        (axes[0], "eight_one_one", "8-1-1", False),
        (axes[1], "eight_one_one", "8-1-1", True),
        (axes[2], "wsd", "WSD", False),
        (axes[3], "wsd", "WSD", True),
    )
    full_windows = {}
    for axis, shape, title, zoomed in panels:
        fixed_batch, fixed_lr = curves[shape]
        medium, light, dark = colors[shape]
        time = fixed_batch["intrinsic_time"].to_numpy(dtype=np.float64)
        observed = fixed_batch["observed_validation_ce"].to_numpy(dtype=np.float64)
        prefix = time <= fork + 1e-12
        tail_batch = time >= fork - 1e-12
        lr_time = fixed_lr["intrinsic_time"].to_numpy(dtype=np.float64)
        lr_observed = fixed_lr["observed_validation_ce"].to_numpy(dtype=np.float64)
        tail_lr = lr_time >= fork - 1e-12
        theory = fixed_batch["predicted_validation_ce"].to_numpy(dtype=np.float64)
        theory_mask = time >= fit_start
        prefix_width = 1.8 if zoomed else 1.7
        validation_width = 2.0 if zoomed else 1.8
        light_validation_width = 3.0 if zoomed else 2.6
        theory_width = 3.0 if zoomed else 2.6

        axis.plot(
            time[prefix],
            observed[prefix],
            color="#8b9098",
            linewidth=prefix_width,
            alpha=0.88,
            label="Shared prefix",
        )
        axis.plot(
            time[tail_batch],
            observed[tail_batch],
            color=medium,
            linewidth=validation_width,
            alpha=0.90,
            label="Fixed batch / LR schedule",
        )
        axis.plot(
            lr_time[tail_lr],
            lr_observed[tail_lr],
            color=light,
            linewidth=light_validation_width,
            alpha=0.82,
            label="Fixed LR / batch schedule",
        )
        axis.plot(
            time[theory_mask],
            theory[theory_mask],
            color=dark,
            linewidth=theory_width,
            label="Theory",
        )
        axis.axvline(fork, color="#5d626a", linewidth=0.9, linestyle=":")
        zoom_start = 9500.0
        if zoomed:
            axis.set_xlim(zoom_start, float(time[-1]))
            zoom_values = np.concatenate(
                [
                    observed[time >= zoom_start],
                    lr_observed[lr_time >= zoom_start],
                    theory[time >= zoom_start],
                ]
            )
            low = float(np.min(zoom_values))
            high = float(np.max(zoom_values))
            pad = 0.08 * (high - low)
            axis.set_ylim(low - pad, high + pad)
            axis.set_xticks(np.linspace(zoom_start, float(time[-1]), 3))
            # Keep the extreme y tick labels away from the lower-left and
            # upper-left corners, where the enlarged x/y labels would collide.
            axis.set_yticks(np.linspace(low, high, 3))
            axis.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
            axis.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
            full_axis, full_right, full_low, full_high = full_windows[shape]
            for source_y, target_y in (
                (full_high, high + pad),
                (full_low, low - pad),
            ):
                figure.add_artist(
                    ConnectionPatch(
                        xyA=(full_right, source_y),
                        xyB=(zoom_start, target_y),
                        coordsA="data",
                        coordsB="data",
                        axesA=full_axis,
                        axesB=axis,
                        color=dark,
                        linewidth=1.6,
                        alpha=0.66,
                        zorder=2,
                    )
                )
        else:
            axis.set_xlim(float(time[0]), float(time[-1]))
            axis.set_ylim(3.35, 11.30)
            axis.set_xticks(np.linspace(float(time[0]), float(time[-1]), 3))
            axis.set_yticks(np.linspace(4.0, 11.0, 3))
            schematic_start = 9000.0
            schematic_low = 3.45
            schematic_high = 4.55
            axis.add_patch(
                Rectangle(
                    (schematic_start, schematic_low),
                    float(time[-1]) - schematic_start,
                    schematic_high - schematic_low,
                    facecolor=to_rgba(dark, 0.07),
                    edgecolor=dark,
                    linewidth=1.7,
                    linestyle=(0, (3, 2)),
                    zorder=5,
                )
            )
            full_windows[shape] = (
                axis,
                float(time[-1]),
                schematic_low,
                schematic_high,
            )
        axis.set_title(title, fontsize=25)
        axis.set_xlabel(r"Intrinsic time $\sum \eta$", fontsize=25)
        axis.grid(alpha=0.12)
        axis.tick_params(labelsize=21.5)

    axes[0].set_ylabel("Risk", fontsize=25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.105),
        fontsize=25,
    )
    figure.suptitle("124M nanoGPT SGD, 2.5B tokens", fontsize=33)
    figure.subplots_adjust(left=0.055, right=0.995, top=0.79, bottom=0.27, wspace=0.29)
    figure.savefig(directory / "validation_fits_full_prefix.png", dpi=240, bbox_inches="tight")
    figure.savefig(directory / "validation_fits_full_prefix.pdf", bbox_inches="tight")
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        default=root / "experiments/results/nanogpt124m-e2e-sgd-unified-phase-surrogate-v002",
    )
    args = parser.parse_args(argv)
    directory = args.directory.expanduser().resolve()
    render(directory)
    print(directory / "validation_fits_full_prefix.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
