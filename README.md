# Paper experiment code

This repository is the reproducibility package for every experiment figure in the current manuscript. It contains the experiment implementations, frozen configurations, validation traces used by the plots, cluster launch material, paper-ready plotting code, and the 13 exact PDF figures included by the manuscript.

The release is organized by experiment family:

- `iclr2027/experiments/theorem_validation/`: spectral constructions and the rapid-target diagnostic.
- `iclr2027/experiments/preserve_change_destroy/`: preserve/change/destroy and finite-width Volterra bridge experiments.
- `iclr2027/experiments/fixed_noise_early_stopping/`: finite-width noisy early stopping.
- `fixed_noise_validation/`: accepted deterministic-equivalent Volterra source curves used by the noisy plots.
- `experiments/nanogpt_local/`: 30M/124M/300M nanoGPT training, schedule, optimizer, validation, and fitting code.
- `experiments/configs/`, `experiments/protocols/`, `experiments/chtc/`: frozen language-model configurations, protocols, and HTCondor materializers.
- `data/language_model/`: lightweight validation traces and fitted predictions sufficient to redraw all language-model figures.
- `paper_figures/`: exact PDFs included by the manuscript.

See [EXPERIMENTS.md](EXPERIMENTS.md) for the figure-to-code-to-data map.

## Quick start

The frozen environment used for release validation is Python 3.12.13 with the versions in `requirements.txt`.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/verify_release.py
.venv/bin/python scripts/reproduce_paper_figures.py
.venv/bin/python scripts/verify_reproduced_figures.py
```

The reproduction script writes to `reproduced_figures/`. A single family can be redrawn with, for example:

```bash
.venv/bin/python scripts/reproduce_paper_figures.py --group spectral
.venv/bin/python scripts/reproduce_paper_figures.py --group fixed-noise
.venv/bin/python scripts/reproduce_paper_figures.py --group language-model
```

The `preserve` group materializes the audited paper PDFs bundled with the accepted numerical outputs. The full numerical runs that produced them are exposed as the corresponding `run_*.py` entry points in `iclr2027/experiments/preserve_change_destroy/`; these runs are substantially more expensive than redrawing a figure.

## Language-model reruns

The nanoGPT code is complete, but a fresh training run requires:

1. the OpenWebText corpus prepared with `experiments/nanogpt_local/prepare_data.py` and the relevant cooldown-corpus builder;
2. CUDA-capable hardware matching the frozen protocol closely enough for the numerical contract being tested;
3. a writable scratch/output location supplied through the launch configuration.

The repository intentionally excludes training corpora, model checkpoints, scheduler logs, authorization files, and machine-specific caches. The bundled validation traces are enough to redraw the paper figures without those large or private artifacts. Details are in [data/language_model/README.md](data/language_model/README.md).

## Validation

`scripts/verify_release.py` checks:

- all 13 manuscript figures are present;
- their SHA-256 checksums match `paper_figures.sha256.json`;
- no checkpoint-like file or file above GitHub's 100 MB per-file limit is included;
- no known local user path or common credential marker remains in text files.

After reproduction, `scripts/verify_reproduced_figures.py` renders the reference and regenerated PDFs with Poppler and checks that every first page is pixel-identical. PDF byte hashes can differ because Matplotlib records creation metadata; rendered equality is the relevant figure check.

The scientific source directories also retain their original unit tests. The release smoke checks compile every Python source file and run the fast theorem-validation and bridge test suites.

The same checks and a full figure redraw are configured in `.github/workflows/reproducibility.yml` for pushes and pull requests.

## Scope and provenance

The map follows the figures included by the current `iclr2027/main.tex`, not every historical experiment in the larger research workspace. Derived CSV/NPZ files are frozen snapshots of the accepted runs. Configuration hashes and numerical gates remain in the source summaries; local absolute paths have been replaced by publication-safe snapshot labels.

The language-model claims are single-seed and externally scoped exactly as recorded in the protocols and summaries. The 124M surrogate is fitted on the fixed-batch 8-1-1 trajectory and transferred without refitting to the other schedule/factorization trajectories.

## Before publishing

- Choose and add a `LICENSE` file. No license was inferred on the author's behalf.
- Replace this repository title with the final paper title if desired.
- Add the final anonymous or camera-ready citation metadata after the submission status is known.
