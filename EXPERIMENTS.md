# Experiment and figure manifest

This table is the authoritative map from the current manuscript figures to their generating code and frozen inputs.

| Manuscript PDF | Experiment | Primary code | Bundled input/results |
|---|---|---|---|
| `nanogpt30m_theta_schedule_response.pdf` | 30M plain-SGD power-law schedule sweep over 11 exponents | `experiments/nanogpt_local/e2e_sgd_theta_powerlaw_30m_v001.py`, dense/high extensions, `scripts/plot_nanogpt30m_theta.py` | `data/language_model/nanogpt30m_theta/` |
| `nanogpt_factorization_collapse_sgd_muon.pdf` | 124M SGD and 300M hybrid-Muon schedule factorization | `experiments/nanogpt_local/e2e_sgd_rpath_factorization_124m_2p5b_v002.py`, `e2e_300m_muon_square_rpath_v001.py`, `plot_124m_sgd_300m_muon_risk_four_panel_v001.py` | `data/language_model/nanogpt124m_sgd_factorization/`, `nanogpt300m_muon_factorization/` |
| `nanogpt124m_sgd_surrogate_fit_transfer.pdf` | Frozen seven-parameter 124M SGD surrogate and transfer | `experiments/nanogpt_local/e2e_sgd_unified_phase_surrogate_124m_2p5b_v002.py`, `e2e_sgd_unified_phase_surrogate_124m_2p5b_v002_wsd_validation.py`, `plot_124m_sgd_unified_full_prefix_v002.py` | `data/language_model/nanogpt124m_surrogate/` |
| `nanogpt124m_intrinsic_time_theory_transfer.pdf` | Four-trajectory intrinsic-time comparison | `experiments/nanogpt_local/plot_124m_v002_four_validation_and_fits.py` | `data/language_model/nanogpt124m_surrogate/` |
| `spectral_six_constructions_powerlike.pdf` | Four power-like spectral constructions | `iclr2027/experiments/theorem_validation/spectral_six_construction_paper_style.py` | selected `iclr2027/experiments/theorem_validation/artifacts/*/curves.csv` |
| `spectral_six_constructions_boundaries.pdf` | Rapid/gapped boundary spectral constructions | same as above | same as above |
| `spectral_iff_rapid_target.pdf` | Rapid-target risk/forcing/memory diagnostic over 20 seeds | `iclr2027/experiments/theorem_validation/spectral_rapid_target_triptych.py` | `spectral_rapid_target_t1e3_w524288_seed31501_v001` through `seed31520_v001` |
| `preserve_change_destroy_continuum_diagnostic.pdf` | Preserve/change/destroy continuum schedule response | `iclr2027/experiments/preserve_change_destroy/run_exact_power_continuum_m100000_sigma25.py` and dependencies | `artifacts/exact_power_continuum_m100000_sigma25/` |
| `volterra_three_layer_bridge.pdf` | True SGD / finite-width Volterra / deterministic-equivalent bridge | `run_experiment_0_finite_rf_sgd_bridge.py`, `run_experiment_0b_finite_w_de_bridge.py`, `plot_experiment_0c_three_layer_bridge.py` | `artifacts/experiment_0_finite_rf_sgd_bridge/`, `artifacts/experiment_0b_finite_w_de_bridge/`, `artifacts/experiment_0c_three_layer_bridge/` |
| `fb_volterra_three_layer_bridge.pdf` | Finite-bulk three-layer bridge | `iclr2027/experiments/preserve_change_destroy/run_fb_three_layer_bridge.py` | `artifacts/fb_three_layer_bridge_full/` |
| `fixed_noise_im_early_stopping_composite.pdf` | Fixed-noise early-stopping trajectories in the three IM subregimes | `iclr2027/experiments/fixed_noise_early_stopping/plot_im_early_stopping_composite.py` | `fixed_noise_validation/artifacts/results/adaptive_real_axis_*` |
| `fixed_noise_im_noise_width_stopping_dynamics.pdf` | Noise-width stopping phase diagram | `iclr2027/experiments/fixed_noise_early_stopping/plot_im_noise_width_stopping_dynamics.py` | same accepted support-adaptive Volterra curves and terminal plateaus |
| `fixed_noise_compute_optimal_representative.pdf` | PLRF pullback plus representative clean/noisy compute-optimal frontiers | `fixed_noise_validation/src/plot_figure5_three_panel_preview.py` and its three helper modules | seven accepted full-width curve grids in `fixed_noise_validation/artifacts/results/` |

## Frozen language-model protocols

- 30M model: 6 layers, 6 heads, width 384, context 256, 30,036,864 parameters, plain FP32 SGD, no momentum, no weight decay, no clipping. The shared prefix has 196,608 updates at batch size 8. The 11 tail exponents are 0, 0.125, 0.25, 0.375, 0.5, 0.75, 1, 1.25, 1.5, 1.75, and 2.
- 124M SGD comparison: 2.5B-token campaign with WSD 80/20 and 8-1-1 tails; fixed-batch/varying-learning-rate is compared with fixed-learning-rate/varying-batch.
- 300M hybrid-Muon comparison: 6.5B-token campaign using the optimizer clock recorded by the protocol.
- 124M surrogate: fitted only on the fixed-batch 8-1-1 trajectory and evaluated without refitting on the matched factorization and WSD trajectories. The accepted fit records `q_K = 1.0170727555666113` and `q_F = 0.37415173978518584`.

The detailed numerical contracts, seeds, launch resources, and claim boundaries are preserved in `experiments/protocols/`, `experiments/configs/`, and the JSON summaries.
