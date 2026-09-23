# 300M hybrid-Muon prospective LR calibration v002

This is an `A2_EXTERNAL` end-to-end stability pilot, not a theorem-facing
ratio experiment or an authorization to run the 6.5B formal prefix. The
completed v001 pilot remains `short_window_no_go`: its low-LR improvement was
measured only after forking from a model trained at `1.5e-4`. Do not relabel
that result or reuse its model as a low-LR from-initialization prefix.

## Frozen initial grid

- Same 303,473,664-parameter nanoGPT, 256-token context, global batch 256,
  micro-batch 32, BF16/Flash runtime, hybrid Muon with auxiliary FP32 AdamW,
  exact initialization seed, first 8,192 source-tape updates, and fixed
  1,024-context validation probe as v001 and the 6.5B campaign.
- Two independent from-initialization arms: nominal LR `7.5e-5` and `1.0e-4`.
  Both use the same 256-update linear warmup and clip `33.504` as a nearly
  non-clipping control. Actual clipping events must still be recorded.
- Fixed validation ticks are 0, 256, and every 512 updates from 512 through
  8,192. Record **every-update** training CE, preclip global gradient norm,
  clipping bit, and step duration. Keep fixed-probe validation CE primary.
- Save a checkpoint at 8,192 containing model parameters, the complete
  hybrid-Muon/aux-AdamW optimizer state, RNG state, exact LR/clip, source-tape
  provenance, and the fixed-probe CE anchor. Do not use another optimizer's
  prefix state or a checkpoint trained under a different LR/clip.

## Frozen prospective decision

For each arm, require finite model and FP32 optimizer state; finite training
and fixed-probe CE; post-warmup clipping fraction at most 10%; and mean CE
over ticks 6,656/7,168/7,680/8,192 strictly below the mean over ticks
4,096/4,608/5,120/5,632. These are **four-point means**, not one terminal
sample. A failure of both arms is `no_go`. If both pass but their last-four
means differ by at most `0.01` CE, extend **both**, using their respective
full checkpoints and the same next 8,192 source-tape updates, to 16,384.
Otherwise the lower last-four mean among passing arms is a *provisional LR
candidate*, not a formal-prefix hyperparameter freeze.

For extension, keep each arm's LR, clip, warmup history, optimizer and RNG
state unchanged. Validation is evaluated every 512 updates from the 8,192
anchor through 16,384. Apply the same four-point trend rule to ticks
12,288/12,800/13,312/13,824 versus 14,848/15,360/15,872/16,384. If both
pass and remain within `0.01`, select the lower LR as a conservative
provisional candidate and explicitly report the near-tie; do not claim that
LR was optimized. Full formal prefix submission is separately authorized.

## Conditional clipping check

Keep clip `33.504` fixed while isolating LR. Report clipping rates separately
for the 256-update startup and the mature segment; a whole-run rate can hide
startup sensitivity. If high-clip arms remain finite and their mature
clipping rate is negligible, do not expand a clipping grid. If a lower clip
is needed, run at most one prospective **from-initialization** matched
control at the provisionally selected LR and clip `10.0`, with the same
8,192-update tape/probe. Compare its full validation trajectory and
startup/mature clipping rates with the existing high-clip arm. A clip chosen
from a mature checkpoint-only branch is not valid for reusing the full-prefix
checkpoint; any changed clipping setup must be trained from initialization.

The high-clip value is a v001 measured control, not an assertion of the
best regularizer. The two LRs, tie band, fixed-probe ticks, and gate above
are frozen before v002 results are inspected. The v001 short-window result
must be reported alongside v002, including its failed center-LR gate.

## Resource and launch boundary

Measured v001 H200 training step duration was about 0.373 seconds. Two
8,192-update arms therefore require about 1.70 H200 GPU-hours of training
compute; extending both to 16,384 makes the cumulative training estimate
3.39 H200 GPU-hours, plus validation, checkpoint, and transfer overhead.
The first 8,192 source-tape windows are packed/uploaded once to the CHTC
OSDF shared staging cache and referenced by both jobs. The packed train
input exceeds 1 GB, so it is not sent from the ordinary `/home` submit
directory. Conditional extension packs only the future 8,192 windows and
reuses the **exact initial code bundle**.
The full 6.5B cache remains not uploaded.
This offline modification does not submit any GPU job.
