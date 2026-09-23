# 124M 2.5B-token BF16 plain-SGD ratio-path factorization protocol v002

## Scope

This is a single-seed, end-to-end external-validity experiment.  Its primary
question is whether two implementations of the same time-varying ratio path
produce similar fixed-probe validation-loss trajectories when compared at
the same intrinsic time.  It is `A2_EXTERNAL`, is not theorem-facing, and by
itself cannot establish global `B/eta` sufficiency, Equation (21) optimality,
or an LLM memory kernel.

The checked-in package is launch-false.  Local materialization, validation,
or plotting does not authorize an upload or GPU submission.

The checked-in v002 campaign-contract JSON is included in every execution
bundle.  Its exact file SHA-256 is frozen into the schedule manifest, every
phase config, each result, and the prefix checkpoint; runtime validation also
checks its semantic contents before training.

This v002 campaign preserves the v001 model, data, tape, seed, optimizer,
`B`, `eta`, ratio paths, and token horizon.  It changes the numerical
execution path to BF16 autocast, TF32-enabled matrix math, and Flash-only
SDPA.  Before launch, its measurement-only validation grid was also expanded
to match the point counts and per-phase normalized-progress density of the
completed 300M SGD refinement experiment.  Neither change alters the training
updates.  The numerical change nevertheless defines a new trajectory: all
v002 prefix and tail artifacts use v002 schemas, and no v001 FP32 prefix
checkpoint is compatible.  The shared prefix must be retrained once under
v002 before any v002 tail runs.

## Frozen model and optimizer

- nanoGPT decoder: 12 layers, 12 attention heads, width 768.
- GPT-2 tokenizer with padded model vocabulary 50,304.
- Context/block size 256, zero dropout, no biases, tied token/head weights.
- Exact trainable parameter count: 123,783,936.  This is lower than the
  block-1024 124M model by 589,824 positional-embedding parameters.
- BF16 autocast covers both training and fixed-probe validation forward
  passes.  Model parameters and accumulated parameter gradients remain
  float32.  Plain SGD is stateless here, so there are no momentum or variance
  tensors whose dtype could differ.
- TF32 is enabled for eligible float32 matrix math, with PyTorch float32
  matmul precision set to `high`.
- Flash SDPA is the only enabled attention backend; math,
  memory-efficient, and cuDNN SDPA are disabled.  Deterministic algorithms
  are disabled, and `torch.compile` remains disabled.
- Plain SGD with no momentum, dampening, weight decay, clipping, or optimizer
  state.
- Initial constant-r realization: `B=4`, `eta=0.005`, hence `r=800`.

## Exact token accounting

An exact 2,500,000,000-token horizon is incompatible with an exact 80/20
split and the four-update macro construction without a residual partial
batch.  The frozen horizon is therefore the nearest lower multiple satisfying
both constraints:

- 2,441,400 canonical B4 updates;
- 9,765,600 contexts;
- 2,499,993,600 tokens per complete logical trajectory;
- shortfall from nominal 2.5B: 6,400 tokens (2.56e-6 relative);
- shared prefix: 1,953,120 updates, 7,812,480 contexts, 1,999,994,880
  tokens, and intrinsic time 9,765.6;
- each tail: 1,953,120 contexts and 499,998,720 tokens;
- actual campaign compute when the prefix is executed once and four tails are
  executed: 3,999,989,760 tokens.

There is no residual B1 update and no masked or partial batch at the endpoint.

## Ratio-path construction

One macro adds `dT=0.02`.  Its integer `g` lies in 4 through 40.

- Fixed batch / LR schedule: perform `g` updates with `B=4` and
  `eta=0.02/g`.
- Fixed LR / batch schedule: perform four updates with `B=g` and
  `eta=0.005`.

Both consume exactly `4g` contexts, add exactly the same `dT`, and implement
the same `r=B/eta=200g` at every completed macro.  They consume identical
contiguous slices of one frozen future tape.  Equality is asserted at macro
boundaries; optimizer update counts are intentionally allowed to differ.

The WSD source path is constant for the first 80%, followed by exponential
learning-rate decay by a factor of ten over the final 20%.  The 8-1-1 source
path is constant over the first 80%, is lower by `sqrt(10)` over the next
10%, and is lower by ten over the final 10%.  The continuous clocks are
inverted onto the `dT=0.02` grid and cumulative source counts are balanced to
integers.  Only the tail is rescaled by less than one macro of endpoint error.

Frozen derived endpoints are:

| Shape | Total macros | Tail macros | Terminal T | FBLR tail updates | FLRBS tail updates |
|---|---:|---:|---:|---:|---:|
| WSD 80/20 | 535,993 | 47,713 | 10,719.86 | 488,280 | 190,852 |
| 8-1-1 | 513,685 | 25,405 | 10,273.70 | 488,280 | 101,620 |

The two factorizations of one shape have identical terminal intrinsic time.
WSD and 8-1-1 do not: comparisons between shapes use exact common-T probe
ticks through `T=10,273.70`.  Their endpoint-at-equal-token comparison is a
separate estimand and must not be described as an equal-T comparison.

## Data and tape

- Train stream: exactly 2,500,000,256 uint16 GPT-2 tokens from the same frozen
  OpenWebText revision, with a new hash-pinned source-shard manifest.  The
  extra aligned block supplies the right-boundary next-token lookahead; the
  nominal usable aligned domain is exactly 2,500,000,000 tokens.
- Validation stream: 16,777,216 document-disjoint uint16 tokens.
- Probe: the same 1,024 distinct-document fixed validation contexts.
- Training tape: one PCG64 permutation of eligible stride-256 contexts,
  seed 2,026,090,804, without context reuse within a trajectory.
- The exact prefix slice is consumed once.  Every tail consumes the same
  exact future slice, partitioned differently only inside matched macros.

The expanded corpus, validation stream, probe, and tape are uploaded to a
SHA-addressed shared cache once.  Prefix and tail jobs refer to the same
cached objects; they must not re-upload independent dataset copies.

The checked-in planning manifest hash-binds the locally verified contiguous
shard prefix 0--23, but remains intentionally not materialization-ready because
its capacity fields are null.  The completed offline capacity scan froze shards
0--22 as the first complete prefix meeting all three train/validation/probe
targets in the checked-in complete manifest.  Materialize the immutable corpus
only from that complete manifest; the planning manifest must remain ineligible.
Neither the scanner nor the builder has a network or GPU-submission path.

## Two-stage execution

1. Run a response-free H200 B0: 512 B4 updates plus 64 direct B40 updates.
   It checks `eta=0.005` stability, the maximum scheduled batch, throughput,
   peak memory, the BF16/TF32/Flash-only plain-SGD runtime, and a 0.25
   H200-hour ceiling.  The raw runtime record must show BF16 autocast, FP32
   parameters, TF32 enabled in both matmul and cuDNN flags, Flash enabled,
   all alternative SDPA backends disabled, no compilation, and empty SGD
   state.
2. `freeze-data` requires and binds the complete capacity-certified source
   manifest, then emits only the B0 config.  A prefix config cannot be
   materialized until the complete B0 result and its diagnostic trace pass
   offline validation.  The validator replays loss, update-time, and
   evaluation-time summaries from the raw trace.  Their B4/B40 update-time
   p90 and maximum measured validation-evaluation time derive, rather than
   guess, each later job's predicted and hard GPU-hour budgets.  The hard
   estimate applies a 1.5x multiplier to measured work plus ten minutes of
   fixed overhead.
3. Run one single-H200 constant-r prefix producer under v002.  It writes a
   roughly 495 MB FP32-parameter model checkpoint, prefix per-update training
   CE, fixed-probe evaluations, and a small result manifest.  Reusing the
   completed v001 prefix is prohibited because its weights were produced by
   a different numerical trajectory.
4. Collect the prefix once and freeze both its file SHA-256 and its decoded
   model-state SHA-256.  Before any tail config is created, validate the full
   model tensor inventory plus the prefix training trace, validation grid, and
   checkpoint/evaluation validation anchor.  The checkpoint binds the v002
   campaign-contract hash, numerical-runtime ID, and full runtime spec.
5. Materialize four tail configs only after step 4.  Run them concurrently on
   four H200 GPUs, one process and one GPU per tail, with no DDP.

If the B0-derived prefix duration approaches the scheduler's single-job wall
limit, stop before the prefix and add coarse checkpoint/resume.  Otherwise,
keep the one-checkpoint implementation; do not add resume machinery merely as
an unmeasured precaution.

The four tail configs must bind the identical v002 prefix checkpoint and data
hashes.  Replaying the prefix independently in four jobs, or attaching a v001
prefix to a v002 tail, is a protocol violation.

Because Flash execution is not declared bitwise deterministic across jobs,
tail reload uses a preregistered numerical check rather than requiring the
fresh validation vector to have the same byte hash as the saved vector.  The
maximum per-context absolute difference must be at most `1e-3` and the mean
absolute difference at most `1e-4`.  The checkpoint-saved float32 validation
vector—not the freshly recomputed vector—is retained as row zero in every
tail, preserving one byte-identical common origin.  Both observed reload
errors and thresholds are written to the tail result.  This reload policy and
both thresholds are part of the hash-bound campaign contract and are repeated
in the prefix checkpoint; changing them invalidates the schedule/config and
checkpoint lineage.

## Measurements and plots

The primary metric is mean cross-entropy on the fixed 1,024-context validation
probe.  Its evaluation grid is a measurement-only, segment-normalized-progress
map of `nanogpt300m-e2e-sgd-rpath-refinement-v001`:

- the shared prefix has exactly 6,000 evaluations, matching the 300M
  distribution of 1,903 broadly spaced early points followed by 4,097 much
  denser pre-fork points;
- every WSD tail has exactly 7,755 evaluations including the fork anchor;
- every 8-1-1 tail has exactly 4,130 evaluations including the fork anchor;
- the two factorizations of each shape use byte-identical macro ticks; and
- the complete 8-1-1 grid is the leading common-time prefix of the WSD grid,
  so cross-shape comparisons need no interpolation.

The mapping preserves each phase's relative sampling positions rather than
the 300M experiment's absolute `dT=0.005` spacing.  This is necessary because
the 124M campaign spans roughly 24.6 times more intrinsic time.  Mapping the
absolute spacing would change the requested point count and make validation
cost dominate by another factor of roughly 24.6.

Every optimizer update retains its raw training batch CE in a separate
compressed NPZ trace.  Coordinates are reconstructed from the frozen
schedule rather than duplicated millions of times.

For this campaign the plot names are explicitly:

- primary/Lei-style view: fixed-probe validation CE versus intrinsic time or
  update, with no uncertainty band required;
- Blake-style diagnostic: training CE with 16-update and 64-update mean
  lines, optionally against log10 tokens.

These are visual conventions, not claims of exact reproduction of either
paper.  Raw metric names remain unambiguous in the artifacts.

## Required gates

- exact 123,783,936-parameter model; B0 must reproduce the seeded
  initialization hash twice inside the pinned Linux container, then freeze
  that hash into the prefix and tail contracts;
- H200 device and one visible GPU; BF16 autocast for train and validation,
  FP32 parameters, TF32 on, Flash-only SDPA, deterministic algorithms off,
  and compilation off;
- finite losses and parameters; empty SGD optimizer state;
- at prefix and tail completion, finite FP32 parameters, finite FP32
  gradients, an empty SGD state, and the unchanged runtime backend contract
  are rechecked before a result can be marked completed;
- shared prefix checkpoint file and model-state hashes exact in all tails;
- checkpoint cursor, tokens, update count, and intrinsic time exact;
- tail load reproduces the saved prefix validation anchor within maximum
  absolute error `1e-3` and mean absolute error `1e-4`, while every tail uses
  the saved anchor bytes for its first evaluation row;
- v002 campaign-contract hash, schemas, and numerical-runtime ID exact; v001
  prefix checkpoints rejected and a newly trained v002 prefix required;
- complete per-update training-CE and fixed-probe evaluation artifacts;
- same-shape macro T/context/ratio grids exact across factorizations;
- launch authorization remains false until an explicit later user command.
