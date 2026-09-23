"""Exploratory extension of the 30M theta grid around/above theta=1.5.

This wrapper changes the enumerated theta values, their exact discrete update
counts, and the length/hash of the deterministic no-replacement future tape
needed by the longest arm.  Theta=2 needs more contexts than the 2.5B corpus,
so the tape is extended into the already-frozen 5B corpus while preserving
every existing 2.5B tape row exactly.  Model, optimizer, checkpoint, validation
probe, intrinsic-time endpoint, and evaluation grid are unchanged.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from experiments.nanogpt_local import e2e_sgd_theta_powerlaw_30m_v001 as core


HIGH_THETAS = (1.25, 1.75, 2.0)
HIGH_TAIL_UPDATES = {
    1.25: 201_066,
    1.75: 638_210,
    2.0: 1_151_656,
}
HIGH_TAGS = {
    1.25: "theta1p25",
    1.75: "theta1p75",
    2.0: "theta2",
}
HIGH_CORPUS_TOKENS = 5_000_007_936
BASE_2P5B_CORPUS_TOKENS = 2_500_000_256
TRAINING_TAPE_SECOND_EXTENSION_SEED = 2_026_090_804
HIGH_DATA_IDENTITY_SHA256 = (
    "e926b419e4769b9b36b7a7e5fe958dbc8f2c668f63a319e4700ca261fbfc5072"
)
HIGH_TRAINING_TAPE_SHA256 = (
    "2f240b20cf523abdc50f9143bfd3b374226d82db0e5f93124bd268e6a9b55dab"
)
HIGH_FUTURE_TAPE_SHA256 = (
    "d9e42b4efe5b8c7ed37334b13d1fd389ded91e13cc31ab807adf8d84e5ada4a0"
)


def _install_high_grid() -> None:
    core.THETAS = HIGH_THETAS
    core.EXPECTED_TAIL_UPDATES = HIGH_TAIL_UPDATES
    core.MAX_TAIL_UPDATES = max(HIGH_TAIL_UPDATES.values())
    core.MAX_FUTURE_CONTEXTS = core.MAX_TAIL_UPDATES * core.BATCH_SIZE
    core.TAPE_CONTEXTS = core.PREFIX_CONTEXTS + core.MAX_FUTURE_CONTEXTS
    core.CORPUS_TOKENS = HIGH_CORPUS_TOKENS
    core.EXPANDED_DATA_IDENTITY_SHA256 = HIGH_DATA_IDENTITY_SHA256
    core.EXPECTED_TRAINING_TAPE_SHA256 = HIGH_TRAINING_TAPE_SHA256
    core.EXPECTED_FUTURE_TAPE_SHA256 = HIGH_FUTURE_TAPE_SHA256

    def theta_tag(theta: float) -> str:
        return HIGH_TAGS[core._theta(theta)]

    def make_training_tape(token_count: int = HIGH_CORPUS_TOKENS) -> np.ndarray:
        """Extend the frozen 2.5B tape only when theta=2 needs more rows."""

        eligible = core._eligible_aligned_starts(int(token_count))
        if eligible < core.TAPE_CONTEXTS:
            raise core.ThetaPowerlawError(
                f"corpus exposes {eligible} aligned contexts, need {core.TAPE_CONTEXTS}"
            )
        legacy_eligible = core._eligible_aligned_starts(core.LEGACY_CORPUS_TOKENS)
        base_2p5b_eligible = core._eligible_aligned_starts(BASE_2P5B_CORPUS_TOKENS)
        legacy_contexts = (
            core.PREFIX_UPDATES + core.LEGACY_MAX_TAIL_UPDATES
        ) * core.BATCH_SIZE

        legacy_generator = np.random.Generator(
            np.random.PCG64(core.TRAINING_TAPE_SEED)
        )
        legacy_selected = legacy_generator.permutation(legacy_eligible)[
            :legacy_contexts
        ]

        remaining = core.TAPE_CONTEXTS - legacy_contexts
        extension_generator = np.random.Generator(
            np.random.PCG64(core.TRAINING_TAPE_EXTENSION_SEED)
        )
        base_extension_pool = extension_generator.permutation(
            base_2p5b_eligible - legacy_eligible
        )
        base_take = min(remaining, len(base_extension_pool))
        parts = [
            legacy_selected,
            base_extension_pool[:base_take] + legacy_eligible,
        ]
        remaining -= base_take
        if remaining:
            second_extension_generator = np.random.Generator(
                np.random.PCG64(TRAINING_TAPE_SECOND_EXTENSION_SEED)
            )
            second_extension_pool = second_extension_generator.permutation(
                eligible - base_2p5b_eligible
            )
            if remaining > len(second_extension_pool):
                raise core.ThetaPowerlawError(
                    "5B-only aligned starts cannot extend the frozen 2.5B tape"
                )
            parts.append(
                second_extension_pool[:remaining] + base_2p5b_eligible
            )

        selected = np.concatenate(parts)
        offsets = np.ascontiguousarray(selected, dtype=np.int64) * core.ALIGNED_STRIDE
        tape = np.ascontiguousarray(offsets.reshape(-1, core.BATCH_SIZE))
        if (
            tape.shape
            != (core.PREFIX_UPDATES + core.MAX_TAIL_UPDATES, core.BATCH_SIZE)
            or core._array_sha256(tape) != core.EXPECTED_TRAINING_TAPE_SHA256
            or core._array_sha256(tape[: core.PREFIX_UPDATES])
            != core.EXPECTED_PREFIX_TAPE_SHA256
            or core._array_sha256(tape[core.PREFIX_UPDATES :])
            != core.EXPECTED_FUTURE_TAPE_SHA256
        ):
            raise core.ThetaPowerlawError("frozen staged PCG64 training tape changed")
        return tape

    original_schedule_manifest_unsigned = core._schedule_manifest_unsigned
    original_tape_manifest = core._tape_manifest

    def schedule_manifest_unsigned() -> core.Json:
        manifest = original_schedule_manifest_unsigned()
        manifest["tape"].update(
            {
                "algorithm": (
                    "preserve the 700M X<=10 PCG64 tape; preserve the 2.5B "
                    "extension; then extend without replacement over 5B-only "
                    "aligned starts"
                ),
                "base_2p5b_corpus_tokens": BASE_2P5B_CORPUS_TOKENS,
                "second_extension_seed": TRAINING_TAPE_SECOND_EXTENSION_SEED,
                "preserves_every_existing_2p5b_tape_row": True,
            }
        )
        return manifest

    def tape_manifest(tape: np.ndarray) -> core.Json:
        manifest = original_tape_manifest(tape)
        manifest.update(
            {
                "algorithm": (
                    "frozen 2.5B tape plus a disjoint 5B-only extension"
                ),
                "base_2p5b_corpus_tokens": BASE_2P5B_CORPUS_TOKENS,
                "second_extension_seed": TRAINING_TAPE_SECOND_EXTENSION_SEED,
            }
        )
        return manifest

    core._theta_tag = theta_tag
    core.make_training_tape = make_training_tape
    core._schedule_manifest_unsigned = schedule_manifest_unsigned
    core._tape_manifest = tape_manifest
    core._tail_schedule.cache_clear()


_install_high_grid()


def main(argv: Sequence[str] | None = None) -> int:
    return core.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
