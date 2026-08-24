#!/usr/bin/env python3
"""Paper-faithful orchestrator for the final spectral-fingerprint manuscript.

This runner intentionally mirrors the analyses reported in the final paper.
It is not a reconstruction of every historical notebook cell.

Canonical path:
  paper-from-residuals
      H1 content reproducibility + LibriSeVoc primary false-pair control
      H2 cross-language persistence
      H3a reuse of the validated controlled exact-checkpoint summary + H3b recomputation
      H4 global confirmatory analysis + family-specific exploratory contrasts
      manuscript consistency checks, selected figures, release validation

Raw pairing/extraction scripts are kept in the repository for reproducibility,
but they are not silently executed by the manuscript runner. The historical
full physical MLAAD/M-AILABS audit was not completed and is therefore not a
stage of this canonical pipeline.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
SRC = REPO / "src"
ROOT = Path(os.environ.get("FINGERPRINT_OUTPUT_ROOT", "/content/drive/MyDrive/fingerprint_q1_outputs"))

S = {
    "h1_controlled": SRC / "03_h1_controls/H1_CONTROLLED_DATASETS_481BINS_FINAL_v1.py",
    "h1_mlaad": SRC / "03_h1_controls/Q1_03_MLAAD_GENERATOR_STABILITY_v2_NEW_STORY.py",
    "false_lsv": SRC / "03_h1_controls/Q1_07_LIBRISEVOC_FALSE_PAIR_v3_NEW_STORY.py",
    "false_mlaad": SRC / "03_h1_controls/Q1_08_MLAAD_NEGATIVE_PAIR_CONTROL_v1.py",
    "h4_relaxed_sensitivity": SRC / "09_supplementary/H4_RELAXED_PROTOCOL_SENSITIVITY_v1.py",
    "h2": SRC / "04_h2/H2_CROSS_LANGUAGE_FINAL_v1.py",
    "h3a_raw": SRC / "04_h3/H3A_CONTROLLED_FROM_RAW_FINAL_v1.py",
    "h3": SRC / "04_h3/H3_FINAL_MANUSCRIPT_v1.py",
    "h4": SRC / "04_h4/H4_GLOBAL_CONFIRMATORY_FINAL_v1.py",
    "h4_family": SRC / "04_h4/H4_FAMILY_SPECIFIC_EXPLORATORY_FINAL_v1.py",
    "master": SRC / "05_master/MASTER_Q1_MANUSCRIPT_FINAL_v3_2.py",
    "figures": SRC / "07_figures/GENERATE_SELECTED_MANUSCRIPT_FIGURES.py",
    "validate": SRC / "08_validation/VALIDATE_RELEASE.py",
    "verify_results": SRC / "08_validation/VERIFY_RECOMPUTED_RESULTS.py",
}


def run(path: Path, *args: object, env: dict[str, str] | None = None) -> None:
    cmd = [sys.executable, str(path), *map(str, args)]
    print("\n" + "=" * 108)
    print("RUN:", " ".join(cmd))
    print("=" * 108, flush=True)
    subprocess.run(cmd, check=True, env=env)


def first_existing(candidates: list[Path], label: str) -> Path:
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(label + " not found. Tried:\n  " + "\n  ".join(map(str, candidates)))


def env_root() -> dict[str, str]:
    env = os.environ.copy()
    env["FINGERPRINT_OUTPUT_ROOT"] = str(ROOT)
    return env


def preflight() -> None:
    print(f"[INFO] repository: {REPO}")
    print(f"[INFO] output root: {ROOT}")
    for name, path in S.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing canonical script {name}: {path}")


def mlaad_inputs() -> tuple[Path, Path]:
    phase = next(
        (
            p
            for p in [
                ROOT / "phase1a/phase1a_mlaad_spectral_residuals_v2_new_story",
                ROOT / "phase1a/phase1a_mlaad_spectral_residuals_v1",
            ]
            if (p / "fingerprints_pair_level_strict.parquet").is_file()
            and (p / "frequency_axis.csv").is_file()
        ),
        None,
    )
    if phase is None:
        raise FileNotFoundError("MLAAD STRICT residual parquet/frequency axis not found")
    return phase / "fingerprints_pair_level_strict.parquet", phase / "frequency_axis.csv"


def librisevoc_inputs() -> Path:
    for folder in [
        ROOT / "q1_harmonized/v3_new_story/librisevoc/full",
        ROOT / "q1_harmonized/v2/librisevoc/full",
    ]:
        required = [
            folder / "fingerprints_pair_level_harmonized.parquet",
            folder / "real_index.parquet",
            folder / "frequency_axis.csv",
            folder / "spectra/fake_log_power_db.npy",
            folder / "spectra/real_log_power_db.npy",
        ]
        if all(p.is_file() for p in required):
            return folder
    raise FileNotFoundError("LibriSeVoc v2 harmonized residual/spectra artifacts not found")


def stage_h1_core() -> None:
    """Only the H1 analyses reported in the main paper."""
    env = env_root()
    run(S["h1_controlled"], env=env)

    mlaad, axis = mlaad_inputs()
    run(
        S["h1_mlaad"],
        "--input", mlaad,
        "--frequency-axis", axis,
        "--output-dir", ROOT / "q1_03/mlaad_generator_stability_manuscript_final",
        "--mode", "full",
        "--split-repeats-full", "200",
        env=env,
    )

    lsv = librisevoc_inputs()
    run(
        S["false_lsv"],
        "--pair-level", lsv / "fingerprints_pair_level_harmonized.parquet",
        "--real-index", lsv / "real_index.parquet",
        "--frequency-axis", lsv / "frequency_axis.csv",
        "--fake-spectra", lsv / "spectra/fake_log_power_db.npy",
        "--real-spectra", lsv / "spectra/real_log_power_db.npy",
        "--output-dir", ROOT / "q1_07/librisevoc_false_pair_manuscript_final",
        "--mode", "full",
        "--bootstraps-full", "5000",
        env=env,
    )


def stage_h3() -> None:
    """H3b is recomputed; H3a is read from its validated controlled-run summary."""
    run(S["h3"], "--root", ROOT, env=env_root())


def stage_h4() -> None:
    env = env_root()
    run(S["h4"], env=env)
    run(S["h4_family"], "--root", ROOT, env=env)


def manuscript_tail() -> None:
    env = env_root()
    run(S["verify_results"], "--repo", REPO, "--root", ROOT, env=env)
    run(S["master"], "--root", ROOT, env=env)
    run(S["figures"], "--root", ROOT, env=env)
    run(S["validate"], "--repo", REPO, env=env)


def stage_paper() -> None:
    """Recompute exactly the manuscript analysis layer from saved artifacts."""
    env = env_root()
    stage_h1_core()
    run(S["h2"], "--root", ROOT, env=env)
    stage_h3()
    stage_h4()
    manuscript_tail()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "preflight",
            "h1",
            "h2",
            "h3a-controlled",
            "h3",
            "h4",
            "figures",
            "validate",
            "paper-from-residuals",
            "supplementary-mlaad-negative-control",
            "supplementary-h4-relaxed-sensitivity",
        ],
    )
    args = parser.parse_args()
    preflight()
    env = env_root()

    if args.stage == "preflight":
        return 0
    if args.stage == "h1":
        stage_h1_core()
    elif args.stage == "h2":
        run(S["h2"], "--root", ROOT, env=env)
    elif args.stage == "h3a-controlled":
        # Reproduction utility for the controlled experiment reported in H3a.
        # It is deliberately explicit because it requires raw corpora/checkpoints.
        env["H3A_MODE"] = "full"
        run(S["h3a_raw"], env=env)
    elif args.stage == "h3":
        stage_h3()
    elif args.stage == "h4":
        stage_h4()
    elif args.stage == "figures":
        run(S["figures"], "--root", ROOT, env=env)
    elif args.stage == "validate":
        run(S["master"], "--registry-only", env=env)
        run(S["validate"], "--repo", REPO, env=env)
    elif args.stage == "paper-from-residuals":
        stage_paper()
    elif args.stage == "supplementary-mlaad-negative-control":
        mlaad, _ = mlaad_inputs()
        run(S["false_mlaad"], "--pair-level", mlaad, env=env)
    elif args.stage == "supplementary-h4-relaxed-sensitivity":
        run(S["h4_relaxed_sensitivity"], env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
