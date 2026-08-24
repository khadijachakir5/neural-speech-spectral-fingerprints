#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final manuscript H4 exploratory family-specific coherence analysis.

This script starts from the *same* 52 language-adjusted generator profiles used
by confirmatory H4. It does not modify the confirmatory verdict.

For each family F:
  Delta_F = mean cosine(within F) - mean cosine(F, outside F)

Inference:
  * generator-level bootstrap CI (10,000 draws),
  * one-sided generator-label/subset permutation test (10,000 Monte Carlo
    draws unless the complete size-preserving subset space is <=50,000),
  * Benjamini-Hochberg correction across exactly four families.

The confirmatory H4 status is locked to INSUFFICIENT_EVIDENCE.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
from common.manuscript_stats import bh_adjust, stable_seed  # noqa: E402

MASTER_SEED = 20260711
N_PERM = 10_000
N_BOOT = 10_000
EXACT_SUBSET_ENUMERATION_MAX = 50_000
EXPECTED_N = 52
EXPECTED_BINS = 481
LOCKED_CONFIRMATORY_STATUS = "INSUFFICIENT_EVIDENCE"
FAMILY_ORDER = [
    "Classical phase reconstruction",
    "GAN vocoder",
    "Integrated GAN/VITS",
    "Neural codec decoder",
]
EXPECTED_COUNTS = {
    "Classical phase reconstruction": 3,
    "GAN vocoder": 26,
    "Integrated GAN/VITS": 17,
    "Neural codec decoder": 6,
}


def first_existing(paths: Iterable[Path], label: str) -> Path:
    for p in paths:
        p = Path(p)
        if p.is_file():
            return p
    raise FileNotFoundError(label + " not found. Tried:\n" + "\n".join(map(str, paths)))


def normalize_family(x: str) -> str:
    s = str(x).strip()
    m = {
        "Classical phase rebuild": "Classical phase reconstruction",
        "Classical phase reconstruction": "Classical phase reconstruction",
        "GAN": "GAN vocoder",
        "GAN vocoder": "GAN vocoder",
        "Integrated GAN (VITS)": "Integrated GAN/VITS",
        "Integrated GAN/VITS": "Integrated GAN/VITS",
        "Neural codec decoder": "Neural codec decoder",
    }
    return m.get(s, s)


def cosine_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n <= 1e-15] = 1.0
    z = x / n
    return z @ z.T


def mean_upper_triangle(sim: np.ndarray, idx: np.ndarray) -> float:
    if len(idx) < 2:
        return float("nan")
    sub = sim[np.ix_(idx, idx)]
    vals = sub[np.triu_indices(len(idx), 1)]
    return float(vals.mean()) if len(vals) else float("nan")


def delta_for_subset(sim: np.ndarray, idx_in: np.ndarray, idx_out: np.ndarray):
    within = mean_upper_triangle(sim, idx_in)
    outside = float(np.mean(sim[np.ix_(idx_in, idx_out)]))
    return within, outside, float(within - outside)


def bootstrap_delta(sim: np.ndarray, idx_in: np.ndarray, idx_out: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(N_BOOT):
        a = rng.choice(idx_in, size=len(idx_in), replace=True)
        b = rng.choice(idx_out, size=len(idx_out), replace=True)
        # Duplicated bootstrap draws must not create artificial self-similarity.
        w = [float(sim[a[i], a[j]]) for i in range(len(a)) for j in range(i + 1, len(a)) if a[i] != a[j]]
        if not w:
            continue
        cross = float(np.mean(sim[np.ix_(a, b)]))
        vals.append(float(np.mean(w) - cross))
    if not vals:
        return [float("nan"), float("nan")], 0
    q = np.quantile(np.asarray(vals, dtype=float), [0.025, 0.975])
    return [float(q[0]), float(q[1])], len(vals)


def permutation_test(sim: np.ndarray, n_family: int, observed: float, seed: int):
    n_total = sim.shape[0]
    all_idx = np.arange(n_total, dtype=int)
    space = math.comb(n_total, n_family)
    null = []
    exact = space <= EXACT_SUBSET_ENUMERATION_MAX
    if exact:
        for comb in itertools.combinations(range(n_total), n_family):
            a = np.fromiter(comb, dtype=int)
            keep = np.ones(n_total, dtype=bool); keep[a] = False
            b = all_idx[keep]
            null.append(delta_for_subset(sim, a, b)[2])
        arr = np.asarray(null, dtype=float)
        p = float(np.mean(arr >= observed))
        method = "exact_all_subsets_preserving_family_size"
    else:
        rng = np.random.default_rng(seed)
        for _ in range(N_PERM):
            a = np.sort(rng.choice(all_idx, size=n_family, replace=False))
            keep = np.ones(n_total, dtype=bool); keep[a] = False
            b = all_idx[keep]
            null.append(delta_for_subset(sim, a, b)[2])
        arr = np.asarray(null, dtype=float)
        p = float((1 + np.sum(arr >= observed)) / (len(arr) + 1))
        method = "monte_carlo_random_subsets_preserving_family_size"
    return p, method, int(len(arr)), int(space)


def run(gen_path: Path, axis_path: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    axis = pd.read_csv(axis_path)
    if not {"column_name", "frequency_hz"}.issubset(axis.columns):
        raise RuntimeError("frequency_axis.csv must contain column_name and frequency_hz")
    band = axis[(axis.frequency_hz >= 80.0) & (axis.frequency_hz <= 7600.0)].copy()
    if len(band) != EXPECTED_BINS:
        raise RuntimeError(f"Expected 481 inferential bins, found {len(band)}")
    res = band.column_name.astype(str).tolist()
    df = pd.read_parquet(gen_path)
    family_col = "waveform_family" if "waveform_family" in df.columns else "canonical_family"
    gen_col = "independent_generator_id"
    missing = [c for c in [gen_col, family_col, *res] if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in generator profile table: {missing[:8]}")
    df = df[[gen_col, family_col, *res]].copy()
    df[family_col] = df[family_col].map(normalize_family)
    if len(df) != EXPECTED_N or df[gen_col].nunique() != EXPECTED_N:
        raise RuntimeError(f"Expected exactly 52 generator-level profiles, found {len(df)}")
    counts = df[family_col].value_counts().to_dict()
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"Family counts mismatch: {counts} != {EXPECTED_COUNTS}")

    x = df[res].to_numpy(float)
    sim = cosine_matrix(x)
    labels = df[family_col].astype(str).to_numpy()
    rows = []
    for fam in FAMILY_ORDER:
        idx_in = np.flatnonzero(labels == fam)
        idx_out = np.flatnonzero(labels != fam)
        w, cross, delta = delta_for_subset(sim, idx_in, idx_out)
        ci, n_boot_eff = bootstrap_delta(sim, idx_in, idx_out, stable_seed("H4F_BOOT", fam, MASTER_SEED))
        p, method, n_null, space = permutation_test(sim, len(idx_in), delta, stable_seed("H4F_PERM", fam, MASTER_SEED))
        rows.append({
            "family": fam,
            "n_generators": len(idx_in),
            "within_family_mean_cosine": w,
            "outside_family_mean_cosine": cross,
            "delta_F": delta,
            "ci95_low": ci[0],
            "ci95_high": ci[1],
            "p_one_sided": p,
            "permutation_method": method,
            "n_null": n_null,
            "permutation_space_size": space,
            "n_bootstrap_effective": n_boot_eff,
        })
    result = pd.DataFrame(rows)
    result["q_BH_4_families"] = bh_adjust(result.p_one_sided.to_numpy(float))
    result["significant_BH_0.05"] = result.q_BH_4_families < 0.05
    result.to_csv(out / "H4_FAMILY_SPECIFIC_FINAL.csv", index=False)

    sig = result.loc[result["significant_BH_0.05"], "family"].tolist()
    summary = {
        "version": "H4-FAMILY-SPECIFIC-EXPLORATORY-FINAL-v1.0.0",
        "role": "EXPLORATORY_ONLY",
        "confirmatory_H4_status_locked": LOCKED_CONFIRMATORY_STATUS,
        "confirmatory_status_changed_by_this_script": False,
        "n_generators": EXPECTED_N,
        "family_counts": EXPECTED_COUNTS,
        "analysis_bins": EXPECTED_BINS,
        "metric": "cosine similarity on final language-adjusted generator profiles p_g(f)",
        "contrast": "within-family mean cosine minus mean cosine to generators outside family",
        "multiplicity": "Benjamini-Hochberg across exactly four family-specific tests",
        "BH_significant_families": sig,
        "results": result.to_dict(orient="records"),
    }
    (out / "H4_FAMILY_SPECIFIC_FINAL_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(result.to_string(index=False))
    print("Confirmatory H4 remains:", LOCKED_CONFIRMATORY_STATUS)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/drive/MyDrive/fingerprint_q1_outputs")
    ap.add_argument("--generator-profiles")
    ap.add_argument("--axis")
    ap.add_argument("--output-dir")
    a = ap.parse_args(); root = Path(a.root)
    gen = Path(a.generator_profiles) if a.generator_profiles else first_existing([
        root / "phase1b/phase1b_family_fingerprints_v2/strict/fingerprints_generator_level_adjusted.parquet",
        root / "phase1b/phase1b_family_fingerprints_v3_new_story/strict/fingerprints_generator_level_adjusted.parquet",
    ], "final H4 generator-level adjusted profiles")
    axis = Path(a.axis) if a.axis else first_existing([
        root / "phase1a/phase1a_mlaad_spectral_residuals_v1/frequency_axis.csv",
        root / "phase1a/phase1a_mlaad_spectral_residuals_v2_new_story/frequency_axis.csv",
    ], "MLAAD frequency axis")
    out = Path(a.output_dir) if a.output_dir else root / "H4_FAMILY_SPECIFIC_EXPLORATORY_FINAL_v1"
    run(gen, axis, out)


if __name__ == "__main__":
    main()
