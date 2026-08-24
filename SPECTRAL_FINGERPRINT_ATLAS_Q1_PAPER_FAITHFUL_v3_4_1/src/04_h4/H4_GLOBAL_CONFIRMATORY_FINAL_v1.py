#!/usr/bin/env python3


# Purpose: Test H4 family-level structure with generator-level inference, same-language adjustment, permutations, bootstrap intervals, LOGO validation, and Holm correction.

from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


VERSION = "H4-GLOBAL-CONFIRMATORY-FINAL-v1.0.0"

ROOT = Path(os.environ.get("FINGERPRINT_OUTPUT_ROOT", "/content/drive/MyDrive/fingerprint_q1_outputs"))
_PHASE1A_CANDIDATES = [
    ROOT / "phase1a/phase1a_mlaad_spectral_residuals_v1",
    ROOT / "phase1a/phase1a_mlaad_spectral_residuals_v2_new_story",
]
PHASE1A_DIR = next((d for d in _PHASE1A_CANDIDATES if (d / "fingerprints_pair_level_strict.parquet").is_file()), _PHASE1A_CANDIDATES[0])
STRICT_INPUT = PHASE1A_DIR / "fingerprints_pair_level_strict.parquet"
FREQUENCY_AXIS_INPUT = PHASE1A_DIR / "frequency_axis.csv"
PHASE1A_SUMMARY_INPUT = PHASE1A_DIR / "phase1a_summary.json"

OUTPUT_DIR = Path(os.environ.get(
    "FINGERPRINT_H4_OUTPUT_DIR",
    str(ROOT / "phase1b/phase1b_family_fingerprints_v2")
))

FORCE_REBUILD = False
RANDOM_SEED = 20260711


N_PERMUTATIONS = 5_000
N_LOGO_PERMUTATIONS = 1_000
N_BOOTSTRAPS = 2_000

ANALYSIS_MIN_HZ = 80.0
ANALYSIS_MAX_HZ = 7_600.0


MIN_LANGUAGE_REFERENCE_GENERATORS = 2
ALLOW_GLOBAL_LANGUAGE_FALLBACK = False


ALPHA = 0.05
CONFIRMATORY_MULTIPLICITY_METHOD = "holm"
CONFIRMATORY_TEST_NAMES = (
    "family_similarity",
    "multivariate_centroid",
    "dispersion",
    "logo_balanced_accuracy",
    "logo_macro_f1",
)


import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

GENERATOR_COL = "independent_generator_id"
FAMILY_COL = "waveform_family"
LANGUAGE_COL = "language"
PAIR_ID_COL = "pair_id"
STATUS_COL = "status"

EXPECTED_STRICT_ROWS = 62_079
EXPECTED_GENERATORS = 52
EXPECTED_FAMILIES = 4
EXPECTED_STORED_BINS = 513


PREFERRED_FAMILY_ORDER = [
    "Classical phase rebuild",
    "GAN vocoder",
    "Integrated GAN (VITS)",
    "Neural codec decoder",
]


def stable_json_hash(payload: Any) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, default=str)
    os.replace(temporary, path)


def atomic_csv_dump(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def atomic_parquet_dump(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd", engine="pyarrow")
    os.replace(temporary, path)


def config_payload() -> Dict[str, Any]:
    return {
        "version": VERSION,
        "random_seed": RANDOM_SEED,
        "n_permutations": N_PERMUTATIONS,
        "n_logo_permutations": N_LOGO_PERMUTATIONS,
        "n_bootstraps": N_BOOTSTRAPS,
        "analysis_min_hz": ANALYSIS_MIN_HZ,
        "analysis_max_hz": ANALYSIS_MAX_HZ,
        "min_language_reference_generators": MIN_LANGUAGE_REFERENCE_GENERATORS,
        "allow_global_language_fallback": ALLOW_GLOBAL_LANGUAGE_FALLBACK,
        "alpha": ALPHA,
        "confirmatory_multiplicity_method": CONFIRMATORY_MULTIPLICITY_METHOD,
        "confirmatory_test_names": list(CONFIRMATORY_TEST_NAMES),
        "unit_of_inference": "independent_generator_id",
        "primary_protocol": "strict",
        "language_role": "nuisance_control_only",
        "family_test": "generator_level_permutation_after_cross_fitted_language_adjustment",
        "classifier": "leave_one_generator_out_nearest_family_median_cosine",
    }


def environment_report() -> Dict[str, Any]:
    packages = {}
    for name in ["numpy", "pandas", "scipy", "sklearn", "pyarrow"]:
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            packages[name] = f"ERROR:{type(exc).__name__}:{exc}"
    return {
        "version": VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


def family_order(observed: Sequence[str]) -> List[str]:
    observed_set = set(map(str, observed))
    ordered = [f for f in PREFERRED_FAMILY_ORDER if f in observed_set]
    ordered.extend(sorted(observed_set - set(ordered)))
    return ordered




def empirical_p_upper(observed: float, null: np.ndarray) -> float:
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if not np.isfinite(observed) or len(null) == 0:
        return float("nan")
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:

    p = np.asarray(p_values, dtype=float)
    result = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return result
    pv = p[valid]
    order = np.argsort(pv, kind="mergesort")
    ranked = pv[order]
    m = len(ranked)
    adjusted_ranked = np.empty(m, dtype=float)
    running = 0.0
    for i, value in enumerate(ranked):
        candidate = (m - i) * float(value)
        running = max(running, candidate)
        adjusted_ranked[i] = min(running, 1.0)
    restored = np.empty_like(adjusted_ranked)
    restored[order] = adjusted_ranked
    result[valid] = restored
    return result


def derived_seed(master_seed: int, label: str) -> int:

    payload = f"{int(master_seed)}::{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32 - 1)


def adjust_confirmatory_pvalues(
    similarity_report: Mapping[str, Any],
    multivariate_report: Mapping[str, Any],
    logo_report: Mapping[str, Any],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    raw = {
        "family_similarity": float(similarity_report["permutation_p_upper"]),
        "multivariate_centroid": float(multivariate_report["permutation_p_pseudo_f"]),
        "dispersion": float(multivariate_report["dispersion_test"]["permutation_p"]),
        "logo_balanced_accuracy": float(logo_report["balanced_accuracy_permutation_p"]),
        "logo_macro_f1": float(logo_report["macro_f1_permutation_p"]),
    }
    if tuple(raw) != tuple(CONFIRMATORY_TEST_NAMES):
        raise RuntimeError("The confirmatory family does not match the frozen configuration")
    adjusted_values = holm_adjust(list(raw.values()))
    adjusted = {name: float(value) for name, value in zip(raw, adjusted_values)}
    table = pd.DataFrame({
        "test": list(raw),
        "p_raw": list(raw.values()),
        "p_holm": list(adjusted.values()),
    })
    table["significant_raw_0_05"] = table["p_raw"] < ALPHA
    table["significant_holm_0_05"] = table["p_holm"] < ALPHA
    report = {
        "method": CONFIRMATORY_MULTIPLICITY_METHOD,
        "alpha": float(ALPHA),
        "scope": list(CONFIRMATORY_TEST_NAMES),
        "raw_p_values": raw,
        "adjusted_p_values": adjusted,
        "n_tests": int(len(raw)),
    }
    return report, table


def robust_feature_scaler_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    center = np.median(x, axis=0)
    mad = np.median(np.abs(x - center[None, :]), axis=0)
    scale = 1.4826 * mad
    fallback = np.std(x, axis=0, ddof=1)
    scale = np.where(scale > 1e-8, scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    return center.astype(np.float64), scale.astype(np.float64)


def robust_feature_scaler_transform(
    x: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return (x - center[None, :]) / scale[None, :]


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return x / norms




def read_frequency_axis() -> pd.DataFrame:
    if not FREQUENCY_AXIS_INPUT.is_file():
        raise FileNotFoundError(f"Axis frequency not found: {FREQUENCY_AXIS_INPUT}")
    axis = pd.read_csv(FREQUENCY_AXIS_INPUT)
    required = {"bin_index", "column_name", "frequency_hz"}
    missing = required - set(axis.columns)
    if missing:
        raise RuntimeError(f"frequency_axis.csv incomplete: {sorted(missing)}")
    axis = axis.sort_values("bin_index").reset_index(drop=True)
    if len(axis) != EXPECTED_STORED_BINS:
        raise RuntimeError(f"Axis frequency: {len(axis)} bins; {EXPECTED_STORED_BINS} expected")
    expected_names = [f"res_{i:04d}" for i in range(EXPECTED_STORED_BINS)]
    if axis["column_name"].astype(str).tolist() != expected_names:
        raise RuntimeError("Residual column names do not match 0..512")
    return axis


def parquet_residual_columns(path: Path) -> List[str]:
    schema_names = pq.ParquetFile(path).schema.names
    residual = sorted(
        [name for name in schema_names if name.startswith("res_")],
        key=lambda name: int(name.split("_")[1]),
    )
    return residual


def load_pair_level(
    path: Path,
    protocol: str,
    expected_rows: int,
    residual_columns: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input {protocol} not found: {path}")

    observed_residual = parquet_residual_columns(path)
    if list(observed_residual) != list(residual_columns):
        raise RuntimeError(f"{protocol}: unexpected residual columns")

    metadata_columns = [
        PAIR_ID_COL,
        GENERATOR_COL,
        FAMILY_COL,
        LANGUAGE_COL,
        STATUS_COL,
    ]
    table = pq.read_table(path, columns=metadata_columns + list(residual_columns))
    frame = table.to_pandas()

    if len(frame) != expected_rows:
        raise RuntimeError(
            f"{protocol}: {len(frame):,} rows observed; {expected_rows:,} expected"
        )
    if frame[PAIR_ID_COL].astype(str).duplicated().any():
        raise RuntimeError(f"{protocol}: pair_id duplicate")
    if not frame[STATUS_COL].astype(str).eq("ok").all():
        bad = int((~frame[STATUS_COL].astype(str).eq("ok")).sum())
        raise RuntimeError(f"{protocol}: {bad} rows with status != ok")

    for column in [PAIR_ID_COL, GENERATOR_COL, FAMILY_COL, LANGUAGE_COL]:
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise RuntimeError(f"{protocol}: empty values in {column}")

    values = frame[list(residual_columns)].to_numpy(dtype=np.float32, copy=False)
    if values.shape != (expected_rows, EXPECTED_STORED_BINS):
        raise RuntimeError(f"{protocol}: matrix residual de shape {values.shape}")
    if not np.isfinite(values).all():
        raise RuntimeError(f"{protocol}: NaN or Inf in residuals")

    generator_family = frame[[GENERATOR_COL, FAMILY_COL]].drop_duplicates()
    if generator_family[GENERATOR_COL].duplicated().any():
        raise RuntimeError(f"{protocol}: a generator belongs to multiple families")

    report = {
        "protocol": protocol,
        "input": str(path),
        "input_sha256": sha256_file(path),
        "n_pairs": int(len(frame)),
        "n_generators": int(frame[GENERATOR_COL].nunique()),
        "n_families": int(frame[FAMILY_COL].nunique()),
        "n_languages": int(frame[LANGUAGE_COL].nunique()),
        "n_bins": int(len(residual_columns)),
        "family_generator_counts": {
            str(k): int(v)
            for k, v in generator_family.groupby(FAMILY_COL)[GENERATOR_COL]
            .nunique().sort_index().items()
        },
        "language_generator_counts": {
            str(k): int(v)
            for k, v in frame[[GENERATOR_COL, LANGUAGE_COL]].drop_duplicates()
            .groupby(LANGUAGE_COL)[GENERATOR_COL].nunique().sort_index().items()
        },
    }

    if report["n_generators"] != EXPECTED_GENERATORS:
        raise RuntimeError(
            f"{protocol}: {report['n_generators']} generators; {EXPECTED_GENERATORS} expected"
        )
    if report["n_families"] != EXPECTED_FAMILIES:
        raise RuntimeError(
            f"{protocol}: {report['n_families']} families; {EXPECTED_FAMILIES} expected"
        )

    return frame, report


def aggregate_generator_language(
    pair_frame: pd.DataFrame,
    residual_columns: Sequence[str],
) -> pd.DataFrame:
    keys = [GENERATOR_COL, FAMILY_COL, LANGUAGE_COL]
    grouped = pair_frame.groupby(keys, observed=True, sort=True)
    medians = grouped[list(residual_columns)].median().reset_index()
    counts = grouped.size().rename("n_pairs").reset_index()
    cells = medians.merge(counts, on=keys, how="left", validate="one_to_one")
    cells["n_pairs"] = cells["n_pairs"].astype(np.int32)
    cells[list(residual_columns)] = cells[list(residual_columns)].astype(np.float32)

    mapping = cells[[GENERATOR_COL, FAMILY_COL]].drop_duplicates()
    if mapping[GENERATOR_COL].duplicated().any():
        raise RuntimeError("Aggregation: a generator belongs to multiple families")
    return cells.sort_values([FAMILY_COL, GENERATOR_COL, LANGUAGE_COL]).reset_index(drop=True)


def language_reference(
    cells: pd.DataFrame,
    x: np.ndarray,
    language: str,
    excluded_generator: Optional[str],
) -> Tuple[np.ndarray, str, int]:
    generators = cells[GENERATOR_COL].astype(str).to_numpy()
    languages = cells[LANGUAGE_COL].astype(str).to_numpy()

    mask = languages == str(language)
    if excluded_generator is not None:
        mask &= generators != str(excluded_generator)

    n_generators = len(np.unique(generators[mask]))
    if n_generators >= MIN_LANGUAGE_REFERENCE_GENERATORS:
        return np.median(x[mask], axis=0), "language", int(n_generators)

    if ALLOW_GLOBAL_LANGUAGE_FALLBACK:
        fallback = np.ones(len(cells), dtype=bool)
        if excluded_generator is not None:
            fallback &= generators != str(excluded_generator)
        n_global = len(np.unique(generators[fallback]))
        if n_global < 1:
            raise RuntimeError("Unable to build a language-specific or global reference")
        return np.median(x[fallback], axis=0), "global_fallback", int(n_global)

    raise RuntimeError(
        "Language-specific reference unavailable without global fallback : "
        f"language={language!r}, excluded_generator={excluded_generator!r}, "
        f"n_other_generators={n_generators}. "
        "Correct the population or explicitly enable ALLOW_GLOBAL_LANGUAGE_FALLBACK."
    )

def cross_fitted_language_adjustment(
    cells: pd.DataFrame,
    residual_columns: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:

    x = cells[list(residual_columns)].to_numpy(dtype=np.float64, copy=False)
    adjusted = np.empty_like(x)
    sources: List[str] = []
    reference_sizes: List[int] = []

    for row_index, row in cells[[GENERATOR_COL, LANGUAGE_COL]].iterrows():
        ref, source, n_ref = language_reference(
            cells,
            x,
            language=str(row[LANGUAGE_COL]),
            excluded_generator=str(row[GENERATOR_COL]),
        )
        adjusted[row_index] = x[row_index] - ref
        sources.append(source)
        reference_sizes.append(n_ref)

    metadata_part = cells[[GENERATOR_COL, FAMILY_COL, LANGUAGE_COL, "n_pairs"]].copy()
    metadata_part["language_reference_source"] = sources
    metadata_part["language_reference_generators"] = np.asarray(reference_sizes, dtype=np.int32)
    residual_part = pd.DataFrame(
        adjusted.astype(np.float32),
        columns=list(residual_columns),
        index=metadata_part.index,
    )
    adjusted_cells = pd.concat([metadata_part, residual_part], axis=1)

    grouped = adjusted_cells.groupby(
        [GENERATOR_COL, FAMILY_COL], observed=True, sort=True
    )
    generator = grouped[list(residual_columns)].median().reset_index()
    n_languages = grouped[LANGUAGE_COL].nunique().rename("n_languages").reset_index()
    n_pairs = grouped["n_pairs"].sum().rename("n_pairs_total").reset_index()
    languages = grouped[LANGUAGE_COL].agg(
        lambda values: "|".join(sorted(set(map(str, values))))
    ).rename("languages").reset_index()

    generator = generator.merge(n_languages, on=[GENERATOR_COL, FAMILY_COL], validate="one_to_one")
    generator = generator.merge(n_pairs, on=[GENERATOR_COL, FAMILY_COL], validate="one_to_one")
    generator = generator.merge(languages, on=[GENERATOR_COL, FAMILY_COL], validate="one_to_one")
    generator[list(residual_columns)] = generator[list(residual_columns)].astype(np.float32)

    fallback_count = int((adjusted_cells["language_reference_source"] == "global_fallback").sum())
    if fallback_count and not ALLOW_GLOBAL_LANGUAGE_FALLBACK:
        raise RuntimeError(f"{fallback_count} unauthorized global fallbacks detected")

    report = {
        "n_generator_language_cells": int(len(cells)),
        "n_generators": int(generator[GENERATOR_COL].nunique()),
        "n_multilingual_generators": int((generator["n_languages"] > 1).sum()),
        "n_monolingual_generators": int((generator["n_languages"] == 1).sum()),
        "reference_source_counts": {
            str(k): int(v)
            for k, v in adjusted_cells["language_reference_source"].value_counts().items()
        },
        "minimum_reference_generators": int(min(reference_sizes)),
        "median_reference_generators": float(np.median(reference_sizes)),
        "global_fallback_count": fallback_count,
        "global_fallback_allowed": bool(ALLOW_GLOBAL_LANGUAGE_FALLBACK),
    }
    return adjusted_cells, generator, report


def pairwise_similarity_table(
    generator_frame: pd.DataFrame,
    residual_columns: Sequence[str],
    analysis_mask: np.ndarray,
) -> pd.DataFrame:
    x = generator_frame[list(residual_columns)].to_numpy(dtype=np.float64)[:, analysis_mask]
    x_norm = l2_normalize_rows(x)
    cosine = x_norm @ x_norm.T
    euclidean = cdist(x, x, metric="euclidean")

    
    ranks = np.apply_along_axis(rankdata, 1, x)
    ranks = ranks - ranks.mean(axis=1, keepdims=True)
    ranks = l2_normalize_rows(ranks)
    spearman = ranks @ ranks.T

    generators = generator_frame[GENERATOR_COL].astype(str).to_numpy()
    families = generator_frame[FAMILY_COL].astype(str).to_numpy()
    languages = generator_frame["languages"].astype(str).to_numpy()

    records: List[Dict[str, Any]] = []
    for i in range(len(generator_frame)):
        for j in range(i + 1, len(generator_frame)):
            records.append({
                "generator_a": generators[i],
                "generator_b": generators[j],
                "family_a": families[i],
                "family_b": families[j],
                "languages_a": languages[i],
                "languages_b": languages[j],
                "same_family": bool(families[i] == families[j]),
                "cosine_similarity": float(cosine[i, j]),
                "spearman_similarity": float(spearman[i, j]),
                "euclidean_distance": float(euclidean[i, j]),
            })
    return pd.DataFrame.from_records(records)


def similarity_delta_from_matrix(similarity: np.ndarray, labels: np.ndarray) -> float:
    upper = np.triu_indices(len(labels), k=1)
    same = labels[upper[0]] == labels[upper[1]]
    values = similarity[upper]
    if not same.any() or (~same).sum() == 0:
        return float("nan")
    return float(values[same].mean() - values[~same].mean())


def similarity_delta_from_bootstrap_sample(
    similarity: np.ndarray,
    labels: np.ndarray,
    sampled_original_indices: np.ndarray,
) -> Tuple[float, int, int]:


    sampled = np.asarray(sampled_original_indices, dtype=int)
    pos_i, pos_j = np.triu_indices(len(sampled), k=1)
    orig_i = sampled[pos_i]
    orig_j = sampled[pos_j]
    distinct = orig_i != orig_j
    excluded_self_pairs = int((~distinct).sum())
    if not distinct.any():
        return float("nan"), excluded_self_pairs, 0
    orig_i = orig_i[distinct]
    orig_j = orig_j[distinct]
    values = similarity[orig_i, orig_j]
    same = labels[orig_i] == labels[orig_j]
    if not same.any() or not (~same).any():
        return float("nan"), excluded_self_pairs, int(len(values))
    delta = float(values[same].mean() - values[~same].mean())
    return delta, excluded_self_pairs, int(len(values))


def jackknife_similarity_delta_ci(
    similarity: np.ndarray,
    labels: np.ndarray,
    observed: float,
) -> Tuple[list[float], float]:

    estimates = []
    n = len(labels)
    for leave_out in range(n):
        keep = np.arange(n) != leave_out
        estimates.append(
            similarity_delta_from_matrix(
                similarity[np.ix_(keep, keep)], labels[keep]
            )
        )
    values = np.asarray(estimates, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return [float("nan"), float("nan")], float("nan")
    mean_j = float(values.mean())
    se = float(np.sqrt((len(values) - 1) / len(values) * np.sum((values - mean_j) ** 2)))
    return [float(observed - 1.959963984540054 * se), float(observed + 1.959963984540054 * se)], se


def generator_similarity_test(
    generator_frame: pd.DataFrame,
    residual_columns: Sequence[str],
    analysis_mask: np.ndarray,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    x = generator_frame[list(residual_columns)].to_numpy(dtype=np.float64)[:, analysis_mask]
    similarity = l2_normalize_rows(x) @ l2_normalize_rows(x).T
    labels = generator_frame[FAMILY_COL].astype(str).to_numpy(dtype=object)

    observed = similarity_delta_from_matrix(similarity, labels)
    null = np.empty(N_PERMUTATIONS, dtype=np.float64)
    for b in range(N_PERMUTATIONS):
        permuted = rng.permutation(labels)
        null[b] = similarity_delta_from_matrix(similarity, permuted)

    family_indices = {
        family: np.flatnonzero(labels == family)
        for family in np.unique(labels)
    }
    bootstrap = np.empty(N_BOOTSTRAPS, dtype=np.float64)
    excluded_counts = np.empty(N_BOOTSTRAPS, dtype=np.int32)
    effective_pair_counts = np.empty(N_BOOTSTRAPS, dtype=np.int32)
    for b in range(N_BOOTSTRAPS):
        sampled_parts = [
            rng.choice(indices, size=len(indices), replace=True)
            for indices in family_indices.values()
        ]
        sampled = np.concatenate(sampled_parts)
        delta_b, n_excluded, n_effective = similarity_delta_from_bootstrap_sample(
            similarity, labels, sampled
        )
        bootstrap[b] = delta_b
        excluded_counts[b] = n_excluded
        effective_pair_counts[b] = n_effective

    bootstrap_valid = bootstrap[np.isfinite(bootstrap)]
    if len(bootstrap_valid) < max(100, int(0.9 * N_BOOTSTRAPS)):
        raise RuntimeError(
            f"Insufficient valid bootstrap replicates: {len(bootstrap_valid)}/{N_BOOTSTRAPS}"
        )
    jackknife_ci, jackknife_se = jackknife_similarity_delta_ci(similarity, labels, observed)

    upper = np.triu_indices(len(labels), k=1)
    same = labels[upper[0]] == labels[upper[1]]
    all_values = similarity[upper]

    return {
        "metric": "cosine_similarity_on_cross_fitted_language_adjusted_generator_fingerprints",
        "n_generators": int(len(labels)),
        "mean_intra_family": float(all_values[same].mean()),
        "mean_inter_family": float(all_values[~same].mean()),
        "delta_intra_minus_inter": float(observed),
        "bootstrap_ci95": [
            float(np.quantile(bootstrap_valid, 0.025)),
            float(np.quantile(bootstrap_valid, 0.975)),
        ],
        "bootstrap_method": (
            "stratified generator bootstrap with replacement within family; "
            "pairs between duplicate copies of the same original generator excluded"
        ),
        "bootstrap_valid_repetitions": int(len(bootstrap_valid)),
        "bootstrap_mean_artificial_self_pairs_excluded": float(excluded_counts.mean()),
        "bootstrap_min_effective_pairs": int(effective_pair_counts.min()),
        "jackknife_ci95": jackknife_ci,
        "jackknife_standard_error": float(jackknife_se),
        "permutation_p_upper": empirical_p_upper(observed, null),
        "null_mean": float(np.mean(null)),
        "null_q95": float(np.quantile(null, 0.95)),
        "n_permutations": int(N_PERMUTATIONS),
        "n_bootstraps": int(N_BOOTSTRAPS),
    }


def multivariate_oneway_stats(x: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    labels = np.asarray(labels, dtype=object)
    n = len(labels)
    groups = np.unique(labels)
    k = len(groups)
    grand = x.mean(axis=0)
    ss_total = float(np.sum((x - grand[None, :]) ** 2))
    ss_between = 0.0
    ss_within = 0.0
    for group in groups:
        idx = labels == group
        centroid = x[idx].mean(axis=0)
        ss_between += float(idx.sum() * np.sum((centroid - grand) ** 2))
        ss_within += float(np.sum((x[idx] - centroid[None, :]) ** 2))
    df_between = max(k - 1, 1)
    df_within = max(n - k, 1)
    pseudo_f = (ss_between / df_between) / max(ss_within / df_within, 1e-15)
    r2 = ss_between / max(ss_total, 1e-15)
    return float(pseudo_f), float(r2)


def dispersion_f_stat(x: np.ndarray, labels: np.ndarray) -> Tuple[float, np.ndarray]:
    labels = np.asarray(labels, dtype=object)
    distances = np.empty(len(labels), dtype=np.float64)
    for group in np.unique(labels):
        idx = labels == group
        centroid = x[idx].mean(axis=0)
        distances[idx] = np.linalg.norm(x[idx] - centroid[None, :], axis=1)

    grand = distances.mean()
    ss_between = 0.0
    ss_within = 0.0
    groups = np.unique(labels)
    for group in groups:
        idx = labels == group
        mean_group = distances[idx].mean()
        ss_between += float(idx.sum() * (mean_group - grand) ** 2)
        ss_within += float(np.sum((distances[idx] - mean_group) ** 2))
    df_between = max(len(groups) - 1, 1)
    df_within = max(len(labels) - len(groups), 1)
    f_value = (ss_between / df_between) / max(ss_within / df_within, 1e-15)
    return float(f_value), distances


def multivariate_family_test(
    generator_frame: pd.DataFrame,
    residual_columns: Sequence[str],
    analysis_mask: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    x_raw = generator_frame[list(residual_columns)].to_numpy(dtype=np.float64)[:, analysis_mask]
    center, scale = robust_feature_scaler_fit(x_raw)
    x = robust_feature_scaler_transform(x_raw, center, scale)
    labels = generator_frame[FAMILY_COL].astype(str).to_numpy(dtype=object)

    observed_f, observed_r2 = multivariate_oneway_stats(x, labels)
    observed_dispersion_f, distances = dispersion_f_stat(x, labels)

    null_f = np.empty(N_PERMUTATIONS, dtype=np.float64)
    null_r2 = np.empty(N_PERMUTATIONS, dtype=np.float64)
    null_dispersion = np.empty(N_PERMUTATIONS, dtype=np.float64)
    for b in range(N_PERMUTATIONS):
        permuted = rng.permutation(labels)
        null_f[b], null_r2[b] = multivariate_oneway_stats(x, permuted)
        null_dispersion[b], _ = dispersion_f_stat(x, permuted)

    dispersion_frame = generator_frame[[GENERATOR_COL, FAMILY_COL, "languages"]].copy()
    dispersion_frame["distance_to_family_centroid"] = distances

    report = {
        "test": "euclidean_permanova_equivalent_oneway_on_robust_scaled_language_adjusted_generator_fingerprints",
        "pseudo_f": float(observed_f),
        "r2_family": float(observed_r2),
        "permutation_p_pseudo_f": empirical_p_upper(observed_f, null_f),
        "permutation_p_r2": empirical_p_upper(observed_r2, null_r2),
        "pseudo_f_null_q95": float(np.quantile(null_f, 0.95)),
        "dispersion_test": {
            "f_statistic": float(observed_dispersion_f),
            "permutation_p": empirical_p_upper(observed_dispersion_f, null_dispersion),
            "interpretation": (
                "A small p-value indicates unequal within-family dispersions; "
                "in that case the centroid-separation test must be interpreted cautiously."
            ),
        },
        "n_permutations": int(N_PERMUTATIONS),
    }
    return report, dispersion_frame




def build_fold_generator_vectors(
    cells: pd.DataFrame,
    residual_columns: Sequence[str],
    held_out_generator: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, Any]]:

    residual_columns = list(residual_columns)
    all_x = cells[residual_columns].to_numpy(dtype=np.float64, copy=False)
    all_generators = cells[GENERATOR_COL].astype(str).to_numpy()

    train_mask = all_generators != str(held_out_generator)
    test_mask = ~train_mask
    if not test_mask.any():
        raise RuntimeError(f"LOGO: generator missing: {held_out_generator}")

    train_cells = cells.loc[train_mask].reset_index(drop=True)
    train_x_raw = all_x[train_mask]
    test_cells = cells.loc[test_mask].reset_index(drop=True)
    test_x_raw = all_x[test_mask]

    train_generators_cell = train_cells[GENERATOR_COL].astype(str).to_numpy()
    train_languages_cell = train_cells[LANGUAGE_COL].astype(str).to_numpy()
    global_train_reference = np.median(train_x_raw, axis=0)

    language_refs: Dict[str, np.ndarray] = {}
    language_ref_sizes: Dict[str, int] = {}
    test_fallback_count = 0
    for language in sorted(set(test_cells[LANGUAGE_COL].astype(str))):
        idx = train_languages_cell == language
        n_generators = len(np.unique(train_generators_cell[idx]))
        if n_generators >= MIN_LANGUAGE_REFERENCE_GENERATORS:
            language_refs[language] = np.median(train_x_raw[idx], axis=0)
            language_ref_sizes[language] = int(n_generators)
        elif ALLOW_GLOBAL_LANGUAGE_FALLBACK:
            language_refs[language] = global_train_reference
            language_ref_sizes[language] = 0
            test_fallback_count += int((test_cells[LANGUAGE_COL].astype(str) == language).sum())
        else:
            raise RuntimeError(
                f"LOGO {held_out_generator}: no training reference in language {language!r}"
            )

    train_adjusted = np.empty_like(train_x_raw)
    train_fallback_count = 0
    train_reference_sizes: list[int] = []
    for i, row in train_cells[[GENERATOR_COL, LANGUAGE_COL]].iterrows():
        generator = str(row[GENERATOR_COL])
        language = str(row[LANGUAGE_COL])
        mask_ref = (train_languages_cell == language) & (train_generators_cell != generator)
        n_ref = len(np.unique(train_generators_cell[mask_ref]))
        if n_ref >= MIN_LANGUAGE_REFERENCE_GENERATORS:
            reference = np.median(train_x_raw[mask_ref], axis=0)
        elif ALLOW_GLOBAL_LANGUAGE_FALLBACK:
            mask_global = train_generators_cell != generator
            reference = np.median(train_x_raw[mask_global], axis=0)
            train_fallback_count += 1
        else:
            raise RuntimeError(
                f"LOGO {held_out_generator}: training cell has no other within-language reference "
                f"(generator={generator!r}, language={language!r})"
            )
        train_reference_sizes.append(int(n_ref))
        train_adjusted[i] = train_x_raw[i] - reference

    test_adjusted = np.empty_like(test_x_raw)
    for i, language in enumerate(test_cells[LANGUAGE_COL].astype(str)):
        test_adjusted[i] = test_x_raw[i] - language_refs[language]

    train_records = []
    for generator, idx_frame in train_cells.groupby(GENERATOR_COL, sort=True).groups.items():
        idx = np.asarray(list(idx_frame), dtype=int)
        family_values = train_cells.loc[idx, FAMILY_COL].astype(str).unique()
        if len(family_values) != 1:
            raise RuntimeError(f"LOGO: ambiguous family for {generator}")
        train_records.append((
            str(generator),
            str(family_values[0]),
            np.median(train_adjusted[idx], axis=0),
        ))

    train_generator_ids = [item[0] for item in train_records]
    y_train = np.asarray([item[1] for item in train_records], dtype=object)
    x_train = np.vstack([item[2] for item in train_records])

    test_family_values = test_cells[FAMILY_COL].astype(str).unique()
    if len(test_family_values) != 1:
        raise RuntimeError(f"LOGO: ambiguous test family for {held_out_generator}")
    y_test = np.asarray([str(test_family_values[0])], dtype=object)
    x_test = np.median(test_adjusted, axis=0, keepdims=True)

    diagnostics = {
        "held_out_generator": str(held_out_generator),
        "test_languages": sorted(test_cells[LANGUAGE_COL].astype(str).unique().tolist()),
        "train_language_fallback_cells": int(train_fallback_count),
        "test_language_fallback_cells": int(test_fallback_count),
        "minimum_train_reference_generators": int(min(train_reference_sizes)) if train_reference_sizes else 0,
        "test_language_reference_sizes": {str(k): int(v) for k, v in language_ref_sizes.items()},
        "global_fallback_allowed": bool(ALLOW_GLOBAL_LANGUAGE_FALLBACK),
    }
    return x_train, y_train, x_test, y_test, train_generator_ids, diagnostics



def precompute_logo_folds(
    cells: pd.DataFrame,
    residual_columns: Sequence[str],
    analysis_mask: np.ndarray,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    folds: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []
    generators = sorted(cells[GENERATOR_COL].astype(str).unique())

    for index, held_out in enumerate(generators, start=1):
        x_train, y_train, x_test, y_test, train_ids, diagnostic = build_fold_generator_vectors(
            cells,
            residual_columns,
            held_out_generator=held_out,
        )
        x_train_band = x_train[:, analysis_mask]
        x_test_band = x_test[:, analysis_mask]
        scaler_center, scaler_scale = robust_feature_scaler_fit(x_train_band)
        z_train = l2_normalize_rows(
            robust_feature_scaler_transform(x_train_band, scaler_center, scaler_scale)
        )
        z_test = l2_normalize_rows(
            robust_feature_scaler_transform(x_test_band, scaler_center, scaler_scale)
        )
        folds.append({
            "held_out_generator": held_out,
            "train_generator_ids": np.asarray(train_ids, dtype=object),
            "z_train": z_train,
            "y_train": y_train,
            "z_test": z_test,
            "y_test": y_test,
        })
        diagnostics.append(diagnostic)
        if index % 10 == 0 or index == len(generators):
            print(f"[LOGO] Precomputing folds: {index}/{len(generators)}")
    return folds, diagnostics


def evaluate_logo_folds(
    folds: Sequence[Mapping[str, Any]],
    generator_to_label: Mapping[str, str],
    families: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    true_labels: List[str] = []
    predicted_labels: List[str] = []

    for fold in folds:
        train_ids = np.asarray(fold["train_generator_ids"], dtype=object)
        y_train = np.asarray([generator_to_label[str(g)] for g in train_ids], dtype=object)
        true = str(generator_to_label[str(fold["held_out_generator"])])
        z_train = np.asarray(fold["z_train"])
        z_test = np.asarray(fold["z_test"])
        scores: Dict[str, float] = {}
        for family in families:
            idx = y_train == family
            if not idx.any():
                scores[str(family)] = float("-inf")
                continue
            prototype = np.median(z_train[idx], axis=0, keepdims=True)
            prototype = l2_normalize_rows(prototype)[0]
            scores[str(family)] = float(z_test[0] @ prototype)
        predicted = max(scores, key=scores.get)
        true_labels.append(true)
        predicted_labels.append(predicted)
        row = {
            "held_out_generator": str(fold["held_out_generator"]),
            "true_family": true,
            "predicted_family": predicted,
            "correct": bool(true == predicted),
        }
        for family in families:
            row[f"score__{family}"] = float(scores.get(family, float("-inf")))
        records.append(row)

    ba = balanced_accuracy_score(true_labels, predicted_labels)
    macro_f1 = f1_score(
        true_labels,
        predicted_labels,
        labels=list(families),
        average="macro",
        zero_division=0,
    )
    matrix = confusion_matrix(true_labels, predicted_labels, labels=list(families))
    report = {
        "balanced_accuracy": float(ba),
        "macro_f1": float(macro_f1),
        "accuracy": float(np.mean(np.asarray(true_labels) == np.asarray(predicted_labels))),
        "n_generators": int(len(true_labels)),
        "family_order": list(families),
        "confusion_matrix": matrix.tolist(),
    }
    return pd.DataFrame.from_records(records), report


def logo_analysis(
    cells: pd.DataFrame,
    residual_columns: Sequence[str],
    analysis_mask: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    families = family_order(cells[FAMILY_COL].astype(str).unique())
    mapping_frame = cells[[GENERATOR_COL, FAMILY_COL]].drop_duplicates().sort_values(GENERATOR_COL)
    generator_ids = mapping_frame[GENERATOR_COL].astype(str).to_numpy(dtype=object)
    labels = mapping_frame[FAMILY_COL].astype(str).to_numpy(dtype=object)
    observed_mapping = dict(zip(generator_ids, labels))

    folds, diagnostics = precompute_logo_folds(cells, residual_columns, analysis_mask)
    predictions, report = evaluate_logo_folds(folds, observed_mapping, families)

    null_ba = np.empty(N_LOGO_PERMUTATIONS, dtype=np.float64)
    null_f1 = np.empty(N_LOGO_PERMUTATIONS, dtype=np.float64)
    for b in range(N_LOGO_PERMUTATIONS):
        permuted = rng.permutation(labels)
        perm_mapping = dict(zip(generator_ids, permuted))
        _, perm_report = evaluate_logo_folds(folds, perm_mapping, families)
        null_ba[b] = perm_report["balanced_accuracy"]
        null_f1[b] = perm_report["macro_f1"]
        if (b + 1) % 200 == 0 or b + 1 == N_LOGO_PERMUTATIONS:
            print(f"[LOGO NULL] {b + 1}/{N_LOGO_PERMUTATIONS}")

    report.update({
        "classifier": "nearest_family_median_cosine",
        "validation": "leave_one_generator_out",
        "language_adjustment": "recomputed_inside_each_fold_without_test_generator",
        "balanced_accuracy_permutation_p": empirical_p_upper(report["balanced_accuracy"], null_ba),
        "macro_f1_permutation_p": empirical_p_upper(report["macro_f1"], null_f1),
        "balanced_accuracy_null_q95": float(np.quantile(null_ba, 0.95)),
        "macro_f1_null_q95": float(np.quantile(null_f1, 0.95)),
        "n_permutations": int(N_LOGO_PERMUTATIONS),
    })

    confusion = pd.DataFrame(
        report["confusion_matrix"],
        index=[f"true__{family}" for family in families],
        columns=[f"pred__{family}" for family in families],
    ).reset_index(names="true_family")
    diagnostic_frame = pd.DataFrame.from_records(diagnostics)
    return predictions, confusion, diagnostic_frame, report


















def evidence_decision(
    similarity_report: Mapping[str, Any],
    multivariate_report: Mapping[str, Any],
    logo_report: Mapping[str, Any],
    multiplicity_report: Mapping[str, Any],
) -> Dict[str, Any]:
    adjusted = multiplicity_report["adjusted_p_values"]
    similarity_positive = (
        similarity_report["delta_intra_minus_inter"] > 0
        and adjusted["family_similarity"] < ALPHA
    )
    centroid_positive = adjusted["multivariate_centroid"] < ALPHA
    dispersion_problem = adjusted["dispersion"] < ALPHA
    logo_positive = (
        adjusted["logo_balanced_accuracy"] < ALPHA
        and adjusted["logo_macro_f1"] < ALPHA
    )

    
    if similarity_positive and centroid_positive and logo_positive and not dispersion_problem:
        status = "SUPPORTED"
    else:
        status = "INSUFFICIENT_EVIDENCE"

    return {
        "status": status,
        "alpha": float(ALPHA),
        "multiplicity_method": multiplicity_report["method"],
        "decision_uses_adjusted_p_values": True,
        "similarity_positive": bool(similarity_positive),
        "multivariate_centroid_positive": bool(centroid_positive),
        "dispersion_difference_detected": bool(dispersion_problem),
        "logo_above_permutation_null": bool(logo_positive),
        "important_note": (
            "This decision concerns shared family-level spectral structure after "
            "same-language cross-fitted adjustment and Holm correction. It does not "
            "establish cross-language persistence, which requires a separate phase."
        ),
    }


def run_protocol(
    protocol: str,
    input_path: Path,
    expected_rows: int,
    residual_columns: Sequence[str],
    analysis_mask: np.ndarray,
    protocol_seed: int,
) -> Dict[str, Any]:
    protocol_start = time.time()
    rng_similarity = np.random.default_rng(derived_seed(protocol_seed, "similarity"))
    rng_multivariate = np.random.default_rng(derived_seed(protocol_seed, "multivariate"))
    rng_logo = np.random.default_rng(derived_seed(protocol_seed, "logo"))
    protocol_dir = OUTPUT_DIR / protocol
    protocol_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 100)
    print(f"PHASE 1B — PROTOCOL {protocol.upper()}")
    print("=" * 100)

    pair_frame, input_report = load_pair_level(
        input_path,
        protocol,
        expected_rows,
        residual_columns,
    )
    print(f"[LOADING] {len(pair_frame):,} pairs, {input_report['n_generators']} generators")

    cells = aggregate_generator_language(pair_frame, residual_columns)
    atomic_parquet_dump(cells, protocol_dir / "fingerprints_generator_language.parquet")
    print(f"[AGGREGATION] {len(cells):,} cellules generator-language")

    adjusted_cells, generators, adjustment_report = cross_fitted_language_adjustment(
        cells,
        residual_columns,
    )
    atomic_parquet_dump(
        adjusted_cells,
        protocol_dir / "fingerprints_generator_language_adjusted.parquet",
    )
    atomic_parquet_dump(
        generators,
        protocol_dir / "fingerprints_generator_level_adjusted.parquet",
    )
    print(
        f"[LANGUAGE ADJUSTMENT] {len(generators)} generators; "
        f"multilingues={adjustment_report['n_multilingual_generators']}"
    )

    pairwise = pairwise_similarity_table(generators, residual_columns, analysis_mask)
    atomic_csv_dump(pairwise, protocol_dir / "pairwise_generator_similarity.csv")
    similarity_report = generator_similarity_test(
        generators,
        residual_columns,
        analysis_mask,
        rng_similarity,
    )
    print(
        f"[SIMILARITY] Δ={similarity_report['delta_intra_minus_inter']:.4f}; "
        f"p={similarity_report['permutation_p_upper']:.6f}"
    )

    multivariate_report, dispersion_frame = multivariate_family_test(
        generators,
        residual_columns,
        analysis_mask,
        rng_multivariate,
    )
    atomic_csv_dump(dispersion_frame, protocol_dir / "family_dispersion_by_generator.csv")
    print(
        f"[MULTIVARIATE] pseudo-F={multivariate_report['pseudo_f']:.3f}; "
        f"R²={multivariate_report['r2_family']:.4f}; "
        f"p={multivariate_report['permutation_p_pseudo_f']:.6f}"
    )

    logo_predictions, logo_confusion, logo_diagnostics, logo_report = logo_analysis(
        cells,
        residual_columns,
        analysis_mask,
        rng_logo,
    )
    atomic_csv_dump(logo_predictions, protocol_dir / "logo_predictions.csv")
    atomic_csv_dump(logo_confusion, protocol_dir / "logo_confusion_matrix.csv")
    atomic_csv_dump(logo_diagnostics, protocol_dir / "logo_fold_diagnostics.csv")
    print(
        f"[LOGO] BA={logo_report['balanced_accuracy']:.4f}; "
        f"macro-F1={logo_report['macro_f1']:.4f}; "
        f"p_BA={logo_report['balanced_accuracy_permutation_p']:.6f}"
    )

    multiplicity_report, multiplicity_table = adjust_confirmatory_pvalues(
        similarity_report, multivariate_report, logo_report
    )
    atomic_csv_dump(multiplicity_table, protocol_dir / "confirmatory_multiplicity_holm.csv")
    similarity_report["permutation_p_holm"] = multiplicity_report["adjusted_p_values"]["family_similarity"]
    multivariate_report["permutation_p_pseudo_f_holm"] = multiplicity_report["adjusted_p_values"]["multivariate_centroid"]
    multivariate_report["dispersion_test"]["permutation_p_holm"] = multiplicity_report["adjusted_p_values"]["dispersion"]
    logo_report["balanced_accuracy_permutation_p_holm"] = multiplicity_report["adjusted_p_values"]["logo_balanced_accuracy"]
    logo_report["macro_f1_permutation_p_holm"] = multiplicity_report["adjusted_p_values"]["logo_macro_f1"]
    print("[HOLM]", multiplicity_report["adjusted_p_values"])

    decision = evidence_decision(
        similarity_report, multivariate_report, logo_report, multiplicity_report
    )
    elapsed = time.time() - protocol_start
    summary = {
        "version": VERSION,
        "protocol": protocol,
        "status": "COMPLETE",
        "elapsed_seconds": float(elapsed),
        "elapsed_minutes": float(elapsed / 60.0),
        "input_report": input_report,
        "aggregation": adjustment_report,
        "analysis_band_hz": [float(ANALYSIS_MIN_HZ), float(ANALYSIS_MAX_HZ)],
        "analysis_bins": int(analysis_mask.sum()),
        "similarity_test": similarity_report,
        "multivariate_family_test": multivariate_report,
        "logo": logo_report,
        "confirmatory_multiplicity": multiplicity_report,
        "evidence_decision": decision,
        "outputs": {
            "generator_language": str(protocol_dir / "fingerprints_generator_language.parquet"),
            "generator_language_adjusted": str(protocol_dir / "fingerprints_generator_language_adjusted.parquet"),
            "generator_level_adjusted": str(protocol_dir / "fingerprints_generator_level_adjusted.parquet"),
            "pairwise_similarity": str(protocol_dir / "pairwise_generator_similarity.csv"),
            "logo_predictions": str(protocol_dir / "logo_predictions.csv"),
            "logo_confusion": str(protocol_dir / "logo_confusion_matrix.csv"),
            "confirmatory_multiplicity": str(protocol_dir / "confirmatory_multiplicity_holm.csv"),
        },
    }
    atomic_json_dump(summary, protocol_dir / "phase1b_protocol_summary.json")

    del pair_frame, cells, adjusted_cells, generators, pairwise
    gc.collect()
    return summary




def main() -> int:
    start = time.time()
    print("=" * 100)
    print("H4 — MLAAD STRICT GLOBAL FAMILY-LEVEL ANALYSIS")
    print("Only the confirmatory STRICT population used for the final global H4 verdict is run here.")
    print("=" * 100)

    for required in [STRICT_INPUT, FREQUENCY_AXIS_INPUT]:
        if not required.is_file():
            raise FileNotFoundError(f"Required file not found: {required}")

    if FORCE_REBUILD and OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    configuration = config_payload()
    configuration_hash = stable_json_hash(configuration)
    run_metadata = {
        "version": VERSION,
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "config": configuration,
        "config_hash": configuration_hash,
        "strict_input": str(STRICT_INPUT),
        "strict_input_sha256": sha256_file(STRICT_INPUT),
        "frequency_axis_input": str(FREQUENCY_AXIS_INPUT),
        "frequency_axis_sha256": sha256_file(FREQUENCY_AXIS_INPUT),
        "scope": "MLAAD STRICT confirmatory global H4 only",
    }
    if PHASE1A_SUMMARY_INPUT.is_file():
        run_metadata["phase1a_summary"] = str(PHASE1A_SUMMARY_INPUT)
        run_metadata["phase1a_summary_sha256"] = sha256_file(PHASE1A_SUMMARY_INPUT)

    # Historical Phase1B metadata may also contain RELAXED-sensitivity fields.
    # For faithful reuse, compatibility is checked only on the raw inputs that
    # determine the final STRICT analysis; packaging/version fields are ignored.
    metadata_path = OUTPUT_DIR / "run_metadata.json"
    if metadata_path.is_file() and not FORCE_REBUILD:
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        critical = ["strict_input_sha256", "frequency_axis_sha256"]
        mismatch = {
            key: {"previous": previous.get(key), "current": run_metadata.get(key)}
            for key in critical
            if previous.get(key) not in {None, run_metadata.get(key)}
        }
        if mismatch:
            raise RuntimeError(f"Existing H4 STRICT inputs do not match current inputs: {mismatch}")
    atomic_json_dump(run_metadata, OUTPUT_DIR / "paper_faithful_run_metadata.json")
    atomic_json_dump(environment_report(), OUTPUT_DIR / "paper_faithful_environment.json")

    frequency_axis = read_frequency_axis()
    residual_columns = frequency_axis["column_name"].astype(str).tolist()
    analysis_mask = (
        (frequency_axis["frequency_hz"].to_numpy(dtype=float) >= ANALYSIS_MIN_HZ)
        & (frequency_axis["frequency_hz"].to_numpy(dtype=float) <= ANALYSIS_MAX_HZ)
    )
    if int(analysis_mask.sum()) != 481:
        raise RuntimeError(f"Expected 481 final inferential bins, found {int(analysis_mask.sum())}")

    strict_summary_path = OUTPUT_DIR / "strict" / "phase1b_protocol_summary.json"
    if strict_summary_path.is_file() and not FORCE_REBUILD:
        strict_summary = json.loads(strict_summary_path.read_text(encoding="utf-8"))
        if strict_summary.get("status") != "COMPLETE":
            raise RuntimeError("Existing STRICT summary is incomplete")
        if str(strict_summary.get("protocol", "strict")).lower() != "strict":
            raise RuntimeError("Existing H4 summary is not the STRICT protocol")
        if int(strict_summary.get("analysis_bins", -1)) != 481:
            raise RuntimeError("Existing H4 STRICT summary does not use the final 481-bin band")
        print("[RESUME] Existing final STRICT result reused after input/band compatibility checks.")
    else:
        strict_summary = run_protocol(
            "strict",
            STRICT_INPUT,
            EXPECTED_STRICT_ROWS,
            residual_columns,
            analysis_mask,
            protocol_seed=RANDOM_SEED,
        )

    completion = {
        "version": VERSION,
        "status": "COMPLETE",
        "completed_utc": pd.Timestamp.utcnow().isoformat(),
        "elapsed_seconds": float(time.time() - start),
        "strict_summary": str(strict_summary_path),
        "confirmatory_decision": strict_summary["evidence_decision"]["status"],
        "scope": "final manuscript global H4",
    }
    atomic_json_dump(completion, OUTPUT_DIR / ".H4_PAPER_FAITHFUL_COMPLETE.json")

    print("\n" + "=" * 100)
    print("H4 STRICT COMPLETE")
    print(f"Confirmatory decision: {strict_summary['evidence_decision']['status']}")
    print(f"Outputs: {OUTPUT_DIR}")
    print(f"Elapsed: {(time.time() - start) / 60.0:.1f} min")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n" + "=" * 100, file=sys.stderr)
        print("[FAILURE PHASE 1B]", file=sys.stderr)
        traceback.print_exc()
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            atomic_json_dump(
                {
                    "version": VERSION,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "traceback": traceback.format_exc(),
                    "failed_utc": pd.Timestamp.utcnow().isoformat(),
                },
                OUTPUT_DIR / "fatal_error.json",
            )
        except Exception:
            pass
        raise
