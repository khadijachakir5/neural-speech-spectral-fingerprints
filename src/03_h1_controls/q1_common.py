#!/usr/bin/env python3


# Purpose: Provide shared statistical primitives for correlations, multiplicity correction, exact tests, aggregation, and cross-fitted language adjustment.

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def stable_int_seed(*parts: Any, modulo: int = 2**32 - 1) -> int:
    text = "||".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(text).hexdigest()[:16], 16) % modulo


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def stable_json_hash(payload: Any) -> str:
    raw = json.dumps(
        payload, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def atomic_csv_dump(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def atomic_parquet_dump(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False, compression="zstd", engine="pyarrow")
    os.replace(tmp, path)


def environment_report(version: str) -> dict[str, Any]:
    
    distributions = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
        "pyarrow": "pyarrow",
        "matplotlib": "matplotlib",
    }
    packages: dict[str, str] = {}
    for label, distribution in distributions.items():
        try:
            packages[label] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            packages[label] = "UNAVAILABLE"
    return {
        "version": version,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


def residual_columns_from_schema(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    names = pq.ParquetFile(path).schema.names
    columns = [name for name in names if name.startswith("res_")]
    columns.sort(key=lambda name: int(name.split("_")[1]))
    if not columns:
        raise RuntimeError(f"No residual column in {path}")
    return columns


def read_frequency_axis(path: Path, expected_columns: Sequence[str]) -> pd.DataFrame:
    axis = pd.read_csv(path)
    required = {"column_name", "frequency_hz"}
    missing = required - set(axis.columns)
    if missing:
        raise RuntimeError(f"Axis frequency incomplete: {sorted(missing)}")
    axis = axis.copy()
    if "bin_index" in axis.columns:
        axis = axis.sort_values("bin_index")
    else:
        axis = axis.set_index("column_name").loc[list(expected_columns)].reset_index()
    if axis["column_name"].astype(str).tolist() != list(expected_columns):
        mapping = axis.set_index("column_name")
        missing = [column for column in expected_columns if column not in mapping.index]
        if missing:
            raise RuntimeError(f"Bins missing de l'axis: {missing[:10]}")
        axis = mapping.loc[list(expected_columns)].reset_index()
    return axis.reset_index(drop=True)


def frequency_mask(axis: pd.DataFrame, low_hz: float = 80.0, high_hz: float = 7600.0) -> np.ndarray:
    hz = axis["frequency_hz"].to_numpy(dtype=float)
    mask = (hz >= low_hz) & (hz <= high_hz)
    if int(mask.sum()) < 2:
        raise RuntimeError("Empty analysis band")
    return mask


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ac = a - np.mean(a)
    bc = b - np.mean(b)
    den = float(np.linalg.norm(ac) * np.linalg.norm(bc))
    if den <= 1e-15:
        return 0.0
    return float(np.dot(ac, bc) / den)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 1e-15:
        return 0.0
    return float(np.dot(a, b) / den)


def correlation_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    centered = x - x.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms = np.where(norms > 1e-15, norms, 1.0)
    return np.clip((centered / norms) @ (centered / norms).T, -1.0, 1.0)


def cosine_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms > 1e-15, norms, 1.0)
    z = x / norms
    return np.clip(z @ z.T, -1.0, 1.0)


def percentile_ci(values: Iterable[float], confidence: float = 0.95) -> list[float]:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return [float("nan"), float("nan")]
    alpha = 1.0 - confidence
    return [float(np.quantile(x, alpha / 2)), float(np.quantile(x, 1 - alpha / 2))]


def empirical_p_upper(observed: float, null: Sequence[float]) -> float:
    x = np.asarray(null, dtype=float)
    x = x[np.isfinite(x)]
    if not np.isfinite(observed) or not len(x):
        return float("nan")
    return float((1 + np.sum(x >= observed)) / (len(x) + 1))


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    pv = p[valid]
    if not len(pv):
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = (len(ranked) - np.arange(len(ranked))) * ranked
    adjusted = np.maximum.accumulate(adjusted)
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    out[valid] = restored
    return out


def bh_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    pv = p[valid]
    if not len(pv):
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    out[valid] = restored
    return out


def aggregate_generator_language(
    pair_frame: pd.DataFrame,
    residual_columns: Sequence[str],
) -> pd.DataFrame:
    keys = ["independent_generator_id", "waveform_family", "language"]
    grouped = pair_frame.groupby(keys, observed=True, sort=True)
    medians = grouped[list(residual_columns)].median().reset_index()
    counts = grouped.size().rename("n_pairs").reset_index()
    cells = medians.merge(counts, on=keys, how="left", validate="one_to_one")
    return cells.sort_values(keys).reset_index(drop=True)


def language_adjust_cell_vectors(
    metadata: pd.DataFrame,
    vectors: np.ndarray,
    minimum_reference_generators: int = 2,
) -> tuple[np.ndarray, pd.DataFrame]:


    generators = metadata["independent_generator_id"].astype(str).to_numpy()
    languages = metadata["language"].astype(str).to_numpy()
    vectors = np.asarray(vectors, dtype=np.float64)
    adjusted = np.empty_like(vectors)
    diagnostic_rows: list[dict[str, Any]] = []

    for i, (generator, language) in enumerate(zip(generators, languages)):
        same_language = (languages == language) & (generators != generator)
        n_ref = len(np.unique(generators[same_language]))
        if n_ref >= minimum_reference_generators:
            reference = np.median(vectors[same_language], axis=0)
            source = "language"
        else:
            fallback = generators != generator
            if not fallback.any():
                raise RuntimeError("Global reference unavailable")
            reference = np.median(vectors[fallback], axis=0)
            source = "global_fallback"
            n_ref = len(np.unique(generators[fallback]))
        adjusted[i] = vectors[i] - reference
        diagnostic_rows.append({
            "row_index": i,
            "independent_generator_id": generator,
            "language": language,
            "reference_source": source,
            "n_reference_generators": int(n_ref),
        })
    return adjusted, pd.DataFrame(diagnostic_rows)


def aggregate_adjusted_cells_to_generators(
    metadata: pd.DataFrame,
    adjusted_vectors: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    metadata = metadata.reset_index(drop=True).copy()
    adjusted_vectors = np.asarray(adjusted_vectors, dtype=np.float64)
    records: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    for generator, idx in metadata.groupby("independent_generator_id", sort=True).groups.items():
        indices = np.asarray(list(idx), dtype=int)
        families = metadata.loc[indices, "waveform_family"].astype(str).unique()
        if len(families) != 1:
            raise RuntimeError(f"Ambiguous family for {generator}")
        languages = sorted(metadata.loc[indices, "language"].astype(str).unique())
        records.append({
            "independent_generator_id": str(generator),
            "waveform_family": str(families[0]),
            "n_languages": int(len(languages)),
            "languages": "|".join(languages),
        })
        vectors.append(np.median(adjusted_vectors[indices], axis=0))
    return pd.DataFrame(records), np.vstack(vectors)


def validate_unique_mapping(frame: pd.DataFrame, key: str, value: str) -> None:
    counts = frame[[key, value]].drop_duplicates().groupby(key)[value].nunique()
    if not counts.eq(1).all():
        bad = counts[counts.ne(1)].index.astype(str).tolist()
        raise RuntimeError(f"Ambiguous mapping {key}->{value}: {bad[:10]}")
