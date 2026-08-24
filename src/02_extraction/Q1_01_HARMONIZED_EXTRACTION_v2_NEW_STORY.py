#!/usr/bin/env python3


# Purpose: Extract the harmonized residual spectral representation for WaveFake and LibriSeVoc using the same frozen signal-processing definition as MLAAD.

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "soundfile": "soundfile",
    "pyarrow": "pyarrow",
}

missing_packages = [
    pip_name
    for module_name, pip_name in REQUIRED_PACKAGES.items()
    if importlib.util.find_spec(module_name) is None
]
if missing_packages:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *missing_packages]
    )

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from scipy.fft import rfft
from scipy.signal import resample_poly
from scipy.signal.windows import hann

try:
    from google.colab import drive

    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive", force_remount=False)
    else:
        print("[DRIVE] Google Drive is already mounted.")
except Exception:
    
    pass


VERSION = "Q1-HARMONIZED-SPECTRAL-EXTRACTION-v2.0.0-NEW-STORY"
DEFAULT_MASTER_SEED = 20260711
DEFAULT_BOOTSTRAP_SEEDS = (20260711, 20260729, 20260817)

DEFAULT_OUTPUT_ROOT = Path(
    "/content/drive/MyDrive/fingerprint_q1_outputs/q1_harmonized/v3_new_story"
)

MLAAD_REFERENCE = Path(
    "/content/drive/MyDrive/fingerprint_q1_outputs/phase1a/"
    "phase1a_mlaad_spectral_residuals_v2_new_story/"
    "fingerprints_pair_level_strict.parquet"
)


@dataclass(frozen=True)
class SpectralConfig:


    target_sr: int = 16_000
    n_fft: int = 1_024
    hop_length: int = 256
    remove_dc: bool = True
    epsilon_power: float = 1e-12
    vad_reference_percentile: float = 95.0
    vad_top_db: float = 40.0
    vad_abs_db: float = -80.0
    min_active_frames: int = 3
    centering_min_hz: float = 80.0
    centering_max_hz: float = 7_600.0

    def validate(self) -> None:
        if self.target_sr <= 0:
            raise ValueError("target_sr must be positive")
        if self.n_fft <= 0 or self.n_fft % 2:
            raise ValueError("n_fft must be a strictly positive even integer")
        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if self.hop_length > self.n_fft:
            raise ValueError("hop_length must not exceed n_fft")
        if not 0.0 <= self.vad_reference_percentile <= 100.0:
            raise ValueError("vad_reference_percentile must be in [0, 100]")
        if self.min_active_frames < 1:
            raise ValueError("min_active_frames must be >= 1")
        if self.centering_min_hz < 0:
            raise ValueError("centering_min_hz must be >= 0")
        nyquist = self.target_sr / 2.0
        if self.centering_max_hz > nyquist:
            raise ValueError("centering_max_hz exceeds Nyquist")
        if self.centering_min_hz >= self.centering_max_hz:
            raise ValueError("invalid centering band")


@dataclass(frozen=True)
class StabilityConfig:
    bootstrap_seeds: Tuple[int, ...] = DEFAULT_BOOTSTRAP_SEEDS
    bootstrap_repeats_full: int = 500
    bootstrap_repeats_quick: int = 50
    split_repeats_full: int = 200
    split_repeats_quick: int = 30
    content_blocks: int = 50
    confidence_level: float = 0.95

    def validate(self) -> None:
        if not self.bootstrap_seeds:
            raise ValueError("At least one bootstrap seed is required")
        if len(set(self.bootstrap_seeds)) != len(self.bootstrap_seeds):
            raise ValueError("Bootstrap seeds must be unique")
        if self.bootstrap_repeats_full < 1 or self.bootstrap_repeats_quick < 1:
            raise ValueError("The number of bootstrap replicates must be >= 1")
        if self.split_repeats_full < 1 or self.split_repeats_quick < 1:
            raise ValueError("The number of splits must be >= 1")
        if self.content_blocks < 4:
            raise ValueError("content_blocks must be >= 4")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    manifest: str
    expected_dataset: str
    expected_domain: Optional[str]
    expected_manifest_sha256: Optional[str]
    expected_pairs: Optional[int]
    expected_originals: Optional[int]
    expected_generators: Tuple[str, ...]
    quick_originals: int


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "wavefake_ljspeech": DatasetSpec(
        key="wavefake_ljspeech",
        manifest=(
            "/content/drive/MyDrive/fingerprint_q1_outputs/"
            "phase0_wavefake_final_v2/"
            "wavefake_ljspeech_manifest_confirmatory.parquet"
        ),
        expected_dataset="wavefake",
        expected_domain="ljspeech",
        expected_manifest_sha256=None,
        expected_pairs=91_700,
        expected_originals=13_100,
        expected_generators=(
            "FBMelGAN_LJ",
            "HiFiGAN_LJ",
            "MBMelGAN_LJ",
            "MelGANLarge_LJ",
            "MelGAN_LJ",
            "PWG_LJ",
            "WaveGlow_LJ",
        ),
        quick_originals=100,
    ),
    "wavefake_jsut": DatasetSpec(
        key="wavefake_jsut",
        manifest=(
            "/content/drive/MyDrive/fingerprint_q1_outputs/"
            "phase0_wavefake_final_v2/"
            "wavefake_jsut_manifest_confirmatory.parquet"
        ),
        expected_dataset="wavefake",
        expected_domain="jsut",
        expected_manifest_sha256=None,
        expected_pairs=10_000,
        expected_originals=5_000,
        expected_generators=("MBMelGAN_JSUT", "PWG_JSUT"),
        quick_originals=100,
    ),
    "librisevoc": DatasetSpec(
        key="librisevoc",
        manifest=(
            "/content/drive/MyDrive/fingerprint_q1_outputs/"
            "phase0_librisevoc_final_v2/librisevoc_manifest_confirmatory_balanced.parquet"
        ),
        expected_dataset="librisevoc",
        expected_domain=None,
        
        
        expected_manifest_sha256=None,
        expected_pairs=72_174,
        expected_originals=12_029,
        expected_generators=(
            "WaveNet_LS",
            "WaveRNN_LS",
            "MelGAN_LS",
            "PWG_LS",
            "WaveGrad_LS",
            "DiffWave_LS",
        ),
        quick_originals=100,
    ),
}

REQUIRED_MANIFEST_COLUMNS = {
    "pair_id",
    "dataset",
    "independent_generator_id",
    "waveform_architecture",
    "waveform_family",
    "language",
    "original_id",
    "fake_path",
    "real_path",
    "qc_status",
}

OPTIONAL_METADATA_COLUMNS = [
    "domain",
    "generator_key",
    "pipeline_type",
    "acoustic_model",
    "representation",
    "taxonomy_confidence",
    "speaker_id",
    "fake_sha256",
    "real_sha256",
    "fake_duration",
    "real_duration",
    "duration_ratio",
    "active_duration_fake",
    "active_duration_real",
    "silence_ratio_fake",
    "silence_ratio_real",
    "clipping_rate_fake",
    "clipping_rate_real",
]

STATUS_PENDING = np.uint8(0)
STATUS_OK = np.uint8(1)
STATUS_FAIL = np.uint8(2)

_WORKER_CONFIG: Optional[SpectralConfig] = None
_FREQUENCIES_HZ: Optional[np.ndarray] = None
_WINDOW: Optional[np.ndarray] = None


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(payload: Any) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def stable_int_hash(text: str, modulo: int) -> int:
    if modulo <= 0:
        raise ValueError("modulo must be positive")
    digest = hashlib.sha256(str(text).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % modulo


def selection_sha256(frame: pd.DataFrame) -> str:
    columns = [
        "pair_id",
        "original_id",
        "independent_generator_id",
        "fake_path",
        "real_path",
    ]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Columns missing for the selection hash : {missing}")
    stable = (
        frame[columns]
        .astype(str)
        .sort_values(columns, kind="mergesort")
        .reset_index(drop=True)
    )
    hashed = pd.util.hash_pandas_object(stable, index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            payload,
            stream,
            indent=2,
            ensure_ascii=False,
            default=str,
            allow_nan=False,
        )
    os.replace(temporary, path)


def atomic_parquet_dump(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(
        temporary,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    os.replace(temporary, path)


def atomic_csv_dump(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(payload), ensure_ascii=False, default=str) + "\n")


def read_jsonl_deduplicated(path: Path, key: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and key in value:
                rows.append(value)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=[key], keep="last")


def environment_report() -> Dict[str, Any]:
    versions: Dict[str, str] = {}
    for name in ["numpy", "pandas", "scipy", "soundfile", "pyarrow"]:
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:
            versions[name] = f"ERROR:{type(exc).__name__}:{exc}"
    return {
        "version": VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
    }


def parse_path_rewrites(values: Sequence[str]) -> List[Tuple[str, str]]:
    rewrites: List[Tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid rewrite {value!r}. Format expected : OLD=NEW"
            )
        old, new = value.split("=", 1)
        if not old:
            raise ValueError("The OLD root of a path rewrite cannot be empty")
        rewrites.append((old, new))
    return rewrites


def rewrite_path(path_text: str, rewrites: Sequence[Tuple[str, str]]) -> str:
    result = str(path_text)
    for old, new in rewrites:
        if result.startswith(old):
            return new + result[len(old) :]
    return result


def iter_slices(length: int, size: int) -> Iterator[Tuple[int, int]]:
    if size < 1:
        raise ValueError("Chunk size must be >= 1")
    for start in range(0, length, size):
        yield start, min(start + size, length)


def percentile_ci(values: Sequence[float], confidence: float) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan")
    alpha = 1.0 - confidence
    low, high = np.quantile(array, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def pearson_correlation(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator <= 1e-15:
        return 0.0
    return float(np.dot(a, b) / denominator)


def create_or_open_memmap(
    path: Path,
    shape: Tuple[int, ...],
    dtype: Any,
    fill_value: Optional[Any] = None,
) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_dtype = np.dtype(dtype)
    if path.exists():
        array = np.load(path, mmap_mode="r+")
        if tuple(array.shape) != tuple(shape):
            raise RuntimeError(
                f"Shape incompatible for {path}: {array.shape} != {shape}"
            )
        if np.dtype(array.dtype) != expected_dtype:
            raise RuntimeError(
                f"Dtype incompatible for {path}: {array.dtype} != {expected_dtype}"
            )
        return array
    array = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=expected_dtype,
        shape=shape,
    )
    if fill_value is not None:
        array[...] = fill_value
        array.flush()
    return array


def clean_hash_series(series: pd.Series) -> pd.Series:
    output = series.astype(str).str.strip().str.lower()
    return output.where(~output.isin({"", "nan", "none", "null", "<na>"}))


def audit_audio_hashes(
    frame: pd.DataFrame,
    output_dir: Path,
    strict: bool,
) -> Dict[str, Any]:
    controls_dir = output_dir / "controls"
    controls_dir.mkdir(parents=True, exist_ok=True)
    required = {"real_sha256", "fake_sha256"}
    if not required.issubset(frame.columns):
        report = {
            "status": "NOT_AVAILABLE",
            "strict": bool(strict),
            "missing_columns": sorted(required - set(frame.columns)),
        }
        atomic_csv_dump(
            pd.DataFrame(columns=["collision_type", "sha256", "details"]),
            controls_dir / "audio_hash_duplicate_audit.csv",
        )
        atomic_json_dump(report, controls_dir / "audio_hash_duplicate_audit.json")
        return report

    work = frame.copy()
    work["real_sha256_clean"] = clean_hash_series(work["real_sha256"])
    work["fake_sha256_clean"] = clean_hash_series(work["fake_sha256"])
    details: List[Dict[str, Any]] = []

    real_unique = (
        work[["original_id", "real_sha256_clean"]]
        .dropna()
        .drop_duplicates()
    )
    for sha, group in real_unique.groupby("real_sha256_clean", sort=True):
        if group["original_id"].nunique() > 1:
            details.append(
                {
                    "collision_type": "real_hash_multiple_original_ids",
                    "sha256": str(sha),
                    "n_original_ids": int(group["original_id"].nunique()),
                    "details": "|".join(
                        sorted(group["original_id"].astype(str).unique())
                    ),
                }
            )

    fake_unique = (
        work[
            [
                "pair_id",
                "original_id",
                "independent_generator_id",
                "fake_sha256_clean",
            ]
        ]
        .dropna()
        .drop_duplicates()
    )
    for sha, group in fake_unique.groupby("fake_sha256_clean", sort=True):
        if (
            group["original_id"].nunique() > 1
            or group["independent_generator_id"].nunique() > 1
        ):
            details.append(
                {
                    "collision_type": "fake_hash_cross_content_or_generator",
                    "sha256": str(sha),
                    "n_original_ids": int(group["original_id"].nunique()),
                    "n_generators": int(
                        group["independent_generator_id"].nunique()
                    ),
                    "details": (
                        "originals="
                        + "|".join(sorted(group["original_id"].astype(str).unique()))
                        + ";generators="
                        + "|".join(
                            sorted(
                                group["independent_generator_id"]
                                .astype(str)
                                .unique()
                            )
                        )
                    ),
                }
            )

    real_hashes = set(real_unique["real_sha256_clean"].dropna().astype(str))
    fake_hashes = set(fake_unique["fake_sha256_clean"].dropna().astype(str))
    for sha in sorted(real_hashes.intersection(fake_hashes)):
        details.append(
            {
                "collision_type": "fake_hash_equals_real_hash",
                "sha256": sha,
                "details": "exact waveform hash appears in fake and real sets",
            }
        )

    detail_frame = pd.DataFrame(details)
    if detail_frame.empty:
        detail_frame = pd.DataFrame(
            columns=[
                "collision_type",
                "sha256",
                "n_original_ids",
                "n_generators",
                "details",
            ]
        )
    atomic_csv_dump(
        detail_frame,
        controls_dir / "audio_hash_duplicate_audit.csv",
    )
    report = {
        "status": "PASS" if detail_frame.empty else "COLLISIONS_DETECTED",
        "strict": bool(strict),
        "n_collision_records": int(len(detail_frame)),
        "collision_type_counts": {
            str(key): int(value)
            for key, value in detail_frame["collision_type"].value_counts().items()
        }
        if len(detail_frame)
        else {},
    }
    atomic_json_dump(report, controls_dir / "audio_hash_duplicate_audit.json")
    if strict and len(detail_frame):
        raise RuntimeError(
            "The SHA-256 audit detected collisions that may create "
            f"fuite. Consulter {controls_dir / 'audio_hash_duplicate_audit.csv'}"
        )
    return report


def normalize_dataset_name(value: str) -> str:
    return str(value).strip().lower().replace("_", "").replace("-", "")


def prepare_manifest(
    spec: DatasetSpec,
    mode: str,
    seed: int,
    rewrites: Sequence[Tuple[str, str]],
    strict_hash_audit: bool,
    output_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    manifest_path = Path(spec.manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found : {manifest_path}")

    observed_manifest_hash = sha256_file(manifest_path)
    if (
        spec.expected_manifest_sha256
        and observed_manifest_hash.lower()
        != spec.expected_manifest_sha256.strip().lower()
    ):
        raise RuntimeError(
            "The manifest does not match the expected frozen SHA-256.\n"
            f"Observed : {observed_manifest_hash}\n"
            f"Expected: {spec.expected_manifest_sha256}"
        )

    frame = pd.read_parquet(manifest_path)
    missing = sorted(REQUIRED_MANIFEST_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Required columns missing: {missing}")
    if frame.empty:
        raise RuntimeError("The manifest is empty")
    if frame["pair_id"].astype(str).duplicated().any():
        raise RuntimeError("Duplicate pair_id values in the manifest")

    observed_dataset_values = {
        normalize_dataset_name(value) for value in frame["dataset"].astype(str).unique()
    }
    expected_dataset = normalize_dataset_name(spec.expected_dataset)
    if expected_dataset == "wavefake":
        dataset_ok = all(value.startswith("wavefake") for value in observed_dataset_values)
    else:
        dataset_ok = observed_dataset_values == {expected_dataset}
    if not dataset_ok:
        raise RuntimeError(
            f"Unexpected dataset: {sorted(observed_dataset_values)}; "
            f"expected={expected_dataset}"
        )

    if spec.expected_domain is not None:
        if "domain" not in frame.columns:
            raise ValueError("The domain column is required for WaveFake")
        observed_domains = set(frame["domain"].astype(str).str.strip().str.lower())
        if observed_domains != {spec.expected_domain.lower()}:
            raise RuntimeError(
                f"Unexpected domain: {sorted(observed_domains)}; "
                f"expected={spec.expected_domain}"
            )

    frame = frame.loc[
        frame["qc_status"].astype(str).str.strip().str.lower().eq("ok")
    ].copy()
    if frame.empty:
        raise RuntimeError("No pair has qc_status='ok'")

    for column in [
        "pair_id",
        "original_id",
        "independent_generator_id",
        "language",
        "fake_path",
        "real_path",
        "waveform_architecture",
        "waveform_family",
    ]:
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise RuntimeError(f"Empty values in {column}")

    for column in ["fake_path", "real_path"]:
        frame[column] = frame[column].map(lambda x: rewrite_path(x, rewrites))

    observed_generators = set(frame["independent_generator_id"].astype(str))
    expected_generators = set(spec.expected_generators)
    if observed_generators != expected_generators:
        raise RuntimeError(
            "Unexpected generator population.\n"
            f"Observed : {sorted(observed_generators)}\n"
            f"Expected: {sorted(expected_generators)}"
        )

    duplicate = int(
        frame.duplicated(["independent_generator_id", "original_id"]).sum()
    )
    if duplicate:
        raise RuntimeError(
            f"{duplicate} duplicate(s) (independent_generator_id, original_id)"
        )

    
    hash_report = audit_audio_hashes(
        frame,
        output_dir,
        strict=strict_hash_audit,
    )

    n_generators = len(expected_generators)
    coverage = frame.groupby("original_id")["independent_generator_id"].nunique()
    complete_ids = np.asarray(
        sorted(coverage[coverage == n_generators].index.astype(str)),
        dtype=str,
    )
    if len(complete_ids) == 0:
        raise RuntimeError("No original_id is shared by all generators")

    if mode == "quick":
        take = min(int(spec.quick_originals), len(complete_ids))
        rng = np.random.default_rng(seed)
        complete_ids = np.sort(rng.choice(complete_ids, size=take, replace=False))

    selected = frame.loc[frame["original_id"].astype(str).isin(complete_ids)].copy()
    selected = selected.sort_values(
        ["original_id", "independent_generator_id", "pair_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    counts = selected["independent_generator_id"].value_counts().sort_index()
    if counts.nunique() != 1:
        raise RuntimeError("The selection is not balanced by generator")
    expected_rows_from_intersection = len(complete_ids) * n_generators
    if len(selected) != expected_rows_from_intersection:
        raise RuntimeError(
            f"Incomplete selection : {len(selected)} rows, "
            f"{expected_rows_from_intersection} expected"
        )

    if mode == "full":
        if spec.expected_originals is not None and len(complete_ids) != spec.expected_originals:
            raise RuntimeError(
                f"{spec.key}: {len(complete_ids):,} complete originals; "
                f"{spec.expected_originals:,} expected"
            )
        if spec.expected_pairs is not None and len(selected) != spec.expected_pairs:
            raise RuntimeError(
                f"{spec.key}: {len(selected):,} pairs; "
                f"{spec.expected_pairs:,} expected"
            )

    selected.insert(0, "extraction_row", np.arange(len(selected), dtype=np.int64))

    
    real_conflicts = (
        selected.groupby("original_id")["real_path"].nunique(dropna=False)
    )
    if (real_conflicts > 1).any():
        raise RuntimeError(
            f"{int((real_conflicts > 1).sum())} original_id values map to multiple real_path values"
        )
    if "real_sha256" in selected.columns:
        real_hash_conflicts = (
            selected.groupby("original_id")["real_sha256"].nunique(dropna=False)
        )
        if (real_hash_conflicts > 1).any():
            raise RuntimeError(
                f"{int((real_hash_conflicts > 1).sum())} original_id values map to multiple real_sha256 values"
            )

    report = {
        "dataset_key": spec.key,
        "mode": mode,
        "manifest": str(manifest_path),
        "manifest_sha256": observed_manifest_hash,
        "selection_sha256": selection_sha256(selected),
        "n_qc_ok_before_intersection": int(len(frame)),
        "n_originals_complete": int(len(complete_ids)),
        "n_pairs_selected": int(len(selected)),
        "n_generators": int(n_generators),
        "generators": sorted(expected_generators),
        "pairs_per_generator": {str(k): int(v) for k, v in counts.items()},
        "n_languages": int(selected["language"].nunique()),
        "languages": sorted(selected["language"].astype(str).unique()),
        "hash_audit": hash_report,
    }
    return selected, report


def validate_paths_sample(frame: pd.DataFrame, n: int, seed: int) -> None:
    sample_n = min(int(n), len(frame))
    sample = frame.sample(n=sample_n, random_state=seed, replace=False)
    missing: List[str] = []
    for row in sample.itertuples(index=False):
        if not Path(str(row.fake_path)).is_file():
            missing.append(str(row.fake_path))
        if not Path(str(row.real_path)).is_file():
            missing.append(str(row.real_path))
    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(
            f"{len(missing)} missing path(s) in the validation sample.\n"
            f"{preview}"
        )


def build_real_index(pair_frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["original_id", "real_path"]
    for optional in ["real_sha256", "language", "speaker_id"]:
        if optional in pair_frame.columns:
            columns.append(optional)
    real_frame = (
        pair_frame[columns]
        .drop_duplicates(subset=["original_id"], keep="first")
        .sort_values("original_id", kind="mergesort")
        .reset_index(drop=True)
    )
    real_frame.insert(0, "real_index", np.arange(len(real_frame), dtype=np.int64))
    mapping = dict(
        zip(
            real_frame["original_id"].astype(str),
            real_frame["real_index"].astype(int),
        )
    )
    output = pair_frame.copy()
    output["real_index"] = output["original_id"].astype(str).map(mapping)
    if output["real_index"].isna().any():
        raise RuntimeError("real_index unresolved")
    output["real_index"] = output["real_index"].astype(np.int64)
    return output, real_frame


def _init_worker(config_dict: Mapping[str, Any]) -> None:
    global _WORKER_CONFIG, _FREQUENCIES_HZ, _WINDOW
    _WORKER_CONFIG = SpectralConfig(**dict(config_dict))
    _WORKER_CONFIG.validate()
    _FREQUENCIES_HZ = np.fft.rfftfreq(
        _WORKER_CONFIG.n_fft,
        d=1.0 / _WORKER_CONFIG.target_sr,
    ).astype(np.float32)
    _WINDOW = hann(_WORKER_CONFIG.n_fft, sym=False).astype(np.float32)


def load_audio_mono_resampled(
    path_text: str,
    config: SpectralConfig,
) -> Tuple[np.ndarray, int]:
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"file_not_found:{path}")
    audio, sample_rate = sf.read(
        str(path),
        dtype="float32",
        always_2d=False,
    )
    if audio.ndim == 2:
        audio = audio.mean(axis=1, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"unsupported_audio_shape:{audio.shape}")
    if audio.size == 0:
        raise ValueError("empty_audio")
    if not np.isfinite(audio).all():
        raise ValueError("nan_or_inf_samples")
    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise ValueError(f"invalid_sample_rate:{sample_rate}")
    original_sample_rate = sample_rate
    if sample_rate != config.target_sr:
        divisor = math.gcd(sample_rate, config.target_sr)
        audio = resample_poly(
            audio,
            up=config.target_sr // divisor,
            down=sample_rate // divisor,
        ).astype(np.float32, copy=False)
    if config.remove_dc:
        audio = audio - np.float32(audio.mean(dtype=np.float64))
    return np.asarray(audio, dtype=np.float32), original_sample_rate


def frame_audio(audio: np.ndarray, config: SpectralConfig) -> np.ndarray:
    if audio.size < config.n_fft:
        audio = np.pad(audio, (0, config.n_fft - audio.size))
    frames = np.lib.stride_tricks.sliding_window_view(audio, config.n_fft)
    frames = frames[:: config.hop_length]
    if len(frames) == 0:
        frames = audio[: config.n_fft][None, :]
    return frames


def active_frame_mask(
    frames: np.ndarray,
    config: SpectralConfig,
) -> Tuple[np.ndarray, bool]:
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    frame_db = 20.0 * np.log10(rms + 1e-12)
    reference_db = float(
        np.percentile(frame_db, config.vad_reference_percentile)
    )
    threshold_db = max(reference_db - config.vad_top_db, config.vad_abs_db)
    active = frame_db >= threshold_db
    fallback_used = False
    if int(active.sum()) < config.min_active_frames:
        fallback_used = True
        number_to_keep = min(
            max(config.min_active_frames, 1),
            len(frames),
        )
        top_indices = np.argsort(rms)[-number_to_keep:]
        active = np.zeros(len(frames), dtype=bool)
        active[top_indices] = True
    return active, fallback_used


def extract_log_spectrum(
    path_text: str,
    config: SpectralConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    audio, original_sample_rate = load_audio_mono_resampled(path_text, config)
    duration_s = float(audio.size / config.target_sr)
    frames = frame_audio(audio, config)
    active, fallback_used = active_frame_mask(frames, config)
    active_frames = np.asarray(frames[active], dtype=np.float32)
    window = hann(config.n_fft, sym=False).astype(np.float32)
    windowed = active_frames * window[None, :]
    spectrum = rfft(windowed, n=config.n_fft, axis=1)
    power = (
        spectrum.real.astype(np.float32) ** 2
        + spectrum.imag.astype(np.float32) ** 2
    )
    power /= np.float32(np.sum(window.astype(np.float64) ** 2))
    log_power_db = 10.0 * np.log10(power + config.epsilon_power)
    robust_log_spectrum = np.median(log_power_db, axis=0).astype(np.float32)
    expected_bins = config.n_fft // 2 + 1
    if robust_log_spectrum.shape != (expected_bins,):
        raise RuntimeError(
            f"invalid_spectrum_shape:{robust_log_spectrum.shape}"
        )
    if not np.isfinite(robust_log_spectrum).all():
        raise ValueError("non_finite_log_spectrum")
    return robust_log_spectrum, {
        "active_frames": int(active.sum()),
        "total_frames": int(len(frames)),
        "vad_fallback_used": bool(fallback_used),
        "duration_s": duration_s,
        "original_sample_rate": int(original_sample_rate),
    }


def _spectrum_worker(task: Tuple[int, str]) -> Tuple[int, Optional[np.ndarray], Dict[str, Any]]:
    index, path_text = task
    if _WORKER_CONFIG is None:
        raise RuntimeError("Worker not initialized")
    try:
        spectrum, stats = extract_log_spectrum(path_text, _WORKER_CONFIG)
        return index, spectrum, {"status": "ok", "error": "", **stats}
    except Exception as exc:
        return index, None, {
            "status": "fail",
            "error": f"{type(exc).__name__}:{exc}",
            "active_frames": -1,
            "total_frames": -1,
            "vad_fallback_used": False,
            "duration_s": np.nan,
            "original_sample_rate": -1,
        }


def band_mean(
    values: np.ndarray,
    frequencies_hz: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> float:
    mask = (frequencies_hz >= low_hz) & (frequencies_hz < high_hz)
    if not mask.any():
        return float("nan")
    return float(np.mean(values[mask]))


def initialize_real_arrays(
    folder: Path,
    n_real: int,
    n_bins: int,
) -> Dict[str, np.memmap]:
    return {
        "spectrum": create_or_open_memmap(
            folder / "real_log_power_db.npy",
            (n_real, n_bins),
            np.float32,
            np.nan,
        ),
        "status": create_or_open_memmap(
            folder / "real_status.npy",
            (n_real,),
            np.uint8,
            STATUS_PENDING,
        ),
        "active_frames": create_or_open_memmap(
            folder / "real_active_frames.npy",
            (n_real,),
            np.int32,
            -1,
        ),
        "total_frames": create_or_open_memmap(
            folder / "real_total_frames.npy",
            (n_real,),
            np.int32,
            -1,
        ),
        "fallback": create_or_open_memmap(
            folder / "real_vad_fallback.npy",
            (n_real,),
            np.uint8,
            0,
        ),
        "duration": create_or_open_memmap(
            folder / "real_duration_s.npy",
            (n_real,),
            np.float32,
            np.nan,
        ),
        "original_sr": create_or_open_memmap(
            folder / "real_original_sample_rate.npy",
            (n_real,),
            np.int32,
            -1,
        ),
    }


def initialize_pair_arrays(
    folder: Path,
    n_pairs: int,
    n_bins: int,
) -> Dict[str, np.memmap]:
    return {
        "fake_spectrum": create_or_open_memmap(
            folder / "fake_log_power_db.npy",
            (n_pairs, n_bins),
            np.float32,
            np.nan,
        ),
        "residual": create_or_open_memmap(
            folder / "residual_centered_db.npy",
            (n_pairs, n_bins),
            np.float32,
            np.nan,
        ),
        "status": create_or_open_memmap(
            folder / "pair_status.npy",
            (n_pairs,),
            np.uint8,
            STATUS_PENDING,
        ),
        "active_frames": create_or_open_memmap(
            folder / "fake_active_frames.npy",
            (n_pairs,),
            np.int32,
            -1,
        ),
        "total_frames": create_or_open_memmap(
            folder / "fake_total_frames.npy",
            (n_pairs,),
            np.int32,
            -1,
        ),
        "fallback": create_or_open_memmap(
            folder / "fake_vad_fallback.npy",
            (n_pairs,),
            np.uint8,
            0,
        ),
        "duration": create_or_open_memmap(
            folder / "fake_duration_s.npy",
            (n_pairs,),
            np.float32,
            np.nan,
        ),
        "original_sr": create_or_open_memmap(
            folder / "fake_original_sample_rate.npy",
            (n_pairs,),
            np.int32,
            -1,
        ),
        "offset": create_or_open_memmap(
            folder / "broadband_offset_db.npy",
            (n_pairs,),
            np.float32,
            np.nan,
        ),
        "l2": create_or_open_memmap(
            folder / "residual_l2.npy",
            (n_pairs,),
            np.float32,
            np.nan,
        ),
        "lf": create_or_open_memmap(
            folder / "residual_lf_mean_db.npy",
            (n_pairs,),
            np.float32,
            np.nan,
        ),
        "mf": create_or_open_memmap(
            folder / "residual_mf_mean_db.npy",
            (n_pairs,),
            np.float32,
            np.nan,
        ),
        "hf": create_or_open_memmap(
            folder / "residual_hf_mean_db.npy",
            (n_pairs,),
            np.float32,
            np.nan,
        ),
    }


def flush_arrays(arrays: Mapping[str, np.memmap]) -> None:
    for array in arrays.values():
        array.flush()


def reset_failed_real(arrays: Mapping[str, np.memmap]) -> int:
    failed = np.asarray(arrays["status"]) == STATUS_FAIL
    count = int(failed.sum())
    if count:
        arrays["spectrum"][failed, :] = np.nan
        arrays["status"][failed] = STATUS_PENDING
        arrays["active_frames"][failed] = -1
        arrays["total_frames"][failed] = -1
        arrays["fallback"][failed] = 0
        arrays["duration"][failed] = np.nan
        arrays["original_sr"][failed] = -1
        flush_arrays(arrays)
    return count


def reset_failed_pairs(arrays: Mapping[str, np.memmap]) -> int:
    failed = np.asarray(arrays["status"]) == STATUS_FAIL
    count = int(failed.sum())
    if count:
        arrays["fake_spectrum"][failed, :] = np.nan
        arrays["residual"][failed, :] = np.nan
        arrays["status"][failed] = STATUS_PENDING
        for key in ["active_frames", "total_frames", "original_sr"]:
            arrays[key][failed] = -1
        arrays["fallback"][failed] = 0
        for key in ["duration", "offset", "l2", "lf", "mf", "hf"]:
            arrays[key][failed] = np.nan
        flush_arrays(arrays)
    return count


def run_real_extraction(
    real_frame: pd.DataFrame,
    output_dir: Path,
    config: SpectralConfig,
    workers: int,
    chunk_size: int,
    retry_failed: bool,
) -> Dict[str, np.memmap]:
    folder = output_dir / "spectra"
    folder.mkdir(parents=True, exist_ok=True)
    n_bins = config.n_fft // 2 + 1
    arrays = initialize_real_arrays(folder, len(real_frame), n_bins)
    errors_path = output_dir / "logs" / "real_errors.jsonl"

    if retry_failed:
        count = reset_failed_real(arrays)
        if count:
            print(f"[RETRY REAL] {count} row(s) reset(s) en pending")

    start_time = time.time()
    processed_session = 0
    total_pending_initial = int((np.asarray(arrays["status"]) == STATUS_PENDING).sum())

    with ProcessPoolExecutor(
        max_workers=max(1, int(workers)),
        initializer=_init_worker,
        initargs=(asdict(config),),
    ) as executor:
        for chunk_id, (start, end) in enumerate(
            iter_slices(len(real_frame), chunk_size),
            start=1,
        ):
            pending_indices = [
                index
                for index in range(start, end)
                if int(arrays["status"][index]) == int(STATUS_PENDING)
            ]
            if not pending_indices:
                continue
            tasks = [
                (index, str(real_frame.iloc[index]["real_path"]))
                for index in pending_indices
            ]
            for index, spectrum, stats in executor.map(
                _spectrum_worker,
                tasks,
                chunksize=1,
            ):
                if spectrum is None:
                    arrays["status"][index] = STATUS_FAIL
                    append_jsonl(
                        errors_path,
                        {
                            "real_index": int(index),
                            "original_id": str(real_frame.iloc[index]["original_id"]),
                            "path": str(real_frame.iloc[index]["real_path"]),
                            "error": stats["error"],
                        },
                    )
                else:
                    arrays["spectrum"][index, :] = spectrum
                    arrays["active_frames"][index] = int(stats["active_frames"])
                    arrays["total_frames"][index] = int(stats["total_frames"])
                    arrays["fallback"][index] = np.uint8(
                        bool(stats["vad_fallback_used"])
                    )
                    arrays["duration"][index] = np.float32(stats["duration_s"])
                    arrays["original_sr"][index] = int(stats["original_sample_rate"])
                    arrays["status"][index] = STATUS_OK
                processed_session += 1
            flush_arrays(arrays)
            elapsed = time.time() - start_time
            rate = processed_session / elapsed if elapsed > 0 else 0.0
            remaining = total_pending_initial - processed_session
            eta = remaining / rate if rate > 0 else float("inf")
            print(
                f"[REAL] chunk {chunk_id} — session={processed_session:,}/"
                f"{total_pending_initial:,} — {rate:.2f} audio/s — "
                f"ETA={eta/60.0:.1f} min"
            )
            gc.collect()

    return arrays


def run_pair_extraction(
    pair_frame: pd.DataFrame,
    real_arrays: Mapping[str, np.memmap],
    output_dir: Path,
    config: SpectralConfig,
    workers: int,
    chunk_size: int,
    retry_failed: bool,
) -> Dict[str, np.memmap]:
    folder = output_dir / "spectra"
    n_bins = config.n_fft // 2 + 1
    arrays = initialize_pair_arrays(folder, len(pair_frame), n_bins)
    errors_path = output_dir / "logs" / "pair_errors.jsonl"
    frequencies = np.fft.rfftfreq(
        config.n_fft,
        d=1.0 / config.target_sr,
    ).astype(np.float32)
    centering_mask = (
        (frequencies >= config.centering_min_hz)
        & (frequencies <= config.centering_max_hz)
    )
    if not centering_mask.any():
        raise RuntimeError("Empty centering band")

    if retry_failed:
        count = reset_failed_pairs(arrays)
        if count:
            print(f"[RETRY PAIRS] {count} row(s) reset(s) en pending")

    start_time = time.time()
    processed_session = 0
    total_pending_initial = int((np.asarray(arrays["status"]) == STATUS_PENDING).sum())

    with ProcessPoolExecutor(
        max_workers=max(1, int(workers)),
        initializer=_init_worker,
        initargs=(asdict(config),),
    ) as executor:
        for chunk_id, (start, end) in enumerate(
            iter_slices(len(pair_frame), chunk_size),
            start=1,
        ):
            pending_indices = [
                index
                for index in range(start, end)
                if int(arrays["status"][index]) == int(STATUS_PENDING)
            ]
            if not pending_indices:
                continue
            tasks = [
                (index, str(pair_frame.iloc[index]["fake_path"]))
                for index in pending_indices
            ]
            for index, fake_spectrum, stats in executor.map(
                _spectrum_worker,
                tasks,
                chunksize=1,
            ):
                real_index = int(pair_frame.iloc[index]["real_index"])
                if fake_spectrum is None:
                    arrays["status"][index] = STATUS_FAIL
                    append_jsonl(
                        errors_path,
                        {
                            "extraction_row": int(index),
                            "pair_id": str(pair_frame.iloc[index]["pair_id"]),
                            "path": str(pair_frame.iloc[index]["fake_path"]),
                            "error": stats["error"],
                        },
                    )
                elif int(real_arrays["status"][real_index]) != int(STATUS_OK):
                    arrays["status"][index] = STATUS_FAIL
                    append_jsonl(
                        errors_path,
                        {
                            "extraction_row": int(index),
                            "pair_id": str(pair_frame.iloc[index]["pair_id"]),
                            "path": str(pair_frame.iloc[index]["fake_path"]),
                            "error": "real_spectrum_unavailable",
                        },
                    )
                else:
                    real_spectrum = np.asarray(
                        real_arrays["spectrum"][real_index],
                        dtype=np.float32,
                    )
                    raw_residual = fake_spectrum - real_spectrum
                    offset = float(np.median(raw_residual[centering_mask]))
                    centered = (
                        raw_residual - np.float32(offset)
                    ).astype(np.float32)
                    if not np.isfinite(centered).all():
                        arrays["status"][index] = STATUS_FAIL
                        append_jsonl(
                            errors_path,
                            {
                                "extraction_row": int(index),
                                "pair_id": str(pair_frame.iloc[index]["pair_id"]),
                                "path": str(pair_frame.iloc[index]["fake_path"]),
                                "error": "non_finite_centered_residual",
                            },
                        )
                    else:
                        arrays["fake_spectrum"][index, :] = fake_spectrum
                        arrays["residual"][index, :] = centered
                        arrays["active_frames"][index] = int(stats["active_frames"])
                        arrays["total_frames"][index] = int(stats["total_frames"])
                        arrays["fallback"][index] = np.uint8(
                            bool(stats["vad_fallback_used"])
                        )
                        arrays["duration"][index] = np.float32(stats["duration_s"])
                        arrays["original_sr"][index] = int(
                            stats["original_sample_rate"]
                        )
                        arrays["offset"][index] = np.float32(offset)
                        arrays["l2"][index] = np.float32(
                            np.sqrt(np.mean(centered.astype(np.float64) ** 2))
                        )
                        arrays["lf"][index] = np.float32(
                            band_mean(centered, frequencies, 0.0, 1_000.0)
                        )
                        arrays["mf"][index] = np.float32(
                            band_mean(centered, frequencies, 1_000.0, 4_000.0)
                        )
                        arrays["hf"][index] = np.float32(
                            band_mean(centered, frequencies, 4_000.0, 8_001.0)
                        )
                        arrays["status"][index] = STATUS_OK
                processed_session += 1
            flush_arrays(arrays)
            elapsed = time.time() - start_time
            rate = processed_session / elapsed if elapsed > 0 else 0.0
            remaining = total_pending_initial - processed_session
            eta = remaining / rate if rate > 0 else float("inf")
            ok_total = int((np.asarray(arrays["status"]) == STATUS_OK).sum())
            fail_total = int((np.asarray(arrays["status"]) == STATUS_FAIL).sum())
            print(
                f"[PAIRS] chunk {chunk_id} — session={processed_session:,}/"
                f"{total_pending_initial:,} — {rate:.2f} pair/s — "
                f"ETA={eta/60.0:.1f} min — ok_total={ok_total:,} — "
                f"fail_total={fail_total:,}"
            )
            gc.collect()

    return arrays


def determine_final_balanced_population(
    pair_frame: pd.DataFrame,
    pair_status: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    work = pair_frame.copy()
    work["extraction_status_code"] = np.asarray(pair_status, dtype=np.uint8)
    successful = work.loc[
        work["extraction_status_code"].eq(int(STATUS_OK))
    ].copy()
    n_generators = int(work["independent_generator_id"].nunique())
    coverage = (
        successful.groupby("original_id")["independent_generator_id"].nunique()
    )
    complete_ids = set(coverage[coverage == n_generators].index.astype(str))
    final_frame = successful.loc[
        successful["original_id"].astype(str).isin(complete_ids)
    ].copy()
    final_frame = final_frame.sort_values(
        ["original_id", "independent_generator_id", "pair_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    counts = final_frame["independent_generator_id"].value_counts().sort_index()
    if final_frame.empty:
        raise RuntimeError("No complete final population remains after extraction")
    if counts.nunique() != 1:
        raise RuntimeError("The final population is not balanced")
    report = {
        "n_pairs_selected": int(len(work)),
        "n_pairs_extraction_ok": int(len(successful)),
        "n_pairs_extraction_fail": int(
            (work["extraction_status_code"] == int(STATUS_FAIL)).sum()
        ),
        "n_originals_selected": int(work["original_id"].nunique()),
        "n_originals_final_complete": int(len(complete_ids)),
        "n_originals_excluded_after_extraction": int(
            work["original_id"].nunique() - len(complete_ids)
        ),
        "n_final_balanced_pairs": int(len(final_frame)),
        "pairs_per_generator": {str(k): int(v) for k, v in counts.items()},
    }
    return final_frame, report


def build_real_stats_frame(
    real_frame: pd.DataFrame,
    arrays: Mapping[str, np.memmap],
) -> pd.DataFrame:
    output = real_frame.copy()
    output["extraction_status_code"] = np.asarray(arrays["status"], dtype=np.uint8)
    output["extraction_status"] = output["extraction_status_code"].map(
        {0: "pending", 1: "ok", 2: "fail"}
    )
    output["active_frames_harmonized"] = np.asarray(
        arrays["active_frames"], dtype=np.int32
    )
    output["total_frames_harmonized"] = np.asarray(
        arrays["total_frames"], dtype=np.int32
    )
    output["vad_fallback_harmonized"] = np.asarray(
        arrays["fallback"], dtype=np.uint8
    ).astype(bool)
    output["duration_s_harmonized"] = np.asarray(
        arrays["duration"], dtype=np.float32
    )
    output["original_sample_rate_harmonized"] = np.asarray(
        arrays["original_sr"], dtype=np.int32
    )
    return output


def build_pair_metadata_frame(
    pair_frame: pd.DataFrame,
    pair_arrays: Mapping[str, np.memmap],
    real_arrays: Mapping[str, np.memmap],
) -> pd.DataFrame:
    keep_columns = [
        "extraction_row",
        "real_index",
        "pair_id",
        "dataset",
        "independent_generator_id",
        "waveform_architecture",
        "waveform_family",
        "language",
        "original_id",
        "fake_path",
        "real_path",
    ]
    for column in OPTIONAL_METADATA_COLUMNS:
        if column in pair_frame.columns and column not in keep_columns:
            keep_columns.append(column)
    output = pair_frame[keep_columns].copy()
    output["status_code_harmonized"] = np.asarray(
        pair_arrays["status"], dtype=np.uint8
    )
    output["status_harmonized"] = output["status_code_harmonized"].map(
        {0: "pending", 1: "ok", 2: "fail"}
    )
    output["fake_duration_s_harmonized"] = np.asarray(
        pair_arrays["duration"], dtype=np.float32
    )
    real_indices = output["real_index"].to_numpy(dtype=np.int64)
    output["real_duration_s_harmonized"] = np.asarray(
        real_arrays["duration"][real_indices], dtype=np.float32
    )
    output["duration_ratio_harmonized"] = np.divide(
        output["fake_duration_s_harmonized"].to_numpy(dtype=np.float64),
        output["real_duration_s_harmonized"].to_numpy(dtype=np.float64),
        out=np.full(len(output), np.nan, dtype=np.float64),
        where=output["real_duration_s_harmonized"].to_numpy(dtype=np.float64) > 0,
    )
    output["fake_active_frames_harmonized"] = np.asarray(
        pair_arrays["active_frames"], dtype=np.int32
    )
    output["real_active_frames_harmonized"] = np.asarray(
        real_arrays["active_frames"][real_indices], dtype=np.int32
    )
    output["fake_total_frames_harmonized"] = np.asarray(
        pair_arrays["total_frames"], dtype=np.int32
    )
    output["real_total_frames_harmonized"] = np.asarray(
        real_arrays["total_frames"][real_indices], dtype=np.int32
    )
    output["fake_vad_fallback_harmonized"] = np.asarray(
        pair_arrays["fallback"], dtype=np.uint8
    ).astype(bool)
    output["real_vad_fallback_harmonized"] = np.asarray(
        real_arrays["fallback"][real_indices], dtype=np.uint8
    ).astype(bool)
    output["fake_original_sample_rate_harmonized"] = np.asarray(
        pair_arrays["original_sr"], dtype=np.int32
    )
    output["real_original_sample_rate_harmonized"] = np.asarray(
        real_arrays["original_sr"][real_indices], dtype=np.int32
    )
    output["broadband_offset_db"] = np.asarray(
        pair_arrays["offset"], dtype=np.float32
    )
    output["residual_l2"] = np.asarray(pair_arrays["l2"], dtype=np.float32)
    output["residual_lf_mean_db"] = np.asarray(
        pair_arrays["lf"], dtype=np.float32
    )
    output["residual_mf_mean_db"] = np.asarray(
        pair_arrays["mf"], dtype=np.float32
    )
    output["residual_hf_mean_db"] = np.asarray(
        pair_arrays["hf"], dtype=np.float32
    )
    return output


def write_final_pair_parquet(
    pair_metadata: pd.DataFrame,
    final_rows: np.ndarray,
    residual: np.ndarray,
    output_path: Path,
    n_bins: int,
    chunk_size: int = 2_048,
) -> int:
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if temporary.exists():
        temporary.unlink()
    writer: Optional[pq.ParquetWriter] = None
    rows_written = 0
    residual_columns = [f"res_{index:04d}" for index in range(n_bins)]
    try:
        for start, end in iter_slices(len(final_rows), chunk_size):
            take = final_rows[start:end]
            metadata_chunk = pair_metadata.iloc[take].reset_index(drop=True)
            residual_chunk = pd.DataFrame(
                np.asarray(residual[take, :], dtype=np.float32),
                columns=residual_columns,
            )
            frame = pd.concat([metadata_chunk, residual_chunk], axis=1)
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary,
                    table.schema,
                    compression="zstd",
                    use_dictionary=True,
                )
            writer.write_table(table)
            rows_written += len(frame)
    finally:
        if writer is not None:
            writer.close()
    if rows_written == 0:
        raise RuntimeError("No row was written to the final Parquet file")
    os.replace(temporary, output_path)
    return rows_written


def build_generator_fingerprints(
    final_frame: pd.DataFrame,
    residual: np.ndarray,
    n_bins: int,
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    residual_columns = [f"res_{index:04d}" for index in range(n_bins)]
    for generator_id, group in final_frame.groupby(
        "independent_generator_id",
        sort=True,
    ):
        rows = group["extraction_row"].to_numpy(dtype=np.int64)
        vector = np.median(
            np.asarray(residual[rows, :], dtype=np.float32),
            axis=0,
        ).astype(np.float32)
        families = group["waveform_family"].astype(str).unique()
        architectures = group["waveform_architecture"].astype(str).unique()
        if len(families) != 1 or len(architectures) != 1:
            raise RuntimeError(f"Ambiguous taxonomy for {generator_id}")
        record: Dict[str, Any] = {
            "independent_generator_id": str(generator_id),
            "waveform_architecture": str(architectures[0]),
            "waveform_family": str(families[0]),
            "n_pairs": int(len(group)),
            "n_originals": int(group["original_id"].nunique()),
            "languages": "|".join(sorted(group["language"].astype(str).unique())),
        }
        record.update(
            {column: float(vector[index]) for index, column in enumerate(residual_columns)}
        )
        records.append(record)
    return pd.DataFrame(records)


def build_block_fingerprints(
    final_frame: pd.DataFrame,
    residual: np.ndarray,
    n_blocks: int,
    n_bins: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, str]]:
    generators = sorted(final_frame["independent_generator_id"].astype(str).unique())
    original_ids = np.asarray(sorted(final_frame["original_id"].astype(str).unique()))
    effective_blocks = min(int(n_blocks), len(original_ids))
    if effective_blocks < 4:
        raise RuntimeError("Fewer than four originals are available for bootstrap")
    
    
    hash_order = sorted(
        original_ids.tolist(),
        key=lambda value: hashlib.sha256(str(value).encode("utf-8")).hexdigest(),
    )
    block_by_original = {
        original_id: rank % effective_blocks
        for rank, original_id in enumerate(hash_order)
    }
    block_vectors = np.full(
        (len(generators), n_blocks, n_bins),
        np.nan,
        dtype=np.float32,
    )
    block_counts = np.zeros((len(generators), n_blocks), dtype=np.int32)
    family_by_generator: Dict[str, str] = {}

    for generator_index, generator_id in enumerate(generators):
        group = final_frame.loc[
            final_frame["independent_generator_id"].astype(str).eq(generator_id)
        ].copy()
        families = group["waveform_family"].astype(str).unique()
        if len(families) != 1:
            raise RuntimeError(f"Ambiguous family for {generator_id}")
        family_by_generator[generator_id] = str(families[0])
        group["content_block"] = group["original_id"].astype(str).map(block_by_original)
        for block_id, block_group in group.groupby("content_block", sort=True):
            rows = block_group["extraction_row"].to_numpy(dtype=np.int64)
            block_vectors[generator_index, int(block_id), :] = np.median(
                np.asarray(residual[rows, :], dtype=np.float32),
                axis=0,
            )
            block_counts[generator_index, int(block_id)] = int(len(rows))

    valid_blocks = np.all(block_counts > 0, axis=0)
    if int(valid_blocks.sum()) < 4:
        raise RuntimeError(
            "Fewer than four not-empty shared blocks are available for bootstrap"
        )
    return (
        block_vectors[:, valid_blocks, :],
        block_counts[:, valid_blocks],
        np.flatnonzero(valid_blocks).astype(np.int32),
        generators,
        family_by_generator,
    )


def run_stability_analysis(
    final_frame: pd.DataFrame,
    residual: np.ndarray,
    output_dir: Path,
    stability: StabilityConfig,
    mode: str,
    n_bins: int,
) -> Dict[str, Any]:
    stability.validate()
    stability_dir = output_dir / "stability"
    stability_dir.mkdir(parents=True, exist_ok=True)
    block_vectors, block_counts, block_ids, generators, family_by_generator = (
        build_block_fingerprints(
            final_frame,
            residual,
            stability.content_blocks,
            n_bins,
        )
    )
    np.savez_compressed(
        stability_dir / "content_block_fingerprints.npz",
        generator_ids=np.asarray(generators, dtype="U"),
        block_ids=block_ids,
        block_vectors=block_vectors,
        block_counts=block_counts,
    )

    full_fingerprints = np.median(block_vectors, axis=1).astype(np.float64)
    bootstrap_repeats = (
        stability.bootstrap_repeats_quick
        if mode == "quick"
        else stability.bootstrap_repeats_full
    )
    split_repeats = (
        stability.split_repeats_quick
        if mode == "quick"
        else stability.split_repeats_full
    )
    n_blocks_observed = block_vectors.shape[1]

    bootstrap_rows: List[Dict[str, Any]] = []
    for bootstrap_seed in stability.bootstrap_seeds:
        rng = np.random.default_rng(int(bootstrap_seed))
        for repeat in range(bootstrap_repeats):
            sampled = rng.choice(
                n_blocks_observed,
                size=n_blocks_observed,
                replace=True,
            )
            bootstrap_fp = np.median(block_vectors[:, sampled, :], axis=1)
            for generator_index, generator_id in enumerate(generators):
                correlation = pearson_correlation(
                    bootstrap_fp[generator_index],
                    full_fingerprints[generator_index],
                )
                rmse = float(
                    np.sqrt(
                        np.mean(
                            (
                                bootstrap_fp[generator_index].astype(np.float64)
                                - full_fingerprints[generator_index]
                            )
                            ** 2
                        )
                    )
                )
                bootstrap_rows.append(
                    {
                        "bootstrap_seed": int(bootstrap_seed),
                        "repeat": int(repeat),
                        "independent_generator_id": generator_id,
                        "waveform_family": family_by_generator[generator_id],
                        "correlation_to_full": correlation,
                        "rmse_to_full_db": rmse,
                    }
                )
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    atomic_csv_dump(
        bootstrap_frame,
        stability_dir / "bootstrap_stability_long.csv",
    )

    bootstrap_summary_rows: List[Dict[str, Any]] = []
    for generator_id, group in bootstrap_frame.groupby(
        "independent_generator_id",
        sort=True,
    ):
        corr_ci = percentile_ci(
            group["correlation_to_full"],
            stability.confidence_level,
        )
        rmse_ci = percentile_ci(
            group["rmse_to_full_db"],
            stability.confidence_level,
        )
        bootstrap_summary_rows.append(
            {
                "independent_generator_id": str(generator_id),
                "waveform_family": str(group["waveform_family"].iloc[0]),
                "n_bootstrap_total": int(len(group)),
                "correlation_mean": float(group["correlation_to_full"].mean()),
                "correlation_median": float(group["correlation_to_full"].median()),
                "correlation_ci_low": corr_ci[0],
                "correlation_ci_high": corr_ci[1],
                "rmse_mean_db": float(group["rmse_to_full_db"].mean()),
                "rmse_ci_low_db": rmse_ci[0],
                "rmse_ci_high_db": rmse_ci[1],
            }
        )
    bootstrap_summary = pd.DataFrame(bootstrap_summary_rows)
    atomic_csv_dump(
        bootstrap_summary,
        stability_dir / "bootstrap_stability_summary.csv",
    )

    split_rows: List[Dict[str, Any]] = []
    master_split_rng = np.random.default_rng(
        stable_int_hash("split-block-reproducibility", 2**32 - 1)
        + int(stability.bootstrap_seeds[0])
    )
    for repeat in range(split_repeats):
        permutation = master_split_rng.permutation(n_blocks_observed)
        split_point = n_blocks_observed // 2
        first_blocks = permutation[:split_point]
        second_blocks = permutation[split_point:]
        if len(first_blocks) == 0 or len(second_blocks) == 0:
            raise RuntimeError("Empty block split")
        first_fp = np.median(block_vectors[:, first_blocks, :], axis=1)
        second_fp = np.median(block_vectors[:, second_blocks, :], axis=1)

        same_correlations = [
            pearson_correlation(first_fp[index], second_fp[index])
            for index in range(len(generators))
        ]
        different_correlations: List[float] = []
        for first_index in range(len(generators)):
            for second_index in range(len(generators)):
                if first_index == second_index:
                    continue
                different_correlations.append(
                    pearson_correlation(
                        first_fp[first_index],
                        second_fp[second_index],
                    )
                )
        split_rows.append(
            {
                "repeat": int(repeat),
                "n_blocks_first": int(len(first_blocks)),
                "n_blocks_second": int(len(second_blocks)),
                "mean_same_generator_correlation": float(
                    np.mean(same_correlations)
                ),
                "mean_different_generator_correlation": float(
                    np.mean(different_correlations)
                ),
                "generator_specificity_delta": float(
                    np.mean(same_correlations)
                    - np.mean(different_correlations)
                ),
                "minimum_same_generator_correlation": float(
                    np.min(same_correlations)
                ),
            }
        )
    split_frame = pd.DataFrame(split_rows)
    atomic_csv_dump(
        split_frame,
        stability_dir / "split_block_reproducibility.csv",
    )
    delta_ci = percentile_ci(
        split_frame["generator_specificity_delta"],
        stability.confidence_level,
    )
    same_ci = percentile_ci(
        split_frame["mean_same_generator_correlation"],
        stability.confidence_level,
    )
    report = {
        "bootstrap": {
            "seeds": [int(seed) for seed in stability.bootstrap_seeds],
            "repeats_per_seed": int(bootstrap_repeats),
            "n_bootstrap_total_per_generator": int(
                bootstrap_repeats * len(stability.bootstrap_seeds)
            ),
            "content_blocks_requested": int(stability.content_blocks),
            "content_blocks_nonempty_common": int(n_blocks_observed),
            "confidence_level": float(stability.confidence_level),
            "minimum_generator_correlation_ci_low": float(
                bootstrap_summary["correlation_ci_low"].min()
            ),
            "maximum_generator_rmse_ci_high_db": float(
                bootstrap_summary["rmse_ci_high_db"].max()
            ),
        },
        "split_block_reproducibility": {
            "n_repeats": int(split_repeats),
            "same_generator_correlation_mean": float(
                split_frame["mean_same_generator_correlation"].mean()
            ),
            "same_generator_correlation_ci": list(same_ci),
            "different_generator_correlation_mean": float(
                split_frame["mean_different_generator_correlation"].mean()
            ),
            "generator_specificity_delta_mean": float(
                split_frame["generator_specificity_delta"].mean()
            ),
            "generator_specificity_delta_ci": list(delta_ci),
            "generator_specificity_supported": bool(delta_ci[0] > 0),
            "scope_note": (
                "Content-block reproducibility diagnostic. "
                "This result is not the confirmatory family test."
            ),
        },
    }
    atomic_json_dump(report, stability_dir / "stability_summary.json")
    return report


def validate_against_mlaad_reference(
    reference_path: Path,
    config: SpectralConfig,
    sample_size: int,
    seed: int,
    output_path: Path,
    atol: float = 2e-5,
    rtol: float = 2e-5,
) -> Dict[str, Any]:
    if not reference_path.is_file():
        report = {
            "status": "SKIPPED_REFERENCE_NOT_FOUND",
            "reference": str(reference_path),
        }
        atomic_json_dump(report, output_path)
        return report

    parquet = pq.ParquetFile(reference_path)
    residual_columns = sorted(
        [name for name in parquet.schema.names if name.startswith("res_")],
        key=lambda name: int(name.split("_")[1]),
    )
    expected_bins = config.n_fft // 2 + 1
    if len(residual_columns) != expected_bins:
        raise RuntimeError(
            f"MLAAD reference: {len(residual_columns)} bins, {expected_bins} expected"
        )
    columns = ["pair_id", "fake_path", "real_path"] + residual_columns
    table = pq.read_table(reference_path, columns=columns)
    frame = table.to_pandas()
    sample_n = min(int(sample_size), len(frame))
    sample = frame.sample(n=sample_n, random_state=seed, replace=False)
    differences: List[float] = []
    failed_rows: List[Dict[str, Any]] = []
    for row in sample.itertuples(index=False):
        fake_spectrum, _ = extract_log_spectrum(str(row.fake_path), config)
        real_spectrum, _ = extract_log_spectrum(str(row.real_path), config)
        raw = fake_spectrum - real_spectrum
        frequencies = np.fft.rfftfreq(
            config.n_fft,
            d=1.0 / config.target_sr,
        )
        mask = (
            (frequencies >= config.centering_min_hz)
            & (frequencies <= config.centering_max_hz)
        )
        calculated = raw - np.float32(np.median(raw[mask]))
        expected = np.asarray(
            [getattr(row, column) for column in residual_columns],
            dtype=np.float32,
        )
        max_abs = float(np.max(np.abs(calculated - expected)))
        differences.append(max_abs)
        if not np.allclose(calculated, expected, atol=atol, rtol=rtol):
            failed_rows.append(
                {
                    "pair_id": str(row.pair_id),
                    "max_absolute_difference": max_abs,
                }
            )
    report = {
        "status": "PASS" if not failed_rows else "FAIL",
        "reference": str(reference_path),
        "reference_sha256": sha256_file(reference_path),
        "sample_size": int(sample_n),
        "seed": int(seed),
        "atol": float(atol),
        "rtol": float(rtol),
        "maximum_absolute_difference": float(max(differences)) if differences else None,
        "median_maximum_absolute_difference": float(np.median(differences))
        if differences
        else None,
        "n_failed": int(len(failed_rows)),
        "failed_rows": failed_rows[:20],
    }
    atomic_json_dump(report, output_path)
    if failed_rows:
        raise RuntimeError(
            "Numerical validation against MLAAD failed. "
            f"Consulter {output_path}"
        )
    return report


def config_payload(
    spectral: SpectralConfig,
    stability: StabilityConfig,
    mode: str,
    master_seed: int,
    workers: int,
    chunk_size: int,
    strict_hash_audit: bool,
    minimum_success_rate: float,
) -> Dict[str, Any]:
    return {
        "version": VERSION,
        "spectral": asdict(spectral),
        "stability": {
            **asdict(stability),
            "bootstrap_seeds": list(stability.bootstrap_seeds),
        },
        "mode": mode,
        "master_seed": int(master_seed),
        "workers": int(workers),
        "chunk_size": int(chunk_size),
        "strict_hash_audit": bool(strict_hash_audit),
        "minimum_success_rate": float(minimum_success_rate),
        "residual_definition": (
            "median_active_log_power_fake_db - median_active_log_power_real_db "
            "- median_80_7600Hz_offset"
        ),
        "unit_of_balance": "original_id shared across all generators",
        "bootstrap_unit": "deterministic content block",
    }


def ensure_run_compatibility(
    output_dir: Path,
    metadata: Mapping[str, Any],
    force: bool,
) -> None:
    if force and output_dir.exists():
        print(f"[FORCE] Removing de {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "run_metadata.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        critical = [
            "version",
            "dataset_key",
            "manifest_sha256",
            "selection_sha256",
            "config_hash",
            "source_code_sha256",
        ]
        mismatches = {
            key: {
                "previous": previous.get(key),
                "current": metadata.get(key),
            }
            for key in critical
            if previous.get(key) != metadata.get(key)
        }
        if mismatches:
            raise RuntimeError(
                "The output directory corresponds to a different input/configuration. "
                "Use --force or a new --output-root. Details: "
                + json.dumps(mismatches, ensure_ascii=False)
            )
    else:
        spectra_dir = output_dir / "spectra"
        if spectra_dir.exists() and any(spectra_dir.glob("*.npy")):
            raise RuntimeError(
                "Extraction tables exist without run_metadata.json. "
                "Their provenance cannot be verified; use --force."
            )
        atomic_json_dump(dict(metadata), metadata_path)


def run_dataset(
    spec: DatasetSpec,
    args: argparse.Namespace,
    spectral: SpectralConfig,
    stability: StabilityConfig,
    source_code_hash: str,
) -> Dict[str, Any]:
    dataset_start = time.time()
    output_dir = Path(args.output_root) / spec.key / args.mode
    if args.force and output_dir.exists():
        print(f"[FORCE] Removing de {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 100)
    print(f"Q1 HARMONIZED EXTRACTION — {spec.key.upper()} — {args.mode.upper()}")
    print("=" * 100)

    pair_frame, manifest_report = prepare_manifest(
        spec,
        mode=args.mode,
        seed=args.seed,
        rewrites=args.path_rewrite,
        strict_hash_audit=not args.allow_hash_collisions,
        output_dir=output_dir,
    )
    pair_frame, real_frame = build_real_index(pair_frame)

    configuration = config_payload(
        spectral,
        stability,
        args.mode,
        args.seed,
        args.workers,
        args.chunk_size,
        not args.allow_hash_collisions,
        args.minimum_success_rate,
    )
    metadata = {
        "version": VERSION,
        "dataset_key": spec.key,
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "manifest": manifest_report["manifest"],
        "manifest_sha256": manifest_report["manifest_sha256"],
        "selection_sha256": manifest_report["selection_sha256"],
        "config": configuration,
        "config_hash": stable_json_hash(configuration),
        "source_code_sha256": source_code_hash,
    }
    ensure_run_compatibility(
        output_dir,
        metadata,
        force=False,
    )
    
    atomic_json_dump(metadata, output_dir / "run_metadata.json")
    atomic_json_dump(environment_report(), output_dir / "environment.json")
    atomic_json_dump(manifest_report, output_dir / "manifest_report.json")

    atomic_parquet_dump(pair_frame, output_dir / "pair_order.parquet")
    atomic_parquet_dump(real_frame, output_dir / "real_index.parquet")

    frequencies = np.fft.rfftfreq(
        spectral.n_fft,
        d=1.0 / spectral.target_sr,
    ).astype(np.float32)
    frequency_axis = pd.DataFrame(
        {
            "bin_index": np.arange(len(frequencies), dtype=np.int32),
            "column_name": [f"res_{index:04d}" for index in range(len(frequencies))],
            "frequency_hz": frequencies,
        }
    )
    atomic_csv_dump(frequency_axis, output_dir / "frequency_axis.csv")

    validate_paths_sample(pair_frame, args.path_check_sample, args.seed)
    print(
        f"[SELECTION] {len(pair_frame):,} pairs = "
        f"{pair_frame['original_id'].nunique():,} originals × "
        f"{pair_frame['independent_generator_id'].nunique()} generators"
    )

    real_arrays = run_real_extraction(
        real_frame,
        output_dir,
        spectral,
        workers=args.workers,
        chunk_size=args.chunk_size,
        retry_failed=args.retry_failed,
    )
    pair_arrays = run_pair_extraction(
        pair_frame,
        real_arrays,
        output_dir,
        spectral,
        workers=args.workers,
        chunk_size=args.chunk_size,
        retry_failed=args.retry_failed,
    )

    real_success_rate = float(
        np.mean(np.asarray(real_arrays["status"], dtype=np.uint8) == STATUS_OK)
    )
    pair_success_rate = float(
        np.mean(np.asarray(pair_arrays["status"], dtype=np.uint8) == STATUS_OK)
    )
    if real_success_rate < args.minimum_success_rate:
        raise RuntimeError(
            f"Real-audio gate failed : {real_success_rate:.2%} < "
            f"{args.minimum_success_rate:.2%}"
        )
    if pair_success_rate < args.minimum_success_rate:
        raise RuntimeError(
            f"Pair gate failed : {pair_success_rate:.2%} < "
            f"{args.minimum_success_rate:.2%}"
        )

    final_frame, final_report = determine_final_balanced_population(
        pair_frame,
        np.asarray(pair_arrays["status"], dtype=np.uint8),
    )
    atomic_parquet_dump(
        final_frame,
        output_dir / "final_balanced_manifest.parquet",
    )
    real_stats = build_real_stats_frame(real_frame, real_arrays)
    atomic_parquet_dump(real_stats, output_dir / "real_extraction_stats.parquet")
    pair_metadata = build_pair_metadata_frame(pair_frame, pair_arrays, real_arrays)
    atomic_parquet_dump(
        pair_metadata,
        output_dir / "pair_extraction_manifest.parquet",
    )

    final_rows = final_frame["extraction_row"].to_numpy(dtype=np.int64)
    rows_written = write_final_pair_parquet(
        pair_metadata,
        final_rows,
        pair_arrays["residual"],
        output_dir / "fingerprints_pair_level_harmonized.parquet",
        n_bins=len(frequencies),
    )
    if rows_written != len(final_frame):
        raise RuntimeError(
            f"Parquet final incomplete : {rows_written}/{len(final_frame)}"
        )

    generator_fingerprints = build_generator_fingerprints(
        final_frame,
        pair_arrays["residual"],
        n_bins=len(frequencies),
    )
    atomic_parquet_dump(
        generator_fingerprints,
        output_dir / "generator_fingerprints_harmonized.parquet",
    )

    stability_report = run_stability_analysis(
        final_frame,
        pair_arrays["residual"],
        output_dir,
        stability,
        args.mode,
        n_bins=len(frequencies),
    )

    errors_real = read_jsonl_deduplicated(
        output_dir / "logs" / "real_errors.jsonl",
        "real_index",
    )
    errors_pair = read_jsonl_deduplicated(
        output_dir / "logs" / "pair_errors.jsonl",
        "extraction_row",
    )
    if not errors_real.empty:
        failed_real_indices = set(
            np.flatnonzero(np.asarray(real_arrays["status"]) == STATUS_FAIL).tolist()
        )
        errors_real = errors_real.loc[
            errors_real["real_index"].astype(int).isin(failed_real_indices)
        ].copy()
    if not errors_real.empty:
        atomic_csv_dump(errors_real, output_dir / "real_failures.csv")
    else:
        atomic_csv_dump(
            pd.DataFrame(columns=["real_index", "original_id", "path", "error"]),
            output_dir / "real_failures.csv",
        )
    if not errors_pair.empty:
        failed_pair_indices = set(
            np.flatnonzero(np.asarray(pair_arrays["status"]) == STATUS_FAIL).tolist()
        )
        errors_pair = errors_pair.loc[
            errors_pair["extraction_row"].astype(int).isin(failed_pair_indices)
        ].copy()
    if not errors_pair.empty:
        atomic_csv_dump(errors_pair, output_dir / "pair_failures.csv")
    else:
        atomic_csv_dump(
            pd.DataFrame(columns=["extraction_row", "pair_id", "path", "error"]),
            output_dir / "pair_failures.csv",
        )

    elapsed = time.time() - dataset_start
    summary = {
        "version": VERSION,
        "status": "COMPLETE",
        "dataset_key": spec.key,
        "mode": args.mode,
        "elapsed_seconds": float(elapsed),
        "elapsed_hours": float(elapsed / 3600.0),
        "manifest": manifest_report,
        "spectral_config": asdict(spectral),
        "stability_config": {
            **asdict(stability),
            "bootstrap_seeds": list(stability.bootstrap_seeds),
        },
        "real_extraction": {
            "n_real": int(len(real_frame)),
            "n_ok": int((np.asarray(real_arrays["status"]) == STATUS_OK).sum()),
            "n_fail": int((np.asarray(real_arrays["status"]) == STATUS_FAIL).sum()),
            "success_rate": real_success_rate,
            "minimum_success_rate": float(args.minimum_success_rate),
            "vad_fallback_count": int(np.asarray(real_arrays["fallback"]).sum()),
        },
        "pair_extraction": {**final_report, "success_rate": pair_success_rate, "minimum_success_rate": float(args.minimum_success_rate)},
        "fake_vad_fallback_count": int(np.asarray(pair_arrays["fallback"]).sum()),
        "stability": stability_report,
        "outputs": {
            "pair_level": str(
                output_dir / "fingerprints_pair_level_harmonized.parquet"
            ),
            "generator_level": str(
                output_dir / "generator_fingerprints_harmonized.parquet"
            ),
            "pair_order": str(output_dir / "pair_order.parquet"),
            "real_index": str(output_dir / "real_index.parquet"),
            "frequency_axis": str(output_dir / "frequency_axis.csv"),
            "spectra_folder": str(output_dir / "spectra"),
        },
    }
    atomic_json_dump(summary, output_dir / "extraction_summary.json")
    atomic_json_dump(
        {
            "version": VERSION,
            "status": "COMPLETE",
            "dataset_key": spec.key,
            "mode": args.mode,
            "summary": str(output_dir / "extraction_summary.json"),
            "completed_utc": pd.Timestamp.utcnow().isoformat(),
        },
        output_dir / ".HARMONIZED_EXTRACTION_COMPLETE.json",
    )

    print("-" * 100)
    print(f"[COMPLETE] {spec.key}")
    print(f"Final balanced pairs : {len(final_frame):,}")
    print(f"Final originals            : {final_frame['original_id'].nunique():,}")
    print(f"Generators                : {final_frame['independent_generator_id'].nunique()}")
    print(f"Real-audio failures               : {summary['real_extraction']['n_fail']}")
    print(f"Pair failures              : {final_report['n_pairs_extraction_fail']}")
    print(f"Output                     : {output_dir}")
    print("-" * 100)
    return summary


def _is_jupyter_kernel_argument_list(argv: Sequence[str]) -> bool:

    values = list(argv)
    for index, value in enumerate(values):
        value = str(value)
        if value == "-f" and index + 1 < len(values):
            candidate = str(values[index + 1])
            if candidate.endswith(".json") and "kernel-" in Path(candidate).name:
                return True
        if value.startswith("-f=") or value.startswith("--f="):
            candidate = value.split("=", 1)[1]
            if candidate.endswith(".json") and "kernel-" in Path(candidate).name:
                return True
    return False


def _strip_jupyter_kernel_arguments(argv: Sequence[str]) -> List[str]:

    values = list(argv)
    cleaned: List[str] = []
    index = 0
    while index < len(values):
        value = str(values[index])
        if value == "-f" and index + 1 < len(values):
            candidate = str(values[index + 1])
            if candidate.endswith(".json") and "kernel-" in Path(candidate).name:
                index += 2
                continue
        if value.startswith("-f=") or value.startswith("--f="):
            candidate = value.split("=", 1)[1]
            if candidate.endswith(".json") and "kernel-" in Path(candidate).name:
                index += 1
                continue
        cleaned.append(value)
        index += 1
    return cleaned


def resolve_source_code_hash() -> Tuple[str, str]:

    candidates: List[Path] = []
    file_value = globals().get("__file__")
    if file_value:
        candidates.append(Path(str(file_value)).expanduser())
    candidates.extend([
        Path("/content/drive/MyDrive/Q1_01_HARMONIZED_EXTRACTION_v1_0_1.py"),
        Path("/content/drive/MyDrive/Q1_01_HARMONIZED_EXTRACTION.py"),
    ])
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            if resolved.is_file():
                return sha256_file(resolved), str(resolved)
        except Exception:
            continue
    
    fallback_payload = {
        "version": VERSION,
        "execution_context": "interactive_notebook_source_unavailable",
    }
    return stable_json_hash(fallback_payload), "INTERACTIVE_NOTEBOOK_FALLBACK"


def parse_bootstrap_seeds(text: str) -> Tuple[int, ...]:
    values = [item.strip() for item in str(text).split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Seed list is empty")
    try:
        seeds = tuple(int(value) for value in values)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Seeds must be comma-separated integers"
        ) from exc
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("Seeds must be unique")
    return seeds


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Harmonized Q1 spectral extraction for WaveFake and LibriSeVoc"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_SPECS),
        default=list(DATASET_SPECS),
        help="Datasets to run",
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="full",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=DEFAULT_MASTER_SEED)
    parser.add_argument(
        "--bootstrap-seeds",
        type=parse_bootstrap_seeds,
        default=DEFAULT_BOOTSTRAP_SEEDS,
        help="Comma-separated integers",
    )
    parser.add_argument("--bootstrap-repeats-full", type=int, default=500)
    parser.add_argument("--bootstrap-repeats-quick", type=int, default=50)
    parser.add_argument("--split-repeats-full", type=int, default=200)
    parser.add_argument("--split-repeats-quick", type=int, default=30)
    parser.add_argument("--content-blocks", type=int, default=50)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--path-check-sample", type=int, default=80)
    parser.add_argument("--minimum-success-rate", type=float, default=1.0)
    parser.add_argument(
        "--path-rewrite",
        action="append",
        default=[],
        metavar="OLD=NEW",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--retry-failed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--allow-hash-collisions",
        action="store_true",
        help="Do not stop the pipeline if the hash audit detects a collision",
    )
    parser.add_argument(
        "--validate-mlaad-reference",
        action="store_true",
    )
    parser.add_argument(
        "--mlaad-reference",
        type=Path,
        default=MLAAD_REFERENCE,
    )
    parser.add_argument(
        "--mlaad-validation-sample",
        type=int,
        default=64,
    )
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    effective_argv = _strip_jupyter_kernel_arguments(effective_argv)
    return parser.parse_args(effective_argv)


def validate_arguments(args: argparse.Namespace) -> None:
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")
    if args.path_check_sample < 1:
        raise ValueError("--path-check-sample must be >= 1")
    if args.mlaad_validation_sample < 1:
        raise ValueError("--mlaad-validation-sample must be >= 1")
    if not 0.0 < args.minimum_success_rate <= 1.0:
        raise ValueError("--minimum-success-rate must be in (0, 1]")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    validate_arguments(args)
    args.output_root = args.output_root.expanduser().resolve()
    args.path_rewrite = parse_path_rewrites(args.path_rewrite)

    spectral = SpectralConfig()
    spectral.validate()
    stability = StabilityConfig(
        bootstrap_seeds=tuple(args.bootstrap_seeds),
        bootstrap_repeats_full=args.bootstrap_repeats_full,
        bootstrap_repeats_quick=args.bootstrap_repeats_quick,
        split_repeats_full=args.split_repeats_full,
        split_repeats_quick=args.split_repeats_quick,
        content_blocks=args.content_blocks,
        confidence_level=args.confidence_level,
    )
    stability.validate()

    source_code_hash, source_code_path = resolve_source_code_hash()
    print(f"Source code   : {source_code_path}")
    print(f"SHA-256 code : {source_code_hash}")
    start = time.time()

    print("=" * 100)
    print("Q1 — HARMONIZED SPECTRAL EXTRACTION")
    print(f"Version       : {VERSION}")
    print(f"Mode          : {args.mode}")
    print(f"Datasets      : {', '.join(args.datasets)}")
    print(f"Master seed   : {args.seed}")
    print(f"Seeds bootstrap: {list(stability.bootstrap_seeds)}")
    print(f"Outputs       : {args.output_root}")
    print("=" * 100)

    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(
        environment_report(),
        args.output_root / "environment.json",
    )

    mlaad_validation_report: Optional[Dict[str, Any]] = None
    if args.validate_mlaad_reference:
        print("[MLAAD VALIDATION] Numerical implementation check...")
        mlaad_validation_report = validate_against_mlaad_reference(
            args.mlaad_reference.expanduser().resolve(),
            spectral,
            sample_size=args.mlaad_validation_sample,
            seed=args.seed,
            output_path=args.output_root / "mlaad_reference_validation.json",
        )
        print(
            "[VALIDATION MLAAD] "
            + str(mlaad_validation_report.get("status", "UNKNOWN"))
        )

    summaries: Dict[str, Any] = {}
    for dataset_key in args.datasets:
        spec = DATASET_SPECS[dataset_key]
        summaries[dataset_key] = run_dataset(
            spec,
            args,
            spectral,
            stability,
            source_code_hash,
        )

    final_report = {
        "version": VERSION,
        "status": "COMPLETE",
        "mode": args.mode,
        "master_seed": int(args.seed),
        "bootstrap_seeds": [int(seed) for seed in stability.bootstrap_seeds],
        "datasets": summaries,
        "mlaad_reference_validation": mlaad_validation_report,
        "elapsed_seconds": float(time.time() - start),
        "completed_utc": pd.Timestamp.utcnow().isoformat(),
    }
    atomic_json_dump(
        final_report,
        args.output_root / f"harmonized_run_summary_{args.mode}.json",
    )
    atomic_json_dump(
        {
            "version": VERSION,
            "status": "COMPLETE",
            "mode": args.mode,
            "summary": str(
                args.output_root / f"harmonized_run_summary_{args.mode}.json"
            ),
            "completed_utc": pd.Timestamp.utcnow().isoformat(),
        },
        args.output_root / f".HARMONIZED_RUN_{args.mode.upper()}_COMPLETE.json",
    )

    print("\n" + "=" * 100)
    print("HARMONIZED EXTRACTION COMPLETE")
    for dataset_key, summary in summaries.items():
        final_pairs = summary["pair_extraction"]["n_final_balanced_pairs"]
        final_originals = summary["pair_extraction"]["n_originals_final_complete"]
        print(
            f"{dataset_key:24s} : {final_pairs:,} pairs, "
            f"{final_originals:,} originals"
        )
    print(f"Temps total : {(time.time() - start) / 60.0:.1f} min")
    print(f"Outputs     : {args.output_root}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    try:
        if _is_jupyter_kernel_argument_list(sys.argv[1:]):
            print(
                "[COLAB] The code was executed directly as a notebook cell. "
                "No automatic run is started to avoid an unintended FULL execution."
            )
            print(
                "Lancez maintenant :\n"
                "main([\"--mode\", \"quick\", \"--datasets\", "
                "\"wavefake_ljspeech\", \"wavefake_jsut\", \"librisevoc\", "
                "\"--validate-mlaad-reference\"])"
            )
        else:
            raise SystemExit(main())
    except Exception as exc:
        print("\n" + "=" * 100, file=sys.stderr)
        print("[Q1_01_HARMONIZED_EXTRACTION FAILURE]", file=sys.stderr)
        traceback.print_exc()
        try:
            fallback = Path(
                os.environ.get(
                    "Q1_HARMONIZED_ERROR_DIR",
                    "/content/drive/MyDrive/fingerprint_q1_outputs/q1_harmonized/v3_new_story",
                )
            )
            fallback.mkdir(parents=True, exist_ok=True)
            atomic_json_dump(
                {
                    "version": VERSION,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "traceback": traceback.format_exc(),
                    "failed_utc": pd.Timestamp.utcnow().isoformat(),
                },
                fallback / "fatal_error.json",
            )
        except Exception:
            pass
        raise
