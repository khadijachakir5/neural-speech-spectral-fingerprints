#!/usr/bin/env python3


# Purpose: Extract harmonized pair-level residual spectral fingerprints for MLAAD STRICT and RELAXED populations with checkpointed execution.

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


VERSION = "MLAAD-PHASE1A-SPECTRAL-RESIDUALS-v1.1-NEW-STORY"

_A2Z_ROOT = Path(os.environ.get("FINGERPRINT_OUTPUT_ROOT", "/content/drive/MyDrive/fingerprint_q1_outputs"))
PHASE0B_DIR = _A2Z_ROOT / "phase0b/phase0b_mlaad_taxonomy_v1_2_new_story"

STRICT_MANIFEST = PHASE0B_DIR / "mlaad_phase1_strict_confirmatory.parquet"
RELAXED_MANIFEST = PHASE0B_DIR / "mlaad_phase1_relaxed_confirmatory.parquet"

OUTPUT_DIR = _A2Z_ROOT / "phase1a/phase1a_mlaad_spectral_residuals_v2_new_story"


WORKERS = 2
CHUNK_SIZE = 128
FORCE_REBUILD = False
RETRY_FAILED_ON_RESUME = True
MIN_SUCCESS_RATE = 1.0


STRICT_EXPECTED_POPULATION = True
EXPECTED_STRICT_PAIRS = 62_079
EXPECTED_RELAXED_PAIRS = 64_625
EXPECTED_GENERATORS = 52
EXPECTED_FAMILIES = 4


TARGET_SR = 16_000
N_FFT = 1_024
HOP_LENGTH = 256
REMOVE_DC = True
EPSILON_POWER = 1e-12


VAD_REFERENCE_PERCENTILE = 95.0
VAD_TOP_DB = 40.0
VAD_ABS_DB = -80.0
MIN_ACTIVE_FRAMES = 3


CENTERING_MIN_HZ = 80.0
CENTERING_MAX_HZ = 7_600.0


WORKER_AUDIO_CACHE_SIZE = 256


import importlib.util
import subprocess

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


FREQUENCIES_HZ = np.fft.rfftfreq(N_FFT, d=1.0 / TARGET_SR).astype(np.float32)
N_BINS = int(len(FREQUENCIES_HZ))
RESIDUAL_COLUMNS = [f"res_{index:04d}" for index in range(N_BINS)]

CORE_METADATA_COLUMNS = [
    "pair_id",
    "independent_generator_id",
    "waveform_family",
    "waveform_architecture",
    "pipeline_type",
    "acoustic_model",
    "representation",
    "taxonomy_confidence",
    "language",
    "original_id",
    "fake_path",
    "real_path",
]

WORKER_CONFIG: Dict[str, Any] = {}


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


def atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, default=str)
    os.replace(temporary, path)


def atomic_parquet_dump(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    dataframe.to_parquet(
        temporary,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    os.replace(temporary, path)


def iter_slices(length: int, size: int) -> Iterator[Tuple[int, int]]:
    if size < 1:
        raise ValueError("CHUNK_SIZE must be >= 1")
    for start in range(0, length, size):
        yield start, min(start + size, length)


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def environment_report() -> Dict[str, Any]:
    versions = {}
    for name in ["numpy", "pandas", "scipy", "soundfile", "pyarrow"]:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            versions[name] = f"ERROR:{type(exc).__name__}:{exc}"
    return {
        "version": VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
    }


def config_payload() -> Dict[str, Any]:
    return {
        "version": VERSION,
        "target_sr": TARGET_SR,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "remove_dc": REMOVE_DC,
        "epsilon_power": EPSILON_POWER,
        "vad_reference_percentile": VAD_REFERENCE_PERCENTILE,
        "vad_top_db": VAD_TOP_DB,
        "vad_abs_db": VAD_ABS_DB,
        "min_active_frames": MIN_ACTIVE_FRAMES,
        "centering_min_hz": CENTERING_MIN_HZ,
        "centering_max_hz": CENTERING_MAX_HZ,
        "residual_definition": (
            "median_active_log_power_fake_db - "
            "median_active_log_power_real_db - broadband_median_offset"
        ),
        "frequency_bins": N_BINS,
    }


def validate_manifest(
    dataframe: pd.DataFrame,
    protocol: str,
    expected_pairs: Optional[int],
) -> Dict[str, Any]:
    required = set(CORE_METADATA_COLUMNS)
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(
            f"Manifest {protocol}: required columns missing: {missing}"
        )

    if dataframe.empty:
        raise RuntimeError(f"Manifest {protocol} empty")

    if dataframe["pair_id"].astype(str).duplicated().any():
        duplicates = int(dataframe["pair_id"].astype(str).duplicated().sum())
        raise RuntimeError(
            f"Manifest {protocol}: {duplicates} duplicate pair_id values"
        )

    if expected_pairs is not None and len(dataframe) != int(expected_pairs):
        raise RuntimeError(
            f"Manifest {protocol}: {len(dataframe):,} rows observed; "
            f"{expected_pairs:,} expected."
        )

    empty_paths = (
        dataframe["fake_path"].map(normalize_text).eq("")
        | dataframe["real_path"].map(normalize_text).eq("")
    )
    if empty_paths.any():
        raise RuntimeError(
            f"Manifest {protocol}: {int(empty_paths.sum())} pair(s) "
            "without path fake or real."
        )

    generator_family = dataframe[
        ["independent_generator_id", "waveform_family"]
    ].drop_duplicates()
    if generator_family["independent_generator_id"].duplicated().any():
        raise RuntimeError(
            f"Manifest {protocol}: a generator belongs to multiple families."
        )

    family_generator_counts = (
        generator_family.groupby("waveform_family")[
            "independent_generator_id"
        ].nunique()
    )
    if (family_generator_counts < 3).any():
        invalid = family_generator_counts[family_generator_counts < 3].to_dict()
        raise RuntimeError(
            f"Manifest {protocol}: non-confirmatory families detected: {invalid}"
        )

    return {
        "protocol": protocol,
        "n_pairs": int(len(dataframe)),
        "n_generators": int(
            dataframe["independent_generator_id"].astype(str).nunique()
        ),
        "n_families": int(dataframe["waveform_family"].astype(str).nunique()),
        "n_languages": int(dataframe["language"].astype(str).nunique()),
        "family_generator_counts": {
            str(key): int(value)
            for key, value in family_generator_counts.sort_index().items()
        },
    }


def compare_overlapping_metadata(
    strict: pd.DataFrame,
    relaxed: pd.DataFrame,
) -> None:
    overlap_ids = sorted(
        set(strict["pair_id"].astype(str))
        .intersection(set(relaxed["pair_id"].astype(str)))
    )
    if not overlap_ids:
        raise RuntimeError("No pair_id is shared between STRICT and RELAXED")

    compare_columns = [
        "pair_id",
        "independent_generator_id",
        "waveform_family",
        "waveform_architecture",
        "language",
        "original_id",
        "fake_path",
        "real_path",
    ]
    left = (
        strict.loc[strict["pair_id"].astype(str).isin(overlap_ids), compare_columns]
        .copy()
        .sort_values("pair_id")
        .reset_index(drop=True)
    )
    right = (
        relaxed.loc[
            relaxed["pair_id"].astype(str).isin(overlap_ids), compare_columns
        ]
        .copy()
        .sort_values("pair_id")
        .reset_index(drop=True)
    )
    for column in compare_columns:
        if not left[column].astype(str).equals(right[column].astype(str)):
            raise RuntimeError(
                f"STRICT/RELAXED inconsistency for column {column}"
            )


def build_union_manifest(
    strict: pd.DataFrame,
    relaxed: pd.DataFrame,
) -> pd.DataFrame:
    strict_ids = set(strict["pair_id"].astype(str))
    relaxed_ids = set(relaxed["pair_id"].astype(str))

    relaxed_base = relaxed.copy()
    relaxed_base["in_strict"] = relaxed_base["pair_id"].astype(str).isin(strict_ids)
    relaxed_base["in_relaxed"] = True

    strict_only_ids = strict_ids - relaxed_ids
    if strict_only_ids:
        strict_only = strict.loc[
            strict["pair_id"].astype(str).isin(strict_only_ids)
        ].copy()
        strict_only["in_strict"] = True
        strict_only["in_relaxed"] = False
        union = pd.concat([relaxed_base, strict_only], ignore_index=True)
    else:
        union = relaxed_base

    if union["pair_id"].astype(str).duplicated().any():
        raise RuntimeError("Union STRICT/RELAXED: Duplicate pair_id values")

    for column in CORE_METADATA_COLUMNS:
        union[column] = union[column].map(normalize_text)

    
    union = union.sort_values(
        ["real_path", "independent_generator_id", "pair_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    union.insert(0, "union_index", np.arange(len(union), dtype=np.int64))
    return union


def load_manifests() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not STRICT_MANIFEST.is_file():
        raise FileNotFoundError(f"Manifest STRICT not found: {STRICT_MANIFEST}")
    if not RELAXED_MANIFEST.is_file():
        raise FileNotFoundError(f"Manifest RELAXED not found: {RELAXED_MANIFEST}")

    strict = pd.read_parquet(STRICT_MANIFEST)
    relaxed = pd.read_parquet(RELAXED_MANIFEST)

    strict_expected = (
        EXPECTED_STRICT_PAIRS if STRICT_EXPECTED_POPULATION else None
    )
    relaxed_expected = (
        EXPECTED_RELAXED_PAIRS if STRICT_EXPECTED_POPULATION else None
    )

    strict_report = validate_manifest(strict, "strict", strict_expected)
    relaxed_report = validate_manifest(relaxed, "relaxed", relaxed_expected)

    compare_overlapping_metadata(strict, relaxed)
    union = build_union_manifest(strict, relaxed)

    if STRICT_EXPECTED_POPULATION:
        if strict_report["n_generators"] != EXPECTED_GENERATORS:
            raise RuntimeError(
                f"STRICT: {strict_report['n_generators']} generators; "
                f"{EXPECTED_GENERATORS} expected."
            )
        if relaxed_report["n_generators"] != EXPECTED_GENERATORS:
            raise RuntimeError(
                f"RELAXED: {relaxed_report['n_generators']} generators; "
                f"{EXPECTED_GENERATORS} expected."
            )
        if strict_report["n_families"] != EXPECTED_FAMILIES:
            raise RuntimeError(
                f"STRICT: {strict_report['n_families']} families; "
                f"{EXPECTED_FAMILIES} expected."
            )
        if relaxed_report["n_families"] != EXPECTED_FAMILIES:
            raise RuntimeError(
                f"RELAXED: {relaxed_report['n_families']} families; "
                f"{EXPECTED_FAMILIES} expected."
            )

    report = {
        "strict_manifest": str(STRICT_MANIFEST),
        "strict_manifest_sha256": sha256_file(STRICT_MANIFEST),
        "relaxed_manifest": str(RELAXED_MANIFEST),
        "relaxed_manifest_sha256": sha256_file(RELAXED_MANIFEST),
        "strict": strict_report,
        "relaxed": relaxed_report,
        "union_pairs": int(len(union)),
        "strict_is_subset_of_relaxed": bool(
            set(strict["pair_id"].astype(str)).issubset(
                set(relaxed["pair_id"].astype(str))
            )
        ),
    }
    return union, report


def _init_worker(worker_config: Mapping[str, Any]) -> None:
    global WORKER_CONFIG
    WORKER_CONFIG = dict(worker_config)


def _load_audio_mono_resampled(path_text: str) -> Tuple[np.ndarray, int]:
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

    if sample_rate != TARGET_SR:
        divisor = math.gcd(sample_rate, TARGET_SR)
        audio = resample_poly(
            audio,
            up=TARGET_SR // divisor,
            down=sample_rate // divisor,
        ).astype(np.float32, copy=False)

    if REMOVE_DC:
        audio = audio - np.float32(audio.mean(dtype=np.float64))

    return np.asarray(audio, dtype=np.float32), sample_rate


def _frame_audio(audio: np.ndarray) -> np.ndarray:
    if audio.size < N_FFT:
        audio = np.pad(audio, (0, N_FFT - audio.size))
    frames = np.lib.stride_tricks.sliding_window_view(audio, N_FFT)
    frames = frames[::HOP_LENGTH]
    if len(frames) == 0:
        frames = audio[:N_FFT][None, :]
    return frames


def _active_frame_mask(frames: np.ndarray) -> Tuple[np.ndarray, bool]:
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    frame_db = 20.0 * np.log10(rms + 1e-12)
    reference_db = float(np.percentile(frame_db, VAD_REFERENCE_PERCENTILE))
    threshold_db = max(reference_db - VAD_TOP_DB, VAD_ABS_DB)
    active = frame_db >= threshold_db

    fallback_used = False
    if int(active.sum()) < MIN_ACTIVE_FRAMES:
        fallback_used = True
        number_to_keep = min(max(MIN_ACTIVE_FRAMES, 1), len(frames))
        top_indices = np.argsort(rms)[-number_to_keep:]
        active = np.zeros(len(frames), dtype=bool)
        active[top_indices] = True

    return active, fallback_used


@lru_cache(maxsize=WORKER_AUDIO_CACHE_SIZE)
def _cached_log_spectrum(
    path_text: str,
) -> Tuple[np.ndarray, int, int, bool, float, int]:
    audio, original_sample_rate = _load_audio_mono_resampled(path_text)
    duration_s = float(audio.size / TARGET_SR)
    frames = _frame_audio(audio)
    active_mask, fallback_used = _active_frame_mask(frames)
    active_frames = np.asarray(frames[active_mask], dtype=np.float32)

    window = hann(N_FFT, sym=False).astype(np.float32)
    windowed = active_frames * window[None, :]
    spectrum = rfft(windowed, n=N_FFT, axis=1)
    power = (
        spectrum.real.astype(np.float32) ** 2
        + spectrum.imag.astype(np.float32) ** 2
    )
    power /= np.float32(np.sum(window.astype(np.float64) ** 2))

    log_power_db = 10.0 * np.log10(power + EPSILON_POWER)
    robust_log_spectrum = np.median(log_power_db, axis=0).astype(np.float32)

    if robust_log_spectrum.shape != (N_BINS,):
        raise RuntimeError(
            f"invalid_spectrum_shape:{robust_log_spectrum.shape}"
        )
    if not np.isfinite(robust_log_spectrum).all():
        raise ValueError("non_finite_log_spectrum")

    return (
        robust_log_spectrum,
        int(active_mask.sum()),
        int(len(frames)),
        bool(fallback_used),
        duration_s,
        int(original_sample_rate),
    )


def _band_mean(
    values: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> float:
    mask = (FREQUENCIES_HZ >= low_hz) & (FREQUENCIES_HZ < high_hz)
    if not mask.any():
        return float("nan")
    return float(np.mean(values[mask]))


def extract_pair_worker(task: Mapping[str, Any]) -> Dict[str, Any]:
    base = {
        "union_index": int(task["union_index"]),
        "pair_id": normalize_text(task["pair_id"]),
        "independent_generator_id": normalize_text(
            task["independent_generator_id"]
        ),
        "waveform_family": normalize_text(task["waveform_family"]),
        "waveform_architecture": normalize_text(
            task["waveform_architecture"]
        ),
        "pipeline_type": normalize_text(task["pipeline_type"]),
        "acoustic_model": normalize_text(task["acoustic_model"]),
        "representation": normalize_text(task["representation"]),
        "taxonomy_confidence": normalize_text(task["taxonomy_confidence"]),
        "language": normalize_text(task["language"]),
        "original_id": normalize_text(task["original_id"]),
        "fake_path": normalize_text(task["fake_path"]),
        "real_path": normalize_text(task["real_path"]),
        "in_strict": bool(task["in_strict"]),
        "in_relaxed": bool(task["in_relaxed"]),
        "config_hash": normalize_text(task["config_hash"]),
        "source_manifest_hash": normalize_text(task["source_manifest_hash"]),
    }

    empty_residual = {column: np.nan for column in RESIDUAL_COLUMNS}

    try:
        (
            fake_spectrum,
            fake_active_frames,
            fake_total_frames,
            fake_fallback,
            fake_duration_s,
            fake_original_sr,
        ) = _cached_log_spectrum(base["fake_path"])

        (
            real_spectrum,
            real_active_frames,
            real_total_frames,
            real_fallback,
            real_duration_s,
            real_original_sr,
        ) = _cached_log_spectrum(base["real_path"])

        raw_residual = fake_spectrum - real_spectrum

        centering_mask = (
            (FREQUENCIES_HZ >= CENTERING_MIN_HZ)
            & (FREQUENCIES_HZ <= CENTERING_MAX_HZ)
        )
        if not centering_mask.any():
            raise RuntimeError("empty_centering_frequency_band")

        broadband_offset_db = float(
            np.median(raw_residual[centering_mask])
        )
        centered_residual = (
            raw_residual - np.float32(broadband_offset_db)
        ).astype(np.float32)

        if not np.isfinite(centered_residual).all():
            raise ValueError("non_finite_centered_residual")

        residual_values = {
            column: float(centered_residual[index])
            for index, column in enumerate(RESIDUAL_COLUMNS)
        }

        return {
            **base,
            "status": "ok",
            "error": "",
            "fake_duration_s_phase1a": fake_duration_s,
            "real_duration_s_phase1a": real_duration_s,
            "duration_ratio_phase1a": (
                float(fake_duration_s / real_duration_s)
                if real_duration_s > 0
                else np.nan
            ),
            "fake_original_sample_rate": fake_original_sr,
            "real_original_sample_rate": real_original_sr,
            "fake_active_frames": fake_active_frames,
            "real_active_frames": real_active_frames,
            "fake_total_frames": fake_total_frames,
            "real_total_frames": real_total_frames,
            "fake_vad_fallback_used": fake_fallback,
            "real_vad_fallback_used": real_fallback,
            "broadband_offset_db": broadband_offset_db,
            "residual_l2": float(
                np.sqrt(np.mean(centered_residual.astype(np.float64) ** 2))
            ),
            "residual_lf_mean_db": _band_mean(
                centered_residual, 0.0, 1_000.0
            ),
            "residual_mf_mean_db": _band_mean(
                centered_residual, 1_000.0, 4_000.0
            ),
            "residual_hf_mean_db": _band_mean(
                centered_residual, 4_000.0, 8_001.0
            ),
            **residual_values,
        }

    except Exception as exc:
        return {
            **base,
            "status": "fail",
            "error": f"{type(exc).__name__}:{exc}",
            "fake_duration_s_phase1a": np.nan,
            "real_duration_s_phase1a": np.nan,
            "duration_ratio_phase1a": np.nan,
            "fake_original_sample_rate": np.nan,
            "real_original_sample_rate": np.nan,
            "fake_active_frames": np.nan,
            "real_active_frames": np.nan,
            "fake_total_frames": np.nan,
            "real_total_frames": np.nan,
            "fake_vad_fallback_used": False,
            "real_vad_fallback_used": False,
            "broadband_offset_db": np.nan,
            "residual_l2": np.nan,
            "residual_lf_mean_db": np.nan,
            "residual_mf_mean_db": np.nan,
            "residual_hf_mean_db": np.nan,
            **empty_residual,
        }


def part_path(parts_dir: Path, part_number: int) -> Path:
    return parts_dir / f"part_{part_number:06d}.parquet"


def validate_existing_part(
    path: Path,
    expected_pair_ids: Sequence[str],
    configuration_hash: str,
    source_manifest_hash: str,
    retry_failed: bool,
) -> bool:
    if not path.is_file():
        return False
    try:
        frame = pd.read_parquet(
            path,
            columns=[
                "pair_id",
                "status",
                "config_hash",
                "source_manifest_hash",
            ],
        )
        if frame["pair_id"].astype(str).tolist() != list(expected_pair_ids):
            return False
        if not frame["config_hash"].astype(str).eq(configuration_hash).all():
            return False
        if not frame["source_manifest_hash"].astype(str).eq(
            source_manifest_hash
        ).all():
            return False
        if retry_failed and frame["status"].astype(str).eq("fail").any():
            return False
        return True
    except Exception:
        return False


def prepare_output_directory(
    metadata: Mapping[str, Any],
) -> Tuple[Path, Path]:
    if FORCE_REBUILD and OUTPUT_DIR.exists():
        print(f"[FORCE] Removing directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parts_dir = OUTPUT_DIR / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = OUTPUT_DIR / "run_metadata.json"
    if metadata_path.is_file():
        observed = json.loads(metadata_path.read_text(encoding="utf-8"))
        critical_keys = [
            "version",
            "config_hash",
            "strict_manifest_sha256",
            "relaxed_manifest_sha256",
            "union_pair_ids_hash",
        ]
        mismatch = {
            key: {
                "observed": observed.get(key),
                "expected": metadata.get(key),
            }
            for key in critical_keys
            if observed.get(key) != metadata.get(key)
        }
        if mismatch:
            raise RuntimeError(
                "OUTPUT_DIR corresponds to a different input/configuration. "
                "Use a new OUTPUT_DIR or FORCE_REBUILD=True. "
                f"Details: {json.dumps(mismatch, ensure_ascii=False)}"
            )
    else:
        atomic_json_dump(dict(metadata), metadata_path)

    return OUTPUT_DIR, parts_dir


class StreamingParquetWriter:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.temporary_path = output_path.with_suffix(
            output_path.suffix + ".tmp"
        )
        self.writer: Optional[pq.ParquetWriter] = None
        self.schema: Optional[pa.Schema] = None
        self.rows_written = 0

    def write_dataframe(self, dataframe: pd.DataFrame) -> None:
        if dataframe.empty:
            return
        table = pa.Table.from_pandas(dataframe, preserve_index=False)
        if self.writer is None:
            self.schema = table.schema
            self.writer = pq.ParquetWriter(
                self.temporary_path,
                self.schema,
                compression="zstd",
                use_dictionary=True,
            )
        elif table.schema != self.schema:
            table = table.cast(self.schema)
        self.writer.write_table(table)
        self.rows_written += len(dataframe)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            os.replace(self.temporary_path, self.output_path)
        else:
            
            pd.DataFrame().to_parquet(
                self.temporary_path,
                index=False,
                compression="zstd",
            )
            os.replace(self.temporary_path, self.output_path)


def consolidate_parts(
    part_paths: Sequence[Path],
) -> Dict[str, Any]:
    union_output = OUTPUT_DIR / "fingerprints_pair_level_union.parquet"
    strict_output = OUTPUT_DIR / "fingerprints_pair_level_strict.parquet"
    relaxed_output = OUTPUT_DIR / "fingerprints_pair_level_relaxed.parquet"

    union_writer = StreamingParquetWriter(union_output)
    strict_writer = StreamingParquetWriter(strict_output)
    relaxed_writer = StreamingParquetWriter(relaxed_output)

    failures: List[pd.DataFrame] = []

    for index, path in enumerate(part_paths, start=1):
        frame = pd.read_parquet(path)
        frame = frame.sort_values("union_index").reset_index(drop=True)

        union_writer.write_dataframe(frame)

        ok = frame["status"].astype(str).eq("ok")
        strict_writer.write_dataframe(
            frame.loc[ok & frame["in_strict"]].copy()
        )
        relaxed_writer.write_dataframe(
            frame.loc[ok & frame["in_relaxed"]].copy()
        )

        failed = frame.loc[
            ~ok,
            [
                "union_index",
                "pair_id",
                "independent_generator_id",
                "waveform_family",
                "language",
                "fake_path",
                "real_path",
                "error",
                "in_strict",
                "in_relaxed",
            ],
        ].copy()
        if not failed.empty:
            failures.append(failed)

        if index % 50 == 0 or index == len(part_paths):
            print(
                f"[CONSOLIDATION] {index:,}/{len(part_paths):,} parts"
            )

    union_writer.close()
    strict_writer.close()
    relaxed_writer.close()

    if failures:
        failure_frame = pd.concat(failures, ignore_index=True)
    else:
        failure_frame = pd.DataFrame(
            columns=[
                "union_index",
                "pair_id",
                "independent_generator_id",
                "waveform_family",
                "language",
                "fake_path",
                "real_path",
                "error",
                "in_strict",
                "in_relaxed",
            ]
        )
    failure_frame.to_csv(
        OUTPUT_DIR / "phase1a_failures.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "union_output": str(union_output),
        "strict_output": str(strict_output),
        "relaxed_output": str(relaxed_output),
        "n_union_rows": int(union_writer.rows_written),
        "n_strict_ok_rows": int(strict_writer.rows_written),
        "n_relaxed_ok_rows": int(relaxed_writer.rows_written),
        "n_failures": int(len(failure_frame)),
    }


def build_final_report(
    manifest_report: Mapping[str, Any],
    consolidation_report: Mapping[str, Any],
    elapsed_seconds: float,
) -> Dict[str, Any]:
    union_path = Path(consolidation_report["union_output"])
    minimal = pd.read_parquet(
        union_path,
        columns=[
            "pair_id",
            "status",
            "in_strict",
            "in_relaxed",
            "independent_generator_id",
            "waveform_family",
            "language",
            "fake_vad_fallback_used",
            "real_vad_fallback_used",
        ],
    )

    if minimal["pair_id"].astype(str).duplicated().any():
        raise RuntimeError("Final output: duplicate pair_id values")

    expected_union = int(manifest_report["union_pairs"])
    if len(minimal) != expected_union:
        raise RuntimeError(
            f"Final output incomplete: {len(minimal):,}/{expected_union:,}"
        )

    ok = minimal["status"].astype(str).eq("ok")
    success_rate = float(ok.mean()) if len(minimal) else 0.0

    strict_expected = int(manifest_report["strict"]["n_pairs"])
    relaxed_expected = int(manifest_report["relaxed"]["n_pairs"])
    strict_ok = int((ok & minimal["in_strict"]).sum())
    relaxed_ok = int((ok & minimal["in_relaxed"]).sum())

    generator_family = minimal.loc[
        ok, ["independent_generator_id", "waveform_family"]
    ].drop_duplicates()
    family_counts = (
        generator_family.groupby("waveform_family")[
            "independent_generator_id"
        ].nunique()
    )

    report = {
        "version": VERSION,
        "status": "COMPLETE" if success_rate >= MIN_SUCCESS_RATE else "FAILED_GATE",
        "elapsed_seconds": float(elapsed_seconds),
        "elapsed_hours": float(elapsed_seconds / 3600.0),
        "manifest_report": dict(manifest_report),
        "config": config_payload(),
        "config_hash": stable_json_hash(config_payload()),
        "outputs": dict(consolidation_report),
        "n_union_expected": expected_union,
        "n_union_observed": int(len(minimal)),
        "n_ok": int(ok.sum()),
        "n_fail": int((~ok).sum()),
        "success_rate": success_rate,
        "minimum_success_rate": float(MIN_SUCCESS_RATE),
        "strict_expected_pairs": strict_expected,
        "strict_ok_pairs": strict_ok,
        "strict_failed_pairs": int(strict_expected - strict_ok),
        "relaxed_expected_pairs": relaxed_expected,
        "relaxed_ok_pairs": relaxed_ok,
        "relaxed_failed_pairs": int(relaxed_expected - relaxed_ok),
        "n_generators_ok": int(
            minimal.loc[ok, "independent_generator_id"].nunique()
        ),
        "n_families_ok": int(
            minimal.loc[ok, "waveform_family"].nunique()
        ),
        "family_generator_counts_ok": {
            str(key): int(value)
            for key, value in family_counts.sort_index().items()
        },
        "fake_vad_fallback_count": int(
            minimal.loc[ok, "fake_vad_fallback_used"].fillna(False).sum()
        ),
        "real_vad_fallback_count": int(
            minimal.loc[ok, "real_vad_fallback_used"].fillna(False).sum()
        ),
    }

    atomic_json_dump(report, OUTPUT_DIR / "phase1a_summary.json")

    if success_rate < MIN_SUCCESS_RATE:
        raise RuntimeError(
            f"Gate extraction failed: rate de success {success_rate:.2%}, "
            f"minimum {MIN_SUCCESS_RATE:.2%}. Consulter phase1a_failures.csv."
        )
    
    if STRICT_EXPECTED_POPULATION:
        if strict_ok != EXPECTED_STRICT_PAIRS:
            raise RuntimeError(
                f"STRICT extraction incomplete: {strict_ok:,}/{EXPECTED_STRICT_PAIRS:,}."
            )
        if relaxed_ok != EXPECTED_RELAXED_PAIRS:
            raise RuntimeError(
                f"RELAXED extraction incomplete: {relaxed_ok:,}/{EXPECTED_RELAXED_PAIRS:,}."
            )

    return report


def main() -> int:
    start_time = time.time()

    print("=" * 100)
    print("PHASE 1A — MLAAD RESIDUAL SPECTRAL FINGERPRINTS")
    print("STRICT + RELAXED are processed in a single extraction")
    print("Phase 0/0B and language analyses are not rerun.")
    print("=" * 100)

    union, manifest_report = load_manifests()
    print("[MANIFESTES]", json.dumps(manifest_report, indent=2, ensure_ascii=False))

    configuration = config_payload()
    configuration_hash = stable_json_hash(configuration)
    source_manifest_hash = stable_json_hash(
        {
            "strict": manifest_report["strict_manifest_sha256"],
            "relaxed": manifest_report["relaxed_manifest_sha256"],
        }
    )
    union_pair_ids_hash = stable_json_hash(
        union["pair_id"].astype(str).tolist()
    )

    run_metadata = {
        "version": VERSION,
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "config": configuration,
        "config_hash": configuration_hash,
        "strict_manifest": str(STRICT_MANIFEST),
        "strict_manifest_sha256": manifest_report[
            "strict_manifest_sha256"
        ],
        "relaxed_manifest": str(RELAXED_MANIFEST),
        "relaxed_manifest_sha256": manifest_report[
            "relaxed_manifest_sha256"
        ],
        "source_manifest_hash": source_manifest_hash,
        "union_pair_ids_hash": union_pair_ids_hash,
        "n_union_pairs": int(len(union)),
        "workers": int(WORKERS),
        "chunk_size": int(CHUNK_SIZE),
    }

    output_dir, parts_dir = prepare_output_directory(run_metadata)
    atomic_json_dump(environment_report(), output_dir / "environment.json")
    atomic_json_dump(manifest_report, output_dir / "manifest_validation.json")

    frequency_frame = pd.DataFrame(
        {
            "bin_index": np.arange(N_BINS, dtype=np.int32),
            "column_name": RESIDUAL_COLUMNS,
            "frequency_hz": FREQUENCIES_HZ,
        }
    )
    frequency_frame.to_csv(
        output_dir / "frequency_axis.csv",
        index=False,
    )

    worker_config = dict(configuration)
    all_part_paths: List[Path] = []
    total_parts = math.ceil(len(union) / CHUNK_SIZE)
    skipped_parts = 0
    processed_pairs = 0
    session_start = time.time()

    with ProcessPoolExecutor(
        max_workers=max(1, int(WORKERS)),
        initializer=_init_worker,
        initargs=(worker_config,),
    ) as executor:
        for part_number, (start, end) in enumerate(
            iter_slices(len(union), CHUNK_SIZE)
        ):
            chunk = union.iloc[start:end].copy()
            path = part_path(parts_dir, part_number)
            all_part_paths.append(path)

            expected_pair_ids = chunk["pair_id"].astype(str).tolist()
            if validate_existing_part(
                path,
                expected_pair_ids,
                configuration_hash,
                source_manifest_hash,
                retry_failed=RETRY_FAILED_ON_RESUME,
            ):
                skipped_parts += 1
                if (
                    skipped_parts % 50 == 0
                    or part_number + 1 == total_parts
                ):
                    print(
                        f"[RESUME] parts valid skipped: {skipped_parts:,}; "
                        f"progression {part_number + 1:,}/{total_parts:,}"
                    )
                continue

            if path.exists():
                path.unlink()

            records = chunk[CORE_METADATA_COLUMNS + [
                "union_index",
                "in_strict",
                "in_relaxed",
            ]].to_dict(orient="records")

            tasks = []
            for record in records:
                task = dict(record)
                task["config_hash"] = configuration_hash
                task["source_manifest_hash"] = source_manifest_hash
                tasks.append(task)

            results = list(
                executor.map(
                    extract_pair_worker,
                    tasks,
                    chunksize=1,
                )
            )
            result_frame = pd.DataFrame(results)
            result_frame = result_frame.sort_values(
                "union_index"
            ).reset_index(drop=True)

            if result_frame["pair_id"].astype(str).tolist() != expected_pair_ids:
                raise RuntimeError(
                    f"Part {part_number}: invalid pair_id order or identity"
                )

            ok_mask = result_frame["status"].astype(str).eq("ok")
            if ok_mask.any():
                values = result_frame.loc[ok_mask, RESIDUAL_COLUMNS].to_numpy(
                    dtype=np.float32
                )
                if not np.isfinite(values).all():
                    raise RuntimeError(
                        f"Part {part_number}: residuals not finite despite status=ok"
                    )

            atomic_parquet_dump(result_frame, path)
            processed_pairs += len(result_frame)

            elapsed = time.time() - session_start
            rate = processed_pairs / elapsed if elapsed > 0 else 0.0
            remaining_pairs = len(union) - (
                (part_number + 1) * CHUNK_SIZE
            )
            eta_seconds = (
                max(0, remaining_pairs) / rate if rate > 0 else math.inf
            )
            print(
                f"[EXTRACTION] part {part_number + 1:,}/{total_parts:,} — "
                f"session={processed_pairs:,} pairs — "
                f"{rate:.2f} pair/s — ETA={eta_seconds / 60:.1f} min — "
                f"ok={int(ok_mask.sum())}/{len(result_frame)}"
            )

            del result_frame, results, tasks, records, chunk
            gc.collect()

    missing_parts = [str(path) for path in all_part_paths if not path.is_file()]
    if missing_parts:
        raise RuntimeError(
            f"{len(missing_parts)} part(s) missing(s) after extraction"
        )

    print("\n[CONSOLIDATION] Building final Parquet files...")
    consolidation_report = consolidate_parts(all_part_paths)

    elapsed_seconds = time.time() - start_time
    final_report = build_final_report(
        manifest_report,
        consolidation_report,
        elapsed_seconds,
    )

    completion = {
        "version": VERSION,
        "status": "COMPLETE",
        "summary": str(output_dir / "phase1a_summary.json"),
        "strict_output": consolidation_report["strict_output"],
        "relaxed_output": consolidation_report["relaxed_output"],
        "frequency_axis": str(output_dir / "frequency_axis.csv"),
        "completed_utc": pd.Timestamp.utcnow().isoformat(),
    }
    atomic_json_dump(completion, output_dir / ".PHASE1A_COMPLETE.json")

    print("\n" + "=" * 100)
    print("PHASE 1A COMPLETE")
    print(f"Union processed                   : {final_report['n_union_observed']:,}")
    print(f"Success                          : {final_report['n_ok']:,}")
    print(f"Failures                          : {final_report['n_fail']:,}")
    print(f"Rate de success                  : {final_report['success_rate']:.2%}")
    print(f"STRICT exploitable              : {final_report['strict_ok_pairs']:,}")
    print(f"RELAXED exploitable             : {final_report['relaxed_ok_pairs']:,}")
    print(f"Generators                     : {final_report['n_generators_ok']}")
    print(f"Familles                        : {final_report['n_families_ok']}")
    print(f"Temps total                      : {final_report['elapsed_hours']:.2f} h")
    print(f"Output STRICT                    : {consolidation_report['strict_output']}")
    print(f"Output RELAXED                   : {consolidation_report['relaxed_output']}")
    print(f"Axis frequency                  : {output_dir / 'frequency_axis.csv'}")
    print(f"Rapport                          : {output_dir / 'phase1a_summary.json'}")
    print("=" * 100)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("\n" + "=" * 100, file=sys.stderr)
        print("[FAILURE PHASE 1A]", file=sys.stderr)
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
