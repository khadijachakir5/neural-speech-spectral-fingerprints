#!/usr/bin/env python3


# Purpose: Build and validate MLAAD v5 to M-AILABS pairs with strict metadata parsing, unambiguous reference resolution, QC, hashes, and resumable checkpoints.

from __future__ import annotations

import glob
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

warnings.filterwarnings("ignore")

SCRIPT_VERSION = "MLAAD-MAILABS-PHASE0C-v2.2-CANONICAL"


CONFIG: Dict[str, Any] = {
    
    "mlaad_root": Path("/content/drive/MyDrive/DOCTORAT/Mlaad/mlaad_v5"),
    "mailabs_root": Path("/content/drive/MyDrive/DOCTORAT/mailbs"),

    
    "output_dir": Path("/content/drive/MyDrive/fingerprint_q1_outputs/phase0_mlaad_v2_2_canonical"),

    
    "taxonomy_path": None,

    
    "meta_glob_suffix": "fake/*/*/meta.csv",
    "meta_separator": "|",
    "meta_columns": [
        "path",
        "original_file",
        "language",
        "is_original_language",
        "duration",
        "training_data",
        "model_name",
        "architecture",
        "transcript",
    ],
    "require_original_language": True,
    "expected_original_language_rows_full": 84_000,

    
    "min_pairs_per_cell": 30,

    
    "quick_test_n": None,          
    "checkpoint_every": 2000,
    "progress_every": 500,
    "resume": True,
    "force_rebuild": False,

    
    "strict_hash_audit": True,
    "exclude_ambiguous_fake_hash_duplicates": True,
    "exclude_fake_real_hash_collisions": True,
    "exclude_ambiguous_real_hash_duplicates": True,

    
    "local_checkpoint_dir": Path("/content/mlaad_phase0c_v2_1_checkpoint"),
}

QC_DEFAULTS: Dict[str, Any] = {
    "target_sr": 16000,
    "min_duration_s": 0.5,
    "max_duration_s": 30.0,
    "max_clipping_rate": 0.001,
    "max_silence_ratio": 0.85,
    "duration_ratio_bounds": (0.5, 2.0),

    
    "vad_frame_ms": 30,
    "vad_hop_ms": 10,
    "vad_reference_percentile": 95.0,
    "vad_top_db": 40.0,
    "vad_abs_db": -80.0,
}

TAXONOMY_COLUMNS = [
    "pipeline_type",
    "acoustic_model",
    "waveform_architecture",
    "waveform_family",
    "representation",
    "training_language_scope",
    "taxonomy_confidence",
    "taxonomy_source",
]


def stable_id(*parts: Any, length: int = 24) -> str:
    raw = "||".join(map(str, parts)).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:length]


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


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
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def atomic_parquet_dump(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def atomic_csv_dump(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def append_reason(current: Any, reason: str) -> str:
    existing = "" if pd.isna(current) else str(current).strip(";")
    if not existing:
        return reason
    parts = existing.split(";")
    if reason in parts:
        return existing
    return existing + ";" + reason


def parse_bool(series: pd.Series) -> pd.Series:
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    return series.astype(str).str.strip().str.lower().map(mapping)


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def environment_report() -> Dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "soundfile": package_version("soundfile"),
            "librosa": package_version("librosa"),
            "pyarrow": package_version("pyarrow"),
        },
    }


@dataclass
class AudioQCResult:
    path: str
    exists: bool = False
    loadable: bool = False
    sample_rate_original: Optional[int] = None
    duration_s: Optional[float] = None
    active_duration_s: Optional[float] = None
    silence_ratio: Optional[float] = None
    clipping_rate: Optional[float] = None
    has_nan_or_inf: Optional[bool] = None
    sha256: Optional[str] = None
    qc_status: str = "unchecked"
    exclusion_reason: str = ""


def load_audio_safe(path: Path, target_sr: int) -> Tuple[Optional[np.ndarray], Optional[int], Optional[str]]:
    try:
        wav, sr_orig = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1, dtype=np.float32)
        elif wav.ndim != 1:
            raise ValueError(f"shape audio not supported : {wav.shape}")
        wav = np.asarray(wav, dtype=np.float32)
        if sr_orig != target_sr:
            wav = librosa.resample(
                wav,
                orig_sr=int(sr_orig),
                target_sr=int(target_sr),
                res_type="kaiser_best",
            ).astype(np.float32, copy=False)
        return wav, int(sr_orig), None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}:{exc}"


def frame_energy_vad(wav: np.ndarray, sr: int, cfg: Dict[str, Any]) -> Tuple[np.ndarray, float]:
    frame_len = max(1, int(round(sr * cfg["vad_frame_ms"] / 1000.0)))
    hop_len = max(1, int(round(sr * cfg["vad_hop_ms"] / 1000.0)))

    if wav.size < frame_len:
        return np.array([], dtype=bool), 0.0

    frames = np.lib.stride_tricks.sliding_window_view(wav, frame_len)[::hop_len]
    energy = np.mean(np.square(frames, dtype=np.float64), axis=1)
    frame_db = 10.0 * np.log10(energy + 1e-12)

    reference_db = float(np.percentile(frame_db, cfg["vad_reference_percentile"]))
    threshold_db = max(reference_db - cfg["vad_top_db"], cfg["vad_abs_db"])
    active_mask = frame_db >= threshold_db
    active_duration = float(active_mask.sum() * hop_len / sr)
    return active_mask, active_duration


def compute_qc(path: Path, cfg: Dict[str, Any]) -> AudioQCResult:
    result = AudioQCResult(path=str(path))

    if not path.exists():
        result.qc_status = "fail"
        result.exclusion_reason = "file_not_found"
        return result

    result.exists = True
    wav, sr_orig, err = load_audio_safe(path, int(cfg["target_sr"]))

    if err is not None or wav is None:
        result.qc_status = "fail"
        result.exclusion_reason = f"load_error:{err}"
        return result

    result.loadable = True
    result.sample_rate_original = sr_orig

    if wav.size == 0:
        result.qc_status = "fail"
        result.exclusion_reason = "empty_audio"
        return result

    has_nan_inf = bool(~np.isfinite(wav).all())
    result.has_nan_or_inf = has_nan_inf
    if has_nan_inf:
        result.qc_status = "fail"
        result.exclusion_reason = "nan_or_inf_samples"
        return result

    duration_s = float(wav.size / cfg["target_sr"])
    clipping_rate = float(np.mean(np.abs(wav) >= 0.999))
    _, active_duration = frame_energy_vad(wav, int(cfg["target_sr"]), cfg)
    silence_ratio = float(max(0.0, 1.0 - active_duration / duration_s)) if duration_s > 0 else 1.0

    result.duration_s = duration_s
    result.active_duration_s = active_duration
    result.silence_ratio = silence_ratio
    result.clipping_rate = clipping_rate
    result.sha256 = sha256_of_file(path)

    reasons: List[str] = []
    if duration_s < cfg["min_duration_s"]:
        reasons.append("duration_below_min")
    if duration_s > cfg["max_duration_s"]:
        reasons.append("duration_above_max")
    if clipping_rate > cfg["max_clipping_rate"]:
        reasons.append("excessive_clipping")
    if silence_ratio > cfg["max_silence_ratio"]:
        reasons.append("excessive_silence")

    result.qc_status = "fail" if reasons else "ok"
    result.exclusion_reason = ";".join(reasons)
    return result


def duration_ratio_ok(fake_dur: Optional[float], real_dur: Optional[float], cfg: Dict[str, Any]) -> bool:
    if fake_dur is None or real_dur is None or real_dur <= 0:
        return False
    lo, hi = cfg["duration_ratio_bounds"]
    return float(lo) <= float(fake_dur / real_dur) <= float(hi)


def resolve_mailabs_path(
    original_file: str,
    mailabs_root: Path,
    cache: Dict[str, Tuple[str, bool, str]],
) -> Tuple[Path, bool, str]:


    original_file = str(original_file).replace("\\", "/").lstrip("./")
    if original_file in cache:
        p, ok, method = cache[original_file]
        return Path(p), bool(ok), str(method)

    direct = mailabs_root / original_file
    if direct.is_file():
        result = (str(direct), True, "direct")
        cache[original_file] = result
        return direct, True, "direct"

    parts = original_file.split("/", 1)
    if len(parts) != 2:
        result = (str(direct), False, "not_found")
        cache[original_file] = result
        return direct, False, "not_found"

    first, rest = parts
    first_dir = mailabs_root / first
    if not first_dir.is_dir():
        result = (str(direct), False, "not_found")
        cache[original_file] = result
        return direct, False, "not_found"

    subdirs = sorted(p for p in first_dir.iterdir() if p.is_dir())
    matches = [p / rest for p in subdirs if (p / rest).is_file()]
    
    
    if len(matches) == 1:
        candidate = matches[0]
        result = (str(candidate), True, "inserted_second_level_unique")
        cache[original_file] = result
        return candidate, True, "inserted_second_level_unique"
    if len(matches) > 1:
        result = (str(direct), False, "ambiguous_multiple_matches")
        cache[original_file] = result
        return direct, False, "ambiguous_multiple_matches"

    result = (str(direct), False, "not_found")
    cache[original_file] = result
    return direct, False, "not_found"

def infer_speaker_id(real_path: Path, mailabs_root: Path) -> Tuple[str, str]:


    try:
        rel_parts = real_path.resolve().relative_to(mailabs_root.resolve()).parts
    except Exception:
        rel_parts = real_path.parts

    if "by_book" in rel_parts:
        idx = rel_parts.index("by_book")
        if idx + 1 < len(rel_parts):
            return str(rel_parts[idx + 1]), "path_after_by_book"

    return "unknown", "unresolved"


def discover_meta_files(mlaad_root: Path) -> List[Path]:
    pattern = str(mlaad_root / CONFIG["meta_glob_suffix"])
    return [Path(p) for p in sorted(glob.glob(pattern))]


def build_input_signature(meta_files: Sequence[Path]) -> Dict[str, Any]:
    file_records = []
    for p in meta_files:
        stat = p.stat()
        file_records.append(
            {
                "path": str(p),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": sha256_of_file(p),
            }
        )

    relevant_config = {
        "script_version": SCRIPT_VERSION,
        "mlaad_root": str(CONFIG["mlaad_root"]),
        "mailabs_root": str(CONFIG["mailabs_root"]),
        "meta_glob_suffix": CONFIG["meta_glob_suffix"],
        "meta_separator": CONFIG["meta_separator"],
        "meta_columns": CONFIG["meta_columns"],
        "require_original_language": CONFIG["require_original_language"],
        "expected_original_language_rows_full": CONFIG["expected_original_language_rows_full"],
        "quick_test_n": CONFIG["quick_test_n"],
        "qc": QC_DEFAULTS,
    }

    return {
        "meta_files": file_records,
        "meta_files_hash": stable_json_hash(file_records),
        "config_hash": stable_json_hash(relevant_config),
        "config": relevant_config,
    }


def read_all_meta(meta_files: Sequence[Path], mlaad_root: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for file in meta_files:
        try:
            df = pd.read_csv(
                file,
                sep=CONFIG["meta_separator"],
                names=CONFIG["meta_columns"],
                header=None,
                engine="python",
                quoting=3,
                on_bad_lines="error",
            )

            parts = file.parts
            idx = parts.index("fake")
            df["lang_folder"] = parts[idx + 1]
            df["generator_dir"] = parts[idx + 2]
            df["meta_file"] = str(file)
            df["meta_row_number"] = np.arange(len(df), dtype=np.int64)
            frames.append(df)
        except Exception as exc:
            raise RuntimeError(f"Error de reading metadata MLAAD {file}: {type(exc).__name__}: {exc}") from exc

    if not frames:
        raise RuntimeError("No readable meta.csv file.")

    meta = pd.concat(frames, ignore_index=True)
    parsed_original = parse_bool(meta["is_original_language"])
    if parsed_original.isna().any():
        raise RuntimeError(f"{int(parsed_original.isna().sum())} is_original_language values are not parseable in the meta.csv files.")
    meta["is_original_language"] = parsed_original

    
    row_lang = meta["language"].astype(str).str.strip().str.lower()
    folder_lang = meta["lang_folder"].astype(str).str.strip().str.lower()
    language_mismatch = row_lang.ne("") & row_lang.ne(folder_lang)
    if language_mismatch.any():
        sample = meta.loc[language_mismatch, ["meta_file", "path", "language", "lang_folder"]].head(10).to_dict(orient="records")
        raise RuntimeError(f"{int(language_mismatch.sum())} language metadata/directory inconsistencies. Examples: {sample}")

    if CONFIG["require_original_language"]:
        meta = meta.loc[meta["is_original_language"].eq(True)].copy()  
        if CONFIG["quick_test_n"] is None and len(meta) != int(CONFIG["expected_original_language_rows_full"]):
            raise RuntimeError(
                f"Population MLAAD language original unexpected: {len(meta):,}; "
                f"{int(CONFIG['expected_original_language_rows_full']):,} expected."
            )

    meta["source_row_id"] = [
        stable_id(
            "MLAAD_META_ROW",
            row.meta_file,
            row.meta_row_number,
            row.path,
            row.original_file,
            row.generator_dir,
        )
        for row in meta.itertuples(index=False)
    ]

    if meta["source_row_id"].duplicated().any():
        raise RuntimeError("Duplicate source_row_id values after reading meta.csv files.")

    meta = meta.sort_values(
        ["meta_file", "meta_row_number", "source_row_id"]
    ).reset_index(drop=True)
    meta["source_index"] = np.arange(len(meta), dtype=np.int64)

    if CONFIG["quick_test_n"] is not None:
        n = min(int(CONFIG["quick_test_n"]), len(meta))
        meta = meta.head(n).copy().reset_index(drop=True)
        meta["source_index"] = np.arange(len(meta), dtype=np.int64)
        print(f"[MODE TEST] {len(meta)} rows selected.")

    return meta


def checkpoint_paths(output_dir: Path) -> Dict[str, Path]:
    local_dir = Path(CONFIG["local_checkpoint_dir"])
    drive_dir = output_dir / "_checkpoint_v2_1"
    return {
        "local_dir": local_dir,
        "drive_dir": drive_dir,
        "local_rows": local_dir / "rows.parquet",
        "local_meta": local_dir / "metadata.json",
        "drive_rows": drive_dir / "rows.parquet",
        "drive_meta": drive_dir / "metadata.json",
    }


def restore_checkpoint_if_needed(paths: Dict[str, Path]) -> None:
    paths["local_dir"].mkdir(parents=True, exist_ok=True)
    if not paths["local_rows"].exists() and paths["drive_rows"].exists():
        shutil.copy2(paths["drive_rows"], paths["local_rows"])
    if not paths["local_meta"].exists() and paths["drive_meta"].exists():
        shutil.copy2(paths["drive_meta"], paths["local_meta"])


def save_checkpoint(rows_df: pd.DataFrame, metadata: Dict[str, Any], paths: Dict[str, Path]) -> None:
    paths["local_dir"].mkdir(parents=True, exist_ok=True)
    paths["drive_dir"].mkdir(parents=True, exist_ok=True)

    atomic_parquet_dump(rows_df, paths["local_rows"])
    atomic_json_dump(metadata, paths["local_meta"])

    shutil.copy2(paths["local_rows"], paths["drive_rows"])
    shutil.copy2(paths["local_meta"], paths["drive_meta"])


def load_valid_checkpoint(
    paths: Dict[str, Path],
    input_signature: Dict[str, Any],
    expected_source_ids: set,
) -> pd.DataFrame:
    if not CONFIG["resume"]:
        return pd.DataFrame()

    restore_checkpoint_if_needed(paths)
    if not paths["local_rows"].exists() or not paths["local_meta"].exists():
        return pd.DataFrame()

    metadata = json.loads(paths["local_meta"].read_text(encoding="utf-8"))
    if metadata.get("meta_files_hash") != input_signature["meta_files_hash"]:
        raise RuntimeError(
            "The checkpoint corresponds to a different meta.csv list/version. "
            "Use force_rebuild=True or a new output_dir."
        )
    if metadata.get("config_hash") != input_signature["config_hash"]:
        raise RuntimeError(
            "The checkpoint corresponds to a different configuration. "
            "Use force_rebuild=True or a new output_dir."
        )

    rows_df = pd.read_parquet(paths["local_rows"])
    if "source_row_id" not in rows_df.columns:
        raise RuntimeError("Old/incompatible checkpoint: source_row_id is missing.")

    rows_df = rows_df.drop_duplicates("source_row_id", keep="last")
    unknown = set(rows_df["source_row_id"].astype(str)) - expected_source_ids
    if unknown:
        raise RuntimeError("The checkpoint contains rows missing from the current input.")

    print(f"[RESUME] {len(rows_df)} rows valid restored.")
    return rows_df


def cleanup_checkpoint(paths: Dict[str, Path]) -> None:
    for key in ["local_dir", "drive_dir"]:
        p = paths[key]
        if p.exists():
            shutil.rmtree(p)


def load_taxonomy(path_value: Any) -> Optional[pd.DataFrame]:
    if path_value in (None, "", "None"):
        return None

    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Taxonomy not found : {path}")

    taxonomy = pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)

    key_candidates = ["independent_generator_id", "generator_key", "generator_dir"]
    key = next((c for c in key_candidates if c in taxonomy.columns), None)
    if key is None:
        raise ValueError(
            "The taxonomy must contain independent_generator_id, generator_key, or generator_dir."
        )

    taxonomy = taxonomy.rename(columns={key: "independent_generator_id"}).copy()
    taxonomy["independent_generator_id"] = taxonomy["independent_generator_id"].astype(str)

    if taxonomy["independent_generator_id"].duplicated().any():
        dup = taxonomy.loc[
            taxonomy["independent_generator_id"].duplicated(False),
            "independent_generator_id",
        ].tolist()
        raise ValueError(f"Duplicate generators in the taxonomy : {dup[:10]}")

    for col in TAXONOMY_COLUMNS:
        if col not in taxonomy.columns:
            taxonomy[col] = pd.NA

    return taxonomy[["independent_generator_id", *TAXONOMY_COLUMNS]].copy()


def attach_taxonomy(df: pd.DataFrame, taxonomy: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = df.copy()

    for col in TAXONOMY_COLUMNS:
        if col in out.columns:
            out = out.drop(columns=[col])

    if taxonomy is None:
        for col in TAXONOMY_COLUMNS:
            out[col] = pd.Series(pd.NA, index=out.index, dtype="string")
        return out, {
            "taxonomy_attached": False,
            "n_generators_total": int(out["independent_generator_id"].nunique()),
            "n_generators_taxonomized": 0,
            "n_generators_missing_taxonomy": int(out["independent_generator_id"].nunique()),
            "missing_generators": sorted(out["independent_generator_id"].astype(str).unique().tolist()),
        }

    out = out.merge(taxonomy, on="independent_generator_id", how="left", validate="many_to_one")
    generators = out[["independent_generator_id", *TAXONOMY_COLUMNS]].drop_duplicates()
    complete = generators["waveform_family"].notna() & generators["waveform_architecture"].notna()
    missing = generators.loc[~complete, "independent_generator_id"].astype(str).tolist()

    return out, {
        "taxonomy_attached": True,
        "n_generators_total": int(len(generators)),
        "n_generators_taxonomized": int(complete.sum()),
        "n_generators_missing_taxonomy": int((~complete).sum()),
        "missing_generators": sorted(missing),
    }


def apply_hash_audit(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    out = df.copy()
    audit_rows: List[Dict[str, Any]] = []

    
    current_ok = out["qc_status"].eq("ok")

    
    fake_non_null = out.loc[current_ok & out["fake_sha256"].notna()].copy()
    fake_groups = fake_non_null.groupby("fake_sha256", observed=True)
    ambiguous_fake_hashes = []
    for sha, group in fake_groups:
        n_originals = int(group["original_id"].nunique())
        n_generators = int(group["independent_generator_id"].nunique())
        if n_originals > 1 or n_generators > 1:
            ambiguous_fake_hashes.append(str(sha))
            audit_rows.append(
                {
                    "audit_type": "ambiguous_fake_hash",
                    "sha256": str(sha),
                    "n_rows": int(len(group)),
                    "n_original_ids": n_originals,
                    "n_generators": n_generators,
                    "action": "exclude_all_rows" if CONFIG["exclude_ambiguous_fake_hash_duplicates"] else "report_only",
                }
            )

    if ambiguous_fake_hashes and CONFIG["exclude_ambiguous_fake_hash_duplicates"]:
        mask = out["fake_sha256"].isin(ambiguous_fake_hashes) & out["qc_status"].eq("ok")
        out.loc[mask, "qc_status"] = "fail"
        out.loc[mask, "exclusion_reason"] = out.loc[mask, "exclusion_reason"].map(
            lambda x: append_reason(x, "ambiguous_fake_hash_duplicate")
        )

    
    current_ok = out["qc_status"].eq("ok")
    real_non_null = out.loc[current_ok & out["real_sha256"].notna()].copy()
    real_groups = real_non_null.groupby("real_sha256", observed=True)
    ambiguous_real_hashes = []
    for sha, group in real_groups:
        n_originals = int(group["original_id"].nunique())
        if n_originals > 1:
            ambiguous_real_hashes.append(str(sha))
            audit_rows.append(
                {
                    "audit_type": "ambiguous_real_hash",
                    "sha256": str(sha),
                    "n_rows": int(len(group)),
                    "n_original_ids": n_originals,
                    "n_generators": int(group["independent_generator_id"].nunique()),
                    "action": "exclude_all_rows" if CONFIG["exclude_ambiguous_real_hash_duplicates"] else "report_only",
                }
            )

    if ambiguous_real_hashes and CONFIG["exclude_ambiguous_real_hash_duplicates"]:
        mask = out["real_sha256"].isin(ambiguous_real_hashes) & out["qc_status"].eq("ok")
        out.loc[mask, "qc_status"] = "fail"
        out.loc[mask, "exclusion_reason"] = out.loc[mask, "exclusion_reason"].map(
            lambda x: append_reason(x, "ambiguous_real_hash_duplicate")
        )

    
    current_ok = out["qc_status"].eq("ok")
    fake_hashes = set(out.loc[current_ok, "fake_sha256"].dropna().astype(str))
    real_hashes = set(out.loc[current_ok, "real_sha256"].dropna().astype(str))
    collisions = sorted(fake_hashes.intersection(real_hashes))

    for sha in collisions:
        fake_rows = out.loc[out["fake_sha256"].astype(str).eq(sha)]
        real_rows = out.loc[out["real_sha256"].astype(str).eq(sha)]
        audit_rows.append(
            {
                "audit_type": "fake_real_hash_collision",
                "sha256": sha,
                "n_rows": int(len(fake_rows) + len(real_rows)),
                "n_original_ids": int(
                    pd.concat([fake_rows["original_id"], real_rows["original_id"]]).nunique()
                ),
                "n_generators": int(fake_rows["independent_generator_id"].nunique()),
                "action": "exclude_fake_rows" if CONFIG["exclude_fake_real_hash_collisions"] else "report_only",
            }
        )

    if collisions and CONFIG["exclude_fake_real_hash_collisions"]:
        mask = out["fake_sha256"].isin(collisions) & out["qc_status"].eq("ok")
        out.loc[mask, "qc_status"] = "fail"
        out.loc[mask, "exclusion_reason"] = out.loc[mask, "exclusion_reason"].map(
            lambda x: append_reason(x, "fake_real_hash_collision")
        )

    audit_df = pd.DataFrame(audit_rows)
    if audit_df.empty:
        audit_df = pd.DataFrame(
            columns=[
                "audit_type",
                "sha256",
                "n_rows",
                "n_original_ids",
                "n_generators",
                "action",
            ]
        )

    summary = {
        "strict_hash_audit": bool(CONFIG["strict_hash_audit"]),
        "n_ambiguous_fake_hashes": int(len(ambiguous_fake_hashes)),
        "n_ambiguous_real_hashes": int(len(ambiguous_real_hashes)),
        "n_fake_real_hash_collisions": int(len(collisions)),
        "n_rows_excluded_by_hash_audit": int(
            out["exclusion_reason"].fillna("").str.contains(
                "ambiguous_fake_hash_duplicate|ambiguous_real_hash_duplicate|fake_real_hash_collision",
                regex=True,
            ).sum()
        ),
    }

    if CONFIG["strict_hash_audit"]:
        unresolved = []
        if ambiguous_fake_hashes and not CONFIG["exclude_ambiguous_fake_hash_duplicates"]:
            unresolved.append("ambiguous_fake_hashes")
        if ambiguous_real_hashes and not CONFIG["exclude_ambiguous_real_hash_duplicates"]:
            unresolved.append("ambiguous_real_hashes")
        if collisions and not CONFIG["exclude_fake_real_hash_collisions"]:
            unresolved.append("fake_real_hash_collisions")
        if unresolved:
            raise RuntimeError(
                "Strict hash audit: anomalies detected but not excluded: " + ", ".join(unresolved)
            )

    return out, audit_df, summary


def validate_manifest_constraints(df: pd.DataFrame) -> None:
    if df.empty:
        raise RuntimeError("Final manifest is empty.")

    if df["source_row_id"].duplicated().any():
        raise RuntimeError("Duplicate source_row_id values in the final manifest.")

    if df["pair_id"].duplicated().any():
        duplicates = int(df["pair_id"].duplicated().sum())
        raise RuntimeError(f"pair_id duplicate : {duplicates}")

    duplicated_pairs = df.duplicated(["independent_generator_id", "original_id"]).sum()
    if duplicated_pairs:
        raise RuntimeError(
            f"{int(duplicated_pairs)} duplicates (independent_generator_id, original_id)."
        )


def build_reports(
    df: pd.DataFrame,
    output_dir: Path,
    input_signature: Dict[str, Any],
    taxonomy_summary: Dict[str, Any],
    hash_summary: Dict[str, Any],
) -> Dict[str, Any]:
    df_ok = df.loc[df["qc_status"].eq("ok")].copy()

    resolve_stats = {
        str(k): int(v)
        for k, v in df["pairing_method"].fillna("unknown").value_counts().to_dict().items()
    }
    if sum(resolve_stats.values()) != len(df):
        raise RuntimeError("Resolution methods do not cover all rows.")

    exclusion_counts = {
        str(k): int(v)
        for k, v in (
            df.loc[df["qc_status"].eq("fail"), "exclusion_reason"]
            .fillna("unknown")
            .value_counts()
            .to_dict()
            .items()
        )
    }

    cell_counts = (
        df_ok.groupby(
            ["independent_generator_id", "language"],
            observed=True,
        )
        .size()
        .reset_index(name="n_valid_pairs")
    )
    cell_counts["meets_min_pairs"] = cell_counts["n_valid_pairs"] >= CONFIG["min_pairs_per_cell"]
    atomic_csv_dump(cell_counts, output_dir / "mlaad_mailabs_cell_eligibility.csv")

    overlap_all = df.groupby("original_id")["independent_generator_id"].nunique()
    overlap_valid = df_ok.groupby("original_id")["independent_generator_id"].nunique()

    shared_all = overlap_all[overlap_all > 1]
    shared_valid = overlap_valid[overlap_valid > 1]

    overlap_report = {
        "all_candidates": {
            "n_unique_originals": int(overlap_all.size),
            "n_originals_shared_across_generators": int(len(shared_all)),
            "max_generators_sharing_one_original": int(overlap_all.max()) if len(overlap_all) else 0,
            "generator_coverage_distribution": {
                str(k): int(v) for k, v in overlap_all.value_counts().sort_index().to_dict().items()
            },
        },
        "qc_valid_only": {
            "n_unique_originals": int(overlap_valid.size),
            "n_originals_shared_across_generators": int(len(shared_valid)),
            "max_generators_sharing_one_original": int(overlap_valid.max()) if len(overlap_valid) else 0,
            "generator_coverage_distribution": {
                str(k): int(v) for k, v in overlap_valid.value_counts().sort_index().to_dict().items()
            },
        },
    }
    atomic_json_dump(overlap_report, output_dir / "mlaad_mailabs_overlap_report.json")

    generator_language_counts = (
        df_ok.groupby("independent_generator_id")["language"]
        .nunique()
        .sort_values(ascending=False)
        .rename("n_languages")
        .reset_index()
    )
    atomic_csv_dump(generator_language_counts, output_dir / "generator_language_coverage.csv")

    summary = {
        "script_version": SCRIPT_VERSION,
        "report_source": "recomputed_from_final_manifest",
        "meta_files_hash": input_signature["meta_files_hash"],
        "config_hash": input_signature["config_hash"],
        "total_pairs_candidates": int(len(df)),
        "n_ok": int(len(df_ok)),
        "n_fail": int(len(df) - len(df_ok)),
        "fail_rate": float((len(df) - len(df_ok)) / len(df)) if len(df) else None,
        "resolve_method_counts": resolve_stats,
        "exclusion_reason_counts": exclusion_counts,
        "n_generators_detected": int(df["independent_generator_id"].nunique()),
        "n_languages_detected": int(df["language"].nunique()),
        "n_cells_total": int(len(cell_counts)),
        "n_cells_meeting_min_pairs": int(cell_counts["meets_min_pairs"].sum()),
        "n_unique_originals_qc_valid": int(df_ok["original_id"].nunique()),
        "n_originals_shared_across_generators_qc_valid": int(len(shared_valid)),
        "max_generators_sharing_one_original_qc_valid": int(overlap_valid.max()) if len(overlap_valid) else 0,
        "taxonomy": taxonomy_summary,
        "hash_audit": hash_summary,
    }

    atomic_json_dump(summary, output_dir / "mlaad_mailabs_pairing_report.json")
    return summary


def build_and_validate() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    mlaad_root = Path(CONFIG["mlaad_root"])
    mailabs_root = Path(CONFIG["mailabs_root"])
    output_dir = Path(CONFIG["output_dir"])

    if not mlaad_root.exists():
        raise FileNotFoundError(f"mlaad_root not found : {mlaad_root}")
    if not mailabs_root.exists():
        raise FileNotFoundError(f"mailabs_root not found : {mailabs_root}")

    if CONFIG["force_rebuild"] and output_dir.exists():
        print(f"[FORCE] Removing directory de output : {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = checkpoint_paths(output_dir)

    if CONFIG["force_rebuild"]:
        cleanup_checkpoint(paths)

    meta_files = discover_meta_files(mlaad_root)
    if not meta_files:
        raise RuntimeError(
            f"No meta.csv file detected with : {mlaad_root / CONFIG['meta_glob_suffix']}"
        )

    print("=" * 88)
    print("PHASE 0C — MLAAD v5 <-> M-AILABS")
    print("=" * 88)
    print(f"Version       : {SCRIPT_VERSION}")
    print(f"Metadata files detected : {len(meta_files)}")
    print(f"MLAAD         : {mlaad_root}")
    print(f"M-AILABS      : {mailabs_root}")
    print(f"Outputs       : {output_dir}")

    input_signature = build_input_signature(meta_files)
    meta = read_all_meta(meta_files, mlaad_root)
    print(f"Rows retained after original-language filter: {len(meta)}")

    expected_source_ids = set(meta["source_row_id"].astype(str))
    checkpoint_df = load_valid_checkpoint(paths, input_signature, expected_source_ids)
    processed_ids = set(checkpoint_df["source_row_id"].astype(str)) if not checkpoint_df.empty else set()

    rows: List[Dict[str, Any]] = checkpoint_df.to_dict(orient="records") if not checkpoint_df.empty else []
    resolver_cache: Dict[str, Tuple[str, bool, str]] = {}
    real_qc_cache: Dict[str, AudioQCResult] = {}

    remaining = meta.loc[~meta["source_row_id"].isin(processed_ids)].copy()
    total = len(meta)
    already = len(processed_ids)
    print(f"To process : {len(remaining)} ; already processed : {already}")

    start_time = time.time()
    processed_session = 0

    for row in remaining.itertuples(index=False):
        raw_fake = str(row.path).replace("\\", "/").lstrip("./")
        fake_path = mlaad_root / raw_fake

        real_path, real_exists, pairing_method = resolve_mailabs_path(
            str(row.original_file), mailabs_root, resolver_cache
        )

        fake_qc = compute_qc(fake_path, QC_DEFAULTS)

        real_cache_key = str(real_path)
        if real_exists:
            if real_cache_key not in real_qc_cache:
                real_qc_cache[real_cache_key] = compute_qc(real_path, QC_DEFAULTS)
            real_qc = real_qc_cache[real_cache_key]
        else:
            real_qc = AudioQCResult(
                path=str(real_path),
                exists=False,
                loadable=False,
                qc_status="fail",
                exclusion_reason=(
                    "mailabs_resolution_ambiguous"
                    if pairing_method == "ambiguous_multiple_matches"
                    else "mailabs_resolution_failed"
                ),
            )

        pair_status = "ok" if fake_qc.qc_status == "ok" and real_qc.qc_status == "ok" else "fail"
        reasons: List[str] = []
        if fake_qc.qc_status == "fail":
            reasons.append(f"fake:{fake_qc.exclusion_reason}")
        if real_qc.qc_status == "fail":
            reasons.append(f"real:{real_qc.exclusion_reason}")

        duration_ratio = None
        if fake_qc.duration_s is not None and real_qc.duration_s not in (None, 0):
            duration_ratio = float(fake_qc.duration_s / real_qc.duration_s)

        if pair_status == "ok" and not duration_ratio_ok(
            fake_qc.duration_s, real_qc.duration_s, QC_DEFAULTS
        ):
            pair_status = "fail"
            reasons.append("duration_ratio_out_of_bounds")

        generator_key = str(row.generator_dir)
        language = str(row.lang_folder)
        original_id = stable_id("MLAAD", str(row.original_file))
        speaker_id, speaker_source = infer_speaker_id(real_path, mailabs_root)

        record = {
            "source_index": int(row.source_index),
            "source_row_id": str(row.source_row_id),
            "meta_file": str(row.meta_file),
            "meta_row_number": int(row.meta_row_number),
            "pair_id": stable_id("MLAAD", generator_key, str(fake_path), str(real_path)),
            "dataset": "MLAAD_v5",
            "domain": "MLAAD-MAILABS",
            "generator_key": generator_key,
            "independent_generator_id": generator_key,
            "language": language,
            "speaker_id": speaker_id,
            "speaker_id_source": speaker_source,
            "original_id": original_id,
            "original_file_meta": str(row.original_file),
            "fake_path": str(fake_path),
            "real_path": str(real_path),
            "mailabs_first_dir_available": bool((mailabs_root / str(row.original_file).split("/")[0]).exists()),
            "pairing_method": pairing_method,
            "fake_duration": fake_qc.duration_s,
            "real_duration": real_qc.duration_s,
            "duration_ratio": duration_ratio,
            "active_duration_fake": fake_qc.active_duration_s,
            "active_duration_real": real_qc.active_duration_s,
            "clipping_rate_fake": fake_qc.clipping_rate,
            "clipping_rate_real": real_qc.clipping_rate,
            "silence_ratio_fake": fake_qc.silence_ratio,
            "silence_ratio_real": real_qc.silence_ratio,
            "sample_rate_fake_original": fake_qc.sample_rate_original,
            "sample_rate_real_original": real_qc.sample_rate_original,
            "fake_sha256": fake_qc.sha256,
            "real_sha256": real_qc.sha256,
            "model_name_meta": str(row.model_name),
            "architecture_meta": str(row.architecture),
            "training_data_meta": str(row.training_data),
            "transcript_meta": str(row.transcript),
            "qc_status": pair_status,
            "exclusion_reason": ";".join(reasons),
        }

        rows.append(record)
        processed_session += 1
        completed = already + processed_session

        if completed % int(CONFIG["progress_every"]) == 0 or completed == total:
            elapsed = time.time() - start_time
            rate = processed_session / elapsed if elapsed > 0 else 0.0
            eta = (total - completed) / rate if rate > 0 else math.inf
            print(f"[{completed}/{total}] {rate:.2f} pair/s — ETA {eta / 60:.1f} min")

        if completed % int(CONFIG["checkpoint_every"]) == 0:
            ckpt_df = pd.DataFrame(rows).drop_duplicates("source_row_id", keep="last")
            ckpt_meta = {
                "script_version": SCRIPT_VERSION,
                "created_utc": pd.Timestamp.utcnow().isoformat(),
                "meta_files_hash": input_signature["meta_files_hash"],
                "config_hash": input_signature["config_hash"],
                "n_expected_rows": int(total),
                "n_checkpoint_rows": int(len(ckpt_df)),
            }
            save_checkpoint(ckpt_df, ckpt_meta, paths)
            print(f"[CHECKPOINT] {len(ckpt_df)} rows saved.")

    if not rows:
        raise RuntimeError("No pair was built.")

    df = pd.DataFrame(rows)
    df = df.drop_duplicates("source_row_id", keep="last").sort_values("source_index").reset_index(drop=True)

    if len(df) != len(meta):
        missing_ids = expected_source_ids - set(df["source_row_id"].astype(str))
        raise RuntimeError(
            f"Incomplete manifest: {len(df)} rows out of {len(meta)} ; "
            f"{len(missing_ids)} source_row_id values are missing."
        )

    if CONFIG["quick_test_n"] is None and len(df) != int(CONFIG["expected_original_language_rows_full"]):
        raise RuntimeError(f"Manifest MLAAD incomplete: {len(df):,}/{int(CONFIG['expected_original_language_rows_full']):,}")
    n_ambiguous = int(df["pairing_method"].eq("ambiguous_multiple_matches").sum())
    if n_ambiguous:
        print(f"[AUDIT] {n_ambiguous} ambiguous M-AILABS rows have been explicitly marked FAIL (no arbitrary fallback).")

    
    taxonomy = load_taxonomy(CONFIG["taxonomy_path"])
    df, taxonomy_summary = attach_taxonomy(df, taxonomy)

    
    df, hash_audit_df, hash_summary = apply_hash_audit(df)

    
    validate_manifest_constraints(df)

    
    full_manifest_path = output_dir / "mlaad_mailabs_manifest_v2_2_canonical.parquet"
    valid_manifest_path = output_dir / "mlaad_mailabs_manifest_v2_2_canonical_qc_ok.parquet"

    atomic_parquet_dump(df, full_manifest_path)
    atomic_parquet_dump(df.loc[df["qc_status"].eq("ok")].copy(), valid_manifest_path)
    atomic_csv_dump(hash_audit_df, output_dir / "audio_hash_duplicate_audit.csv")
    atomic_json_dump(hash_summary, output_dir / "audio_hash_duplicate_audit_summary.json")
    atomic_json_dump(taxonomy_summary, output_dir / "taxonomy_coverage_report.json")
    atomic_json_dump(environment_report(), output_dir / "environment.json")
    atomic_json_dump(input_signature, output_dir / "input_signature.json")

    summary = build_reports(
        df,
        output_dir,
        input_signature,
        taxonomy_summary,
        hash_summary,
    )

    
    cleanup_checkpoint(paths)

    print("\n" + "=" * 88)
    print("FINAL SUMMARY — MLAAD v5 / M-AILABS")
    print("=" * 88)
    print(f"Candidate pairs : {summary['total_pairs_candidates']}")
    print(f"QC-valid pairs : {summary['n_ok']}")
    print(f"Excluded pairs    : {summary['n_fail']} ({summary['fail_rate']:.2%})")
    print(f"Resolution        : {summary['resolve_method_counts']}")
    print(
        f"Cells >= {CONFIG['min_pairs_per_cell']} pairs : "
        f"{summary['n_cells_meeting_min_pairs']} / {summary['n_cells_total']}"
    )
    print(f"Generators       : {summary['n_generators_detected']}")
    print(f"Languages           : {summary['n_languages_detected']}")
    print(f"Originals valid : {summary['n_unique_originals_qc_valid']}")
    print(
        "Valid originals shared across generators : "
        f"{summary['n_originals_shared_across_generators_qc_valid']}"
    )
    print(f"Taxonomy attached  : {summary['taxonomy']['taxonomy_attached']}")
    print(f"Manifest full : {full_manifest_path}")
    print(f"Manifest QC=ok   : {valid_manifest_path}")
    print(f"Rapport principal : {output_dir / 'mlaad_mailabs_pairing_report.json'}")

    return df, summary


if __name__ == "__main__":
    mlaad_df, mlaad_summary = build_and_validate()
