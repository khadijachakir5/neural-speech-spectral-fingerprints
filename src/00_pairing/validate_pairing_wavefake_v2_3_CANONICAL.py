#!/usr/bin/env python3


# Purpose: Build and validate WaveFake LJSpeech and JSUT fake-to-real pairs with content matching, QC, hashes, and resumable checkpoints.

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional


PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "soundfile": "soundfile",
    "scipy": "scipy",
    "pyarrow": "pyarrow",
}
missing = [pip_name for module, pip_name in PACKAGES.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

warnings.filterwarnings("ignore")


try:
    from google.colab import drive  

    drive.mount("/content/drive", force_remount=False)
except Exception:
    pass


CONFIG = {
    "wavefake_root": Path(
        "/content/drive/MyDrive/Datasets/archive/generated_audio"
    ),
    "ljspeech_real_root": Path(
        "/content/drive/MyDrive/Datasets/LJSpeech-1.1/wavs"
    ),
    "jsut_real_root": Path(
        "/content/drive/MyDrive/Datasets/jsut_ver1.1/basic5000/wav"
    ),
    "output_dir": Path(
        "/content/drive/MyDrive/fingerprint_q1_outputs/phase0_wavefake_v3_canonical"
    ),

    
    "quick_test_n_per_generator": None,

    "checkpoint_every": 2000,
    "progress_every": 1000,

    
    "force_rebuild": False,

    
    "strict_hash_audit": True,
}


QC_CONFIG = {
    "target_sr": 16000,
    "min_duration_s": 0.5,
    "max_duration_s": 30.0,
    "max_clipping_rate": 0.001,
    "max_silence_ratio": 0.85,
    "duration_ratio_bounds": (0.5, 2.0),
    "vad_frame_ms": 30,
    "vad_top_db": 40.0,
    "vad_abs_db": -80.0,
    "vad_reference_percentile": 95.0,
}


WAVEFAKE_GENERATORS = {
    "ljspeech_hifiGAN": {
        "independent_generator_id": "HiFiGAN_LJ",
        "waveform_architecture": "HiFi-GAN",
        "waveform_family": "GAN",
        "language": "en",
        "real_root_key": "ljspeech",
    },
    "ljspeech_melgan": {
        "independent_generator_id": "MelGAN_LJ",
        "waveform_architecture": "MelGAN",
        "waveform_family": "GAN",
        "language": "en",
        "real_root_key": "ljspeech",
    },
    "ljspeech_full_band_melgan": {
        "independent_generator_id": "FBMelGAN_LJ",
        "waveform_architecture": "FB-MelGAN",
        "waveform_family": "GAN",
        "language": "en",
        "real_root_key": "ljspeech",
    },
    "ljspeech_melgan_large": {
        "independent_generator_id": "MelGANLarge_LJ",
        "waveform_architecture": "MelGAN-Large",
        "waveform_family": "GAN",
        "language": "en",
        "real_root_key": "ljspeech",
    },
    "ljspeech_parallel_wavegan": {
        "independent_generator_id": "PWG_LJ",
        "waveform_architecture": "Parallel WaveGAN",
        "waveform_family": "GAN",
        "language": "en",
        "real_root_key": "ljspeech",
    },
    "ljspeech_multi_band_melgan": {
        "independent_generator_id": "MBMelGAN_LJ",
        "waveform_architecture": "MB-MelGAN",
        "waveform_family": "GAN",
        "language": "en",
        "real_root_key": "ljspeech",
    },
    "ljspeech_waveglow": {
        "independent_generator_id": "WaveGlow_LJ",
        "waveform_architecture": "WaveGlow",
        "waveform_family": "Flow",
        "language": "en",
        "real_root_key": "ljspeech",
    },
    "jsut_multi_band_melgan": {
        "independent_generator_id": "MBMelGAN_JSUT",
        "waveform_architecture": "MB-MelGAN",
        "waveform_family": "GAN",
        "language": "ja",
        "real_root_key": "jsut",
    },
    "jsut_parallel_wavegan": {
        "independent_generator_id": "PWG_JSUT",
        "waveform_architecture": "Parallel WaveGAN",
        "waveform_family": "GAN",
        "language": "ja",
        "real_root_key": "jsut",
    },
}

EXPECTED_GENERATORS_BY_DOMAIN = {
    "ljspeech": 7,
    "jsut": 2,
}
EXPECTED_REAL_FILES = {
    "ljspeech": 13100,
    "jsut": 5000,
}

_SUFFIX_RE = re.compile(r"_(?:generated|gen)$")


def normalize_fake_stem(stem: str) -> str:

    return _SUFFIX_RE.sub("", str(stem))


def real_root_for(key: str) -> Path:
    if key == "ljspeech":
        return CONFIG["ljspeech_real_root"]
    if key == "jsut":
        return CONFIG["jsut_real_root"]
    raise KeyError(f"Domaine real inconnu : {key}")


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


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_audio_safe(path: Path, target_sr: int):

    try:
        wav, sr_orig = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1, dtype=np.float32)
        elif wav.ndim != 1:
            raise ValueError(f"shape audio not prise en charge : {wav.shape}")

        wav = np.asarray(wav, dtype=np.float32)
        if sr_orig <= 0:
            raise ValueError(f"sample rate invalid : {sr_orig}")

        if int(sr_orig) != int(target_sr):
            gcd = math.gcd(int(sr_orig), int(target_sr))
            up = int(target_sr) // gcd
            down = int(sr_orig) // gcd
            wav = resample_poly(wav, up=up, down=down).astype(np.float32, copy=False)

        return wav, int(sr_orig), None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def simple_energy_vad(
    wav: np.ndarray,
    sr: int,
    frame_ms: int,
    top_db: float,
    absolute_db: float,
    reference_percentile: float,
):

    frame_len = int(sr * frame_ms / 1000)
    if frame_len <= 0 or len(wav) < frame_len:
        return np.array([], dtype=bool), 0.0

    n_frames = len(wav) // frame_len
    frames = wav[: n_frames * frame_len].reshape(n_frames, frame_len)
    energies = np.mean(frames.astype(np.float64) ** 2, axis=1)
    frame_db = 10.0 * np.log10(energies + 1e-12)
    reference_db = float(np.percentile(frame_db, reference_percentile))

    active_mask = (
        (frame_db >= reference_db - top_db)
        & (frame_db >= absolute_db)
    )
    active_duration = float(active_mask.sum() * frame_len / sr)
    return active_mask, active_duration


def compute_qc(path: Path, cfg: dict = QC_CONFIG) -> AudioQCResult:
    result = AudioQCResult(path=str(path))

    if not path.exists():
        result.qc_status = "fail"
        result.exclusion_reason = "file_not_found"
        return result
    result.exists = True

    wav, sr_orig, error = load_audio_safe(path, int(cfg["target_sr"]))
    if error is not None or wav is None:
        result.qc_status = "fail"
        result.exclusion_reason = f"load_error:{error}"
        return result

    result.loadable = True
    result.sample_rate_original = int(sr_orig)

    if wav.size == 0:
        result.qc_status = "fail"
        result.exclusion_reason = "empty_audio"
        return result

    has_nonfinite = bool(not np.isfinite(wav).all())
    result.has_nan_or_inf = has_nonfinite
    if has_nonfinite:
        result.qc_status = "fail"
        result.exclusion_reason = "nan_or_inf_samples"
        return result

    duration_s = float(len(wav) / int(cfg["target_sr"]))
    clipping_rate = float(np.mean(np.abs(wav) >= 0.999))

    _, active_duration = simple_energy_vad(
        wav=wav,
        sr=int(cfg["target_sr"]),
        frame_ms=int(cfg["vad_frame_ms"]),
        top_db=float(cfg["vad_top_db"]),
        absolute_db=float(cfg["vad_abs_db"]),
        reference_percentile=float(cfg["vad_reference_percentile"]),
    )
    silence_ratio = float(1.0 - active_duration / duration_s) if duration_s > 0 else 1.0
    silence_ratio = float(np.clip(silence_ratio, 0.0, 1.0))

    result.duration_s = duration_s
    result.active_duration_s = active_duration
    result.silence_ratio = silence_ratio
    result.clipping_rate = clipping_rate
    result.sha256 = sha256_of_file(path)

    reasons = []
    if duration_s < float(cfg["min_duration_s"]):
        reasons.append("duration_below_min")
    if duration_s > float(cfg["max_duration_s"]):
        reasons.append("duration_above_max")
    if clipping_rate > float(cfg["max_clipping_rate"]):
        reasons.append("excessive_clipping")
    if silence_ratio > float(cfg["max_silence_ratio"]):
        reasons.append("excessive_silence")

    if reasons:
        result.qc_status = "fail"
        result.exclusion_reason = ";".join(reasons)
    else:
        result.qc_status = "ok"
        result.exclusion_reason = ""

    return result


def duration_ratio_ok(fake_duration: Optional[float], real_duration: Optional[float]) -> bool:
    if fake_duration is None or real_duration is None or real_duration <= 0:
        return False
    ratio = float(fake_duration / real_duration)
    lo, hi = QC_CONFIG["duration_ratio_bounds"]
    return float(lo) <= ratio <= float(hi)


def atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def preflight() -> dict:
    roots = {
        "wavefake_root": CONFIG["wavefake_root"],
        "ljspeech_real_root": CONFIG["ljspeech_real_root"],
        "jsut_real_root": CONFIG["jsut_real_root"],
    }
    for name, path in roots.items():
        print(f"[PREFLIGHT] {name}: {path}")
        if not Path(path).exists():
            raise FileNotFoundError(f"Root not found : {name} -> {path}")

    examples = {
        "LJSpeech": CONFIG["ljspeech_real_root"] / "LJ001-0001.wav",
        "JSUT": CONFIG["jsut_real_root"] / "BASIC5000_0001.wav",
    }
    for corpus, path in examples.items():
        if not path.exists():
            raise FileNotFoundError(f"Control file not found for {corpus}: {path}")

    real_counts = {
        "ljspeech": len(list(CONFIG["ljspeech_real_root"].glob("*.wav"))),
        "jsut": len(list(CONFIG["jsut_real_root"].glob("*.wav"))),
    }
    print(f"[PREFLIGHT] LJSpeech real : {real_counts['ljspeech']}")
    print(f"[PREFLIGHT] JSUT real     : {real_counts['jsut']}")

    for domain, expected in EXPECTED_REAL_FILES.items():
        if real_counts[domain] < expected:
            raise RuntimeError(
                f"Corpus real {domain} incomplete : {real_counts[domain]} files, "
                f"at least {expected} expected."
            )

    generated_counts = {}
    for folder_name in WAVEFAKE_GENERATORS:
        folder = CONFIG["wavefake_root"] / folder_name
        if not folder.exists():
            raise FileNotFoundError(f"Directory generator missing : {folder}")
        count = len(list(folder.glob("*.wav")))
        generated_counts[folder_name] = count
        print(f"[PREFLIGHT] {folder_name}: {count} files")
        if count == 0:
            raise RuntimeError(f"No WAV file in : {folder}")

    print("[PREFLIGHT] PASS — WaveFake directory structure is accessible.")
    return {
        "real_counts": real_counts,
        "generated_counts": generated_counts,
    }


def clean_hash(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip().str.lower()
    return out.where(~out.isin({"", "nan", "none", "null"}))


def audit_manifest(df: pd.DataFrame, output_dir: Path) -> dict:
    critical = {}

    critical["duplicate_pair_id"] = int(df["pair_id"].duplicated().sum())
    critical["duplicate_generator_original"] = int(
        df.duplicated(["independent_generator_id", "original_id"]).sum()
    )

    ok = df.loc[df["qc_status"].eq("ok")].copy()
    critical["missing_fake_hash_in_qc_ok"] = int(ok["fake_sha256"].isna().sum())
    critical["missing_real_hash_in_qc_ok"] = int(ok["real_sha256"].isna().sum())

    real_consistency = (
        ok.groupby(["domain", "original_id"])[["real_path", "real_sha256"]]
        .nunique(dropna=False)
    )
    critical["originals_with_multiple_real_paths"] = int((real_consistency["real_path"] > 1).sum())
    critical["originals_with_multiple_real_hashes"] = int((real_consistency["real_sha256"] > 1).sum())

    work = ok.copy()
    work["fake_hash_clean"] = clean_hash(work["fake_sha256"])
    work["real_hash_clean"] = clean_hash(work["real_sha256"])

    collision_rows = []

    fake_groups = (
        work.dropna(subset=["fake_hash_clean"])
        .groupby("fake_hash_clean")
        .agg(
            n_pairs=("pair_id", "nunique"),
            n_originals=("original_id", "nunique"),
            n_generators=("independent_generator_id", "nunique"),
            generators=("independent_generator_id", lambda s: "|".join(sorted(set(map(str, s))))),
            originals=("original_id", lambda s: "|".join(sorted(set(map(str, s)))[:20])),
        )
    )
    ambiguous_fake = fake_groups[
        (fake_groups["n_originals"] > 1) | (fake_groups["n_generators"] > 1)
    ]
    for sha, row in ambiguous_fake.iterrows():
        collision_rows.append({
            "collision_type": "fake_hash_cross_content_or_generator",
            "sha256": sha,
            "n_pairs": int(row["n_pairs"]),
            "n_originals": int(row["n_originals"]),
            "n_generators": int(row["n_generators"]),
            "details": f"generators={row['generators']}; originals={row['originals']}",
        })

    real_unique = work[["domain", "original_id", "real_hash_clean"]].dropna().drop_duplicates()
    real_groups = real_unique.groupby("real_hash_clean").agg(
        n_originals=("original_id", "nunique"),
        domains=("domain", lambda s: "|".join(sorted(set(map(str, s))))),
        originals=("original_id", lambda s: "|".join(sorted(set(map(str, s)))[:20])),
    )
    ambiguous_real = real_groups[real_groups["n_originals"] > 1]
    for sha, row in ambiguous_real.iterrows():
        collision_rows.append({
            "collision_type": "real_hash_multiple_original_ids",
            "sha256": sha,
            "n_originals": int(row["n_originals"]),
            "details": f"domains={row['domains']}; originals={row['originals']}",
        })

    real_hashes = set(real_unique["real_hash_clean"].dropna())
    fake_hashes = set(work["fake_hash_clean"].dropna())
    cross_hashes = sorted(real_hashes.intersection(fake_hashes))
    for sha in cross_hashes:
        collision_rows.append({
            "collision_type": "fake_hash_equals_real_hash",
            "sha256": sha,
            "details": "exact file hash appears in both real and fake sets",
        })

    collision_df = pd.DataFrame(collision_rows)
    if collision_df.empty:
        collision_df = pd.DataFrame(
            columns=["collision_type", "sha256", "n_pairs", "n_originals", "n_generators", "details"]
        )
    collision_path = output_dir / "wavefake_hash_collision_audit.csv"
    collision_df.to_csv(collision_path, index=False)

    coverage_rows = []
    for domain, group in ok.groupby("domain", sort=True):
        expected_generators = EXPECTED_GENERATORS_BY_DOMAIN[str(domain)]
        coverage = group.groupby("original_id")["independent_generator_id"].nunique()
        counts = coverage.value_counts().sort_index()
        for n_generators, n_originals in counts.items():
            coverage_rows.append({
                "domain": str(domain),
                "n_generators_available": int(n_generators),
                "n_originals": int(n_originals),
                "expected_generators": int(expected_generators),
                "is_complete": bool(int(n_generators) == int(expected_generators)),
            })
    coverage_df = pd.DataFrame(coverage_rows)
    coverage_path = output_dir / "wavefake_coverage_by_domain.csv"
    coverage_df.to_csv(coverage_path, index=False)

    complete_by_domain = {}
    for domain, expected in EXPECTED_GENERATORS_BY_DOMAIN.items():
        g = ok.loc[ok["domain"].eq(domain)]
        coverage = g.groupby("original_id")["independent_generator_id"].nunique()
        complete_by_domain[domain] = int((coverage == expected).sum())

    critical_failure_count = sum(int(v) for v in critical.values())
    hash_collision_count = int(len(collision_df))

    summary = {
        "critical_checks": critical,
        "critical_failure_count": int(critical_failure_count),
        "hash_collision_records": hash_collision_count,
        "complete_originals_by_domain": complete_by_domain,
        "collision_audit_file": str(collision_path),
        "coverage_file": str(coverage_path),
        "status": (
            "PASS"
            if critical_failure_count == 0 and hash_collision_count == 0
            else "FAIL"
        ),
    }

    if critical_failure_count > 0:
        raise RuntimeError(f"Audit structural WaveFake failed : {critical}")
    if CONFIG["strict_hash_audit"] and hash_collision_count > 0:
        raise RuntimeError(
            f"Audit SHA-256 WaveFake failed : {hash_collision_count} collision(s). "
            f"Consulter {collision_path}."
        )

    return summary


def summarize_manifest(df: pd.DataFrame) -> dict:
    total = int(len(df))
    n_ok = int(df["qc_status"].eq("ok").sum())
    n_fail = int(total - n_ok)

    fail_reasons = (
        df.loc[df["qc_status"].eq("fail"), "exclusion_reason"]
        .fillna("unknown")
        .value_counts()
        .to_dict()
    )

    by_generator = {}
    for gen, group in df.groupby("independent_generator_id", sort=True):
        by_generator[str(gen)] = {
            "domain": str(group["domain"].iloc[0]),
            "language": str(group["language"].iloc[0]),
            "waveform_architecture": str(group["waveform_architecture"].iloc[0]),
            "waveform_family": str(group["waveform_family"].iloc[0]),
            "n_candidates": int(len(group)),
            "n_ok": int(group["qc_status"].eq("ok").sum()),
            "n_fail": int(group["qc_status"].eq("fail").sum()),
        }

    by_domain = {}
    for domain, group in df.groupby("domain", sort=True):
        by_domain[str(domain)] = {
            "n_generators": int(group["independent_generator_id"].nunique()),
            "n_candidates": int(len(group)),
            "n_ok": int(group["qc_status"].eq("ok").sum()),
            "n_fail": int(group["qc_status"].eq("fail").sum()),
            "n_unique_originals": int(group["original_id"].nunique()),
        }

    return {
        "total_pairs_candidates": total,
        "n_ok": n_ok,
        "n_fail": n_fail,
        "fail_rate": round(n_fail / total, 6) if total else None,
        "exclusion_reason_counts": fail_reasons,
        "n_generators": int(df["independent_generator_id"].nunique()),
        "n_domains": int(df["domain"].nunique()),
        "n_unique_originals_union": int(df["original_id"].nunique()),
        "by_generator": by_generator,
        "by_domain": by_domain,
    }


def build_candidate_list() -> list:
    candidates = []
    limit = CONFIG["quick_test_n_per_generator"]

    for folder_name, meta in WAVEFAKE_GENERATORS.items():
        gen_dir = CONFIG["wavefake_root"] / folder_name
        fake_files = sorted(gen_dir.glob("*.wav"))
        if limit is not None:
            fake_files = fake_files[: int(limit)]

        print(f"[{folder_name}] {len(fake_files)} files candidates")
        real_root = real_root_for(meta["real_root_key"])

        for fake_path in fake_files:
            candidates.append((folder_name, meta, real_root, fake_path))

    if not candidates:
        raise RuntimeError("No file WaveFake candidate found.")
    return candidates


def candidate_order_hash(candidates: list) -> str:


    records = []
    for folder, meta, real_root, fake_path in candidates:
        original_id = normalize_fake_stem(fake_path.stem)
        real_path = real_root / f"{original_id}.wav"
        def sig(p: Path):
            if not p.exists():
                return (str(p), -1, -1)
            st = p.stat()
            return (str(p), int(st.st_size), int(st.st_mtime_ns))
        records.append((folder, meta["independent_generator_id"], sig(fake_path), sig(real_path)))
    return sha256_text(json.dumps(records, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def checkpoint_config_hash() -> str:
    payload = {
        "protocol": "WaveFake-Phase0C-v2.3-CANONICAL",
        "qc_config": QC_CONFIG,
        "generators": WAVEFAKE_GENERATORS,
        "expected_generators_by_domain": EXPECTED_GENERATORS_BY_DOMAIN,
        "expected_real_files": EXPECTED_REAL_FILES,
        "quick_test_n_per_generator": CONFIG["quick_test_n_per_generator"],
    }
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":")))


def build_and_validate(preflight_summary: dict):
    output_dir = CONFIG["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    out_manifest = output_dir / "wavefake_manifest.parquet"
    report_path = output_dir / "wavefake_pairing_report.json"
    checkpoint_path = output_dir / "_wavefake_checkpoint_rows.parquet"
    checkpoint_meta_path = output_dir / "_wavefake_checkpoint_meta.json"

    if CONFIG["force_rebuild"]:
        for path in [out_manifest, report_path, checkpoint_path, checkpoint_meta_path,
                     output_dir / "wavefake_hash_collision_audit.csv",
                     output_dir / "wavefake_coverage_by_domain.csv"]:
            if path.exists():
                path.unlink()
                print(f"[FORCE] Removed : {path}")

    if out_manifest.exists() and not CONFIG["force_rebuild"] and CONFIG["quick_test_n_per_generator"] is None:
        raise RuntimeError(
            f"A final manifest already exists: {out_manifest}. "
            "The canonical protocol refuses silent reuse. "
            "Use the frozen validated manifest for manuscript analyses, or set "
            "CONFIG['force_rebuild']=True to rebuild from the WAV files."
        )

    candidates = build_candidate_list()
    order_hash = candidate_order_hash(candidates)

    rows = []
    start_idx = 0
    real_qc_cache: Dict[str, AudioQCResult] = {}

    if checkpoint_path.exists() and CONFIG["quick_test_n_per_generator"] is None:
        if not checkpoint_meta_path.exists():
            raise RuntimeError(
                "WaveFake checkpoint has no metadata. Delete the checkpoint "
                "or relancer with force_rebuild=True."
            )
        checkpoint_meta = json.loads(checkpoint_meta_path.read_text(encoding="utf-8"))
        if checkpoint_meta.get("candidate_order_sha256") != order_hash:
            raise RuntimeError(
                "Checkpoint is incompatible with the current inputs. "
                "Relancer with force_rebuild=True."
            )
        if checkpoint_meta.get("config_sha256") != checkpoint_config_hash():
            raise RuntimeError(
                "Checkpoint is incompatible with the current QC/taxonomy configuration. "
                "Relancer with force_rebuild=True."
            )

        checkpoint_df = pd.read_parquet(checkpoint_path)
        rows = checkpoint_df.to_dict(orient="records")
        start_idx = len(rows)
        print(f"[RESUME] {start_idx} pairs restored from the checkpoint.")

        for row in rows:
            real_path = str(row.get("real_path", ""))
            if real_path and real_path not in real_qc_cache:
                real_qc_cache[real_path] = AudioQCResult(
                    path=real_path,
                    exists=True,
                    loadable=True,
                    sample_rate_original=row.get("real_sample_rate_original"),
                    duration_s=row.get("real_duration"),
                    active_duration_s=row.get("active_duration_real"),
                    silence_ratio=row.get("silence_ratio_real"),
                    clipping_rate=row.get("clipping_rate_real"),
                    has_nan_or_inf=False,
                    sha256=row.get("real_sha256"),
                    qc_status=row.get("real_qc_status", "ok"),
                    exclusion_reason=row.get("real_exclusion_reason", ""),
                )

    total = len(candidates)
    if CONFIG["quick_test_n_per_generator"] is None:
        expected_total = 7 * EXPECTED_REAL_FILES["ljspeech"] + 2 * EXPECTED_REAL_FILES["jsut"]
        if total != expected_total:
            raise RuntimeError(f"Population WaveFake unexpected: {total:,} candidates; {expected_total:,} expected.")
    t0 = time.time()
    processed_this_session = 0

    for i, (folder_name, meta, real_root, fake_path) in enumerate(candidates):
        if i < start_idx:
            continue

        original_id = normalize_fake_stem(fake_path.stem)
        real_path = real_root / f"{original_id}.wav"

        fake_qc = compute_qc(fake_path)
        real_key = str(real_path)
        if real_key not in real_qc_cache:
            real_qc_cache[real_key] = compute_qc(real_path)
        real_qc = real_qc_cache[real_key]

        reasons = []
        if fake_qc.qc_status != "ok":
            reasons.append(f"fake:{fake_qc.exclusion_reason}")
        if real_qc.qc_status != "ok":
            reasons.append(f"real:{real_qc.exclusion_reason}")

        pair_status = "ok" if not reasons else "fail"
        if pair_status == "ok" and not duration_ratio_ok(fake_qc.duration_s, real_qc.duration_s):
            pair_status = "fail"
            reasons.append("duration_ratio_out_of_bounds")

        ratio = None
        if fake_qc.duration_s is not None and real_qc.duration_s not in (None, 0):
            ratio = float(fake_qc.duration_s / real_qc.duration_s)

        rows.append({
            "pair_id": f"wavefake__{folder_name}__{original_id}",
            "dataset": "WaveFake",
            "domain": meta["real_root_key"],
            "generator_key": folder_name,
            "independent_generator_id": meta["independent_generator_id"],
            "waveform_architecture": meta["waveform_architecture"],
            "waveform_family": meta["waveform_family"],
            "pipeline_type": "neural-vocoder",
            "language": meta["language"],
            "speaker_id": None,
            "original_id": original_id,
            "fake_path": str(fake_path),
            "real_path": str(real_path),
            "fake_duration": fake_qc.duration_s,
            "real_duration": real_qc.duration_s,
            "duration_ratio": ratio,
            "active_duration_fake": fake_qc.active_duration_s,
            "active_duration_real": real_qc.active_duration_s,
            "clipping_rate_fake": fake_qc.clipping_rate,
            "clipping_rate_real": real_qc.clipping_rate,
            "silence_ratio_fake": fake_qc.silence_ratio,
            "silence_ratio_real": real_qc.silence_ratio,
            "fake_sha256": fake_qc.sha256,
            "real_sha256": real_qc.sha256,
            "sample_rate_original": fake_qc.sample_rate_original,
            "real_sample_rate_original": real_qc.sample_rate_original,
            "fake_qc_status": fake_qc.qc_status,
            "real_qc_status": real_qc.qc_status,
            "fake_exclusion_reason": fake_qc.exclusion_reason,
            "real_exclusion_reason": real_qc.exclusion_reason,
            "qc_status": pair_status,
            "exclusion_reason": ";".join(reasons),
        })

        processed_this_session += 1

        if (i + 1) % int(CONFIG["progress_every"]) == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            rate = processed_this_session / elapsed if elapsed > 0 else 0.0
            remaining = (total - (i + 1)) / rate if rate > 0 else float("inf")
            print(
                f"  [{i + 1}/{total}] {rate:.2f} pair/s — "
                f"ETA {remaining / 60:.1f} min"
            )

        if (
            CONFIG["quick_test_n_per_generator"] is None
            and (i + 1) % int(CONFIG["checkpoint_every"]) == 0
        ):
            atomic_write_parquet(pd.DataFrame(rows), checkpoint_path)
            atomic_write_json(
                {
                    "candidate_order_sha256": order_hash,
                    "config_sha256": checkpoint_config_hash(),
                    "n_candidates": total,
                    "n_rows_checkpointed": len(rows),
                    "created_utc": pd.Timestamp.utcnow().isoformat(),
                },
                checkpoint_meta_path,
            )
            print(f"  [CHECKPOINT] {len(rows)} pairs saved.")

    if not rows:
        raise RuntimeError("No pair was built.")

    df = pd.DataFrame(rows)
    atomic_write_parquet(df, out_manifest)

    audit = audit_manifest(df, output_dir)
    if CONFIG["quick_test_n_per_generator"] is None:
        expected_complete = {"ljspeech": 13100, "jsut": 5000}
        if audit["complete_originals_by_domain"] != expected_complete:
            raise RuntimeError(
                f"Intersection confirmatory WaveFake unexpected: {audit['complete_originals_by_domain']} != {expected_complete}"
            )
    summary = summarize_manifest(df)
    summary.update({
        "protocol": "WaveFake-Phase0C-v2.3-CANONICAL",
        "qc_config": QC_CONFIG,
        "source_roots": {
            "wavefake_root": str(CONFIG["wavefake_root"]),
            "ljspeech_real_root": str(CONFIG["ljspeech_real_root"]),
            "jsut_real_root": str(CONFIG["jsut_real_root"]),
        },
        "preflight": preflight_summary,
        "candidate_order_sha256": order_hash,
        "manifest_sha256": sha256_of_file(out_manifest),
        "audit": audit,
        "governance_note": (
            "LJSpeech and JSUT must be frozen separately because their content sets differ. "
            "No global nine-generator intersection is valid."
        ),
    })
    atomic_write_json(summary, report_path)

    if checkpoint_path.exists():
        checkpoint_path.unlink()
    if checkpoint_meta_path.exists():
        checkpoint_meta_path.unlink()

    print("\n" + "=" * 80)
    print("WAVEFAKE SUMMARY — PHASE 0C v2.3 CANONICAL")
    print("=" * 80)
    print(f"Candidate pairs : {summary['total_pairs_candidates']}")
    print(f"QC-valid pairs      : {summary['n_ok']}")
    print(f"QC-failed pairs    : {summary['n_fail']} ({summary['fail_rate']:.2%})")
    print(f"Generators       : {summary['n_generators']}")
    print(f"Audit             : {audit['status']}")
    print(f"Complete LJSpeech originals: {audit['complete_originals_by_domain']['ljspeech']}")
    print(f"Complete JSUT originals    : {audit['complete_originals_by_domain']['jsut']}")
    print(f"Manifest         : {out_manifest}")
    print(f"Rapport           : {report_path}")
    print("=" * 80)

    return df, summary


if __name__ == "__main__":
    preflight_summary = preflight()
    wavefake_df, wavefake_summary = build_and_validate(preflight_summary)
