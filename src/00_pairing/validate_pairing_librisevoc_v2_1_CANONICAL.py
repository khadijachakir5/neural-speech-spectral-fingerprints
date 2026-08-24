

# Purpose: Build and validate LibriSeVoc self-vocoding fake-to-real pairs with strict QC, duration checks, hashes, and resumable checkpoints.

from pathlib import Path
import sys
import time
import pandas as pd


import hashlib
import json
import os
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np
import soundfile as sf
import librosa

warnings.filterwarnings("ignore")


QC_DEFAULTS = dict(
    target_sr=16000,
    min_duration_s=0.5,
    max_duration_s=30.0,
    max_clipping_rate=0.001,       
    max_silence_ratio=0.85,        
    duration_ratio_bounds=(0.5, 2.0),  
    vad_frame_ms=30,
    vad_energy_percentile=15,      
)


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
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_audio_safe(path: Path, target_sr: int):

    try:
        wav, sr_orig = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr_orig != target_sr:
            wav = librosa.resample(wav, orig_sr=sr_orig, target_sr=target_sr)
        return wav, sr_orig, None
    except Exception as e:
        return None, None, str(e)


def simple_energy_vad(wav: np.ndarray, sr: int, frame_ms: int = 30,
                       energy_percentile: int = 15):


    frame_len = int(sr * frame_ms / 1000)
    if frame_len <= 0 or len(wav) < frame_len:
        return np.array([], dtype=bool), 0.0
    n_frames = len(wav) // frame_len
    frames = wav[: n_frames * frame_len].reshape(n_frames, frame_len)
    energies = np.mean(frames ** 2, axis=1)
    threshold = np.percentile(energies, energy_percentile)
    
    threshold = max(threshold, 1e-8)
    active_mask = energies > threshold
    active_duration = active_mask.sum() * frame_len / sr
    return active_mask, float(active_duration)


def compute_qc(path: Path, cfg: dict = QC_DEFAULTS) -> AudioQCResult:
    result = AudioQCResult(path=str(path))

    if not path.exists():
        result.qc_status = "fail"
        result.exclusion_reason = "file_not_found"
        return result
    result.exists = True

    wav, sr_orig, err = load_audio_safe(path, cfg["target_sr"])
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

    nan_inf = bool(np.isnan(wav).any() or np.isinf(wav).any())
    result.has_nan_or_inf = nan_inf
    if nan_inf:
        result.qc_status = "fail"
        result.exclusion_reason = "nan_or_inf_samples"
        return result

    duration_s = len(wav) / cfg["target_sr"]
    result.duration_s = float(duration_s)

    clipping_rate = float(np.mean(np.abs(wav) >= 0.999))
    result.clipping_rate = clipping_rate

    _, active_duration = simple_energy_vad(
        wav, cfg["target_sr"], cfg["vad_frame_ms"], cfg["vad_energy_percentile"]
    )
    result.active_duration_s = active_duration
    result.silence_ratio = float(1.0 - active_duration / duration_s) if duration_s > 0 else 1.0

    result.sha256 = sha256_of_file(path)

    
    reasons = []
    if duration_s < cfg["min_duration_s"]:
        reasons.append("duration_below_min")
    if duration_s > cfg["max_duration_s"]:
        reasons.append("duration_above_max")
    if clipping_rate > cfg["max_clipping_rate"]:
        reasons.append("excessive_clipping")
    if result.silence_ratio > cfg["max_silence_ratio"]:
        reasons.append("excessive_silence")

    if reasons:
        result.qc_status = "fail"
        result.exclusion_reason = ";".join(reasons)
    else:
        result.qc_status = "ok"
        result.exclusion_reason = ""

    return result


def duration_ratio_ok(fake_dur: float, real_dur: float, cfg: dict = QC_DEFAULTS) -> bool:
    if fake_dur is None or real_dur is None or real_dur <= 0:
        return False
    ratio = fake_dur / real_dur
    lo, hi = cfg["duration_ratio_bounds"]
    return lo <= ratio <= hi


def write_manifest(rows: list, out_path: Path):
    import pandas as pd
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    return df


def write_report(summary: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)


def summarize_manifest(df) -> dict:
    total = len(df)
    n_ok = int((df["qc_status"] == "ok").sum()) if total else 0
    n_fail = total - n_ok
    reasons = (
        df.loc[df["qc_status"] == "fail", "exclusion_reason"]
        .value_counts()
        .to_dict()
        if total else {}
    )
    n_original_shared = 0
    if "original_id" in df.columns and "independent_generator_id" in df.columns:
        counts = df.groupby("original_id")["independent_generator_id"].nunique()
        n_original_shared = int((counts > 1).sum())
    return dict(
        total_pairs_candidates=total,
        n_ok=n_ok,
        n_fail=n_fail,
        fail_rate=round(n_fail / total, 4) if total else None,
        exclusion_reason_counts=reasons,
        n_originals_shared_across_generators=n_original_shared,
    )


CONFIG = dict(
    librisevoc_root=Path("/content/drive/MyDrive/Datasets/LibriSeVoc"),
    real_subdir="gt",
    output_dir=Path("/content/drive/MyDrive/fingerprint_q1_outputs/phase0_librisevoc_v2"),
    max_files_per_generator=None,
    
    quick_test_n=None,        
    checkpoint_every=2000,    
    progress_every=1000,      
    force_rebuild=False,      
)


LIBRISEVOC_GENERATORS = {
    "wavenet":           dict(independent_generator_id="WaveNet_LS",   waveform_architecture="WaveNet",           waveform_family="Autoregressive"),
    "wavernn":           dict(independent_generator_id="WaveRNN_LS",   waveform_architecture="WaveRNN",           waveform_family="Autoregressive"),
    "melgan":            dict(independent_generator_id="MelGAN_LS",    waveform_architecture="MelGAN",            waveform_family="GAN"),          
    "parallel_wave_gan": dict(independent_generator_id="PWG_LS",       waveform_architecture="Parallel WaveGAN",  waveform_family="GAN"),          
    "wavegrad":          dict(independent_generator_id="WaveGrad_LS",  waveform_architecture="WaveGrad",          waveform_family="Diffusion"),
    "diffwave":          dict(independent_generator_id="DiffWave_LS",  waveform_architecture="DiffWave",          waveform_family="Diffusion"),
}


def stable_json_hash(payload) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def candidate_signature(flat_files, real_root: Path) -> str:
    records = []
    for folder_name, meta, fake_path in flat_files:
        fake_stem = fake_path.stem
        original_id = fake_stem[:-4] if fake_stem.endswith("_gen") else fake_stem
        real_path = real_root / f"{original_id}.wav"
        def fsig(p: Path):
            if not p.exists(): return (str(p), -1, -1)
            st=p.stat(); return (str(p), int(st.st_size), int(st.st_mtime_ns))
        records.append((folder_name, meta["independent_generator_id"], fsig(fake_path), fsig(real_path)))
    return stable_json_hash(records)


def config_signature() -> str:
    return stable_json_hash({
        "qc": QC_DEFAULTS,
        "generators": LIBRISEVOC_GENERATORS,
        "real_subdir": CONFIG["real_subdir"],
        "max_files_per_generator": CONFIG["max_files_per_generator"],
        "quick_test_n": CONFIG["quick_test_n"],
        "duration_abs_max_s": 0.25,
    })


def build_and_validate():
    root = CONFIG["librisevoc_root"]
    real_root = root / CONFIG["real_subdir"]
    out_manifest = CONFIG["output_dir"] / "librisevoc_manifest.parquet"
    checkpoint_path = CONFIG["output_dir"] / "_librisevoc_checkpoint_rows.parquet"
    checkpoint_meta_path = CONFIG["output_dir"] / "_librisevoc_checkpoint_meta.json"

    
    if out_manifest.exists() and not CONFIG["force_rebuild"] and not CONFIG["quick_test_n"]:
        try:
            existing_df = pd.read_parquet(out_manifest)
            existing_n_ok = int((existing_df["qc_status"] == "ok").sum())
        except Exception as e:
            print(f"[WARNING] Existing manifest unreadable ({e}) — rebuild complete.")
            existing_n_ok = -1  

        if existing_n_ok > 0:
            raise RuntimeError(
                f"A final manifest already exists: {out_manifest}. "
                "The canonical protocol refuses silent reuse. "
                "Audit this file or set CONFIG['force_rebuild']=True to rebuild it."
            )
        else:
            print(f"[INVALID MANIFEST DETECTED] {out_manifest} has n_ok=0 — rebuilding automatically.")

    if not root.exists():
        raise FileNotFoundError(f"librisevoc_root not found : {root}")
    if not real_root.exists():
        raise FileNotFoundError(f"Directory real not found : {real_root}")

    
    flat_files = []
    for folder_name, meta in LIBRISEVOC_GENERATORS.items():
        gen_dir = root / folder_name
        if not gen_dir.exists():
            raise FileNotFoundError(f"Directory generator required missing : {gen_dir}")
        fake_files = sorted(gen_dir.glob("*.wav"))
        if CONFIG["max_files_per_generator"]:
            fake_files = fake_files[: CONFIG["max_files_per_generator"]]
        print(f"[{folder_name}] {len(fake_files)} files candidates")
        for fp in fake_files:
            flat_files.append((folder_name, meta, fp))

    if CONFIG["quick_test_n"]:
        flat_files = flat_files[: CONFIG["quick_test_n"]]
        print(f"[MODE TEST] limited to {len(flat_files)} pairs — set quick_test_n=None for the full run")
    elif len(flat_files) != 79_206:
        raise RuntimeError(f"LibriSeVoc: {len(flat_files):,} pairs candidates; 79,206 expected (13,201 × 6).")

    input_sig = candidate_signature(flat_files, real_root)
    cfg_sig = config_signature()

    
    rows = []
    start_idx = 0
    if checkpoint_path.exists() and not CONFIG["quick_test_n"]:
        try:
            if not checkpoint_meta_path.exists():
                raise RuntimeError("checkpoint metadata missing")
            meta_ckpt = json.loads(checkpoint_meta_path.read_text(encoding="utf-8"))
            if meta_ckpt.get("input_signature") != input_sig or meta_ckpt.get("config_signature") != cfg_sig:
                raise RuntimeError("checkpoint is incompatible with the current inputs/configuration")
            ckpt_df = pd.read_parquet(checkpoint_path)
            rows = ckpt_df.to_dict(orient="records")
            start_idx = len(rows)
            if start_idx > len(flat_files):
                raise RuntimeError("checkpoint contains more rows than the candidate list")
            print(f"[RESUME] {start_idx} pairs restored from a signed checkpoint.")
        except Exception as e:
            raise RuntimeError(f"Checkpoint LibriSeVoc invalid: {e}. Set force_rebuild=True or delete the checkpoint.")

    total = len(flat_files)
    t0 = time.time()

    for i, (folder_name, meta, fake_path) in enumerate(flat_files):
        if i < start_idx:
            continue

        fake_stem = fake_path.stem  
        original_id = fake_stem[:-4] if fake_stem.endswith("_gen") else fake_stem
        speaker_id = original_id.split("_")[0] if "_" in original_id else None
        real_path = real_root / f"{original_id}.wav"

        fake_qc = compute_qc(fake_path)
        real_qc = compute_qc(real_path)

        pair_qc_status = "ok" if (fake_qc.qc_status == "ok" and real_qc.qc_status == "ok") else "fail"
        reasons = []
        if fake_qc.qc_status == "fail":
            reasons.append(f"fake:{fake_qc.exclusion_reason}")
        if real_qc.qc_status == "fail":
            reasons.append(f"real:{real_qc.exclusion_reason}")

        if pair_qc_status == "ok":
            if not duration_ratio_ok(fake_qc.duration_s, real_qc.duration_s):
                pair_qc_status = "fail"
                reasons.append("duration_ratio_out_of_bounds")
            if fake_qc.duration_s and real_qc.duration_s:
                delta = abs(fake_qc.duration_s - real_qc.duration_s)
                if delta > 0.25:
                    pair_qc_status = "fail"
                    reasons.append(f"self_vocoding_duration_mismatch:{delta:.3f}s")

        rows.append(dict(
            pair_id=f"librisevoc__{folder_name}__{original_id}",
            dataset="LibriSeVoc",
            domain="librispeech",
            generator_key=folder_name,
            independent_generator_id=meta["independent_generator_id"],
            waveform_architecture=meta["waveform_architecture"],
            waveform_family=meta["waveform_family"],
            pipeline_type="self-vocoding",
            language="en",
            speaker_id=speaker_id,
            original_id=original_id,
            fake_path=str(fake_path),
            real_path=str(real_path),
            fake_duration=fake_qc.duration_s,
            real_duration=real_qc.duration_s,
            duration_ratio=(fake_qc.duration_s / real_qc.duration_s) if (fake_qc.duration_s and real_qc.duration_s) else None,
            active_duration_fake=fake_qc.active_duration_s,
            active_duration_real=real_qc.active_duration_s,
            clipping_rate_fake=fake_qc.clipping_rate,
            clipping_rate_real=real_qc.clipping_rate,
            silence_ratio_fake=fake_qc.silence_ratio,
            silence_ratio_real=real_qc.silence_ratio,
            fake_sha256=fake_qc.sha256,
            real_sha256=real_qc.sha256,
            sample_rate_original=fake_qc.sample_rate_original,
            qc_status=pair_qc_status,
            exclusion_reason=";".join(reasons),
        ))

        if (i + 1) % CONFIG["progress_every"] == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            rate = (i + 1 - start_idx) / elapsed if elapsed > 0 else 0
            remaining = (total - (i + 1)) / rate if rate > 0 else float("inf")
            print(f"  [{i+1}/{total}] {rate:.1f} pairs/s — ETA {remaining/60:.1f} min")

        if not CONFIG["quick_test_n"] and (i + 1) % CONFIG["checkpoint_every"] == 0:
            tmp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
            pd.DataFrame(rows).to_parquet(tmp, index=False)
            os.replace(tmp, checkpoint_path)
            meta_tmp = checkpoint_meta_path.with_suffix(checkpoint_meta_path.suffix + ".tmp")
            meta_tmp.write_text(json.dumps({
                "input_signature": input_sig,
                "config_signature": cfg_sig,
                "n_rows": len(rows),
            }, indent=2), encoding="utf-8")
            os.replace(meta_tmp, checkpoint_meta_path)
            print(f"  [CHECKPOINT] {len(rows)} pairs saved -> {checkpoint_path}")

    if not rows:
        raise RuntimeError("No pair was built; check CONFIG and the dataset layout.")

    df = write_manifest(rows, out_manifest)

    
    if not CONFIG["quick_test_n"]:
        for p in [checkpoint_path, checkpoint_meta_path]:
            if p.exists(): p.unlink()
        print("[CLEANUP] Signed checkpoint removed after the full run.")

    contradiction = df["qc_status"].eq("ok") & df["exclusion_reason"].fillna("").astype(str).str.strip().ne("")
    if contradiction.any():
        raise RuntimeError(f"{int(contradiction.sum())} rows have qc_status='ok' but a non-empty exclusion_reason.")
    summary = summarize_manifest(df)
    summary["by_generator"] = (
        df.groupby("independent_generator_id")["qc_status"]
        .value_counts().unstack(fill_value=0).to_dict(orient="index")
    )
    if not CONFIG["quick_test_n"]:
        if len(df) != 79_206:
            raise RuntimeError(f"LibriSeVoc manifest incomplete: {len(df):,}/79,206")
        if summary["n_ok"] != 77_890:
            raise RuntimeError(f"Unexpected LibriSeVoc QC count: {summary['n_ok']:,} pairs OK; 77,890 expected for the corrected v2 protocol.")
    n_generators = df["independent_generator_id"].nunique()
    coverage = df.groupby("original_id")["independent_generator_id"].nunique()
    n_incomplete = int((coverage < n_generators).sum())
    summary["n_generators_detected"] = int(n_generators)
    summary["n_originals_not_covered_by_all_generators"] = n_incomplete

    write_report(summary, CONFIG["output_dir"] / "librisevoc_pairing_report.json")

    print("\n=== LibriSeVoc SUMMARY ===")
    print(f"Candidate pairs : {summary['total_pairs_candidates']}")
    print(f"OK                : {summary['n_ok']}")
    print(f"Failures            : {summary['n_fail']}")
    print(f"Incomplete originals (not covered by all {n_generators} generators) : {n_incomplete}")
    print(f"Manifest: {out_manifest}")

    return df, summary


if __name__ == "__main__":
    build_and_validate()