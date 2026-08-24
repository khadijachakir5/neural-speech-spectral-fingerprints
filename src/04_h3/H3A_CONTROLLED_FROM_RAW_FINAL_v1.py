# ==================================================================================================
# H3a CONTROLLED — SAME EXACT CHECKPOINT × SAME LANGUAGE × TWO CORPORA v1.0.0
#
# Google Colab one-cell experiment
#
# QUESTION
# --------
# Does the EXACT SAME frozen neural vocoder checkpoint preserve its median log-power spectral
# fingerprint when the corpus changes while language is held constant?
#
# CONTROLLED DESIGN
# -----------------
# English corpus A: LJSpeech 1.1
# English corpus B: LibriTTS dev-clean
#
# The SAME downloaded checkpoint file (verified by SHA-256) is used for analysis-synthesis of both.
#
# Primary endpoint:
#   Δ_shape(g) =
#       Pearson(P[g,LJS], P[g,LibriTTS])
#       - symmetrized mean Pearson(P[g,corpus A], P[h!=g,corpus B])
#
# Primary frequency endpoint:
#   Δ_support(g) =
#       Jaccard_top10(P[g,LJS], P[g,LibriTTS])
#       - corresponding different-checkpoint baseline
#
# Primary inference unit = frozen checkpoint, not utterance.
# Exact one-sided sign-flip across checkpoints.
#
# IMPORTANT
# ---------
# - This is a NEW controlled experiment. It does not pretend that old WaveFake/LibriSeVoc checkpoints
#   were identical.
# - Each corpus is self-vocoded from real waveform-derived conditioning features using the same exact
#   checkpoint and the checkpoint's own official preprocessing configuration.
# - Residual:
#       r_raw(f) = log PSD_generated(f) - log PSD_real_reference(f)
#       r(f) = r_raw(f) - median_{80-7600 Hz}(r_raw(f))
# - Fingerprint:
#       P[g,D](f) = median over utterance residuals for checkpoint g and corpus D.
# ==================================================================================================

from __future__ import annotations

import os
import sys
import json
import math
import time
import shutil
import hashlib
import random
import tarfile
import subprocess
import itertools
from pathlib import Path
from datetime import datetime

# --------------------------------------------------------------------------------------------------
# 0. USER CONFIG
# --------------------------------------------------------------------------------------------------

VERSION = "H3A-CONTROLLED-SAME-CHECKPOINT-TWO-CORPORA-v1.1.0-A2Z"
SEED = 20260711

# "quick": scientific smoke/full-pipeline run
# "full" : manuscript-grade larger sample
MODE = os.environ.get("H3A_MODE", "quick").strip().lower()
if MODE not in {"quick", "full"}:
    raise ValueError("H3A_MODE must be quick or full")

N_PER_CORPUS = 50 if MODE == "quick" else 250
MAX_LIBRITTS_PER_SPEAKER = 3 if MODE == "quick" else 8

# Six exact frozen checkpoints, all official ParallelWaveGAN pretrained tags with 22.05 kHz,
# 80–7600 Hz mel range, FFT 1024, hop 256.
# Five are enough to make exact one-sided sign-flip p_min = 1/32 = 0.03125.
CHECKPOINT_TAGS = [
    "ljspeech_parallel_wavegan.v1",
    "ljspeech_melgan.v1",
    "ljspeech_full_band_melgan.v1",
    "ljspeech_multi_band_melgan.v2",
    "ljspeech_hifigan.v1",
    "ljspeech_style_melgan.v1",
]

# If True, download corpora if not already present.
# LJSpeech is the full 1.1 archive; LibriTTS uses the smaller dev-clean archive.
DOWNLOAD_CORPORA_IF_MISSING = True

# If already downloaded elsewhere, you can set these manually.
LJSPEECH_ROOT_OVERRIDE = None
LIBRITTS_ROOT_OVERRIDE = None

# Synthesis/restart behavior
FORCE_RESYNTHESIS = False
INSTALL_DEPS = os.environ.get("H3A_INSTALL_DEPS", "1") == "1"

# Fingerprint extraction: harmonized with existing work
ANALYSIS_SR = 16000
N_FFT = 1024
HOP_LENGTH = 256
FMIN = 80.0
FMAX = 7600.0
EPS = 1e-10

# Bootstrap is over utterances within each checkpoint/corpus, descriptive.
N_CONTENT_BOOT = 1000 if MODE == "full" else 300
# Group inference is across checkpoints.
N_GROUP_BOOT = 10000

# --------------------------------------------------------------------------------------------------
# 1. PATHS
# --------------------------------------------------------------------------------------------------

ROOT = Path(os.environ.get("FINGERPRINT_OUTPUT_ROOT", "/content/drive/MyDrive/fingerprint_q1_outputs"))
OUTROOT = ROOT / "H3A_CONTROLLED_SAME_CHECKPOINT_TWO_CORPORA_v1"
RUN = OUTROOT / ("run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))

CACHE = ROOT / "_controlled_h3a_cache"
CORPORA_ROOT = CACHE / "corpora"
MODEL_ROOT = CACHE / "parallel_wavegan_pretrained"
INPUT_ROOT = CACHE / "selected_input_22050"
PWG_DUMP_ROOT = CACHE / "pwg_dump"
SYNTH_ROOT = CACHE / "synthesized"

RUN.mkdir(parents=True, exist_ok=True)
CORPORA_ROOT.mkdir(parents=True, exist_ok=True)
MODEL_ROOT.mkdir(parents=True, exist_ok=True)
INPUT_ROOT.mkdir(parents=True, exist_ok=True)
PWG_DUMP_ROOT.mkdir(parents=True, exist_ok=True)
SYNTH_ROOT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------------------
# 2. DRIVE + DEPENDENCIES
# --------------------------------------------------------------------------------------------------

try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
except Exception:
    print("[INFO] Google Colab not detected.")

def run_cmd(cmd, check=True, cwd=None):
    print("[CMD]", " ".join(map(str, cmd)))
    return subprocess.run(list(map(str,cmd)), check=check, cwd=cwd)

if INSTALL_DEPS:
    # Avoid hiding failures: install exact ParallelWaveGAN release plus explicit runtime deps.
    run_cmd([
        sys.executable, "-m", "pip", "install", "-q",
        "parallel-wavegan==0.6.1",
        "librosa>=0.10.2",
        "resampy>=0.4.3",
        "soundfile>=0.12",
        "h5py>=3.10",
        "PyYAML>=6",
        "gdown>=5",
        "scipy>=1.11",
        "pandas>=2.0",
    ])

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import h5py
import yaml
import torch
from scipy import signal
from scipy.stats import spearmanr, wasserstein_distance
from parallel_wavegan.utils import download_pretrained_model, PRETRAINED_MODEL_LIST

# --------------------------------------------------------------------------------------------------
# 3. HELPERS
# --------------------------------------------------------------------------------------------------

def banner(s):
    print("\n" + "="*126)
    print(s)
    print("="*126)

def sha256_file(path, block=1024*1024):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        while True:
            b=f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def atomic_csv(df,path):
    path=Path(path)
    tmp=path.with_suffix(path.suffix+".tmp")
    df.to_csv(tmp,index=False)
    os.replace(tmp,path)

def atomic_json(obj,path):
    path=Path(path)
    tmp=path.with_suffix(path.suffix+".tmp")
    with open(tmp,"w",encoding="utf-8") as f:
        json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=True)
    os.replace(tmp,path)

def stable_seed(*parts):
    return int(hashlib.sha256("|".join(map(str,parts)).encode()).hexdigest()[:8],16)

def safe_mean(x):
    a=np.asarray(list(x) if not isinstance(x,np.ndarray) else x,dtype=float).ravel()
    a=a[np.isfinite(a)]
    return float(np.mean(a)) if len(a) else np.nan

def percentile_ci(x):
    a=np.asarray(x,float)
    a=a[np.isfinite(a)]
    if not len(a):
        return [np.nan,np.nan]
    return [float(np.percentile(a,2.5)),float(np.percentile(a,97.5))]

def exact_signflip_p_one_sided(v):
    v=np.asarray(v,float)
    v=v[np.isfinite(v)]
    n=len(v)
    if n==0:
        return np.nan
    obs=float(np.mean(v))
    if n<=20:
        null=[]
        for signs in itertools.product([-1.0,1.0],repeat=n):
            null.append(float(np.mean(v*np.asarray(signs,float))))
        return float(np.mean(np.asarray(null)>=obs-1e-15))
    rng=np.random.default_rng(stable_seed("signflip",SEED,n))
    B=200000
    cnt=0
    for _ in range(B):
        s=rng.choice([-1.0,1.0],size=n)
        cnt += np.mean(v*s)>=obs-1e-15
    return float((cnt+1)/(B+1))

def pearson(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if np.std(a)==0 or np.std(b)==0:
        return np.nan
    return float(np.corrcoef(a,b)[0,1])

def cosine(a,b):
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/den) if den>0 else np.nan

def support_mask(x,q=0.90):
    a=np.abs(np.asarray(x,float))
    return a>=np.quantile(a,q)

def jaccard(a,b,q=0.90):
    ma=support_mask(a,q); mb=support_mask(b,q)
    u=ma|mb
    return float(np.sum(ma&mb)/np.sum(u)) if np.any(u) else np.nan

def signed_jaccard(a,b,q=0.90):
    ma=support_mask(a,q); mb=support_mask(b,q)
    u=ma|mb
    inter=ma&mb&(np.sign(a)==np.sign(b))
    return float(np.sum(inter)/np.sum(u)) if np.any(u) else np.nan

def compare_profiles(a,b,freq):
    a=np.asarray(a,float); b=np.asarray(b,float)
    wa=np.abs(a); wb=np.abs(b)
    out={
        "pearson":pearson(a,b),
        "cosine":cosine(a,b),
        "spearman":float(spearmanr(a,b).statistic),
        "rmse":float(np.sqrt(np.mean((a-b)**2))),
        "mae":float(np.mean(np.abs(a-b))),
        "jaccard_top10":jaccard(a,b,0.90),
        "signed_jaccard_top10":signed_jaccard(a,b,0.90),
        "jaccard_top5":jaccard(a,b,0.95),
        "jaccard_top15":jaccard(a,b,0.85),
    }
    out["wasserstein_hz"]=float(
        wasserstein_distance(freq,freq,u_weights=wa,v_weights=wb)
    ) if wa.sum()>0 and wb.sum()>0 else np.nan
    return out

# --------------------------------------------------------------------------------------------------
# 4. DOWNLOAD / LOCATE CORPORA
# --------------------------------------------------------------------------------------------------

banner("STEP A — CORPORA")

LJS_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
LIBRITTS_URL = "https://www.openslr.org/resources/60/dev-clean.tar.gz"

def wget_resume(url,dst):
    dst=Path(dst)
    if dst.is_file() and dst.stat().st_size>0:
        print("[CACHE]",dst)
        return
    run_cmd(["wget","-c","-O",dst,url])

def extract_tar(archive,outdir):
    outdir=Path(outdir)
    marker=outdir/(".EXTRACTED_"+Path(archive).name.replace(".","_"))
    if marker.exists():
        return
    print("[EXTRACT]",archive)
    with tarfile.open(archive,"r:*") as tf:
        tf.extractall(outdir)
    marker.write_text("ok\n")

if LJSPEECH_ROOT_OVERRIDE:
    LJS_ROOT=Path(LJSPEECH_ROOT_OVERRIDE)
else:
    LJS_ROOT=CORPORA_ROOT/"LJSpeech-1.1"

if LIBRITTS_ROOT_OVERRIDE:
    LIB_ROOT=Path(LIBRITTS_ROOT_OVERRIDE)
else:
    LIB_ROOT=CORPORA_ROOT/"LibriTTS"/"dev-clean"

if not LJS_ROOT.exists() and DOWNLOAD_CORPORA_IF_MISSING:
    arc=CORPORA_ROOT/"LJSpeech-1.1.tar.bz2"
    wget_resume(LJS_URL,arc)
    extract_tar(arc,CORPORA_ROOT)

if not LIB_ROOT.exists() and DOWNLOAD_CORPORA_IF_MISSING:
    arc=CORPORA_ROOT/"libritts-dev-clean.tar.gz"
    wget_resume(LIBRITTS_URL,arc)
    extract_tar(arc,CORPORA_ROOT)

if not LJS_ROOT.exists():
    raise FileNotFoundError(f"LJSpeech root missing: {LJS_ROOT}")
if not LIB_ROOT.exists():
    raise FileNotFoundError(f"LibriTTS root missing: {LIB_ROOT}")

ljs_files=sorted((LJS_ROOT/"wavs").glob("*.wav"))
lib_files=sorted(LIB_ROOT.glob("*/*/*.wav"))

print("LJSpeech wavs:",len(ljs_files))
print("LibriTTS dev-clean wavs:",len(lib_files))
if len(ljs_files)<N_PER_CORPUS or len(lib_files)<N_PER_CORPUS:
    raise RuntimeError("Not enough files for requested N_PER_CORPUS.")

# --------------------------------------------------------------------------------------------------
# 5. DETERMINISTIC SAMPLE SELECTION
# --------------------------------------------------------------------------------------------------

banner("STEP B — SAMPLE SELECTION")

rng=np.random.default_rng(SEED)
ljs_sel=[ljs_files[i] for i in rng.choice(len(ljs_files),size=N_PER_CORPUS,replace=False)]

# LibriTTS: cap files per speaker so the second corpus is not dominated by a few speakers.
by_spk={}
for p in lib_files:
    spk=p.parts[-3]
    by_spk.setdefault(spk,[]).append(p)

lib_candidates=[]
rng2=np.random.default_rng(SEED+17)
for spk in sorted(by_spk):
    arr=by_spk[spk]
    take=min(MAX_LIBRITTS_PER_SPEAKER,len(arr))
    idx=rng2.choice(len(arr),size=take,replace=False)
    lib_candidates.extend([arr[i] for i in idx])

if len(lib_candidates)<N_PER_CORPUS:
    lib_candidates=lib_files
lib_sel=[lib_candidates[i] for i in rng2.choice(len(lib_candidates),size=N_PER_CORPUS,replace=False)]

selection=[]
for corpus,files in [("LJSpeech",ljs_sel),("LibriTTS",lib_sel)]:
    for i,p in enumerate(files):
        selection.append({
            "corpus":corpus,
            "selection_index":i,
            "source_path":str(p),
            "source_sha256":sha256_file(p),
        })
selection=pd.DataFrame(selection)
atomic_csv(selection,RUN/"selected_real_files_manifest.csv")
print(selection.groupby("corpus").size())

# --------------------------------------------------------------------------------------------------
# 6. PREPARE 22.05 kHz INPUTS
# --------------------------------------------------------------------------------------------------

banner("STEP C — PREPARE COMMON VOCODER INPUT RATE")

VOCODER_SR=22050

def read_mono(path):
    y,sr=sf.read(path,dtype="float32",always_2d=False)
    if y.ndim>1:
        y=np.mean(y,axis=1)
    return np.asarray(y,np.float32),int(sr)

def resample(y,orig_sr,target_sr):
    if orig_sr==target_sr:
        return y.astype(np.float32,copy=False)
    try:
        return librosa.resample(
            y,
            orig_sr=orig_sr,
            target_sr=target_sr,
            res_type="kaiser_best",
        ).astype(np.float32)
    except ModuleNotFoundError as e:
        # Robust fallback if the optional resampy backend is unavailable.
        from scipy.signal import resample_poly
        g=math.gcd(int(orig_sr),int(target_sr))
        up=int(target_sr)//g
        down=int(orig_sr)//g
        return resample_poly(y,up,down).astype(np.float32)

prepared_rows=[]
for corpus,files in [("LJSpeech",ljs_sel),("LibriTTS",lib_sel)]:
    d=INPUT_ROOT/corpus
    d.mkdir(parents=True,exist_ok=True)
    for i,p in enumerate(files):
        uid=f"{corpus.lower()}_{i:05d}"
        out=d/f"{uid}.wav"
        if not out.is_file():
            y,sr=read_mono(p)
            y=resample(y,sr,VOCODER_SR)
            peak=float(np.max(np.abs(y))) if len(y) else 0.0
            if peak>0.999:
                y=y/(peak+1e-12)*0.999
            sf.write(out,y,VOCODER_SR,subtype="PCM_16")
        prepared_rows.append({
            "corpus":corpus,
            "uid":uid,
            "source_path":str(p),
            "prepared_path":str(out),
            "prepared_sha256":sha256_file(out),
        })

prepared=pd.DataFrame(prepared_rows)
atomic_csv(prepared,RUN/"prepared_inputs_manifest.csv")

# --------------------------------------------------------------------------------------------------
# 7. DOWNLOAD EXACT CHECKPOINTS + PROVENANCE FREEZE
# --------------------------------------------------------------------------------------------------

banner("STEP D — FREEZE EXACT CHECKPOINTS")

available=set(PRETRAINED_MODEL_LIST.keys())
missing_tags=[t for t in CHECKPOINT_TAGS if t not in available]
if missing_tags:
    raise RuntimeError(
        "Pretrained tags not available in installed ParallelWaveGAN: "
        + str(missing_tags)
        + "\nAvailable examples: "
        + str(sorted(list(available))[:30])
    )

checkpoint_rows=[]
model_info={}

for tag in CHECKPOINT_TAGS:
    download_pretrained_model(tag,str(MODEL_ROOT))
    mdir=MODEL_ROOT/tag
    if not mdir.is_dir():
        # Some versions may return/download under a slightly different nested directory.
        candidates=[p for p in MODEL_ROOT.rglob(tag) if p.is_dir()]
        if candidates:
            mdir=candidates[0]
    configs=sorted(mdir.glob("config*.yml"))+sorted(mdir.glob("config*.yaml"))
    checkpoints=sorted(mdir.glob("checkpoint-*steps.pkl"))
    stats=sorted(mdir.glob("stats.h5"))
    if not configs or not checkpoints or not stats:
        raise RuntimeError(f"Incomplete pretrained model folder for {tag}: {mdir}")
    config=configs[0]
    checkpoint=checkpoints[-1]
    statsf=stats[0]
    cfg=yaml.safe_load(config.read_text())

    fs=int(cfg.get("sampling_rate",cfg.get("fs",VOCODER_SR)))
    fft=int(cfg.get("fft_size",N_FFT))
    hop=int(cfg.get("hop_size",HOP_LENGTH))
    fmin=float(cfg.get("fmin",80))
    fmax=float(cfg.get("fmax",7600))
    if fs!=VOCODER_SR:
        raise RuntimeError(f"{tag}: expected 22050 Hz, got {fs}")

    info={
        "tag":tag,
        "model_dir":str(mdir),
        "config":str(config),
        "checkpoint":str(checkpoint),
        "stats":str(statsf),
        "checkpoint_sha256":sha256_file(checkpoint),
        "config_sha256":sha256_file(config),
        "stats_sha256":sha256_file(statsf),
        "sampling_rate":fs,
        "fft_size":fft,
        "hop_size":hop,
        "fmin":fmin,
        "fmax":fmax,
    }
    model_info[tag]=info
    checkpoint_rows.append(info)

checkpoints_df=pd.DataFrame(checkpoint_rows)
atomic_csv(checkpoints_df,RUN/"frozen_checkpoint_provenance.csv")
print(checkpoints_df[["tag","checkpoint_sha256","sampling_rate","fft_size","hop_size","fmin","fmax"]].to_string(index=False))

# --------------------------------------------------------------------------------------------------
# 8. ANALYSIS-SYNTHESIS WITH SAME EXACT CHECKPOINT FOR BOTH CORPORA
# --------------------------------------------------------------------------------------------------

banner("STEP E — ANALYSIS-SYNTHESIS")

if torch.cuda.is_available():
    print("[GPU]",torch.cuda.get_device_name(0))
else:
    print("[WARN] CUDA GPU not detected. Synthesis will be much slower.")

synth_manifest=[]

for tag in CHECKPOINT_TAGS:
    info=model_info[tag]
    for corpus in ["LJSpeech","LibriTTS"]:
        input_dir=INPUT_ROOT/corpus
        raw_dir=PWG_DUMP_ROOT/tag/corpus/"raw"
        out_dir=SYNTH_ROOT/tag/corpus
        raw_dir.mkdir(parents=True,exist_ok=True)
        out_dir.mkdir(parents=True,exist_ok=True)

        expected=len(list(out_dir.glob("*_gen.wav")))
        if FORCE_RESYNTHESIS or expected < N_PER_CORPUS:
            if FORCE_RESYNTHESIS:
                shutil.rmtree(raw_dir,ignore_errors=True)
                shutil.rmtree(out_dir,ignore_errors=True)
                raw_dir.mkdir(parents=True,exist_ok=True)
                out_dir.mkdir(parents=True,exist_ok=True)

            # Official analysis-synthesis path:
            # waveform -> checkpoint-config features -> on-the-fly normalization -> same checkpoint decode.
            run_cmd([
                "parallel-wavegan-preprocess",
                "--config",info["config"],
                "--rootdir",input_dir,
                "--dumpdir",raw_dir,
            ])
            run_cmd([
                "parallel-wavegan-decode",
                "--checkpoint",info["checkpoint"],
                "--dumpdir",raw_dir,
                "--normalize-before",
                "--outdir",out_dir,
            ])
        else:
            print(f"[CACHE] {tag} / {corpus}: {expected} generated wavs")

        for _,row in prepared[prepared["corpus"].eq(corpus)].iterrows():
            uid=row["uid"]
            candidates=[
                out_dir/f"{uid}_gen.wav",
                out_dir/f"{uid}.wav",
            ]
            gen=next((p for p in candidates if p.is_file()),None)
            if gen is None:
                hits=list(out_dir.glob(f"{uid}*gen*.wav"))
                gen=hits[0] if hits else None
            if gen is None:
                raise FileNotFoundError(f"Generated waveform missing for {tag}/{corpus}/{uid}")

            # Find corresponding PWG raw HDF5, which typically contains the preprocessed waveform.
            h5hits=list(raw_dir.glob(f"{uid}*.h5"))
            raw_h5=h5hits[0] if h5hits else None

            synth_manifest.append({
                "checkpoint_tag":tag,
                "checkpoint_sha256":info["checkpoint_sha256"],
                "corpus":corpus,
                "uid":uid,
                "input_wav":row["prepared_path"],
                "pwg_raw_h5":str(raw_h5) if raw_h5 else "",
                "generated_wav":str(gen),
                "generated_sha256":sha256_file(gen),
            })

synth_manifest=pd.DataFrame(synth_manifest)
atomic_csv(synth_manifest,RUN/"synthesis_manifest.csv")

# Hard identity check: same SHA is used for both corpora for every checkpoint.
identity_check=[]
for tag,sub in synth_manifest.groupby("checkpoint_tag"):
    hashes=sorted(sub["checkpoint_sha256"].unique())
    corpora=sorted(sub["corpus"].unique())
    ok=(len(hashes)==1 and set(corpora)=={"LJSpeech","LibriTTS"})
    identity_check.append({
        "checkpoint_tag":tag,
        "checkpoint_sha256":hashes[0] if len(hashes)==1 else "|".join(hashes),
        "corpora":"|".join(corpora),
        "exact_same_checkpoint_verified":bool(ok),
    })
identity_check=pd.DataFrame(identity_check)
atomic_csv(identity_check,RUN/"exact_checkpoint_identity_check.csv")
if not identity_check["exact_same_checkpoint_verified"].all():
    raise RuntimeError("Exact checkpoint identity check FAILED.")
print("[PASS] Exact same checkpoint SHA-256 verified across both corpora for every tag.")

# --------------------------------------------------------------------------------------------------
# 9. REAL REFERENCE USED BY PREPROCESSING
# --------------------------------------------------------------------------------------------------

banner("STEP F — BUILD TRUE REAL REFERENCES")

def load_preprocessed_wave(h5path,fallback):
    if h5path and Path(h5path).is_file():
        with h5py.File(h5path,"r") as h:
            # Most PWG raw dumps contain "wave"; inspect alternatives defensively.
            for key in ["wave","wav","audio"]:
                if key in h:
                    y=np.asarray(h[key],dtype=np.float32).squeeze()
                    if y.ndim==1 and len(y)>0:
                        return y,VOCODER_SR,"PWG_RAW_H5_"+key
    y,sr=read_mono(fallback)
    return y,sr,"PREPARED_INPUT_FALLBACK"

# --------------------------------------------------------------------------------------------------
# 10. FILE-LEVEL RESIDUAL EXTRACTION
# --------------------------------------------------------------------------------------------------

banner("STEP G — EXTRACT HARMONIZED RESIDUALS")

freq_full=np.fft.rfftfreq(N_FFT,d=1.0/ANALYSIS_SR)
band=(freq_full>=FMIN)&(freq_full<=FMAX)
FREQ=freq_full[band]
print(f"Analysis band actual centers: {FREQ[0]:.3f}–{FREQ[-1]:.3f} Hz | bins={len(FREQ)}")

axis_df=pd.DataFrame({
    "bin_index":np.arange(len(FREQ)),
    "frequency_hz":FREQ,
    "column_name":[f"r_{i:03d}" for i in range(len(FREQ))]
})
RES_COLS=axis_df["column_name"].tolist()
atomic_csv(axis_df,RUN/"frequency_axis.csv")

def median_log_power(y,sr):
    y=np.asarray(y,np.float32)
    y=resample(y,sr,ANALYSIS_SR)
    if len(y)<N_FFT:
        y=np.pad(y,(0,N_FFT-len(y)))
    f,t,Z=signal.stft(
        y,
        fs=ANALYSIS_SR,
        window="hann",
        nperseg=N_FFT,
        noverlap=N_FFT-HOP_LENGTH,
        nfft=N_FFT,
        boundary=None,
        padded=False,
    )
    P=(np.abs(Z)**2)+EPS
    return np.median(np.log(P),axis=1)

res_rows=[]
res_vectors=[]

for idx,row in synth_manifest.iterrows():
    real_y,real_sr,real_source=load_preprocessed_wave(row["pwg_raw_h5"],row["input_wav"])
    gen_y,gen_sr=read_mono(row["generated_wav"])

    # Median-spectrum residual; frame-by-frame duration alignment is not required.
    real_lp=median_log_power(real_y,real_sr)
    gen_lp=median_log_power(gen_y,gen_sr)
    rraw=gen_lp-real_lp
    r=rraw[band]
    r=r-np.median(r)

    res_rows.append({
        "checkpoint_tag":row["checkpoint_tag"],
        "checkpoint_sha256":row["checkpoint_sha256"],
        "corpus":row["corpus"],
        "uid":row["uid"],
        "real_reference_source":real_source,
        "n_real_samples":len(real_y),
        "n_generated_samples":len(gen_y),
    })
    res_vectors.append(r.astype(np.float32))

res_meta=pd.DataFrame(res_rows)
R=np.vstack(res_vectors)
pair_level=pd.concat([res_meta,pd.DataFrame(R,columns=RES_COLS)],axis=1)
pair_level.to_parquet(RUN/"pair_level_controlled_residuals.parquet",index=False)

# QC
qc=res_meta.groupby(["checkpoint_tag","corpus"]).agg(
    n_pairs=("uid","size"),
    real_reference_sources=("real_reference_source",lambda x:"|".join(sorted(set(x)))),
).reset_index()
atomic_csv(qc,RUN/"residual_extraction_qc.csv")
print(qc.to_string(index=False))

# --------------------------------------------------------------------------------------------------
# 11. CHECKPOINT × CORPUS FINGERPRINTS
# --------------------------------------------------------------------------------------------------

banner("STEP H — BUILD FINGERPRINTS")

profiles={}
profile_rows=[]

for (tag,corpus),idxs in res_meta.groupby(["checkpoint_tag","corpus"]).groups.items():
    ii=np.asarray(list(idxs),dtype=int)
    v=np.median(R[ii],axis=0)
    profiles[(tag,corpus)]=v
    profile_rows.append({
        "checkpoint_tag":tag,
        "checkpoint_sha256":res_meta.loc[ii[0],"checkpoint_sha256"],
        "corpus":corpus,
        "n_pairs":len(ii),
    })

profile_meta=pd.DataFrame(profile_rows)
profile_X=np.vstack([profiles[(r["checkpoint_tag"],r["corpus"])] for _,r in profile_meta.iterrows()])
profile_wide=pd.concat([profile_meta,pd.DataFrame(profile_X,columns=RES_COLS)],axis=1)
atomic_csv(profile_wide,RUN/"checkpoint_corpus_fingerprints.csv")

# --------------------------------------------------------------------------------------------------
# 12. H3a PRIMARY TEST: SAME EXACT CHECKPOINT VS DIFFERENT CHECKPOINT BASELINE
# --------------------------------------------------------------------------------------------------

banner("STEP I — H3a PRIMARY CROSS-CORPUS TEST")

rows=[]

for g in CHECKPOINT_TAGS:
    A=profiles[(g,"LJSpeech")]
    B=profiles[(g,"LibriTTS")]
    same=compare_profiles(A,B,FREQ)

    controls=[]
    for h in CHECKPOINT_TAGS:
        if h==g:
            continue
        # Symmetric orientations
        controls.append(compare_profiles(A,profiles[(h,"LibriTTS")],FREQ))
        controls.append(compare_profiles(profiles[(h,"LJSpeech")],B,FREQ))

    baseline={m:safe_mean([x[m] for x in controls]) for m in same}

    rec={
        "checkpoint_tag":g,
        "checkpoint_sha256":model_info[g]["checkpoint_sha256"],
        "n_control_comparisons":len(controls),
    }
    for m in same:
        rec[f"{m}_same_exact"]=same[m]
        rec[f"{m}_different_checkpoint"]=baseline[m]

        if m in ["rmse","mae","wasserstein_hz"]:
            # Positive delta means exact checkpoint is closer/better.
            rec[f"delta_{m}"]=baseline[m]-same[m]
        else:
            rec[f"delta_{m}"]=same[m]-baseline[m]
    rows.append(rec)

per_checkpoint=pd.DataFrame(rows)
atomic_csv(per_checkpoint,RUN/"H3a_per_checkpoint_results.csv")
print(per_checkpoint[[
    "checkpoint_tag",
    "pearson_same_exact","pearson_different_checkpoint","delta_pearson",
    "jaccard_top10_same_exact","jaccard_top10_different_checkpoint","delta_jaccard_top10",
]].to_string(index=False))

# --------------------------------------------------------------------------------------------------
# 13. CONTENT BOOTSTRAP PER CHECKPOINT (DESCRIPTIVE)
# --------------------------------------------------------------------------------------------------

banner("STEP J — CONTENT BOOTSTRAP")

boot_rows=[]

for g in CHECKPOINT_TAGS:
    ia=np.asarray(list(res_meta[(res_meta["checkpoint_tag"].eq(g))&(res_meta["corpus"].eq("LJSpeech"))].index),dtype=int)
    ib=np.asarray(list(res_meta[(res_meta["checkpoint_tag"].eq(g))&(res_meta["corpus"].eq("LibriTTS"))].index),dtype=int)
    rngb=np.random.default_rng(stable_seed("content_boot",g,SEED))
    bp=[]; bj=[]
    for b in range(N_CONTENT_BOOT):
        sa=rngb.choice(ia,size=len(ia),replace=True)
        sb=rngb.choice(ib,size=len(ib),replace=True)
        pa=np.median(R[sa],axis=0)
        pb=np.median(R[sb],axis=0)
        bp.append(pearson(pa,pb))
        bj.append(jaccard(pa,pb,0.90))
    boot_rows.append({
        "checkpoint_tag":g,
        "pearson_same_exact":pearson(profiles[(g,"LJSpeech")],profiles[(g,"LibriTTS")]),
        "pearson_content_boot_ci95_low":percentile_ci(bp)[0],
        "pearson_content_boot_ci95_high":percentile_ci(bp)[1],
        "jaccard_top10_same_exact":jaccard(profiles[(g,"LJSpeech")],profiles[(g,"LibriTTS")],0.90),
        "jaccard_content_boot_ci95_low":percentile_ci(bj)[0],
        "jaccard_content_boot_ci95_high":percentile_ci(bj)[1],
    })

content_boot=pd.DataFrame(boot_rows)
atomic_csv(content_boot,RUN/"H3a_content_bootstrap_per_checkpoint.csv")

# --------------------------------------------------------------------------------------------------
# 14. GROUP INFERENCE ACROSS CHECKPOINTS
# --------------------------------------------------------------------------------------------------

banner("STEP K — CHECKPOINT-LEVEL INFERENCE")

primary_cols=["delta_pearson","delta_jaccard_top10"]
group={}
rngg=np.random.default_rng(stable_seed("group_boot",SEED))

for col in primary_cols:
    v=per_checkpoint[col].to_numpy(float)
    boots=[]
    for _ in range(N_GROUP_BOOT):
        s=rngg.choice(v,size=len(v),replace=True)
        boots.append(float(np.mean(s)))
    group[col]={
        "n_checkpoints":len(v),
        "mean_delta":float(np.mean(v)),
        "median_delta":float(np.median(v)),
        "bootstrap_ci95":percentile_ci(boots),
        "exact_signflip_p_one_sided":exact_signflip_p_one_sided(v),
        "positive_checkpoints":int(np.sum(v>0)),
        "zero_checkpoints":int(np.sum(v==0)),
        "negative_checkpoints":int(np.sum(v<0)),
        "minimum_attainable_exact_p":float(1/(2**len(v))),
    }

shape=group["delta_pearson"]
freq=group["delta_jaccard_top10"]

shape_supported=(
    shape["mean_delta"]>0 and
    shape["bootstrap_ci95"][0]>0 and
    shape["exact_signflip_p_one_sided"]<=ALPHA
)
freq_supported=(
    freq["mean_delta"]>0 and
    freq["bootstrap_ci95"][0]>0 and
    freq["exact_signflip_p_one_sided"]<=ALPHA
)

if shape_supported and freq_supported:
    STATUS="SUPPORTED"
elif shape_supported:
    STATUS="SHAPE_SUPPORTED_FREQUENCY_INSUFFICIENT"
else:
    STATUS="INSUFFICIENT_EVIDENCE"

answer=pd.DataFrame([{
    "question":"Same exact frozen checkpoint + same language: does the spectral fingerprint persist when corpus changes?",
    "status":STATUS,
    "n_exact_checkpoints":len(CHECKPOINT_TAGS),
    "mean_delta_pearson":shape["mean_delta"],
    "delta_pearson_ci95":str(shape["bootstrap_ci95"]),
    "delta_pearson_exact_p":shape["exact_signflip_p_one_sided"],
    "positive_shape_checkpoints":shape["positive_checkpoints"],
    "mean_delta_jaccard_top10":freq["mean_delta"],
    "delta_jaccard_ci95":str(freq["bootstrap_ci95"]),
    "delta_jaccard_exact_p":freq["exact_signflip_p_one_sided"],
    "positive_frequency_checkpoints":freq["positive_checkpoints"],
}])
atomic_csv(answer,RUN/"H3a_CONTROLLED_ANSWER_TABLE.csv")

summary={
    "version":VERSION,
    "mode":MODE,
    "run_dir":str(RUN),
    "question":"Same exact frozen checkpoint, same English language, different corpus.",
    "corpus_A":"LJSpeech 1.1",
    "corpus_B":"LibriTTS dev-clean",
    "n_per_corpus":N_PER_CORPUS,
    "checkpoint_tags":CHECKPOINT_TAGS,
    "exact_checkpoint_identity_all_pass":bool(identity_check["exact_same_checkpoint_verified"].all()),
    "analysis":{
        "sample_rate":ANALYSIS_SR,
        "n_fft":N_FFT,
        "hop_length":HOP_LENGTH,
        "requested_band_hz":[FMIN,FMAX],
        "actual_band_centers_hz":[float(FREQ[0]),float(FREQ[-1])],
        "n_bins":len(FREQ),
        "residual":"median log-power generated minus median log-power real, then median-centered over analysis band",
    },
    "primary_endpoints":{
        "shape":"Pearson delta: exact checkpoint cross-corpus minus symmetrized different-checkpoint baseline",
        "frequency":"top10% Jaccard delta: exact checkpoint cross-corpus minus symmetrized different-checkpoint baseline",
    },
    "group_inference":group,
    "status":STATUS,
    "important_interpretation":{
        "what_is_controlled":"checkpoint identity and language",
        "what_changes":"corpus/content/speaker distribution",
        "unit_of_inference":"checkpoint",
        "not_claimed":"This does not prove independence from every possible dataset shift; it tests two English corpora under a frozen analysis-synthesis protocol.",
    }
}
atomic_json(summary,RUN/"H3a_CONTROLLED_MASTER_SUMMARY.json")

# Normalized summary consumed by the paper-level H3 aggregator.
# The raw controlled experiment remains the source of truth.
compat_summary = {
    "overall": STATUS,
    "shape": group["delta_pearson"],
    "frequency": group["delta_jaccard_top10"],
    "source_raw_summary": str(RUN / "H3a_CONTROLLED_MASTER_SUMMARY.json"),
    "mode": MODE,
    "n_per_corpus": N_PER_CORPUS,
}
atomic_json(compat_summary, RUN / "H3a_controlled_summary.json")
shutil.copy2(RUN / "exact_checkpoint_identity_check.csv", RUN / "H3a_exact_checkpoint_identity_check.csv")

# --------------------------------------------------------------------------------------------------
# 15. FINAL OUTPUT
# --------------------------------------------------------------------------------------------------

banner("FINAL ANSWER")
print(answer.to_string(index=False))
print("\nCheckpoint-level primary results:")
print(per_checkpoint[[
    "checkpoint_tag","delta_pearson","delta_jaccard_top10"
]].to_string(index=False))

print("\nKEY FILES")
for f in [
    "H3a_CONTROLLED_MASTER_SUMMARY.json",
    "H3a_CONTROLLED_ANSWER_TABLE.csv",
    "frozen_checkpoint_provenance.csv",
    "exact_checkpoint_identity_check.csv",
    "selected_real_files_manifest.csv",
    "synthesis_manifest.csv",
    "pair_level_controlled_residuals.parquet",
    "checkpoint_corpus_fingerprints.csv",
    "H3a_per_checkpoint_results.csv",
    "H3a_content_bootstrap_per_checkpoint.csv",
]:
    print(" -",RUN/f)

print("\nOutput directory:",RUN)