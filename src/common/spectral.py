#!/usr/bin/env python3
# Purpose: Provide the canonical spectral configuration and residual-spectrum extraction primitives used by shared validation and robustness controls.

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import soundfile as sf
from scipy.fft import rfft
from scipy.signal import resample_poly
from scipy.signal.windows import hann

@dataclass(frozen=True)
class SpectralConfig:
    target_sr: int = 16000
    n_fft: int = 1024
    hop_length: int = 256
    remove_dc: bool = True
    epsilon_power: float = 1e-12
    vad_reference_percentile: float = 95.0
    vad_top_db: float = 40.0
    vad_abs_db: float = -80.0
    min_active_frames: int = 3
    centering_min_hz: float = 80.0
    centering_max_hz: float = 7600.0


def load_audio(path: str|Path, cfg: SpectralConfig=SpectralConfig()):
    path=Path(path)
    y,sr=sf.read(str(path),dtype='float32',always_2d=False)
    if y.ndim==2: y=y.mean(axis=1,dtype=np.float32)
    if y.ndim!=1 or y.size==0 or not np.isfinite(y).all():
        raise ValueError(f'invalid audio: {path}')
    sr=int(sr)
    if sr!=cfg.target_sr:
        g=math.gcd(sr,cfg.target_sr)
        y=resample_poly(y,cfg.target_sr//g,sr//g).astype(np.float32,copy=False)
    if cfg.remove_dc:
        y=y-np.float32(y.mean(dtype=np.float64))
    return np.asarray(y,np.float32),sr


def frame_audio(y: np.ndarray,cfg: SpectralConfig):
    if len(y)<cfg.n_fft: y=np.pad(y,(0,cfg.n_fft-len(y)))
    f=np.lib.stride_tricks.sliding_window_view(y,cfg.n_fft)[::cfg.hop_length]
    return f if len(f) else y[:cfg.n_fft][None,:]


def active_mask(frames: np.ndarray,cfg: SpectralConfig):
    rms=np.sqrt(np.mean(frames.astype(np.float64)**2,axis=1))
    db=20*np.log10(rms+1e-12)
    ref=float(np.percentile(db,cfg.vad_reference_percentile))
    thr=max(ref-cfg.vad_top_db,cfg.vad_abs_db)
    active=db>=thr
    if int(active.sum())<cfg.min_active_frames:
        k=min(cfg.min_active_frames,len(frames))
        ix=np.argsort(rms)[-k:]
        active=np.zeros(len(frames),dtype=bool); active[ix]=True
    return active


def extract_log_spectrum(path: str|Path,cfg: SpectralConfig=SpectralConfig()):
    y,_=load_audio(path,cfg)
    frames=frame_audio(y,cfg)
    a=active_mask(frames,cfg)
    w=hann(cfg.n_fft,sym=False).astype(np.float32)
    z=rfft(np.asarray(frames[a],np.float32)*w[None,:],n=cfg.n_fft,axis=1)
    p=(z.real.astype(np.float32)**2+z.imag.astype(np.float32)**2)
    p/=np.float32(np.sum(w.astype(np.float64)**2))
    s=np.median(10*np.log10(p+cfg.epsilon_power),axis=0).astype(np.float32)
    if not np.isfinite(s).all(): raise ValueError(f'not-finite spectrum: {path}')
    return s


def center_residual(raw: np.ndarray,cfg: SpectralConfig=SpectralConfig()):
    f=np.fft.rfftfreq(cfg.n_fft,d=1/cfg.target_sr)
    m=(f>=cfg.centering_min_hz)&(f<=cfg.centering_max_hz)
    off=float(np.median(np.asarray(raw)[m]))
    return (np.asarray(raw,np.float32)-np.float32(off)).astype(np.float32),off
