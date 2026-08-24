#!/usr/bin/env python3
"""Small deterministic statistical utilities used by manuscript-final analyses."""
from __future__ import annotations

import hashlib
import itertools
import math
from typing import Iterable, Sequence
import numpy as np


def stable_seed(*parts) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big", signed=False)


def pearson(a, b) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if a.size != b.size or a.size < 2:
        return float("nan")
    aa = a - np.mean(a); bb = b - np.mean(b)
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / den) if den > 0 else float("nan")


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > 0 else float("nan")


def top_fraction_mask(x, fraction: float = 0.10) -> np.ndarray:
    x = np.abs(np.asarray(x, dtype=float))
    k = max(1, int(math.ceil(fraction * x.size)))
    order = np.argsort(-x, kind="mergesort")
    out = np.zeros(x.size, dtype=bool)
    out[order[:k]] = True
    return out


def jaccard_from_values(a, b, fraction: float = 0.10) -> float:
    ma = top_fraction_mask(a, fraction); mb = top_fraction_mask(b, fraction)
    u = np.logical_or(ma, mb).sum()
    return float(np.logical_and(ma, mb).sum() / u) if u else float("nan")


def exact_signflip_p_one_sided(values: Sequence[float]) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    obs = float(np.mean(v))
    null = [float(np.mean(v * np.asarray(s, dtype=float)))
            for s in itertools.product([-1.0, 1.0], repeat=v.size)]
    return float(np.mean(np.asarray(null) >= obs - 1e-15))


def bootstrap_mean_ci(values: Sequence[float], n_boot: int, seed: int) -> list[float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        draws[i] = float(np.mean(rng.choice(v, size=v.size, replace=True)))
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(p))
    if finite.size == 0:
        return out
    order = finite[np.argsort(p[finite])]
    m = len(order); running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        out[idx] = min(1.0, running)
    return out


def bh_adjust(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    if np.any(~np.isfinite(p)):
        raise ValueError("BH requires finite p-values")
    m = len(p); order = np.argsort(p); ranked = p[order]
    adj = np.empty(m, dtype=float); running = 1.0
    for i in range(m - 1, -1, -1):
        running = min(running, ranked[i] * m / (i + 1))
        adj[i] = min(1.0, running)
    out = np.empty(m, dtype=float); out[order] = adj
    return out


def normalize_language(x: str) -> str:
    s = str(x).strip().lower().replace("_", "-")
    aliases = {"english":"en","eng":"en","en-us":"en","en-gb":"en",
               "japanese":"ja","jpn":"ja","jp":"ja",
               "german":"de","deu":"de","french":"fr","fra":"fr",
               "spanish":"es","spa":"es","italian":"it","ita":"it",
               "polish":"pl","pol":"pl","russian":"ru","rus":"ru",
               "ukrainian":"uk","ukr":"uk"}
    return aliases.get(s, s.split("-")[0] if len(s) >= 2 else s)


def h4_confirmatory_decision(delta_family: float, adjusted_p: dict[str, float], alpha: float = 0.05) -> str:
    ok = (
        delta_family > 0
        and adjusted_p["family_similarity"] < alpha
        and adjusted_p["multivariate_centroid"] < alpha
        and adjusted_p["logo_balanced_accuracy"] < alpha
        and adjusted_p["logo_macro_f1"] < alpha
        and adjusted_p["dispersion"] >= alpha
    )
    return "SUPPORTED" if ok else "INSUFFICIENT_EVIDENCE"
