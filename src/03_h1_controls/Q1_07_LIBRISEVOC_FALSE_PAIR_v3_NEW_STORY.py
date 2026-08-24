#!/usr/bin/env python3


# Purpose: Run the LibriSeVoc negative pairing control to verify dependence on the intended bona-fide reference.

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd


import sys
try:
    import q1_common as _q1_common_probe  
except ModuleNotFoundError:
    for _candidate in [
        Path.cwd(),
        Path("/content/drive/MyDrive/Q1_MISSING_ANALYSES"),
        Path("/mnt/data/q1_missing_analyses"),
    ]:
        if (_candidate / "q1_common.py").is_file():
            sys.path.insert(0, str(_candidate))
            break
    try:
        import q1_common as _q1_common_probe  
    except ModuleNotFoundError:
        import base64 as _base64
        import types as _types
        import zlib as _zlib
        _embedded_q1_common = (
        'c-rk+-HzMF6~5O~Oc^vL9WA}<G;S(HK^-&(0=Pk%CcP0B%!nLG<K>?@<gVn6-=&u|`WX9S{m%T5L$0)TivY<*jUY>M&N*|=H|K9='
        '^wIC0bgFr>mgN&sUbFhp?yB<B*+=Z`@fl02OqSaP>)OrP=k#S31i_E3%w@}EBNPaVqAL}vtFB==KJb#~2W3X;{O(;LN=y2;i+DDB'
        'iHA*9rmd{Nl?j##R#aJ+3#O{FW!c@kRy2hyrD}zmV`9vuAOphL_U>JI_pafsz+^PDn02LW@7}TOkt{bD5+&s2Rij>aBs%9zsFv}%'
        '&SlEi5FuJtJ`$UaNZXF2@-FR4u+V!WX1vQ}6|=ANTw72onnX{oSabJI$*faM)u3*8w<m31-Tjk^Jziz2VnS!OX{v%H$)+QTNy21N'
        'R}E$=%c|wt?%B+K+i|ta<=VZPKll(AqU9NHIpb>hnz*ly{uLl|&y{jdb<W#O)fDbs9hA|j<}J<U6#5RYW}rRPSY`M9d3l(#Z?Fmk'
        '*qr?p=wWou{w@G2@a?rPyP`gjhGpH)&C3iQ@L!$HX0uFem}*H*BFi>WLS*6Nnm4Ulm}&IVR||$wwq#EqKmPP7J7X6SJNuHpEoKb='
        'THuS(;OC!%_*Es#u;6v5+GftQs3?k|R+Wh`&~6$;nyV4Oza%N4g*jF`{`9kFA<2v4o!Dn`3-<`4)#BpW<(yqSi`eh2N=|vrk!)lx'
        'LV|ChQJb@L*Oga^l5d35{epe*1$%m~HHU6S^Bo2zlN;r3WV_Sr5m&V+!=PCQ5upZ7!3)Ez>2{cz{m^t`_(uQq{19&FWBe}*2c4o3'
        't=uqpo?#hFp?_=eUX8x);;x352_tX>=FUfLyuRd%3E2vq4td2hy~1NS8h!(vi89XMJ1XqwwLEhqfu>Ea#6c~|=s6RmqA21jm2&xl'
        '=Ss|bCMB@`yhXGx!(blFS+EGAIg-kTcX_)c57D#_`p0B>pkj2yNX$~m)EG&skBpqN!8NqO-&UnvFTKRj5;-I;t}@w#=A}1pk35%D'
        'SNhfH)fP2C(v0LKiRv~Rx!2won`ib33u2H3o-g6my3$QRsF}pQbQT};t$rQC0ewv}hf8Nra{2Tq{yxA)gG#{zk#<rMLJtGS1i4gQ'
        '8{jq#FAy|!7XJmA>;=93%J^xG;<h3kjs9f*-dQ`r@=VEX@WEz7TO`f)H*dPt^N9IW#!<Z$>ee!NquNZn*H)Hd8Po^fG}X-qz=6&}'
        'HdRTrGyxHUJiHbS30;uWY11>Awkw_15mT22gpb%uQ<>p(idh!8GA~_S6sGN1%XKcRc6awL*rCt@bPYISEhGa_ja+x8=UB4aJ_G_?'
        '!h(f+7;;NsDj5ygCvObFV7SbK&t(x=43FMu!P`2osFV|d_hx9`+I3H_IFir80|fuOB*hH{iaX2sTIBPQX;`<hsj+NDp~7eg`L;QX'
        'D}bA9rRC8a_q&bQ!XEm<hY}<9si@myeBT-2-&XC5sw=av0Y~$`>4P7?eg2Q<-@JVO)yuB~KTms$X7vGv_Gz>L)V^W}T~B;u*!D&s'
        'aF4Nj2&2*R8_kH&KDtFR9*vklGsyEKt@5rYRYILqqSBoxcsL1abg+6Mty8;R$v8-CV^23tuV<Q~L|Bri)~^Ag9)00DkSI2D#F|D^'
        'TMa<JqL11?^aC8iNTL>iA|16F4HM-Cv>ZYuL3HU$h^hL9sJ_db7we2`Cf(qKh8=Bm8wOxoU0g=S1zdMwBn-=ohFA&qeOI=!5PIpu'
        'P4L`w+aRt={IwmmK)1~@3h7(2@a?!jTgg4_JOk9G6+I*zNpRGdi9I#vB6G{K=-XDk0D3w61Z)&0F$8y}J1JcVO!hK*yza1p$tVB='
        'BQ_y8sNoni*}Vx&bQTiH65aH2U_3*s(uOoiY}M+qv()dyZI4w2S~OQHMUExW+bhv7*exv@`f4jFa`0K3?5y=6?B@)y*GqJg56DG)'
        'y#_secHNX+^BkEq^q7r6fiiJ|4TVorh9lUbIIq$b#Y8w=t>_X9FtGI+lepDL$YsnGwk?A80TC*g78kMKE!lCwgNB9K&%95`!J@U^'
        'ej>QcxDGG-2=t~ik-B!2;sO+Q!dTC8IrG|bBYK<?kFNh$5`qo2-ZbZr_#T0IYd&H7a(?;NSJ^CroA|$j+lHM6#~;m_s<yp`3B|$p'
        '3$CuTdrzC2yt+YPEZ7F^B|-T4d3-)+JGtGBKL7FAIR<oWmUWD3+cex3s@)sAP^)pQT~gPgi*}fiA1t*jV*#lHt`NLmF0Bp``&fVY'
        'V(F&uor}^Nv7kjl4Tu=FaSw)L@D&;YB&<jFsDoY0OpJ3DiL)rxf;Y5lNUNp^`NA`B&er~`HeXv|a}rFzC|7zk<S6#qP<+!=&;DSh'
        'T#xD2r|5V}NFo<upb)$ax#g$tYw!Eoe8&y|16o9gp<Ke+ZCp~J<Y~knv&qMGI#dMW1=1Jd?BX*IO9Kv;+Q+J&A*+xhQ=kIrPe_Dw'
        'w6sEE6WG}nGUttK55Fpo58-kQmR|xDn0OD9rV+V5PD1#hE!&6tdninUX+Jh`f6BxZNky27w})fwt-l}&mltz(B}ARcLU|hxvQ!!2'
        'zU@xVoP8MY%?{L|`S7K|X>P{n>;ehG2MFY8F6*$L1o&vui2a$}{}}%;XJ;3beYBvFWBuazoq%Kii{kJ`ua33v7&31F#G9jb*P<bS'
        '&{8I;3{7EQ^hbUxZObKX{mKne8+xZ9RiDRy`fTWF4dqdAbipyABZrfhqltRI0=-llSz@CS?rkq>8_Hai_;fE!%#>kJ@{-Pa{5QkS'
        '^Lj@i4|Ca>XTt2h5X}C%;~EFOHk$JZdm8nEClnWu=uus|&23T3MyA*c)JazZ+^|AjZmvaUal_(udG58M2DfRZnjlC#{|j>dkW(aL'
        'v?n+XtB>RTG!$O2Pe?2k$UW86UMyv$(8vVIMdUneSLH>*vsWFQpiZ2J9kYKLWAzvW$9x2cs%s7H20$gbyut=)uH|vNJq4@G3{8(n'
        'ZHDS=>i1Wg;-25t*Y^>^Y-ZLb6OGle+3F1w$nEiFd8LzPeZA6T)+03xdO$;E$Rbm5Io`Iy{~l2hG0#(Ws?f+FMOVZ;O}nDY>6q3L'
        'L`TSVB5<GP?7U9|#xm7bjU$B@w+92`SfV4dHc=O0$K5hm81<IN(9R~v8LwFH{_hj_e8N&ZX1;Gc@q7d2!XKhAJLVWL<%DCb7K=0N'
        '*Ug{byp!<lwh>$CpKL`b^!a|01Bi|T3wuH)o8;JGs?%h2Ff%y;n9oj}0nnub?eYTJ?$kn`fO{%PI`|B3__d(3qGZF-j2`Hd(=9O2'
        'ZkwvB&5HMG#`@#>Ku0U^T9-ZTK`!}=*9R0LlU$$XSW#y0HZ==-H$xdlM>cV3)zMb5ml4yokfx;FEfRe~+Nt0uugG(~eKd+-ng+-l'
        'iBP^=8Y_0y%`(WvrUeLkX|ZI>ph7=|of-a2PH{-+=pcclL=W#U9ke!UBqSuc#um0}?A?(&Ydq!%mT<IHuqQM+RFXz)guXEzmRa>z'
        'k=9Szu0}`XZ#MjiZ0i#Q!E^2G`mQ)_Dy7agvZVl{%{n)R2*x{B%4>l?WJ#Bc*c|u$<jnYqWtF7EG<2_}Y}hGRq-$&k&YbBBY`Mk%'
        'Yu@xr*iA0X&x{f=dpW&Gm+SoQoxUg+`!w%JXT|Ao&fTcT&PwCqJy_aFjOGu7&Ge|oZrhD}I4j1ohM0PY*Dmn5IM|OTcfq5~=cO~t'
        'XZLuLZ_7%xGEJK522P>7-SGtlWwO=ftUr;La~Ae?&wIlneMkhgX+VT;kZK122Qn3T<we1@`kbfRh5@CeC;4M<Mnz{|{o8OBRFa!R'
        '29*$4+?5i^&|_PTCUd8{uJ~zYG<*++%=rAuQP{F%a+F1PGMU{EosaLURn;|`eQ<-}vSD=CsG;Y1zUC<%_?*Dv*k!omig|hPFGwCb'
        '_W6DY2)50uHAjHaQNB_LApgYh{V=%c-;02mCvo(_i0r*f+fU=IYTpz{a$u(rUd{%cd*VqP#Jr{m4{t|m1Q@a>!pNTU@z}|%uoxC-'
        '!qTt$V#slm341RzNYQnVewfgeNBuRhHySkPUHQBnAAk1kke~0v2_VtEp2hL$TBWEI4imq>WcA!fYnNA$#>cD!ku6C5&xeMjdZN<$'
        'R|C`8CkJPf(Nh_=5h;4W2Xp4AI%_h|f`Npxs+Un^w%5n$&LG!RJ=Cxby_F$9awiIInF@8}^a2z1M;GvzA%=AGCr33)AF^9fNBYEZ'
        '9kY=-O0^WkPOiSkxcGp3ffvMpG{lRw+;(^WHb29=?LqJ_1^i0uu8fW=f1>(2ts`$6q?h_cT6iGzA#cc2(Ijf8*coz%&FcIzx)<iX'
        'Z8o+EOs}$@NHF?#-~Fa+AitLqJa=`b4k>*a77oWDwU6@3D4SUPA|1ikszs_E`Vf;@TxXpa^b>ohbNsg?XsI<=x*_kx@Y9OnS0(+%'
        'Knqx{z+$N{FJnpzN!IBfOSWjyBY)!(tqfUQj@-Au|0TJxcpY9uG0Kh~VQWt9mZ4Z#!V>#hY+-adI(n7&@P(-TQKEHxb_@O9o_%?%'
        'HGI2Zx0tfO75fjMncpw'
        )
        _module = _types.ModuleType("q1_common")
        _source = _zlib.decompress(_base64.b85decode(_embedded_q1_common)).decode("utf-8")
        exec(compile(_source, "<embedded q1_common.py>", "exec"), _module.__dict__)
        sys.modules["q1_common"] = _module

from q1_common import (
    atomic_csv_dump,
    atomic_json_dump,
    atomic_parquet_dump,
    bh_adjust,
    environment_report,
    frequency_mask,
    holm_adjust,
    pearson_corr,
    percentile_ci,
    read_frequency_axis,
    residual_columns_from_schema,
    sha256_file,
    stable_int_seed,
    stable_json_hash,
)

VERSION = "Q1_07-LIBRISEVOC-FULL-FALSE-PAIR-CONTROL-v1.0.0"
DEFAULT_SEED = 20260711


def local_or_drive(local: str, drive: str) -> Path:
    p = Path("/mnt/data") / local
    return p if p.exists() else Path(drive)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair-level", type=Path,
        default=local_or_drive(
            "fingerprints_pair_level_harmonized.parquet",
            "/content/drive/MyDrive/fingerprint_q1_outputs/q1_harmonized/v3_new_story/"
            "librisevoc/full/fingerprints_pair_level_harmonized.parquet",
        ),
    )
    parser.add_argument(
        "--real-index", type=Path,
        default=local_or_drive(
            "real_index.parquet",
            "/content/drive/MyDrive/fingerprint_q1_outputs/q1_harmonized/v3_new_story/"
            "librisevoc/full/real_index.parquet",
        ),
    )
    parser.add_argument(
        "--frequency-axis", type=Path,
        default=local_or_drive(
            "frequency_axis.csv",
            "/content/drive/MyDrive/fingerprint_q1_outputs/q1_harmonized/v3_new_story/"
            "librisevoc/full/frequency_axis.csv",
        ),
    )
    parser.add_argument(
        "--fake-spectra", type=Path,
        default=Path(
            "/content/drive/MyDrive/fingerprint_q1_outputs/q1_harmonized/v3_new_story/"
            "librisevoc/full/spectra/fake_log_power_db.npy"
        ),
    )
    parser.add_argument(
        "--real-spectra", type=Path,
        default=Path(
            "/content/drive/MyDrive/fingerprint_q1_outputs/q1_harmonized/v3_new_story/"
            "librisevoc/full/spectra/real_log_power_db.npy"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(
            "/content/drive/MyDrive/fingerprint_q1_outputs/q1_07/"
            "librisevoc_full_false_pair_control_v3_new_story"
        ),
    )
    parser.add_argument("--mode", choices=["quick", "full"], default="full")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--content-blocks", type=int, default=50)
    parser.add_argument("--bootstraps-full", type=int, default=5000)
    parser.add_argument("--bootstraps-quick", type=int, default=500)
    parser.add_argument("--quick-originals", type=int, default=300)
    parser.add_argument("--analysis-min-hz", type=float, default=80.0)
    parser.add_argument("--analysis-max-hz", type=float, default=7600.0)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def build_mismatch_mapping(real_index: pd.DataFrame) -> dict[str, str]:
    real_index = real_index.copy()
    real_index["original_id"] = real_index["original_id"].astype(str)
    if "speaker_id" not in real_index.columns:
        real_index["speaker_id"] = "unknown"
    mapping: dict[str, str] = {}
    for _, group in real_index.groupby("speaker_id", dropna=False, sort=True):
        ids = sorted(group["original_id"].astype(str).unique())
        if len(ids) > 1:
            for a, b in zip(ids, ids[1:] + ids[:1]):
                mapping[a] = b
    all_ids = sorted(real_index["original_id"].astype(str).unique())
    if len(all_ids) < 2:
        raise RuntimeError("Fewer than two originals")
    for i, oid in enumerate(all_ids):
        if oid not in mapping:
            mapping[oid] = all_ids[(i + 1) % len(all_ids)]
        if mapping[oid] == oid:
            mapping[oid] = all_ids[(i + 1) % len(all_ids)]
    return mapping


def build_cross_fitted_block_prototypes(
    pair_meta: pd.DataFrame,
    paired_residuals: np.ndarray,
    n_blocks: int,
) -> tuple[dict[tuple[str, int], np.ndarray], np.ndarray]:
    originals = sorted(pair_meta["original_id"].astype(str).unique())
    block_by_original = {
        oid: stable_int_seed("LIBRISEVOC-FALSE-PAIR-BLOCK", oid) % n_blocks
        for oid in originals
    }
    row_blocks = pair_meta["original_id"].astype(str).map(block_by_original).to_numpy(dtype=int)
    prototypes: dict[tuple[str, int], np.ndarray] = {}

    for generator, idx in pair_meta.groupby("independent_generator_id", sort=True).groups.items():
        indices = np.asarray(list(idx), dtype=int)
        block_fp: dict[int, np.ndarray] = {}
        for block in range(n_blocks):
            take = indices[row_blocks[indices] == block]
            if len(take):
                block_fp[block] = np.median(paired_residuals[take], axis=0)
        valid_blocks = sorted(block_fp)
        if len(valid_blocks) < 4:
            raise RuntimeError(f"Fewer than four blocks for {generator}")
        for block in valid_blocks:
            others = [block_fp[b] for b in valid_blocks if b != block]
            prototypes[(str(generator), int(block))] = np.median(np.vstack(others), axis=0)
    return prototypes, row_blocks


def bootstrap_mean_delta(
    values: np.ndarray,
    repeats: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    out = np.empty(repeats, dtype=float)
    n = len(values)
    for repeat in range(repeats):
        idx = rng.integers(0, n, size=n)
        out[repeat] = float(np.mean(values[idx]))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    start = time.time()
    for name in ["pair_level", "real_index", "frequency_axis", "fake_spectra", "real_spectra"]:
        value = getattr(args, name).expanduser().resolve()
        setattr(args, name, value)
        if not value.is_file():
            raise FileNotFoundError(value)
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise RuntimeError("Output directory is not empty. Use --force.")
    if args.force and args.output_dir.exists():
        import shutil
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    residual_columns = residual_columns_from_schema(args.pair_level)
    axis = read_frequency_axis(args.frequency_axis, residual_columns)
    mask = frequency_mask(axis, args.analysis_min_hz, args.analysis_max_hz)
    selected_columns = [c for c, keep in zip(residual_columns, mask) if keep]
    selected_indices = np.flatnonzero(mask)

    meta_columns = [
        "extraction_row", "real_index", "pair_id", "independent_generator_id",
        "waveform_family", "language", "speaker_id", "original_id",
        "status_harmonized",
    ]
    pair_frame = pd.read_parquet(args.pair_level, columns=meta_columns + selected_columns)
    if not pair_frame["status_harmonized"].astype(str).eq("ok").all():
        raise RuntimeError("Invalid pairs detected")
    pair_frame = pair_frame.sort_values("extraction_row").reset_index(drop=True)
    if not np.array_equal(pair_frame["extraction_row"].to_numpy(dtype=int), np.arange(len(pair_frame))):
        raise RuntimeError("extraction_row does not match 0..N-1")

    real_index = pd.read_parquet(args.real_index)
    mismatch = build_mismatch_mapping(real_index)
    real_row_by_original = dict(zip(
        real_index["original_id"].astype(str), real_index["real_index"].astype(int)
    ))

    if args.mode == "quick":
        originals = sorted(pair_frame["original_id"].astype(str).unique())
        rng = np.random.default_rng(args.seed)
        selected = set(rng.choice(
            originals, size=min(args.quick_originals, len(originals)), replace=False
        ))
        pair_frame = pair_frame.loc[pair_frame["original_id"].astype(str).isin(selected)].copy()
        pair_frame = pair_frame.reset_index(drop=True)

    paired_residuals = pair_frame[selected_columns].to_numpy(dtype=np.float32, copy=False)
    prototypes, row_blocks = build_cross_fitted_block_prototypes(
        pair_frame, paired_residuals, args.content_blocks,
    )

    fake_spectra = np.load(args.fake_spectra, mmap_mode="r")
    real_spectra = np.load(args.real_spectra, mmap_mode="r")
    if fake_spectra.shape[1] != len(residual_columns) or real_spectra.shape[1] != len(residual_columns):
        raise RuntimeError("Dimensions spectrales incompatibles")

    result_rows: list[dict[str, Any]] = []
    for start_row in range(0, len(pair_frame), args.chunk_size):
        chunk = pair_frame.iloc[start_row:start_row + args.chunk_size]
        paired_chunk = paired_residuals[start_row:start_row + len(chunk)].astype(np.float64)
        fake_rows = chunk["extraction_row"].to_numpy(dtype=int)
        wrong_real_rows = np.asarray([
            real_row_by_original[mismatch[str(oid)]]
            for oid in chunk["original_id"].astype(str)
        ], dtype=int)
        fake_band = np.asarray(fake_spectra[fake_rows][:, selected_indices], dtype=np.float64)
        wrong_band = np.asarray(real_spectra[wrong_real_rows][:, selected_indices], dtype=np.float64)
        false_chunk = fake_band - wrong_band
        false_chunk -= np.median(false_chunk, axis=1, keepdims=True)

        for local_i, row in enumerate(chunk.itertuples(index=False)):
            global_i = start_row + local_i
            generator = str(row.independent_generator_id)
            block = int(row_blocks[global_i])
            prototype = prototypes[(generator, block)]
            paired = paired_chunk[local_i]
            false = false_chunk[local_i]
            result_rows.append({
                "pair_id": str(row.pair_id),
                "independent_generator_id": generator,
                "original_id": str(row.original_id),
                "mismatched_original_id": mismatch[str(row.original_id)],
                "content_block": block,
                "paired_rms": float(np.sqrt(np.mean(paired ** 2))),
                "mismatched_rms": float(np.sqrt(np.mean(false ** 2))),
                "paired_corr_to_cross_fitted_generator_fp": pearson_corr(paired, prototype),
                "mismatched_corr_to_cross_fitted_generator_fp": pearson_corr(false, prototype),
            })
        print(f"[FALSE PAIR] {min(start_row + args.chunk_size, len(pair_frame))}/{len(pair_frame)}")

    metrics = pd.DataFrame(result_rows)
    metrics["delta_rms_mismatched_minus_paired"] = metrics["mismatched_rms"] - metrics["paired_rms"]
    metrics["delta_corr_paired_minus_mismatched"] = (
        metrics["paired_corr_to_cross_fitted_generator_fp"]
        - metrics["mismatched_corr_to_cross_fitted_generator_fp"]
    )
    atomic_parquet_dump(metrics, args.output_dir / "false_pair_metrics_full.parquet")

    original_summary = metrics.groupby("original_id", sort=True).agg(
        n_generators=("independent_generator_id", "nunique"),
        paired_rms=("paired_rms", "mean"),
        mismatched_rms=("mismatched_rms", "mean"),
        delta_rms=("delta_rms_mismatched_minus_paired", "mean"),
        paired_corr=("paired_corr_to_cross_fitted_generator_fp", "mean"),
        mismatched_corr=("mismatched_corr_to_cross_fitted_generator_fp", "mean"),
        delta_corr=("delta_corr_paired_minus_mismatched", "mean"),
    ).reset_index()
    generator_summary = metrics.groupby("independent_generator_id", sort=True).agg(
        n_pairs=("pair_id", "size"),
        n_originals=("original_id", "nunique"),
        paired_rms=("paired_rms", "mean"),
        mismatched_rms=("mismatched_rms", "mean"),
        delta_rms=("delta_rms_mismatched_minus_paired", "mean"),
        paired_corr=("paired_corr_to_cross_fitted_generator_fp", "mean"),
        mismatched_corr=("mismatched_corr_to_cross_fitted_generator_fp", "mean"),
        delta_corr=("delta_corr_paired_minus_mismatched", "mean"),
    ).reset_index()
    atomic_csv_dump(original_summary, args.output_dir / "false_pair_metrics_by_original.csv")
    atomic_csv_dump(generator_summary, args.output_dir / "false_pair_metrics_by_generator.csv")

    if not original_summary["n_generators"].eq(
        original_summary["n_generators"].iloc[0]
    ).all():
        raise RuntimeError("Variable number of generators per original")

    rms_test = wilcoxon(
        original_summary["mismatched_rms"], original_summary["paired_rms"],
        alternative="greater", zero_method="wilcox", method="approx",
    )
    corr_test = wilcoxon(
        original_summary["paired_corr"], original_summary["mismatched_corr"],
        alternative="greater", zero_method="wilcox", method="approx",
    )
    raw_p = [float(rms_test.pvalue), float(corr_test.pvalue)]
    holm = holm_adjust(raw_p)
    bh = bh_adjust(raw_p)

    bootstraps = args.bootstraps_quick if args.mode == "quick" else args.bootstraps_full
    boot_rms = bootstrap_mean_delta(
        original_summary["delta_rms"].to_numpy(), bootstraps, args.seed + 701,
    )
    boot_corr = bootstrap_mean_delta(
        original_summary["delta_corr"].to_numpy(), bootstraps, args.seed + 702,
    )

    config = {
        "version": VERSION,
        "mode": args.mode,
        "seed": args.seed,
        "content_blocks": args.content_blocks,
        "bootstraps": bootstraps,
        "analysis_band_hz": [args.analysis_min_hz, args.analysis_max_hz],
        "analysis_bins": int(mask.sum()),
        "mismatch_mapping": "deterministic rotation within speaker, otherwise global rotation",
        "inference_unit": "original_id aggregated across generators",
        "generator_prototype": "cross-fitted by deterministic content block",
    }
    summary = {
        "version": VERSION,
        "status": "COMPLETE",
        "inputs": {
            "pair_level": str(args.pair_level),
            "pair_level_sha256": sha256_file(args.pair_level),
            "real_index": str(args.real_index),
            "real_index_sha256": sha256_file(args.real_index),
            "fake_spectra": str(args.fake_spectra),
            "real_spectra": str(args.real_spectra),
        },
        "config": config,
        "config_hash": stable_json_hash(config),
        "population": {
            "n_pairs": int(len(metrics)),
            "n_originals": int(original_summary["original_id"].nunique()),
            "n_generators": int(metrics["independent_generator_id"].nunique()),
        },
        "rms_control": {
            "paired_mean": float(original_summary["paired_rms"].mean()),
            "mismatched_mean": float(original_summary["mismatched_rms"].mean()),
            "delta_mismatched_minus_paired": float(original_summary["delta_rms"].mean()),
            "delta_bootstrap_ci95": percentile_ci(boot_rms),
            "wilcoxon_p": raw_p[0],
            "p_holm": float(holm[0]),
            "q_bh": float(bh[0]),
        },
        "correlation_control": {
            "paired_mean": float(original_summary["paired_corr"].mean()),
            "mismatched_mean": float(original_summary["mismatched_corr"].mean()),
            "delta_paired_minus_mismatched": float(original_summary["delta_corr"].mean()),
            "delta_bootstrap_ci95": percentile_ci(boot_corr),
            "wilcoxon_p": raw_p[1],
            "p_holm": float(holm[1]),
            "q_bh": float(bh[1]),
        },
        "elapsed_seconds": float(time.time() - start),
        "outputs": {
            "pair_metrics": str(args.output_dir / "false_pair_metrics_full.parquet"),
            "original_summary": str(args.output_dir / "false_pair_metrics_by_original.csv"),
            "generator_summary": str(args.output_dir / "false_pair_metrics_by_generator.csv"),
        },
    }
    atomic_json_dump(summary, args.output_dir / "q1_07_librisevoc_false_pair_summary.json")
    atomic_json_dump(environment_report(VERSION), args.output_dir / "environment.json")
    atomic_json_dump(
        {"version": VERSION, "status": "COMPLETE", "summary": str(args.output_dir / "q1_07_librisevoc_false_pair_summary.json")},
        args.output_dir / ".Q1_07_COMPLETE.json",
    )
    print(json.dumps({
        "population": summary["population"],
        "rms_control": summary["rms_control"],
        "correlation_control": summary["correlation_control"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
