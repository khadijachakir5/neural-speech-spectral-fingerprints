#!/usr/bin/env python3


# Purpose: Test H1 content-disjoint split-half reproducibility of MLAAD generator fingerprints with canonical population and language-reference guardrails.

from __future__ import annotations

import argparse
import gc
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
    aggregate_adjusted_cells_to_generators,
    atomic_csv_dump,
    atomic_json_dump,
    atomic_parquet_dump,
    correlation_matrix,
    environment_report,
    frequency_mask,
    language_adjust_cell_vectors,
    percentile_ci,
    read_frequency_axis,
    residual_columns_from_schema,
    sha256_file,
    stable_int_seed,
    stable_json_hash,
)

VERSION = "Q1_03-MLAAD-GENERATOR-STABILITY-v2.0.0"
DEFAULT_SEED = 20260711
EXPECTED_STRICT_PAIRS = 62_079
EXPECTED_STRICT_GENERATORS = 52
EXPECTED_STRICT_LANGUAGES = 8


def default_path(local_name: str, drive_path: str) -> Path:
    local = Path("/mnt/data") / local_name
    return local if local.exists() else Path(drive_path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=default_path(
            "fingerprints_pair_level_strict.parquet",
            "/content/drive/MyDrive/fingerprint_q1_outputs/phase1a/"
            "phase1a_mlaad_spectral_residuals_v2_new_story/fingerprints_pair_level_strict.parquet",
        ),
    )
    parser.add_argument(
        "--frequency-axis",
        type=Path,
        default=default_path(
            "frequency_axis.csv",
            "/content/drive/MyDrive/fingerprint_q1_outputs/phase1a/"
            "phase1a_mlaad_spectral_residuals_v2_new_story/frequency_axis.csv",
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/content/drive/MyDrive/fingerprint_q1_outputs/q1_03/"
            "mlaad_generator_stability_v2_new_story"
        ),
    )
    parser.add_argument("--mode", choices=["quick", "full"], default="full")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--content-blocks", type=int, default=50)
    parser.add_argument("--split-repeats-full", type=int, default=200)
    parser.add_argument("--split-repeats-quick", type=int, default=30)
    parser.add_argument("--minimum-language-reference-generators", type=int, default=2)
    parser.add_argument("--analysis-min-hz", type=float, default=80.0)
    parser.add_argument("--analysis-max-hz", type=float, default=7600.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def validate_canonical_population(frame: pd.DataFrame) -> None:


    observed = {
        "n_pairs": int(len(frame)),
        "n_generators": int(frame["independent_generator_id"].astype(str).nunique()),
        "n_languages": int(frame["language"].astype(str).nunique()),
    }
    expected = {
        "n_pairs": EXPECTED_STRICT_PAIRS,
        "n_generators": EXPECTED_STRICT_GENERATORS,
        "n_languages": EXPECTED_STRICT_LANGUAGES,
    }
    if observed != expected:
        raise RuntimeError(
            "Canonical MLAAD STRICT population guard failed. "
            f"Observed={observed}; expected={expected}. "
            "Do not coerce the data to match historical counts: audit the upstream "
            "pairing/taxonomy/extraction stage before running H1."
        )


def validate_input(frame: pd.DataFrame, residual_columns: list[str]) -> None:
    required = {
        "pair_id", "original_id", "independent_generator_id",
        "waveform_family", "language", "status",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {sorted(missing)}")
    if not frame["status"].astype(str).eq("ok").all():
        raise RuntimeError("Some rows have status != ok")
    if frame["pair_id"].astype(str).duplicated().any():
        raise RuntimeError("Duplicate pair_id values")
    if len(residual_columns) < 2:
        raise RuntimeError("Insufficient number of bins")
    mapping = frame[["independent_generator_id", "waveform_family"]].drop_duplicates()
    if mapping["independent_generator_id"].duplicated().any():
        raise RuntimeError("A generator belongs to multiple families")

    validate_canonical_population(frame)


def assert_language_only_references(
    diagnostics: pd.DataFrame,
    *,
    half: str,
    repeat: int,
    minimum_reference_generators: int,
) -> None:


    required = {"reference_source", "n_reference_generators",
                "independent_generator_id", "language"}
    missing = required - set(diagnostics.columns)
    if missing:
        raise RuntimeError(
            f"Language-adjustment diagnostics missing columns: {sorted(missing)}"
        )

    bad_source = diagnostics["reference_source"].astype(str).ne("language")
    too_few = (
        pd.to_numeric(diagnostics["n_reference_generators"], errors="coerce")
        < int(minimum_reference_generators)
    )
    bad = diagnostics.loc[bad_source | too_few].copy()
    if not bad.empty:
        preview = bad[
            ["independent_generator_id", "language",
             "reference_source", "n_reference_generators"]
        ].head(10).to_dict(orient="records")
        raise RuntimeError(
            "Confirmatory H1 forbids global language fallback and requires "
            f">={minimum_reference_generators} other same-language generators. "
            f"repeat={repeat}, half={half}, n_bad={len(bad)}, preview={preview}"
        )


def build_cell_block_vectors(
    frame: pd.DataFrame,
    residual_columns: list[str],
    n_blocks_requested: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:

    metadata_rows: list[dict[str, Any]] = []
    vector_rows: list[np.ndarray] = []
    count_rows: list[np.ndarray] = []

    for cell_index, ((generator, family, language), group) in enumerate(
        frame.groupby(
            ["independent_generator_id", "waveform_family", "language"],
            observed=True, sort=True,
        )
    ):
        group = group.reset_index(drop=True)
        n_blocks = min(int(n_blocks_requested), max(4, len(group) // 4))
        
        tokens = (
            group["original_id"].astype(str) + "||" + group["pair_id"].astype(str)
        ).tolist()
        ordered = sorted(
            range(len(group)),
            key=lambda i: stable_int_seed("MLAAD-STABILITY-BLOCK", generator, language, tokens[i]),
        )
        block_by_row = np.empty(len(group), dtype=np.int32)
        for rank, row_index in enumerate(ordered):
            block_by_row[row_index] = rank % n_blocks

        x = group[residual_columns].to_numpy(dtype=np.float32, copy=False)
        block_vectors = np.empty((n_blocks, len(residual_columns)), dtype=np.float32)
        block_counts = np.empty(n_blocks, dtype=np.int32)
        for block in range(n_blocks):
            mask = block_by_row == block
            if not mask.any():
                raise RuntimeError(f"Empty block for {generator}/{language}")
            block_vectors[block] = np.median(x[mask], axis=0)
            block_counts[block] = int(mask.sum())

        metadata_rows.append({
            "cell_index": int(cell_index),
            "independent_generator_id": str(generator),
            "waveform_family": str(family),
            "language": str(language),
            "n_pairs": int(len(group)),
            "n_blocks": int(n_blocks),
        })
        vector_rows.append(block_vectors)
        count_rows.append(block_counts)

    
    max_blocks = max(v.shape[0] for v in vector_rows)
    n_cells = len(vector_rows)
    n_bins = len(residual_columns)
    vectors = np.full((n_cells, max_blocks, n_bins), np.nan, dtype=np.float32)
    counts = np.zeros((n_cells, max_blocks), dtype=np.int32)
    for i, (v, c) in enumerate(zip(vector_rows, count_rows)):
        vectors[i, : v.shape[0]] = v
        counts[i, : c.shape[0]] = c
    return pd.DataFrame(metadata_rows), vectors, counts


def split_cell_fingerprints(
    metadata: pd.DataFrame,
    block_vectors: np.ndarray,
    repeat: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    first = np.empty((len(metadata), block_vectors.shape[2]), dtype=np.float64)
    second = np.empty_like(first)
    for i, row in metadata.iterrows():
        n_blocks = int(row["n_blocks"])
        rng = np.random.default_rng(stable_int_seed(seed, repeat, row["independent_generator_id"], row["language"]))
        order = rng.permutation(n_blocks)
        split = n_blocks // 2
        a = order[:split]
        b = order[split:]
        if not len(a) or not len(b):
            raise RuntimeError("Empty split")
        first[i] = np.median(block_vectors[i, a], axis=0)
        second[i] = np.median(block_vectors[i, b], axis=0)
    return first, second


def cross_correlation_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ac = a - a.mean(axis=1, keepdims=True)
    bc = b - b.mean(axis=1, keepdims=True)
    an = np.linalg.norm(ac, axis=1, keepdims=True)
    bn = np.linalg.norm(bc, axis=1, keepdims=True)
    an = np.where(an > 1e-15, an, 1.0)
    bn = np.where(bn > 1e-15, bn, 1.0)
    return np.clip((ac / an) @ (bc / bn).T, -1.0, 1.0)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    start = time.time()
    args.input = args.input.expanduser().resolve()
    args.frequency_axis = args.frequency_axis.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if not args.frequency_axis.is_file():
        raise FileNotFoundError(args.frequency_axis)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise RuntimeError("Output directory is not empty. Use --force or a new directory.")
    if args.force and args.output_dir.exists():
        import shutil
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    residual_columns = residual_columns_from_schema(args.input)
    axis = read_frequency_axis(args.frequency_axis, residual_columns)
    mask = frequency_mask(axis, args.analysis_min_hz, args.analysis_max_hz)
    selected_columns = [c for c, keep in zip(residual_columns, mask) if keep]

    metadata_columns = [
        "pair_id", "original_id", "independent_generator_id",
        "waveform_family", "language", "status",
    ]
    frame = pd.read_parquet(args.input, columns=metadata_columns + selected_columns)
    validate_input(frame, selected_columns)

    cell_meta, block_vectors, block_counts = build_cell_block_vectors(
        frame, selected_columns, args.content_blocks,
    )
    atomic_csv_dump(cell_meta, args.output_dir / "generator_language_cell_inventory.csv")
    np.savez_compressed(
        args.output_dir / "generator_language_block_fingerprints.npz",
        block_vectors=block_vectors,
        block_counts=block_counts,
        cell_index=cell_meta["cell_index"].to_numpy(dtype=np.int32),
        generator_ids=cell_meta["independent_generator_id"].astype(str).to_numpy(dtype="U"),
        languages=cell_meta["language"].astype(str).to_numpy(dtype="U"),
    )

    repeats = args.split_repeats_quick if args.mode == "quick" else args.split_repeats_full
    repeat_rows: list[dict[str, Any]] = []
    generator_rows: list[dict[str, Any]] = []
    adjustment_rows: list[pd.DataFrame] = []

    for repeat in range(repeats):
        first_cells, second_cells = split_cell_fingerprints(
            cell_meta, block_vectors, repeat, args.seed,
        )
        first_adjusted, diag_a = language_adjust_cell_vectors(
            cell_meta, first_cells,
            minimum_reference_generators=args.minimum_language_reference_generators,
        )
        second_adjusted, diag_b = language_adjust_cell_vectors(
            cell_meta, second_cells,
            minimum_reference_generators=args.minimum_language_reference_generators,
        )

        
        assert_language_only_references(
            diag_a,
            half="A",
            repeat=repeat,
            minimum_reference_generators=args.minimum_language_reference_generators,
        )
        assert_language_only_references(
            diag_b,
            half="B",
            repeat=repeat,
            minimum_reference_generators=args.minimum_language_reference_generators,
        )

        if repeat == 0:
            diag_a.insert(0, "half", "A")
            diag_b.insert(0, "half", "B")
            adjustment_rows.extend([diag_a, diag_b])

        gen_meta_a, gen_a = aggregate_adjusted_cells_to_generators(cell_meta, first_adjusted)
        gen_meta_b, gen_b = aggregate_adjusted_cells_to_generators(cell_meta, second_adjusted)
        if not gen_meta_a["independent_generator_id"].equals(gen_meta_b["independent_generator_id"]):
            raise RuntimeError("Inconsistent generator order")

        corr = cross_correlation_matrix(gen_a, gen_b)
        same = np.diag(corr)
        off = corr[~np.eye(len(corr), dtype=bool)]
        delta = float(np.mean(same) - np.mean(off))
        repeat_rows.append({
            "repeat": int(repeat),
            "n_generators": int(len(same)),
            "mean_same_generator_correlation": float(np.mean(same)),
            "median_same_generator_correlation": float(np.median(same)),
            "minimum_same_generator_correlation": float(np.min(same)),
            "mean_different_generator_correlation": float(np.mean(off)),
            "generator_specificity_delta": delta,
        })
        for i, generator in enumerate(gen_meta_a["independent_generator_id"].astype(str)):
            different = np.delete(corr[i], i)
            generator_rows.append({
                "repeat": int(repeat),
                "independent_generator_id": generator,
                "waveform_family": str(gen_meta_a.loc[i, "waveform_family"]),
                "n_languages": int(gen_meta_a.loc[i, "n_languages"]),
                "same_generator_correlation": float(corr[i, i]),
                "mean_correlation_to_other_generators": float(np.mean(different)),
                "generator_specificity_delta": float(corr[i, i] - np.mean(different)),
            })

        if (repeat + 1) % 10 == 0 or repeat + 1 == repeats:
            print(f"[SPLIT] {repeat + 1}/{repeats}")

    repeat_frame = pd.DataFrame(repeat_rows)
    generator_detail = pd.DataFrame(generator_rows)
    atomic_csv_dump(repeat_frame, args.output_dir / "split_reproducibility_repeats.csv")
    atomic_csv_dump(generator_detail, args.output_dir / "split_reproducibility_by_generator_repeat.csv")
    if adjustment_rows:
        atomic_csv_dump(pd.concat(adjustment_rows, ignore_index=True), args.output_dir / "language_adjustment_diagnostics_first_repeat.csv")

    generator_summary_rows = []
    for generator, group in generator_detail.groupby("independent_generator_id", sort=True):
        generator_summary_rows.append({
            "independent_generator_id": generator,
            "waveform_family": str(group["waveform_family"].iloc[0]),
            "n_languages": int(group["n_languages"].iloc[0]),
            "same_generator_correlation_mean": float(group["same_generator_correlation"].mean()),
            "same_generator_correlation_ci95_low": percentile_ci(group["same_generator_correlation"])[0],
            "same_generator_correlation_ci95_high": percentile_ci(group["same_generator_correlation"])[1],
            "different_generator_correlation_mean": float(group["mean_correlation_to_other_generators"].mean()),
            "specificity_delta_mean": float(group["generator_specificity_delta"].mean()),
            "specificity_delta_ci95_low": percentile_ci(group["generator_specificity_delta"])[0],
            "specificity_delta_ci95_high": percentile_ci(group["generator_specificity_delta"])[1],
        })
    generator_summary = pd.DataFrame(generator_summary_rows)
    atomic_csv_dump(generator_summary, args.output_dir / "generator_stability_summary.csv")

    config = {
        "version": VERSION,
        "mode": args.mode,
        "seed": args.seed,
        "content_blocks_requested": args.content_blocks,
        "split_repeats": repeats,
        "minimum_language_reference_generators": args.minimum_language_reference_generators,
        "analysis_band_hz": [args.analysis_min_hz, args.analysis_max_hz],
        "analysis_bins": int(mask.sum()),
        "unit_of_inference": "independent_generator_id",
        "language_adjustment": "recomputed independently in each split half",
        "global_language_fallback_allowed": False,
        "canonical_population_guard": {
            "n_pairs": EXPECTED_STRICT_PAIRS,
            "n_generators": EXPECTED_STRICT_GENERATORS,
            "n_languages": EXPECTED_STRICT_LANGUAGES,
        },
    }
    summary = {
        "version": VERSION,
        "status": "COMPLETE",
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "frequency_axis": str(args.frequency_axis),
        "frequency_axis_sha256": sha256_file(args.frequency_axis),
        "config": config,
        "config_hash": stable_json_hash(config),
        "population": {
            "n_pairs": int(len(frame)),
            "n_generators": int(frame["independent_generator_id"].nunique()),
            "n_languages": int(frame["language"].nunique()),
            "n_generator_language_cells": int(len(cell_meta)),
            "minimum_pairs_per_cell": int(cell_meta["n_pairs"].min()),
            "maximum_pairs_per_cell": int(cell_meta["n_pairs"].max()),
            "canonical_guard_passed": True,
            "global_language_fallback_count": 0,
        },
        "split_block_reproducibility": {
            "n_repeats": int(repeats),
            "same_generator_correlation_mean": float(repeat_frame["mean_same_generator_correlation"].mean()),
            "same_generator_correlation_ci95": percentile_ci(repeat_frame["mean_same_generator_correlation"]),
            "different_generator_correlation_mean": float(repeat_frame["mean_different_generator_correlation"].mean()),
            "different_generator_correlation_ci95": percentile_ci(repeat_frame["mean_different_generator_correlation"]),
            "generator_specificity_delta_mean": float(repeat_frame["generator_specificity_delta"].mean()),
            "generator_specificity_delta_ci95": percentile_ci(repeat_frame["generator_specificity_delta"]),
            "minimum_generator_correlation_across_repeats": float(repeat_frame["minimum_same_generator_correlation"].min()),
            "generator_specificity_supported": bool(
                percentile_ci(repeat_frame["generator_specificity_delta"])[0] > 0
            ),
        },
        "elapsed_seconds": float(time.time() - start),
        "outputs": {
            "repeats": str(args.output_dir / "split_reproducibility_repeats.csv"),
            "generator_summary": str(args.output_dir / "generator_stability_summary.csv"),
            "generator_detail": str(args.output_dir / "split_reproducibility_by_generator_repeat.csv"),
            "block_fingerprints": str(args.output_dir / "generator_language_block_fingerprints.npz"),
        },
    }
    atomic_json_dump(summary, args.output_dir / "q1_03_mlaad_generator_stability_summary.json")
    atomic_json_dump(environment_report(VERSION), args.output_dir / "environment.json")
    atomic_json_dump(
        {"version": VERSION, "status": "COMPLETE", "summary": str(args.output_dir / "q1_03_mlaad_generator_stability_summary.json")},
        args.output_dir / ".Q1_03_COMPLETE.json",
    )
    print(json.dumps(summary["split_block_reproducibility"], indent=2, ensure_ascii=False))
    del frame, block_vectors
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
