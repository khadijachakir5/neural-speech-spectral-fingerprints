#!/usr/bin/env python3
"""Final manuscript H3 aggregation: exact-checkpoint H3a + English-matched nominal H3b."""
from __future__ import annotations
import argparse, json, re, os, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import sys
HERE=Path(__file__).resolve(); sys.path.insert(0,str(HERE.parents[1]))
from common.manuscript_stats import pearson, cosine, jaccard_from_values, holm_adjust, stable_seed, normalize_language

MASTER_SEED=20260711; N_PERM=10000; ALPHA=0.05
EXPECTED_ROWS={"wavefake_ljspeech":91700,"librisevoc":72174,"mlaad_strict":62079}


def first_existing(paths, required=True):
    for p in paths:
        p=Path(p)
        if p.is_file(): return p
    if required: raise FileNotFoundError("No candidate exists:\n"+"\n".join(map(str,paths)))
    return None


def arch_key(x):
    z=re.sub(r"[^a-z0-9]+","",str(x).lower())
    aliases=[(("multibandmelgan","mbmelgan"),"multibandmelgan"),(("fullbandmelgan","fbmelgan"),"fullbandmelgan"),
             (("parallelwavegan","pwg"),"parallelwavegan"),(("hifigan",),"hifigan"),(("stylemelgan",),"stylemelgan"),
             (("melgan",),"melgan"),(("griffinlim",),"griffinlim")]
    for pats,key in aliases:
        if any(p in z for p in pats): return key
    return z


def family_key(x):
    s=str(x).lower()
    if "codec" in s: return "Neural codec decoder"
    if "vits" in s or "integrated" in s: return "Integrated GAN/VITS"
    if "phase" in s or "griffin" in s: return "Classical phase reconstruction"
    if "gan" in s: return "GAN vocoder"
    if "diff" in s: return "Diffusion vocoder"
    if "flow" in s: return "Flow-based vocoder"
    if "autoregressive" in s: return "Autoregressive vocoder"
    return str(x)


def load_dataset(key,path,res,english_only=False):
    schema=pd.read_parquet(path,engine="pyarrow").columns.tolist()
    required=["independent_generator_id","waveform_architecture","waveform_family","language"]+res
    missing=[c for c in required if c not in schema]
    if missing: raise RuntimeError(f"{key}: missing columns {missing[:20]}")
    status="status" if "status" in schema else ("status_harmonized" if "status_harmonized" in schema else None)
    cols=required+([status] if status else [])
    df=pd.read_parquet(path,columns=cols)
    if status: df=df[df[status].astype(str).str.lower().eq("ok")].copy()
    if len(df)!=EXPECTED_ROWS[key]: raise RuntimeError(f"{key}: rows {len(df)} != {EXPECTED_ROWS[key]}")
    df["language"]=df.language.astype(str).map(normalize_language)
    if english_only: df=df[df.language.eq("en")].copy()
    df["arch_key"]=df.waveform_architecture.map(arch_key)
    df["family_key"]=df.waveform_family.map(family_key)
    return df


def generator_profiles(df,res,context):
    meta=df.groupby("independent_generator_id",observed=True).agg(
        arch_key=("arch_key","first"), family_key=("family_key","first"),
        waveform_architecture=("waveform_architecture","first")
    ).reset_index()
    vals=df.groupby("independent_generator_id",observed=True)[res].median().reset_index()
    out=meta.merge(vals,on="independent_generator_id",validate="one_to_one")
    out["context"]=context
    return out


def empirical_upper(obs,null):
    arr=np.asarray(null,float); return float((1+np.sum(arr>=obs-1e-15))/(len(arr)+1))


def parse_h3a(obj):
    # Accept the normalized summary emitted by the controlled H3a experiment.
    def find_metric(names):
        for name in names:
            if isinstance(obj,dict) and name in obj and isinstance(obj[name],dict): return obj[name]
        # recursive search by key
        stack=[obj]
        while stack:
            x=stack.pop()
            if isinstance(x,dict):
                for k,v in x.items():
                    if k in names and isinstance(v,dict): return v
                    stack.append(v)
            elif isinstance(x,list): stack.extend(x)
        return {}
    shape=find_metric(["shape","H3a_shape","pearson_shape"])
    support=find_metric(["frequency","support","H3a_frequency","H3a_support","jaccard_support"])
    def norm(m):
        delta=m.get("mean_delta",m.get("delta")); ci=m.get("bootstrap_ci95",m.get("ci95")); p=m.get("exact_signflip_p_one_sided",m.get("p_exact",m.get("p")))
        pos=m.get("positive_checkpoints",m.get("positive"))
        return {"delta":delta,"ci95":ci,"p_exact":p,"positive":pos}
    return {"shape":norm(shape),"support":norm(support),"status":obj.get("overall",obj.get("status","UNKNOWN"))}


def run(root:Path,h3a_summary:Path|None,out:Path):
    out.mkdir(parents=True,exist_ok=True)
    axis=first_existing([root/"phase1a/phase1a_mlaad_spectral_residuals_v1/frequency_axis.csv",root/"phase1a/phase1a_mlaad_spectral_residuals_v2_new_story/frequency_axis.csv"])
    ax=pd.read_csv(axis); sel=ax[(ax.frequency_hz>=80)&(ax.frequency_hz<=7600)]
    if len(sel)!=481: raise RuntimeError(f"Expected 481 bins, got {len(sel)}")
    res=sel.column_name.astype(str).tolist()
    paths={
      "mlaad_strict":first_existing([root/"phase1a/phase1a_mlaad_spectral_residuals_v1/fingerprints_pair_level_strict.parquet",root/"phase1a/phase1a_mlaad_spectral_residuals_v2_new_story/fingerprints_pair_level_strict.parquet"]),
      "wavefake_ljspeech":first_existing([root/"q1_harmonized/v1/wavefake_ljspeech/full/fingerprints_pair_level_harmonized.parquet",root/"q1_harmonized/v3_new_story/wavefake_ljspeech/full/fingerprints_pair_level_harmonized.parquet"]),
      "librisevoc":first_existing([root/"q1_harmonized/v2/librisevoc/full/fingerprints_pair_level_harmonized.parquet",root/"q1_harmonized/v3_new_story/librisevoc/full/fingerprints_pair_level_harmonized.parquet",root/"q1_harmonized/v1/librisevoc/full/fingerprints_pair_level_harmonized.parquet"])
    }
    m=generator_profiles(load_dataset("mlaad_strict",paths["mlaad_strict"],res,True),res,"mlaad_en")
    w=generator_profiles(load_dataset("wavefake_ljspeech",paths["wavefake_ljspeech"],res),res,"wavefake_ljspeech")
    l=generator_profiles(load_dataset("librisevoc",paths["librisevoc"],res),res,"librisevoc")
    profiles=pd.concat([m,w,l],ignore_index=True)

    specs=[("hifigan","mlaad_en","wavefake_ljspeech","HiFi-GAN"),
           ("melgan","librisevoc","mlaad_en","MelGAN"),
           ("melgan","librisevoc","wavefake_ljspeech","MelGAN"),
           ("melgan","mlaad_en","wavefake_ljspeech","MelGAN"),
           ("parallelwavegan","librisevoc","wavefake_ljspeech","Parallel WaveGAN")]
    rows=[]; null_dir=out/"H3b_nulls"; null_dir.mkdir(exist_ok=True)
    for arch,a,b,label in specs:
        sa=profiles[(profiles.context==a)&(profiles.arch_key==arch)]
        sb=profiles[(profiles.context==b)&(profiles.arch_key==arch)]
        if sa.empty or sb.empty: raise RuntimeError(f"Missing nominal architecture group {arch} {a} {b}")
        fam=str(sa.family_key.iloc[0]);
        if str(sb.family_key.iloc[0])!=fam: raise RuntimeError(f"Family mismatch for {arch}")
        va=np.median(sa[res].to_numpy(float),axis=0); vb=np.median(sb[res].to_numpy(float),axis=0)
        obs=pearson(va,vb); obs_cos=cosine(va,vb); obs_j=jaccard_from_values(va,vb,0.10)
        pa=profiles[(profiles.context==a)&(profiles.family_key==fam)]; pb=profiles[(profiles.context==b)&(profiles.family_key==fam)]
        na=len(sa); nb=len(sb)
        if len(pa)<na or len(pb)<nb: raise RuntimeError("Null pool smaller than observed architecture group")
        rng=np.random.default_rng(stable_seed("H3b_perm",arch,a,b,MASTER_SEED)); null=np.empty(N_PERM,float)
        xa=pa[res].to_numpy(float); xb=pb[res].to_numpy(float)
        for i in range(N_PERM):
            ia=rng.choice(len(xa),size=na,replace=False); ib=rng.choice(len(xb),size=nb,replace=False)
            null[i]=pearson(np.median(xa[ia],axis=0),np.median(xb[ib],axis=0))
        p=empirical_upper(obs,null)
        np.save(null_dir/f"{len(rows)+1:02d}_{arch}_{a}_vs_{b}_pearson_null.npy",null)
        rows.append({"canonical_architecture":label,"architecture_match_key":arch,"canonical_family":fam,"dataset_a":a,"dataset_b":b,
                     "language_control":"en","identity_level":"NOMINAL_ARCHITECTURE_ONLY_NOT_SAME_CHECKPOINT","n_generators_a":na,"n_generators_b":nb,
                     "generators_a":"|".join(sorted(sa.independent_generator_id.astype(str))),"generators_b":"|".join(sorted(sb.independent_generator_id.astype(str))),
                     "pearson":obs,"pearson_empirical_p_same_family_null":p,"cosine":obs_cos,"support_jaccard_top10":obs_j,"permutations":N_PERM})
    tab=pd.DataFrame(rows); tab["pearson_p_holm_5_primary"]=holm_adjust(tab.pearson_empirical_p_same_family_null)
    tab["positive_and_holm_significant"]=(tab.pearson>0)&(tab.pearson_p_holm_5_primary<ALPHA)
    tab.to_csv(out/"H3b_FINAL_PRIMARY_ENGLISH_MATCHED.csv",index=False)

    if h3a_summary is None:
        # The paper-faithful source is the controlled exact-checkpoint run itself.
        # Historical recovery/master outputs are intentionally not searched.
        h3a_summary=first_existing(
            sorted(root.glob("H3A_CONTROLLED_SAME_CHECKPOINT_TWO_CORPORA_v1/run_*/H3a_controlled_summary.json"), reverse=True),
            required=False
        )
    h3a=None
    if h3a_summary and h3a_summary.is_file():
        h3a=parse_h3a(json.loads(h3a_summary.read_text(encoding="utf-8")))
    else:
        h3a={"status":"NOT_AVAILABLE","note":"Provide the validated H3a controlled-run summary or run --stage h3a-controlled explicitly."}
    summary={"version":"H3-FINAL-MANUSCRIPT-v1.1.0-A2Z","H3a":h3a,"H3b":{"role":"exploratory English-matched nominal architecture",
             "primary_endpoint":"Pearson","n_comparisons":len(tab),"permutations_per_comparison":N_PERM,
             "multiplicity":"Holm across exactly 5 primary Pearson comparisons","positive_holm_significant":int(tab.positive_and_holm_significant.sum()),
             "status":"NO_SIGNIFICANT_NOMINAL_ARCHITECTURE_PERSISTENCE" if int(tab.positive_and_holm_significant.sum())==0 else "SOME_SIGNIFICANT_NOMINAL_ARCHITECTURE_PERSISTENCE"}}
    (out/"H3_FINAL_SUMMARY.json").write_text(json.dumps(summary,indent=2,default=str),encoding="utf-8")
    print(tab.to_string(index=False)); print(json.dumps(summary,indent=2,default=str))


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",default="/content/drive/MyDrive/fingerprint_q1_outputs"); p.add_argument("--h3a-summary"); p.add_argument("--output-dir")
    a=p.parse_args(); root=Path(a.root); out=Path(a.output_dir) if a.output_dir else root/"H3_FINAL_MANUSCRIPT_v1"
    run(root,Path(a.h3a_summary) if a.h3a_summary else None,out)
if __name__=="__main__": main()
