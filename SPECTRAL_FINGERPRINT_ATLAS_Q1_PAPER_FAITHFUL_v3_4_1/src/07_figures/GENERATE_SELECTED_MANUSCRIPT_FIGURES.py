#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate manuscript figures from recomputed analysis outputs.

Frozen reference_results are used only as a numerical audit target. A figure is
never generated from a frozen number when the corresponding recomputed output
is required for the manuscript pipeline.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FAMILY_ORDER=["Classical phase reconstruction","GAN vocoder","Integrated GAN/VITS","Neural codec decoder"]
FAMILY_SHORT={"Classical phase reconstruction":"Classical","GAN vocoder":"GAN","Integrated GAN/VITS":"VITS","Neural codec decoder":"Codec"}

def normfam(s):
    return {"Classical phase rebuild":"Classical phase reconstruction","Integrated GAN (VITS)":"Integrated GAN/VITS"}.get(str(s),str(s))

def save(fig,p):
    p.parent.mkdir(parents=True,exist_ok=True); fig.savefig(p,bbox_inches="tight"); plt.close(fig); print("[FIG]",p)

def first_existing(paths,label):
    for p in paths:
        p=Path(p)
        if p.is_file(): return p
    raise FileNotFoundError(label+" not found. Tried:\n"+"\n".join(map(str,paths)))

def registry(repo): return json.loads((repo/"reference_results/manuscript_results_v1.json").read_text(encoding="utf-8"))

def close(a,b,tol=5e-4): return abs(float(a)-float(b))<=tol

def assert_ref(name,actual,expected,tol=5e-4):
    if not close(actual,expected,tol): raise RuntimeError(f"{name}: recomputed {actual} != frozen manuscript {expected} (tol={tol})")

def computed_h1(root,ref):
    ext=pd.read_csv(root/"H1_481BINS_50BLOCKS_FINAL_v1/H1_481BINS_50BLOCKS_FINAL_TABLE.csv")
    by={str(r.dataset):r for _,r in ext.iterrows()}
    names={"wavefake_ljspeech":"WaveFake-LJSpeech","wavefake_jsut":"WaveFake-JSUT","librisevoc_v2":"LibriSeVoc-v2"}
    out={}
    for k,label in names.items():
        r=by[label]; out[k]={"delta":float(r.delta_gen),"lo":float(r.delta_ci95_low),"hi":float(r.delta_ci95_high)}
        assert_ref("H1 "+k+" delta",out[k]["delta"],ref["H1"][k]["delta"])
    mp=root/"q1_03/mlaad_generator_stability_manuscript_final/q1_03_mlaad_generator_stability_summary.json"
    m=json.loads(mp.read_text(encoding="utf-8"))["split_block_reproducibility"]
    out["mlaad_strict"]={"delta":float(m["generator_specificity_delta_mean"]),"lo":float(m["generator_specificity_delta_ci95"][0]),"hi":float(m["generator_specificity_delta_ci95"][1])}
    assert_ref("H1 MLAAD delta",out["mlaad_strict"]["delta"],ref["H1"]["mlaad_strict"]["delta"])
    return out

def fig2(root,ref,out):
    h=computed_h1(root,ref); order=[("wavefake_ljspeech","WF-LJSpeech"),("wavefake_jsut","WF-JSUT"),("librisevoc_v2","LibriSeVoc v2"),("mlaad_strict","MLAAD STRICT")]
    vals=[h[k]["delta"] for k,_ in order]; lo=[h[k]["lo"] for k,_ in order]; hi=[h[k]["hi"] for k,_ in order]
    y=np.arange(len(order)); fig,ax=plt.subplots(figsize=(7.3,3.2)); ax.barh(y,vals,xerr=[np.array(vals)-lo,np.array(hi)-vals],capsize=2)
    ax.set_yticks(y,[x[1] for x in order]); ax.invert_yaxis(); ax.set_xlabel(r"Generator-specificity contrast $\Delta_{gen}$"); ax.axvline(0,lw=.8); ax.grid(axis="x",alpha=.2); fig.tight_layout(); save(fig,out/"Fig2_H1_content_reproducibility_bars.pdf")

def fig3(h2dir,out):
    d=pd.read_csv(h2dir/"H2_generator_level_effects.csv")
    pref=["Bark","XTTS v1.1","XTTS v2","WhisperSpeech","Griffin-Lim"]
    def friendly(x):
        s=str(x).lower()
        if "griffin" in s:return "Griffin-Lim"
        if "whisper" in s:return "WhisperSpeech"
        if "xtts_v1.1" in s or "xtts-v1.1" in s:return "XTTS v1.1"
        if "xtts_v2" in s or "xtts-v2" in s:return "XTTS v2"
        if "bark" in s:return "Bark"
        return str(x)
    d["display"]=d.generator.map(friendly); d["ord"]=d.display.map({x:i for i,x in enumerate(pref)}).fillna(99); d=d.sort_values("ord")
    fig,axs=plt.subplots(1,2,figsize=(9.2,3.7),sharey=True); y=np.arange(len(d)); axs[0].barh(y,d.delta_pearson); axs[1].barh(y,d.delta_jaccard)
    axs[0].set_yticks(y,d.display); axs[0].invert_yaxis(); axs[0].set_title("Pearson shape contrast"); axs[1].set_title("Top-10% Jaccard contrast")
    for ax in axs: ax.axvline(0,lw=.8); ax.set_xlabel("Same − matched different (Δ)"); ax.grid(axis="x",alpha=.2)
    fig.tight_layout(); save(fig,out/"Fig3_H2_cross_language_bars.pdf")

def load_h3a(h3dir,ref):
    h=json.loads((h3dir/"H3_FINAL_SUMMARY.json").read_text(encoding="utf-8"))["H3a"]
    for key,rkey in [("shape","shape"),("support","support")]:
        assert_ref("H3a "+key+" delta",h[key]["delta"],ref["H3a"][rkey]["delta"],2e-4)
    return h

def fig4a(h3dir,ref,out):
    h=load_h3a(h3dir,ref); vals=[h["shape"]["delta"],h["support"]["delta"]]; los=[h["shape"]["ci95"][0],h["support"]["ci95"][0]]; his=[h["shape"]["ci95"][1],h["support"]["ci95"][1]]
    fig,ax=plt.subplots(figsize=(5.8,3.35)); x=np.arange(2); ax.bar(x,vals,yerr=[np.array(vals)-los,np.array(his)-vals],capsize=4); ax.set_xticks(x,["Pearson\nshape","Top-10%\nJaccard"]); ax.set_ylabel("Exact-checkpoint Δ"); ax.axhline(0,lw=.8); ax.grid(axis="y",alpha=.2)
    ax.text(.98,.92,"Same frozen checkpoint (n=6)\n6/6 positive (shape), p=1/64\n5/6 positive (support), p=3/64\nBoth co-endpoints supported",transform=ax.transAxes,ha="right",va="top",fontsize=8)
    fig.tight_layout(); save(fig,out/"Fig4a_H3_checkpoint_conceptual.pdf")

def fig4b(h3dir,out):
    tab=pd.read_csv(h3dir/"H3b_FINAL_PRIMARY_ENGLISH_MATCHED.csv"); files=sorted((h3dir/"H3b_nulls").glob("*.npy"))
    if len(files)!=5: raise RuntimeError(f"Expected 5 H3b null arrays, found {len(files)}")
    fig,axs=plt.subplots(1,5,figsize=(13.0,3.15))
    for i,(ax,(_,r),nf) in enumerate(zip(axs,tab.iterrows(),files)):
        z=np.load(nf); ax.hist(z,bins=45); ax.axvline(float(r.pearson),lw=1.6)
        arch=str(r.canonical_architecture); a=str(r.dataset_a).replace("wavefake_ljspeech","WF-LJS").replace("mlaad_en","MLAAD-en").replace("librisevoc","LSV"); b=str(r.dataset_b).replace("wavefake_ljspeech","WF-LJS").replace("mlaad_en","MLAAD-en").replace("librisevoc","LSV")
        ax.set_title(f"{arch}\n{a} ↔ {b}",fontsize=8); ax.set_xlabel("Pearson",fontsize=8); ax.tick_params(labelsize=7)
        if i==0: ax.set_ylabel("Null count",fontsize=8)
    fig.suptitle("Nominal architecture only: observed Pearson vs same-family permutation null (10,000 draws each) — 0/5 significant after Holm",fontsize=9); fig.tight_layout(rect=[0,0,1,.9]); save(fig,out/"Fig4b_H3_nominal_null_histograms.pdf")

def _axis_cols(axis):
    band=axis[(axis.frequency_hz>=80)&(axis.frequency_hz<=7600)]
    if len(band)!=481: raise RuntimeError(f"Expected 481 bins, found {len(band)}")
    return band.column_name.astype(str).tolist(),band.frequency_hz.to_numpy(float)

def _cosine(x):
    x=np.asarray(x,float); n=np.linalg.norm(x,axis=1,keepdims=True); n[n<1e-15]=1; z=x/n; return z@z.T

def h4_global(h4dir,ref):
    s=json.loads((h4dir/"phase1b_protocol_summary.json").read_text(encoding="utf-8")); d=s["similarity_test"]; actual={"delta":float(d["delta_intra_minus_inter"]),"ci":list(map(float,d["bootstrap_ci95"]))}
    assert_ref("H4 global delta",actual["delta"],ref["H4_global"]["delta_family"],1e-8); return actual

def fig5(ref,h4dir,h4fdir,axispath,out):
    glob=h4_global(h4dir,ref); g=pd.read_parquet(h4dir/"fingerprints_generator_level_adjusted.parquet"); famcol="waveform_family" if "waveform_family" in g else "canonical_family"; g[famcol]=g[famcol].map(normfam)
    axis=pd.read_csv(axispath); cols,_=_axis_cols(axis); g=g.sort_values([famcol,"independent_generator_id"]).reset_index(drop=True); sim=_cosine(g[cols].to_numpy(float))
    f=pd.read_csv(h4fdir/"H4_FAMILY_SPECIFIC_FINAL.csv"); f.family=f.family.map(normfam); globalrow={"family":"Global family contrast","delta_F":glob["delta"],"ci95_low":glob["ci"][0],"ci95_high":glob["ci"][1]}; plot=pd.concat([pd.DataFrame([globalrow]),f[["family","delta_F","ci95_low","ci95_high"]]],ignore_index=True)
    preds=pd.read_csv(h4dir/"logo_predictions.csv"); preds.true_family=preds.true_family.map(normfam); preds.predicted_family=preds.predicted_family.map(normfam); cm=pd.crosstab(pd.Categorical(preds.true_family,FAMILY_ORDER),pd.Categorical(preds.predicted_family,FAMILY_ORDER),dropna=False).to_numpy()
    fig,axs=plt.subplots(1,3,figsize=(13.5,4.2),gridspec_kw={"width_ratios":[1.05,1.25,1]})
    ax=axs[0]; y=np.arange(len(plot)); x=plot.delta_F.to_numpy(float); lo=plot.ci95_low.to_numpy(float); hi=plot.ci95_high.to_numpy(float); ax.errorbar(x,y,xerr=[x-lo,hi-x],fmt="o",capsize=3); ax.axvline(0,ls="--",lw=.8); ax.set_yticks(y,["Global"]+[FAMILY_SHORT.get(x,x) for x in plot.family.iloc[1:]]); ax.invert_yaxis(); ax.set_xlabel("Similarity contrast (Δ)"); ax.set_title("(a) Global and family-specific\ncoherence")
    ax=axs[1]; im=ax.imshow(sim,vmin=-1,vmax=1,aspect="auto"); counts=[sum(g[famcol]==f) for f in FAMILY_ORDER]; edges=np.cumsum([0]+counts); centers=[(edges[i]+edges[i+1]-1)/2 for i in range(4)]; ax.set_xticks(centers,[FAMILY_SHORT[f] for f in FAMILY_ORDER],rotation=45,ha="right",fontsize=8); ax.set_yticks(centers,[FAMILY_SHORT[f] for f in FAMILY_ORDER],fontsize=8); [ax.axhline(e-.5,lw=.7) for e in edges[1:-1]]; [ax.axvline(e-.5,lw=.7) for e in edges[1:-1]]; ax.set_title("(b) 52×52 generator similarity\n(cosine, 481 bins, adjusted)"); fig.colorbar(im,ax=ax,fraction=.046,pad=.04,label="Cosine similarity")
    ax=axs[2]; row=cm/cm.sum(axis=1,keepdims=True); im2=ax.imshow(row,vmin=0,vmax=1); ax.set_xticks(range(4),[FAMILY_SHORT[f] for f in FAMILY_ORDER],rotation=45,ha="right",fontsize=8); ax.set_yticks(range(4),[FAMILY_SHORT[f] for f in FAMILY_ORDER],fontsize=8); ax.set_xlabel("Predicted family"); ax.set_ylabel("True family"); ax.set_title("(c) Leave-one-generator-out\nclassification")
    for i in range(4):
        for j in range(4): ax.text(j,i,str(int(cm[i,j])),ha="center",va="center",fontsize=9)
    fig.colorbar(im2,ax=ax,fraction=.046,pad=.04,label="Row-normalized proportion"); fig.tight_layout(); save(fig,out/"Fig5_H4_family_organization.pdf")

def fig6(h4dir,axispath,out):
    g=pd.read_parquet(h4dir/"fingerprints_generator_level_adjusted.parquet"); famcol="waveform_family" if "waveform_family" in g else "canonical_family"; g[famcol]=g[famcol].map(normfam); axis=pd.read_csv(axispath); cols,freq=_axis_cols(axis)
    fig,axs=plt.subplots(2,2,figsize=(10.5,6.6),sharex=True,sharey=True); axs=axs.ravel()
    for ax,fam in zip(axs,FAMILY_ORDER):
        part=g[g[famcol]==fam]; x=part[cols].to_numpy(float)
        for row in x: ax.plot(freq,row,lw=.55,alpha=.45)
        ax.plot(freq,np.median(x,axis=0),lw=2.2); ax.axhline(0,ls="--",lw=.7); ax.set_title(f"{fam} (n={len(part)})",fontsize=9); ax.grid(alpha=.15)
    axs[2].set_xlabel("Frequency (Hz)"); axs[3].set_xlabel("Frequency (Hz)"); axs[0].set_ylabel(r"Adjusted residual $p_g(f)$ (dB)"); axs[2].set_ylabel(r"Adjusted residual $p_g(f)$ (dB)"); fig.suptitle("All 52 MLAAD generators: language-adjusted profiles used in H4. Thin: generator; thick: family median.",fontsize=10); fig.tight_layout(rect=[0,0,1,.95]); save(fig,out/"Fig6_H4_adjusted_profiles_by_family.pdf")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",default="/content/drive/MyDrive/fingerprint_q1_outputs"); ap.add_argument("--output-dir"); a=ap.parse_args(); repo=Path(__file__).resolve().parents[2]; root=Path(a.root); out=Path(a.output_dir) if a.output_dir else repo/"paper_figures/generated"; ref=registry(repo)
    h2=root/"H2_CROSS_LANGUAGE_FINAL_v1"; h3=root/"H3_FINAL_MANUSCRIPT_v1"; h4=first_existing([root/"phase1b/phase1b_family_fingerprints_v2/strict/phase1b_protocol_summary.json"],"H4 strict summary").parent; h4f=root/"H4_FAMILY_SPECIFIC_EXPLORATORY_FINAL_v1"; axis=first_existing([root/"phase1a/phase1a_mlaad_spectral_residuals_v2_new_story/frequency_axis.csv",root/"phase1a/phase1a_mlaad_spectral_residuals_v1/frequency_axis.csv"],"MLAAD frequency axis")
    fig2(root,ref,out); fig3(h2,out); fig4a(h3,ref,out); fig4b(h3,out); fig5(ref,h4,h4f,axis,out); fig6(h4,axis,out); return 0
if __name__=="__main__": raise SystemExit(main())
