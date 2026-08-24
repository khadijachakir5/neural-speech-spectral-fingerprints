#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import pandas as pd

def close(a,b,tol): return abs(float(a)-float(b))<=tol

def check(name,a,b,tol):
    if not close(a,b,tol): raise RuntimeError(f"{name}: {a} != {b} within {tol}")
    print(f"[PASS] {name}: {float(a):.12g}")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--repo',type=Path,required=True); ap.add_argument('--root',type=Path,required=True); a=ap.parse_args(); ref=json.loads((a.repo/'reference_results/manuscript_results_v1.json').read_text()); root=a.root
    # H1 three controlled populations
    t=pd.read_csv(root/'H1_481BINS_50BLOCKS_FINAL_v1/H1_481BINS_50BLOCKS_FINAL_TABLE.csv'); mp={'WaveFake-LJSpeech':'wavefake_ljspeech','WaveFake-JSUT':'wavefake_jsut','LibriSeVoc-v2':'librisevoc_v2'}
    for _,r in t.iterrows():
        if r.dataset in mp: check('H1 '+mp[r.dataset],r.delta_gen,ref['H1'][mp[r.dataset]]['delta'],5e-4)
    m=json.loads((root/'q1_03/mlaad_generator_stability_manuscript_final/q1_03_mlaad_generator_stability_summary.json').read_text())['split_block_reproducibility']; check('H1 MLAAD',m['generator_specificity_delta_mean'],ref['H1']['mlaad_strict']['delta'],5e-4)
    # H2
    h2=json.loads((root/'H2_CROSS_LANGUAGE_FINAL_v1/H2_FINAL_SUMMARY.json').read_text())
    check('H2 shape',h2['shape']['mean_delta'],ref['H2']['shape']['delta'],1e-10); check('H2 support',h2['support']['mean_delta'],ref['H2']['support']['delta'],1e-10)
    # H3
    h3=json.loads((root/'H3_FINAL_MANUSCRIPT_v1/H3_FINAL_SUMMARY.json').read_text())['H3a']; check('H3a shape',h3['shape']['delta'],ref['H3a']['shape']['delta'],2e-4); check('H3a support',h3['support']['delta'],ref['H3a']['support']['delta'],2e-4)
    h3b=pd.read_csv(root/'H3_FINAL_MANUSCRIPT_v1/H3b_FINAL_PRIMARY_ENGLISH_MATCHED.csv')
    if len(h3b)!=5: raise RuntimeError('H3b must contain 5 primary comparisons')
    # H4
    h4=json.loads((root/'phase1b/phase1b_family_fingerprints_v2/strict/phase1b_protocol_summary.json').read_text()); check('H4 global delta',h4['similarity_test']['delta_intra_minus_inter'],ref['H4_global']['delta_family'],1e-10)
    if h4['evidence_decision']['status']!='INSUFFICIENT_EVIDENCE': raise RuntimeError('H4 confirmatory verdict changed')
    fam=pd.read_csv(root/'H4_FAMILY_SPECIFIC_EXPLORATORY_FINAL_v1/H4_FAMILY_SPECIFIC_FINAL.csv')
    if len(fam)!=4: raise RuntimeError('H4 family-specific table must have 4 rows')
    codec=fam[fam.family.astype(str).str.contains('codec',case=False)].iloc[0]; check('H4 codec delta',codec.delta_F,0.793211,5e-4)
    print('[PASS] Recomputed manuscript outputs agree with frozen numerical registry.')
if __name__=='__main__': main()
