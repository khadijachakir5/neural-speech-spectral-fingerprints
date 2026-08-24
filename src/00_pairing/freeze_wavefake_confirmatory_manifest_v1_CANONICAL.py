#!/usr/bin/env python3


# Purpose: Freeze separate balanced WaveFake LJSpeech and JSUT confirmatory manifests deterministically.

from pathlib import Path
import hashlib, json, os
import pandas as pd

INPUT = Path('/content/drive/MyDrive/fingerprint_q1_outputs/phase0_wavefake_v3_canonical/wavefake_manifest.parquet')
OUTPUT = Path('/content/drive/MyDrive/fingerprint_q1_outputs/phase0_wavefake_final_v2')
SPECS = {
    'ljspeech': dict(n_generators=7, n_originals=13100, n_pairs=91700, output='wavefake_ljspeech_manifest_confirmatory.parquet'),
    'jsut': dict(n_generators=2, n_originals=5000, n_pairs=10000, output='wavefake_jsut_manifest_confirmatory.parquet'),
}
REQUIRED = {'pair_id','dataset','domain','independent_generator_id','original_id','fake_path','real_path','fake_sha256','real_sha256','qc_status','exclusion_reason'}

def sha256_file(path: Path, chunk=1<<20):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def selection_hash(df):
    cols=['pair_id','domain','original_id','independent_generator_id','fake_path','real_path']
    x=df[cols].astype(str).sort_values(cols, kind='mergesort')
    return hashlib.sha256('\n'.join(x.agg('||'.join,axis=1)).encode()).hexdigest()

def atomic_parquet(df,path):
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); df.to_parquet(tmp,index=False); os.replace(tmp,path)

def atomic_json(obj,path):
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf-8'); os.replace(tmp,path)

def main():
    if not INPUT.is_file(): raise FileNotFoundError(INPUT)
    df=pd.read_parquet(INPUT)
    missing=REQUIRED-set(df.columns)
    if missing: raise RuntimeError(f'Missing columns: {sorted(missing)}')
    if len(df)!=101700: raise RuntimeError(f'101,700 pairs expected, {len(df):,} observed')
    if df['pair_id'].astype(str).duplicated().any(): raise RuntimeError('Duplicate pair_id values')
    if df.duplicated(['independent_generator_id','original_id']).any(): raise RuntimeError('(generator, original_id) duplicate')
    contradiction=df['qc_status'].astype(str).str.lower().eq('ok') & df['exclusion_reason'].fillna('').astype(str).str.strip().ne('')
    if contradiction.any(): raise RuntimeError(f'{int(contradiction.sum())} QC=ok rows with non-empty exclusion_reason')
    source_sha=sha256_file(INPUT)
    reports={}
    for domain,spec in SPECS.items():
        d=df[df['domain'].astype(str).str.lower().eq(domain)].copy()
        ok=d[d['qc_status'].astype(str).str.lower().eq('ok')].copy()
        if len(ok)!=spec['n_pairs']:
            raise RuntimeError(f'{domain}: {len(ok):,} QC-ok, {spec["n_pairs"]:,} expected')
        if ok['independent_generator_id'].nunique()!=spec['n_generators']:
            raise RuntimeError(f'{domain}: number de generators incorrect')
        
        if (ok.groupby('original_id')['real_path'].nunique()!=1).any(): raise RuntimeError(f'{domain}: original_id -> multiple real_path')
        if (ok.groupby('original_id')['real_sha256'].nunique()!=1).any(): raise RuntimeError(f'{domain}: original_id -> multiple real_sha256')
        coverage=ok.groupby('original_id')['independent_generator_id'].nunique()
        complete=set(coverage[coverage.eq(spec['n_generators'])].index.astype(str))
        final=ok[ok['original_id'].astype(str).isin(complete)].copy().sort_values(['original_id','independent_generator_id','pair_id'],kind='mergesort').reset_index(drop=True)
        if len(complete)!=spec['n_originals'] or len(final)!=spec['n_pairs']:
            raise RuntimeError(f'{domain}: freeze {len(complete):,} originals/{len(final):,} pairs; expected {spec["n_originals"]:,}/{spec["n_pairs"]:,}')
        counts=final['independent_generator_id'].value_counts()
        if counts.nunique()!=1 or int(counts.iloc[0])!=spec['n_originals']: raise RuntimeError(f'{domain}: imbalance')
        out=OUTPUT/spec['output']; atomic_parquet(final,out)
        out_sha=sha256_file(out)
        rep=dict(status='VALIDATED_AND_FROZEN',domain=domain,source_manifest=str(INPUT),source_manifest_sha256=source_sha,output_manifest=str(out),output_manifest_sha256=out_sha,selection_sha256=selection_hash(final),n_originals=len(complete),n_pairs=len(final),n_generators=spec['n_generators'],pairs_per_generator={str(k):int(v) for k,v in counts.sort_index().items()})
        atomic_json(rep, OUTPUT/f'{domain}_confirmatory_report.json'); reports[domain]=rep
        print(f'[PASS] {domain}: {len(complete):,} originals × {spec["n_generators"]} = {len(final):,} pairs')
    atomic_json({'status':'VALIDATED_AND_FROZEN','source_manifest_sha256':source_sha,'domains':reports}, OUTPUT/'wavefake_confirmatory_freeze_report.json')
    print('[PASS] WAVEFAKE CANONICAL FREEZE COMPLETE')

if __name__=='__main__': main()
