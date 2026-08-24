#!/usr/bin/env python3


# Purpose: Run a constrained MLAAD negative pairing control with incorrect within-stratum M-AILABS references.

from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'common'))
from spectral import SpectralConfig, extract_log_spectrum, center_residual
sys.path.insert(0,str(HERE))
from q1_common import holm_adjust, pearson_corr, percentile_ci, sha256_file

SEED=20260711

def stable_int(*x): return int(hashlib.sha256('||'.join(map(str,x)).encode()).hexdigest()[:16],16)%(2**32-1)
def atomic_json(o,p):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(o,indent=2,ensure_ascii=False,default=str),encoding='utf-8'); os.replace(t,p)
def atomic_csv(d,p):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+'.tmp'); d.to_csv(t,index=False); os.replace(t,p)

def parse_args():
    r=Path('/content/drive/MyDrive/fingerprint_q1_outputs')
    p=argparse.ArgumentParser()
    p.add_argument('--pair-level',type=Path,default=r/'phase1a/phase1a_mlaad_spectral_residuals_v2_new_story/fingerprints_pair_level_strict.parquet')
    p.add_argument('--strict-manifest',type=Path,default=r/'phase0b/phase0b_mlaad_taxonomy_v1_2_new_story/mlaad_phase1_strict_confirmatory.parquet')
    p.add_argument('--output-dir',type=Path,default=r/'q1_08/mlaad_negative_pair_control_v1')
    p.add_argument('--bootstraps',type=int,default=5000)
    p.add_argument('--prototype-folds',type=int,default=10)
    p.add_argument('--seed',type=int,default=SEED)
    p.add_argument('--force',action='store_true')
    return p.parse_args()

def mismatch_map(real_meta: pd.DataFrame):
    x=real_meta.copy(); x['original_id']=x.original_id.astype(str); x['language']=x.language.astype(str)
    if 'speaker_id' not in x: x['speaker_id']='unknown'
    x['speaker_id']=x.speaker_id.fillna('unknown').astype(str)
    out={}; source={}
    for (la,sp),g in x.groupby(['language','speaker_id'],sort=True):
        ids=sorted(g.original_id.unique())
        if len(ids)>1:
            shift=1+stable_int('mismatch',la,sp)% (len(ids)-1)
            for i,a in enumerate(ids): out[a]=ids[(i+shift)%len(ids)]; source[a]='same_language_same_speaker'
    for la,g in x.groupby('language',sort=True):
        ids=sorted(g.original_id.unique())
        if len(ids)<2: continue
        shift=1+stable_int('mismatch_lang',la)% (len(ids)-1)
        for i,a in enumerate(ids):
            if a not in out: out[a]=ids[(i+shift)%len(ids)]; source[a]='same_language'
    if any(a==b for a,b in out.items()): raise RuntimeError('self mismatch generated')
    return out,source

def resumable_boot(values,n,seed,path):
    values=np.asarray(values,float); path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): arr=np.load(path,mmap_mode='r+')
    else:
        arr=np.lib.format.open_memmap(path,mode='w+',dtype=np.float64,shape=(n,)); arr[:]=np.nan; arr.flush()
    rng=np.random.default_rng(seed)
    
    for i in range(n):
        if np.isfinite(arr[i]): continue
        rr=np.random.default_rng(stable_int(seed,i))
        arr[i]=np.mean(rr.choice(values,size=len(values),replace=True));
        if (i+1)%250==0: arr.flush(); print(f'[BOOT] {i+1}/{n}')
    arr.flush(); return np.asarray(arr)

def main():
    a=parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    summary_path=a.output_dir/'q1_08_mlaad_negative_pair_summary.json'
    if summary_path.exists() and not a.force:
        print('[RESUME] COMPLETE summary already exists:',summary_path); return 0
    pair=pd.read_parquet(a.pair_level)
    res=[c for c in pair.columns if c.startswith('res_')]
    req={'pair_id','original_id','independent_generator_id','language','real_path','broadband_offset_db'}
    if req-set(pair.columns): raise RuntimeError(f'missing columns: {sorted(req-set(pair.columns))}')
    man=pd.read_parquet(a.strict_manifest)
    joincols=['pair_id']+[c for c in ['speaker_id'] if c in man.columns]
    if len(joincols)>1: pair=pair.merge(man[joincols].drop_duplicates('pair_id'),on='pair_id',how='left',validate='one_to_one')
    pair['speaker_id']=pair.get('speaker_id',pd.Series('unknown',index=pair.index)).fillna('unknown').astype(str)
    
    real_meta=pair[['original_id','real_path','language','speaker_id']].drop_duplicates()
    if (real_meta.groupby('original_id').real_path.nunique()!=1).any(): raise RuntimeError('original_id maps to multiple real paths')
    mm,ms=mismatch_map(real_meta)
    if len(mm)<real_meta.original_id.nunique(): raise RuntimeError('some originals have no same-language mismatch candidate')
    real_by_id=real_meta.drop_duplicates('original_id').set_index('original_id').real_path.astype(str).to_dict()
    needed=sorted(set(real_by_id.values()))
    cache_dir=a.output_dir/'real_spectrum_cache'; cache_dir.mkdir(exist_ok=True)
    cfg=SpectralConfig(); spectra={}
    for i,p in enumerate(needed,1):
        key=hashlib.sha256(p.encode()).hexdigest(); f=cache_dir/f'{key}.npy'
        if f.exists(): s=np.load(f)
        else:
            s=extract_log_spectrum(p,cfg); tmp=f.with_suffix('.tmp.npy'); np.save(tmp,s); os.replace(tmp,f)
        spectra[p]=np.asarray(s,np.float32)
        if i%1000==0 or i==len(needed): print(f'[REAL SPECTRA] {i}/{len(needed)}')
    R=pair[res].to_numpy(np.float32); offsets=pair.broadband_offset_db.to_numpy(np.float32)
    
    pair['_fold']=[stable_int('proto',x,a.seed)%a.prototype_folds for x in pair.original_id.astype(str)]
    proto={}
    for (g,l),sub in pair.groupby(['independent_generator_id','language'],sort=True):
        idx=sub.index.to_numpy()
        for fold in range(a.prototype_folds):
            tr=idx[pair.loc[idx,'_fold'].to_numpy()!=fold]
            if len(tr): proto[(str(g),str(l),fold)]=np.median(R[tr],axis=0)
    rows=[]
    for i,row in pair.iterrows():
        oid=str(row.original_id); correct_path=str(row.real_path); wrong_oid=mm[oid]; wrong_path=real_by_id[wrong_oid]
        correct=R[i].astype(np.float32)
        raw_correct=correct+np.float32(row.broadband_offset_db)
        fake_s=raw_correct+spectra[correct_path]
        wrong,_=center_residual(fake_s-spectra[wrong_path],cfg)
        pr=proto.get((str(row.independent_generator_id),str(row.language),int(row._fold)))
        if pr is None: continue
        rows.append({
            'pair_id':str(row.pair_id),'original_id':oid,'generator':str(row.independent_generator_id),'language':str(row.language),
            'mismatched_original_id':wrong_oid,'mismatch_constraint':ms[oid],
            'paired_rms':float(np.sqrt(np.mean(correct.astype(np.float64)**2))),
            'mismatched_rms':float(np.sqrt(np.mean(wrong.astype(np.float64)**2))),
            'paired_corr_to_cross_fitted_generator_language_fp':pearson_corr(correct,pr),
            'mismatched_corr_to_cross_fitted_generator_language_fp':pearson_corr(wrong,pr),
        })
    d=pd.DataFrame(rows)
    d['delta_rms']=d.mismatched_rms-d.paired_rms
    d['delta_corr']=d.paired_corr_to_cross_fitted_generator_language_fp-d.mismatched_corr_to_cross_fitted_generator_language_fp
    atomic_csv(d,a.output_dir/'pair_level_negative_control.csv')
    orig=d.groupby('original_id',sort=True).agg(delta_rms=('delta_rms','mean'),delta_corr=('delta_corr','mean'),paired_rms=('paired_rms','mean'),mismatched_rms=('mismatched_rms','mean'),paired_corr=('paired_corr_to_cross_fitted_generator_language_fp','mean'),mismatched_corr=('mismatched_corr_to_cross_fitted_generator_language_fp','mean')).reset_index()
    atomic_csv(orig,a.output_dir/'original_level_negative_control.csv')
    wr=wilcoxon(orig.mismatched_rms,orig.paired_rms,alternative='greater',zero_method='wilcox')
    wc=wilcoxon(orig.paired_corr,orig.mismatched_corr,alternative='greater',zero_method='wilcox')
    raw=np.array([float(wr.pvalue),float(wc.pvalue)]); holm=holm_adjust(raw)
    br=resumable_boot(orig.delta_rms.to_numpy(),a.bootstraps,a.seed+701,a.output_dir/'bootstrap_delta_rms.npy')
    bc=resumable_boot(orig.delta_corr.to_numpy(),a.bootstraps,a.seed+702,a.output_dir/'bootstrap_delta_corr.npy')
    ci_r=percentile_ci(br); ci_c=percentile_ci(bc)
    pass_r=float(orig.delta_rms.mean())>0 and ci_r[0]>0 and holm[0]<0.05
    pass_c=float(orig.delta_corr.mean())>0 and ci_c[0]>0 and holm[1]<0.05
    out={
        'version':'Q1_08_MLAAD_NEGATIVE_PAIR_CONTROL_v1','status':'PASS' if pass_r and pass_c else 'CONTROL_NOT_CONFIRMED',
        'interpretation':'Supplementary pairing-sensitivity control; not an H1-H4 endpoint.',
        'unit_of_inference':'original_id','n_pairs':len(d),'n_originals':len(orig),'seed':a.seed,'bootstraps':a.bootstraps,
        'mismatch_rule':'same language; same speaker/group when possible; deterministic not-self rotation',
        'rms':{'paired_mean':float(orig.paired_rms.mean()),'mismatched_mean':float(orig.mismatched_rms.mean()),'delta_mean':float(orig.delta_rms.mean()),'bootstrap_ci95':ci_r,'wilcoxon_p_numeric':float(wr.pvalue),'p_underflow_to_zero':bool(float(wr.pvalue)==0.0),'holm_p':float(holm[0])},
        'correlation':{'paired_mean':float(orig.paired_corr.mean()),'mismatched_mean':float(orig.mismatched_corr.mean()),'delta_mean':float(orig.delta_corr.mean()),'bootstrap_ci95':ci_c,'wilcoxon_p_numeric':float(wc.pvalue),'p_underflow_to_zero':bool(float(wc.pvalue)==0.0),'holm_p':float(holm[1])},
        'inputs':{'pair_level':str(a.pair_level),'pair_level_sha256':sha256_file(a.pair_level),'strict_manifest':str(a.strict_manifest),'strict_manifest_sha256':sha256_file(a.strict_manifest)},
    }
    atomic_json(out,summary_path); print(json.dumps(out,indent=2,ensure_ascii=False)); return 0
if __name__=='__main__': raise SystemExit(main())
