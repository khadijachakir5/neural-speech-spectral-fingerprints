import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def ref(): return json.loads((ROOT/'reference_results/manuscript_results_v1.json').read_text())

def test_h1_final_481_values():
    r=ref(); assert r['representation']['analysis_bins']==481
    assert abs(r['H1']['wavefake_ljspeech']['delta']-0.6930)<1e-9
    assert abs(r['H1']['wavefake_jsut']['delta']-1.0466)<1e-9
    assert abs(r['H1']['librisevoc_v2']['delta']-1.1261)<1e-9
    assert abs(r['H1']['mlaad_strict']['delta']-0.9693)<1e-9

def test_h2_both_coprimary_pass():
    r=ref()['H2']
    for k in ['shape','support']:
        assert r[k]['delta']>0
        assert r[k]['ci95'][0]>0
        assert r[k]['p_exact']==0.03125
        assert r[k]['positive_generators']=='5/5'

def test_h3_final_logic():
    r=ref(); assert r['H3a']['status']=='SUPPORTED'
    assert r['H3a']['shape']['p_exact']==0.015625
    assert r['H3a']['support']['p_exact']==0.046875
    h=r['H3b']; assert len(h)==5
    assert sum(x['pearson']>0 and x['p_holm']<.05 for x in h)==0

def test_h4_global_and_exploratory_separation():
    r=ref(); assert r['H4_global']['status']=='INSUFFICIENT_EVIDENCE'
    p=r['H4_global']['p_holm']
    assert p['family_similarity']>=.05 and p['multivariate_centroid']>=.05
    assert p['dispersion']<.05 and p['logo_balanced_accuracy']<.05 and p['logo_macro_f1']<.05
    sig=[x['family'] for x in r['H4_family_specific'] if x['q_bh']<.05]
    assert sig==['Neural codec decoder']
