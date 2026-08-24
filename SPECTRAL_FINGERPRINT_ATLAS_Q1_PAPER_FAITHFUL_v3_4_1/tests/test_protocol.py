import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_protocol_core_values():
    p=json.loads((ROOT/'config/protocol_v3_2.json').read_text())
    assert p['version']=='3.2.0-manuscript-final'
    assert p['master_seed']==20260711
    assert p['spectral_representation']['n_fft']==1024
    assert p['analysis_band']['n_bins']==481
    assert p['analysis_band']['stored_one_sided_bins']==513
    assert p['H2']['language_adjustment']=='none'
    assert p['H2']['multilingual_generators']==5
    assert p['H3b']['primary_comparisons']==5
    assert p['H3b']['primary_endpoint']=='Pearson'
    assert 'Holm' in p['H3b']['multiplicity']
    assert 'otherwise INSUFFICIENT_EVIDENCE' in p['H4']['decision_rule']
    assert p['H4']['family_specific_exploratory']['does_not_change_confirmatory_H4'] is True

def test_audit_qualification_locked():
    p=json.loads((ROOT/'config/protocol_v3_2.json').read_text())
    a=p['audit_status']
    assert a['strict_structural_pairing_validation']=='complete'
    assert a['mlaad_mailabs_physical_audit_unique_paths_checked']==57999
    assert a['mlaad_mailabs_physical_audit_unique_paths_total']==146140
    assert a['failures_in_checked_paths']==0
    assert a['full_physical_audit_completed'] is False
