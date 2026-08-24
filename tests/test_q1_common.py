# Purpose: Validate the shared statistical primitives used across the hypothesis-testing pipeline.

import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/03_h1_controls'))
import q1_common as q

def test_pearson_identity():
    x=np.arange(10,dtype=float)
    assert abs(q.pearson_corr(x,x)-1.0)<1e-12

def test_holm_known_example():
    p=np.array([0.01,0.03,0.04])
    got=q.holm_adjust(p)
    
    assert np.allclose(got,[0.03,0.06,0.06])

def test_seed_determinism():
    assert q.stable_int_seed('abc',20260711)==q.stable_int_seed('abc',20260711)
