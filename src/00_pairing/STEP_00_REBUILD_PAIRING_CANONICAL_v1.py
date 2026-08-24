#!/usr/bin/env python3


# Purpose: Rebuild the canonical Phase 0 pairing manifests in dependency order before downstream spectral analysis.

from pathlib import Path
import subprocess, sys
ROOT=Path(__file__).resolve().parent
STEPS=[
 'validate_pairing_wavefake_v2_3_CANONICAL.py',
 'freeze_wavefake_confirmatory_manifest_v1_CANONICAL.py',
 'validate_pairing_librisevoc_v2_1_CANONICAL.py',
 'freeze_librisevoc_confirmatory_manifest_v2_CANONICAL.py',
 'validate_pairing_mlaad_mailabs_v2_2_CANONICAL.py',
]
for i,name in enumerate(STEPS,1):
    p=ROOT/name
    print('\n'+'='*100); print(f'STEP {i}/{len(STEPS)} — {name}'); print('='*100)
    subprocess.run([sys.executable,str(p)],check=True)
print('\n[PASS] PHASE 0 PAIRING/QC REBUILD COMPLETE. Do not start spectral extraction yet; retain the logs for the final independent audit of these new outputs.')
