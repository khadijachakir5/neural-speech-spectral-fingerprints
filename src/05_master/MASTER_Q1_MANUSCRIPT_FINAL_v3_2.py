#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Manuscript-final registry and result-consistency validator.

This is deliberately *not* a new hypothesis-testing engine. It locks the
manuscript-grade numerical registry and checks any available stage summaries
against it. The scientific analyses remain in their dedicated H1/H2/H3/H4
scripts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

VERSION = "MASTER-Q1-MANUSCRIPT-FINAL-v3.2.0"
TOL = 5e-5


def load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def close(a, b, tol=TOL):
    return math.isfinite(float(a)) and abs(float(a)-float(b)) <= tol


def nested(obj, *keys):
    cur=obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur=cur[k]
    return cur


def first_existing(paths: Iterable[Path]):
    for p in paths:
        if p.is_file(): return p
    return None


def check(label, value, target, checks, tol=TOL):
    ok = value is not None and close(value,target,tol)
    checks.append({"check":label,"pass":bool(ok),"observed":value,"expected":target})


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("FINGERPRINT_OUTPUT_ROOT","/content/drive/MyDrive/fingerprint_q1_outputs"))
    ap.add_argument("--registry-only", action="store_true", help="validate bundled protocol/results without external data outputs")
    ap.add_argument("--output-dir")
    a=ap.parse_args()
    repo=Path(__file__).resolve().parents[2]
    protocol=load_json(repo/"config/protocol_v3_2.json")
    ref=load_json(repo/"reference_results/manuscript_results_v1.json")
    root=Path(a.root)
    out=Path(a.output_dir) if a.output_dir else root/"MASTER_Q1_MANUSCRIPT_FINAL_v3_2"
    if a.registry_only:
        out=repo/"_release_validation"
    out.mkdir(parents=True,exist_ok=True)

    checks=[]
    checks += [
      {"check":"protocol_version","pass":protocol.get("version")=="3.2.0-manuscript-final"},
      {"check":"master_seed","pass":protocol.get("master_seed")==20260711},
      {"check":"analysis_481_bins","pass":protocol["analysis_band"]["n_bins"]==481},
      {"check":"H2_unadjusted","pass":protocol["H2"]["language_adjustment"]=="none"},
      {"check":"H3b_five_Pearson_Holm","pass":protocol["H3b"]["primary_comparisons"]==5 and "Holm" in protocol["H3b"]["multiplicity"] and protocol["H3b"]["primary_endpoint"]=="Pearson"},
      {"check":"H4_conjunctive_no_partial_support","pass":"INSUFFICIENT_EVIDENCE" in protocol["H4"]["decision_rule"] and "otherwise" in protocol["H4"]["decision_rule"]},
      {"check":"H4_reference_locked_insufficient","pass":ref["H4_global"]["status"]=="INSUFFICIENT_EVIDENCE"},
      {"check":"H4_codec_exploratory_only_one_BH_signal","pass":sum(float(r["q_bh"])<0.05 for r in ref["H4_family_specific"])==1 and next(r for r in ref["H4_family_specific"] if r["family"]=="Neural codec decoder")["q_bh"]<0.05},
    ]

    sources={}
    if not a.registry_only:
        h2=first_existing([root/"H2_CROSS_LANGUAGE_FINAL_v1/H2_FINAL_SUMMARY.json"])
        h3=first_existing([root/"H3_FINAL_MANUSCRIPT_v1/H3_FINAL_SUMMARY.json"])
        h4=first_existing([
            root/"phase1b/phase1b_family_fingerprints_v2/strict/phase1b_protocol_summary.json",
            root/"phase1b/phase1b_family_fingerprints_v3_new_story/strict/phase1b_protocol_summary.json",
        ])
        h4f=first_existing([root/"H4_FAMILY_SPECIFIC_EXPLORATORY_FINAL_v1/H4_FAMILY_SPECIFIC_FINAL_SUMMARY.json"])
        sources={"H2":str(h2) if h2 else None,"H3":str(h3) if h3 else None,"H4":str(h4) if h4 else None,"H4_family_specific":str(h4f) if h4f else None}
        if h2:
            x=load_json(h2)
            check("H2_shape_delta",nested(x,"shape","mean_delta"),ref["H2"]["shape"]["delta"],checks)
            check("H2_support_delta",nested(x,"support","mean_delta"),ref["H2"]["support"]["delta"],checks)
        if h3:
            x=load_json(h3)
            # Compare the paper-level H3 summary emitted by H3_FINAL_MANUSCRIPT_v1.
            for label, keys, target in [
                ("H3a_shape",("H3a","shape","mean_delta"),ref["H3a"]["shape"]["delta"]),
                ("H3a_support",("H3a","support","mean_delta"),ref["H3a"]["support"]["delta"]),
            ]:
                v=nested(x,*keys)
                if v is not None: check(label,v,target,checks)
        if h4:
            x=load_json(h4)
            # phase1b schema: similarity_test / multivariate / evidence_decision
            status=nested(x,"evidence_decision","status")
            checks.append({"check":"H4_actual_status","pass":status=="INSUFFICIENT_EVIDENCE","observed":status,"expected":"INSUFFICIENT_EVIDENCE"})
            v=nested(x,"similarity_test","delta_intra_minus_inter")
            if v is not None: check("H4_actual_delta_family",v,ref["H4_global"]["delta_family"],checks)
        if h4f:
            x=load_json(h4f)
            checks.append({"check":"H4F_does_not_change_confirmatory","pass":x.get("confirmatory_status_changed_by_this_script") is False})

    passed=all(bool(c["pass"]) for c in checks)
    summary={"version":VERSION,"pass":passed,"registry":str(repo/"reference_results/manuscript_results_v1.json"),"external_sources":sources,"checks":checks}
    (out/"MASTER_MANUSCRIPT_FINAL_SUMMARY.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    for c in checks: print(("PASS" if c["pass"] else "FAIL"),"—",c["check"])
    if not passed: raise SystemExit(2)
    print("\nMANUSCRIPT-FINAL REGISTRY: PASS")
    return 0

if __name__=="__main__": raise SystemExit(main())
