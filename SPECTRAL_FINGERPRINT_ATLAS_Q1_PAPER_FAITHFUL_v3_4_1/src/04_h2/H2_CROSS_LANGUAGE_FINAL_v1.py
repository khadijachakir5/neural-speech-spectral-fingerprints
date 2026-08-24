#!/usr/bin/env python3
"""Final manuscript H2: unadjusted cross-language persistence on MLAAD STRICT."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

import sys
HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
from common.manuscript_stats import (pearson, jaccard_from_values, exact_signflip_p_one_sided,
                                     bootstrap_mean_ci, stable_seed, normalize_language)

MASTER_SEED = 20260711
EXPECTED_ROWS = 62079
EXPECTED_GENERATORS = 52
EXPECTED_MULTILINGUAL = 5
EXPECTED_LANGUAGE_PAIRS = 99
N_BOOT = 10000


def first_existing(paths):
    for p in paths:
        p = Path(p)
        if p.is_file(): return p
    raise FileNotFoundError("None of the candidate files exists:\n" + "\n".join(map(str, paths)))


def infer_residual_columns(axis: pd.DataFrame):
    if not {"column_name", "frequency_hz"}.issubset(axis.columns):
        raise RuntimeError("frequency_axis.csv must contain column_name and frequency_hz")
    sel = axis[(axis.frequency_hz >= 80.0) & (axis.frequency_hz <= 7600.0)]
    if len(sel) != 481:
        raise RuntimeError(f"Expected 481 inferential bins, found {len(sel)}")
    return sel.column_name.astype(str).tolist(), sel.frequency_hz.to_numpy(float)


def run(input_path: Path, axis_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    axis = pd.read_csv(axis_path); res, _ = infer_residual_columns(axis)
    base = ["independent_generator_id", "language"] + res
    df = pd.read_parquet(input_path, columns=base)
    if len(df) != EXPECTED_ROWS: raise RuntimeError(f"MLAAD STRICT rows {len(df)} != {EXPECTED_ROWS}")
    df["independent_generator_id"] = df.independent_generator_id.astype(str)
    df["language"] = df.language.astype(str).map(normalize_language)
    if df.independent_generator_id.nunique() != EXPECTED_GENERATORS:
        raise RuntimeError("MLAAD generator count mismatch")

    cells = df.groupby(["independent_generator_id","language"], observed=True, sort=True)[res].median().reset_index()
    lookup = {(str(r.independent_generator_id), str(r.language)): cells.loc[i,res].to_numpy(float)
              for i, r in cells[["independent_generator_id","language"]].iterrows()}
    langs_by_gen = cells.groupby("independent_generator_id", observed=True).language.agg(lambda x: sorted(set(x))).to_dict()
    multilingual = sorted([g for g, ls in langs_by_gen.items() if len(ls) >= 2])
    if len(multilingual) != EXPECTED_MULTILINGUAL:
        raise RuntimeError(f"Expected 5 multilingual generators, found {len(multilingual)}: {multilingual}")

    rows = []
    import itertools
    for g in multilingual:
        for l1, l2 in itertools.combinations(langs_by_gen[g], 2):
            a = lookup[(g,l1)]; b = lookup[(g,l2)]
            same_p = pearson(a,b); same_j = jaccard_from_values(a,b,0.10)
            eligible = [h for h in langs_by_gen if h != g and l1 in langs_by_gen[h] and l2 in langs_by_gen[h]]
            if not eligible: continue
            bp=[]; bj=[]
            for h in eligible:
                h1=lookup[(h,l1)]; h2=lookup[(h,l2)]
                bp.append(0.5*(pearson(a,h2)+pearson(h1,b)))
                bj.append(0.5*(jaccard_from_values(a,h2,0.10)+jaccard_from_values(h1,b,0.10)))
            rows.append({"generator":g,"language_1":l1,"language_2":l2,"n_baseline_generators":len(eligible),
                         "pearson_same":same_p,"pearson_different":float(np.mean(bp)),"delta_pearson":same_p-float(np.mean(bp)),
                         "jaccard_same":same_j,"jaccard_different":float(np.mean(bj)),"delta_jaccard":same_j-float(np.mean(bj))})
    pair = pd.DataFrame(rows)
    if len(pair) != EXPECTED_LANGUAGE_PAIRS:
        raise RuntimeError(f"Expected 99 eligible language-pair comparisons, found {len(pair)}")
    pair.to_csv(output_dir/"H2_language_pair_primary.csv", index=False)

    gen = pair.groupby("generator", observed=True).agg(
        n_language_pairs=("delta_pearson","size"),
        pearson_same=("pearson_same","mean"), pearson_different=("pearson_different","mean"), delta_pearson=("delta_pearson","mean"),
        jaccard_same=("jaccard_same","mean"), jaccard_different=("jaccard_different","mean"), delta_jaccard=("delta_jaccard","mean")
    ).reset_index()
    gen.to_csv(output_dir/"H2_generator_level_effects.csv", index=False)

    shape = gen.delta_pearson.to_numpy(float); support = gen.delta_jaccard.to_numpy(float)
    summary = {
      "version":"H2-CROSS-LANGUAGE-FINAL-v1.0.0","population":"MLAAD STRICT","language_adjustment":"none",
      "n_multilingual_generators":len(gen),"n_language_pair_comparisons":len(pair),
      "shape":{"same":float(gen.pearson_same.mean()),"different":float(gen.pearson_different.mean()),"mean_delta":float(shape.mean()),
               "median_delta":float(np.median(shape)),"bootstrap_ci95":bootstrap_mean_ci(shape,N_BOOT,stable_seed("H2","shape",MASTER_SEED)),
               "exact_signflip_p_one_sided":exact_signflip_p_one_sided(shape),"positive_generators":int((shape>0).sum())},
      "support":{"same":float(gen.jaccard_same.mean()),"different":float(gen.jaccard_different.mean()),"mean_delta":float(support.mean()),
                 "median_delta":float(np.median(support)),"bootstrap_ci95":bootstrap_mean_ci(support,N_BOOT,stable_seed("H2","support",MASTER_SEED)),
                 "exact_signflip_p_one_sided":exact_signflip_p_one_sided(support),"positive_generators":int((support>0).sum())}
    }
    summary["status"] = "SUPPORTED" if all(summary[k]["mean_delta"]>0 and summary[k]["bootstrap_ci95"][0]>0 and summary[k]["exact_signflip_p_one_sided"]<=0.05 for k in ["shape","support"]) else "INSUFFICIENT_EVIDENCE"
    (output_dir/"H2_FINAL_SUMMARY.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))
    return summary


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",default="/content/drive/MyDrive/fingerprint_q1_outputs"); p.add_argument("--input"); p.add_argument("--axis"); p.add_argument("--output-dir")
    a=p.parse_args(); root=Path(a.root)
    inp=Path(a.input) if a.input else first_existing([root/"phase1a/phase1a_mlaad_spectral_residuals_v1/fingerprints_pair_level_strict.parquet",root/"phase1a/phase1a_mlaad_spectral_residuals_v2_new_story/fingerprints_pair_level_strict.parquet"])
    axis=Path(a.axis) if a.axis else first_existing([inp.parent/"frequency_axis.csv",root/"phase1a/phase1a_mlaad_spectral_residuals_v1/frequency_axis.csv"])
    out=Path(a.output_dir) if a.output_dir else root/"H2_CROSS_LANGUAGE_FINAL_v1"
    run(inp,axis,out)
if __name__=="__main__": main()
