#!/usr/bin/env python3
"""Static audit for the paper-faithful code release."""
from __future__ import annotations

import argparse
import json
import py_compile
import re
from pathlib import Path

REQ_FIGS = [
    "Fig1_pipeline.pdf",
    "Fig2_H1_content_reproducibility_bars.pdf",
    "Fig3_H2_cross_language_bars.pdf",
    "Fig4a_H3_checkpoint_conceptual.pdf",
    "Fig4b_H3_nominal_null_histograms.pdf",
    "Fig5_H4_family_organization.pdf",
    "Fig6_H4_adjusted_profiles_by_family.pdf",
]

CANONICAL = [
    "src/03_h1_controls/H1_CONTROLLED_DATASETS_481BINS_FINAL_v1.py",
    "src/03_h1_controls/Q1_03_MLAAD_GENERATOR_STABILITY_v2_NEW_STORY.py",
    "src/03_h1_controls/Q1_07_LIBRISEVOC_FALSE_PAIR_v3_NEW_STORY.py",
    "src/04_h2/H2_CROSS_LANGUAGE_FINAL_v1.py",
    "src/04_h3/H3A_CONTROLLED_FROM_RAW_FINAL_v1.py",
    "src/04_h3/H3_FINAL_MANUSCRIPT_v1.py",
    "src/04_h4/H4_GLOBAL_CONFIRMATORY_FINAL_v1.py",
    "src/04_h4/H4_FAMILY_SPECIFIC_EXPLORATORY_FINAL_v1.py",
    "src/05_master/MASTER_Q1_MANUSCRIPT_FINAL_v3_2.py",
    "src/07_figures/GENERATE_SELECTED_MANUSCRIPT_FIGURES.py",
]

REMOVED_SUPERSEDED = [
    "src/04_h3/H3_FINAL_RECOVERY_AND_COMPLETION_v1.py",
    "src/04_h4/H4_FINAL_RECOVERY_AND_COMPLETION_v1.py",
    "src/04_h4/H4_EXPLORATORY_DEEP_DIVE_v1.py",
    "src/07_figures/COLAB_FINAL_FIGURES_2_TO_6_REFERENCE.py",
    "src/legacy/MASTER_Q1_NEW_STORY_v3_1.py",
    "src/00_pairing/STEP_00_FULL_PAIRING_AUDIT_v2_CANONICAL.py",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    args = parser.parse_args()
    repo = Path(args.repo)
    checks: list[tuple[str, bool, str]] = []

    def ck(name: str, ok: bool, note: str = "") -> None:
        checks.append((name, bool(ok), note))

    protocol = json.loads((repo / "config/protocol_v3_2.json").read_text())
    results = json.loads((repo / "reference_results/manuscript_results_v1.json").read_text())

    ck("package version", (repo / "VERSION").read_text().strip() == "3.4.1-paper-faithful")
    ck("scientific protocol unchanged", protocol["version"] == "3.2.0-manuscript-final")
    ck("481 inferential bins", protocol["analysis_band"]["n_bins"] == 481 and results["representation"]["analysis_bins"] == 481)
    ck("H2 unadjusted", protocol["H2"]["language_adjustment"] == "none")
    ck("H2 five multilingual generators", protocol["H2"]["multilingual_generators"] == 5)
    ck("H3b five Pearson/Holm tests", protocol["H3b"]["primary_comparisons"] == 5 and protocol["H3b"]["primary_endpoint"] == "Pearson" and "Holm" in protocol["H3b"]["multiplicity"])
    ck("H4 locked conjunctive rule", "otherwise INSUFFICIENT_EVIDENCE" in protocol["H4"]["decision_rule"])
    ck("H4 final status", results["H4_global"]["status"] == "INSUFFICIENT_EVIDENCE")
    ck("only codec exploratory BH signal", sum(x["q_bh"] < 0.05 for x in results["H4_family_specific"]) == 1)
    ck("partial physical audit represented honestly", protocol["audit_status"]["full_physical_audit_completed"] is False and protocol["audit_status"]["mlaad_mailabs_physical_audit_unique_paths_checked"] == 57999)

    for rel in CANONICAL:
        path = repo / rel
        ck("canonical exists: " + path.name, path.is_file())
        if path.is_file():
            try:
                py_compile.compile(str(path), doraise=True)
                ok, note = True, ""
            except Exception as exc:
                ok, note = False, str(exc)
            ck("compile: " + path.name, ok, note)

    for rel in REMOVED_SUPERSEDED:
        ck("superseded excluded: " + Path(rel).name, not (repo / rel).exists())

    for fig in REQ_FIGS:
        path = repo / "paper_figures/main" / fig
        ck("figure: " + fig, path.is_file() and path.stat().st_size > 1000)

    run_text = (repo / "run_pipeline.py").read_text(errors="ignore")
    ck("single canonical paper chain", "paper-from-residuals" in run_text)
    ck("no synthetic A2Z raw chain", "full-from-raw" not in run_text and "manuscript-from-residuals" not in run_text)
    ck("supplementary MLAAD control explicit", "supplementary-mlaad-negative-control" in run_text)
    ck("RELAXED H4 sensitivity explicit and non-primary", "supplementary-h4-relaxed-sensitivity" in run_text and (repo / "src/09_supplementary/H4_RELAXED_PROTOCOL_SENSITIVITY_v1.py").is_file())
    ck("no incomplete physical audit stage", "pairing-audit" not in run_text and "with-full-physical-audit" not in run_text)

    active_text = "\n".join((repo / rel).read_text(errors="ignore") for rel in CANONICAL if (repo / rel).is_file())
    ck("no PARTIAL_SUPPORT in canonical engines", "PARTIAL_SUPPORT" not in active_text)
    ck("no 513-bin inferential lock", not bool(re.search(r"EXPECTED_(?:N_)?BINS\s*=\s*513", active_text)))

    readme = (repo / "README.md").read_text(errors="ignore")
    ck("README audit caveat", "57,999" in readme and "146,140" in readme and "not completed" in readme)
    ck("README notebook-provenance boundary", "not a clean executable description" in (repo / "docs/HISTORICAL_NOTEBOOK_PROVENANCE.md").read_text(errors="ignore"))

    failed = [item for item in checks if not item[1]]
    for name, ok, note in checks:
        print(("PASS" if ok else "FAIL"), "—", name, (("— " + note) if note else ""))

    report = {
        "pass": not failed,
        "n_checks": len(checks),
        "n_failed": len(failed),
        "checks": [{"name": n, "pass": ok, "note": note} for n, ok, note in checks],
    }
    (repo / "RELEASE_VALIDATION.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nRelease validation: {len(checks) - len(failed)}/{len(checks)} checks passed")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
