# Paper-faithful cleanup report

## Why this release was rebuilt

The development notebooks are chronological working records. They contain complete reruns, corrected reruns, failed cells, audits, recovery scripts, and exploratory analyses that were later narrowed or removed from the final manuscript. Publishing all of those branches as one active pipeline would overstate what contributed to the final paper.

## Notebook audit used for this cleanup

Five development notebooks supplied for the audit were reviewed:

- `Untitled15(3).ipynb`: 21 large code cells, including repeated complete H1→H4 master pipelines and successive audit/recovery stages;
- `CORRECTIONFINGERPRINT(10).ipynb`: 5 repair/finalization cells centered on corrected LibriSeVoc manifest/provenance handling;
- `article_2_inchaelah (2)(10).ipynb`: 24 code cells, including successive global-analysis/figure builders, one exact duplicate code cell, and an earlier cell that failed because a manifest was absent;
- `fingerprint_mlaad (3)(20260824-151538).ipynb`: 17 code cells spanning successive MLAAD family, lineage, sensitivity, and correction stages;
- `fingerprintlibrisecvox (2)(10).ipynb`: 3 code cells for the corrected LibriSeVoc confirmatory manifest/final audit path.

These notebooks remain historical provenance. They are not treated as the final executable specification.

## Removed from the active release

The following superseded or non-canonical engines were removed:

- `H3_FINAL_RECOVERY_AND_COMPLETION_v1.py`;
- `H4_FINAL_RECOVERY_AND_COMPLETION_v1.py`;
- `H4_EXPLORATORY_DEEP_DIVE_v1.py`;
- `COLAB_FINAL_FIGURES_2_TO_6_REFERENCE.py`;
- `MASTER_Q1_NEW_STORY_v3_1.py`;
- `BUILD_MANUSCRIPT_RELEASE_BUNDLE.py`;
- `STEP_00_FULL_PAIRING_AUDIT_v2_CANONICAL.py` because that exhaustive physical audit was not historically completed;
- the separate A-to-Z layout validator and duplicate A-to-Z documentation.

## H4 cleanup

The previous `PHASE1B_FAMILY_FINGERPRINTS_v2_1_NEW_STORY.py` mixed the final STRICT global H4 test with RELAXED sensitivity, within-language diagnostics, frequency-effect calculations, and intermediate figure production.

The active replacement is `H4_GLOBAL_CONFIRMATORY_FINAL_v1.py`. It retains only the operations needed for the final global H4 decision and the downstream paper objects. The RELAXED population, which the manuscript labels sensitivity-only, is handled by a small wrapper (`src/09_supplementary/H4_RELAXED_PROTOCOL_SENSITIVITY_v1.py`) that reuses this same engine rather than duplicating it.

The global engine retains:

- cross-fitted same-language adjustment;
- generator-level profiles;
- pairwise family similarity;
- multivariate centroid test and dispersion;
- LOGO balanced accuracy and macro-F1;
- joint Holm correction;
- final `INSUFFICIENT_EVIDENCE` decision;
- saved generator profiles / LOGO objects used by the final figures and exploratory family-specific analysis.

The family-specific exploratory analysis remains in its own script, as in the manuscript interpretation.

## Quantitative reduction

Compared with the preceding v3.3.0 package:

- Python files: **39 → 33** (the source-language regression test is restored);
- Python source lines: **23,159 → 16,070**;
- the H4 global engine: **1,790 → 1,328 lines**;
- one canonical paper runner replaces the previous A-to-Z / recovery orchestration.

No two distributed Python files are exact duplicates. The reduction is structural; the final protocol and frozen manuscript results were not changed.

## Final validation

- `pytest`: 14/14 passed;
- manuscript numerical registry validation: PASS;
- paper-faithful release validation: 52/52 checks passed;
- final inferential band: 481 bins;
- H2 language adjustment: none;
- H3b primary family: exactly 5 Pearson tests with Holm correction;
- H4 final status: `INSUFFICIENT_EVIDENCE`;
- only neural codec decoders are BH-significant in the exploratory family-specific H4 analysis.


## v3.4.1 translation-regression patch

The v3.4.0 paper-faithful structure and scientific logic are unchanged. This patch only:

- translates the three mixed-language strings reported in the v3.4.0 audit;
- translates additional short human-facing remnants found by the restored guard after the three initially reported lines;
- restores `tests/test_source_language.py` with word-bounded French-token patterns, including the requested `manquants?`, `insuffisant\w*`, `pointent`, `existent`, isolated `ambigu\b`, plus the additional exact French tokens exposed during the final scan;
- updates release-version metadata and checksums.

No protocol value, seed, dataset rule, statistical test, manuscript numerical registry entry, or inferential decision is changed.

Validation of the v3.4.1 patch:

- Python compilation: PASS;
- `pytest`: 14/14 passed;
- `tests/test_source_language.py`: PASS;
- manuscript numerical registry validation: PASS;
- paper-faithful release validation: 52/52 checks passed;
- text-neutralized AST comparison of every scientifically modified Python file against v3.4.0: PASS;
- `config/protocol_v3_2.json`, `reference_results/manuscript_results_v1.json`, and `run_pipeline.py`: byte-identical to v3.4.0.
