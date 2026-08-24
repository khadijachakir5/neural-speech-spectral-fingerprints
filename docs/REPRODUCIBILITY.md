# Reproducibility notes

## Scientific lock

`config/protocol_v3_2.json` is the final scientific contract. `reference_results/manuscript_results_v1.json` records the final manuscript values used as regression targets. The registry is never used to fabricate missing analysis outputs.

## Canonical final-analysis chain

The final paper analysis layer is rerun from validated saved residual artifacts:

```bash
python run_pipeline.py --stage paper-from-residuals
```

This is intentionally different from publishing the chronological Colab history as if it were one pipeline. The development notebooks contain repeated, failed, repaired, and superseded cells.

## Randomness

Master seed: `20260711`. Additional harmonized seeds: `20260729`, `20260817`. Stage-specific stochastic seeds are deterministic.

## H2

H2 uses **unadjusted** generator-language profiles. H4-style language nuisance subtraction is prohibited for H2 because it would change the hypothesis being tested.

## H3

H3a and H3b remain separate:

- H3a: exact frozen checkpoint, controlled LJSpeech 1.1 vs LibriTTS dev-clean experiment;
- H3b: nominal architecture only, five English-matched Pearson comparisons, Holm correction across exactly five tests.

The H3 aggregator reads the normalized summary emitted by the H3a controlled experiment. It does not search historical recovery/master outputs.

## H4

The confirmatory unit is the generator. The target generator is excluded from its same-language nuisance reference. The global H4 decision uses the five Holm-corrected endpoints frozen in the protocol. Family-specific `Delta_F` contrasts are exploratory and cannot revise global H4.

## Sensitivity-only MLAAD RELAXED population

The final H4 runner is STRICT-only. The broader 64,625-pair RELAXED population is reproduced through a small wrapper that reuses the same H4 functions rather than duplicating the full engine:

```bash
python run_pipeline.py --stage supplementary-h4-relaxed-sensitivity
```

This sensitivity stage cannot alter the primary STRICT H4 decision.

## Pairing audit qualification

Structural pairing validation was completed for the confirmatory population. The physical MLAAD/M-AILABS path audit checked 57,999 of 146,140 unique paths, with zero failures in the checked subset. It was not completed, and this release does not represent it as a completed stage.

## Release integrity

```bash
python -m compileall -q src run_pipeline.py
pytest -q
python src/05_master/MASTER_Q1_MANUSCRIPT_FINAL_v3_2.py --registry-only
python src/08_validation/VALIDATE_RELEASE.py --repo .
sha256sum -c MANIFEST_SHA256.txt
```
