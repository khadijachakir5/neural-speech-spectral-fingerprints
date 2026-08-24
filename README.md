# Spectral Artifact Atlas of Neural Speech Generators

## Paper-faithful code release — v3.4.1

This release contains the **minimal active code needed to reproduce the analyses reported in the final manuscript**, plus the raw pairing/extraction code needed to reconstruct the spectral residual inputs when the datasets are available.

It deliberately does **not** reproduce the chronological structure of the development notebooks. Those notebooks contain failed cells, repair cells, superseded versions, repeated master pipelines, and exploratory branches that were not retained in the final paper. They are provenance material, not the canonical scientific pipeline.

The scientific protocol itself is unchanged from the manuscript-final contract in `config/protocol_v3_2.json`.

## What the final paper actually analyzes

The active paper chain is limited to:

1. **H1 — content reproducibility** in WaveFake–LJSpeech, WaveFake–JSUT, LibriSeVoc v2, and MLAAD STRICT;
2. **primary LibriSeVoc false-pairing control**;
3. **H2 — cross-language persistence** on the five multilingual MLAAD generators, using unadjusted generator–language fingerprints;
4. **H3a — exact-checkpoint cross-corpus persistence** using the controlled LJSpeech 1.1 / LibriTTS dev-clean experiment;
5. **H3b — exploratory nominal-architecture comparisons** across exactly five English-matched Pearson tests with Holm correction;
6. **H4 — global family-level analysis** on MLAAD STRICT, with the prespecified five-endpoint Holm-corrected decision rule;
7. **exploratory family-specific H4 contrasts**, with Benjamini–Hochberg correction across the four families;
8. the **six selected main figures** used by the manuscript.

The MLAAD same-language negative-pairing control is retained only as a **supplementary methodological control**. The broader MLAAD RELAXED population is also available as an explicit **sensitivity-only** stage that reuses the same final H4 engine. Detailed lineage/frequency deep dives that were present in historical notebooks are not active manuscript engines in this release.

## Canonical spectral representation

```text
mono -> 16 kHz -> remove DC
-> periodic Hann, NFFT=1024, hop=256
-> active-frame selection
-> median 10*log10(power + 1e-12)
-> r_raw(f) = S_fake(f) - S_real(f)
-> r(f) = r_raw(f) - median over the analysis band
```

The one-sided FFT contains 513 bins, but **all final inference uses the same 481 bins** with actual centers from **93.75 to 7593.75 Hz**.

## Final analytical populations

| Population | Pairs | Bona fide originals | Generators | Languages |
|---|---:|---:|---:|---:|
| WaveFake–LJSpeech | 91,700 | 13,100 | 7 | 1 |
| WaveFake–JSUT | 10,000 | 5,000 | 2 | 1 |
| LibriSeVoc v2 | 72,174 | 12,029 | 6 | 1 |
| MLAAD STRICT | 62,079 | — | 52 | 8 |

MLAAD STRICT contains 79 generator–language cells and four technological families. MLAAD RELAXED is sensitivity-only and is not a separate main-paper endpoint.

## Frozen manuscript results

`reference_results/manuscript_results_v1.json` is a regression lock, not a substitute for computation. It prevents accidental drift in the released code.

The main locked results are:

- H1 `Delta_gen`: 0.6930, 1.0466, 1.1261, 0.9693;
- H2 Pearson contrast 0.3691 and top-10% Jaccard contrast 0.1711, exact `p=0.03125` for both;
- H3a shape contrast 0.5218 (`p=0.015625`) and support contrast 0.1859 (`p=0.046875`);
- H3b: 0/5 positive-and-Holm-significant primary comparisons;
- H4 global: `Delta_fam=0.0443`, pseudo-`F=1.8002`, `R^2=0.1011`, LOGO balanced accuracy 0.5767, macro-F1 0.4081, final verdict `INSUFFICIENT_EVIDENCE`;
- exploratory H4: only neural codec decoders remain BH-significant (`Delta_F=0.7932`, `q_BH=0.0008`).

## Repository structure

```text
config/protocol_v3_2.json             final scientific contract
reference_results/                    frozen manuscript regression values
src/00_pairing/                       pairing construction/freezing/validation
src/01_taxonomy/                      MLAAD generator taxonomy
src/02_extraction/                    harmonized residual extraction
src/03_h1_controls/                   H1 + pairing controls
src/04_h2/                            final H2
src/04_h3/                            controlled H3a + final H3 aggregation/H3b
src/04_h4/                            global H4 + family-specific exploratory H4
src/05_master/                        manuscript consistency check
src/07_figures/                       one selected-figure generator
src/08_validation/                    release/result validation
src/09_supplementary/                 explicitly non-primary controls/sensitivity
src/common/                           shared spectral/statistical utilities
paper_figures/main/                   selected manuscript PDFs
```

Superseded recovery scripts, historical masters, duplicated figure builders, and non-manuscript H4 deep-dive engines are intentionally excluded from this release.

## Canonical manuscript rerun

The final statistical analysis was assembled from validated saved residual artifacts rather than by pretending that every historical notebook cell formed one uninterrupted A-to-Z run.

```bash
python run_pipeline.py --stage paper-from-residuals
```

The stages can also be run separately:

```bash
python run_pipeline.py --stage h1
python run_pipeline.py --stage h2
python run_pipeline.py --stage h3
python run_pipeline.py --stage h4
python run_pipeline.py --stage figures
python run_pipeline.py --stage validate
```

The controlled H3a synthesis experiment requires the two raw corpora and frozen checkpoints and is therefore explicit:

```bash
python run_pipeline.py --stage h3a-controlled
```

Supplementary/sensitivity stages are explicit and are **not** run by the main-paper chain:

```bash
python run_pipeline.py --stage supplementary-mlaad-negative-control
python run_pipeline.py --stage supplementary-h4-relaxed-sensitivity
```

## Raw pairing and extraction

Raw-data reconstruction code remains under `src/00_pairing/`, `src/01_taxonomy/`, and `src/02_extraction/`. These scripts document how the validated residual inputs were built, but the paper runner does not silently execute them.

This distinction is intentional: the historical exhaustive physical MLAAD/M-AILABS path audit was **not completed**. The recorded physical checks covered **57,999 of 146,140 unique paths**, with no failures among the checked paths. The release therefore does not claim complete physical SHA certification of all paths.

## Figures

The only active main-figure generator is:

```bash
python src/07_figures/GENERATE_SELECTED_MANUSCRIPT_FIGURES.py --root "$FINGERPRINT_OUTPUT_ROOT"
```

Selected PDFs:

```text
Fig1_pipeline.pdf
Fig2_H1_content_reproducibility_bars.pdf
Fig3_H2_cross_language_bars.pdf
Fig4a_H3_checkpoint_conceptual.pdf
Fig4b_H3_nominal_null_histograms.pdf
Fig5_H4_family_organization.pdf
Fig6_H4_adjusted_profiles_by_family.pdf
```

## Validation

```bash
python -m compileall -q src run_pipeline.py
pytest -q
python src/05_master/MASTER_Q1_MANUSCRIPT_FINAL_v3_2.py --registry-only
python src/08_validation/VALIDATE_RELEASE.py --repo .
```

## Historical notebooks

The development notebooks are not used as executable release modules. Their role and the reason for excluding repeated/superseded cells are documented in `docs/HISTORICAL_NOTEBOOK_PROVENANCE.md`.
