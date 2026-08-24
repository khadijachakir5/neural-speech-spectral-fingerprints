# Final manuscript-to-code mapping

This table maps **only analyses retained in the final paper** to active code.

| Final manuscript component | Active code |
|---|---|
| Paired spectral representation | `src/common/spectral.py`; extraction scripts in `src/02_extraction/` |
| Pairing / manifest QC | `src/00_pairing/validate_pairing_*.py`, `freeze_*_manifest*.py`, `STEP_00_REBUILD_PAIRING_CANONICAL_v1.py` |
| H1 controlled populations | `src/03_h1_controls/H1_CONTROLLED_DATASETS_481BINS_FINAL_v1.py` |
| H1 MLAAD | `src/03_h1_controls/Q1_03_MLAAD_GENERATOR_STABILITY_v2_NEW_STORY.py` |
| Primary LibriSeVoc false-pair control | `src/03_h1_controls/Q1_07_LIBRISEVOC_FALSE_PAIR_v3_NEW_STORY.py` |
| Supplementary MLAAD negative pairing | `src/03_h1_controls/Q1_08_MLAAD_NEGATIVE_PAIR_CONTROL_v1.py` |
| H2 cross-language | `src/04_h2/H2_CROSS_LANGUAGE_FINAL_v1.py` |
| H3a exact checkpoint | `src/04_h3/H3A_CONTROLLED_FROM_RAW_FINAL_v1.py` |
| H3b + final H3 aggregation | `src/04_h3/H3_FINAL_MANUSCRIPT_v1.py` |
| H4 global confirmatory | `src/04_h4/H4_GLOBAL_CONFIRMATORY_FINAL_v1.py` |
| H4 family-specific exploratory | `src/04_h4/H4_FAMILY_SPECIFIC_EXPLORATORY_FINAL_v1.py` |
| MLAAD RELAXED protocol sensitivity | `src/09_supplementary/H4_RELAXED_PROTOCOL_SENSITIVITY_v1.py` (reuses the final H4 engine) |
| Final consistency registry | `src/05_master/MASTER_Q1_MANUSCRIPT_FINAL_v3_2.py` |
| Main manuscript figures | `src/07_figures/GENERATE_SELECTED_MANUSCRIPT_FIGURES.py` |
| Release/result validation | `src/08_validation/` |

## Boundary enforced by this release

- Confirmatory: H1, H2, controlled H3a, global MLAAD STRICT H4 according to the frozen conjunctive rule.
- Exploratory: H3b nominal architecture and family-specific H4 contrasts.
- Supplementary control: MLAAD same-language negative pairing.
- Historical lineage/frequency deep-dive engines are not part of the active paper pipeline.
- The incomplete exhaustive physical pairing audit is not represented as a completed manuscript stage.
