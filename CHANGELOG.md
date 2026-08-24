# Changelog

## 3.4.1-paper-faithful

- Fixed the three mixed French/English source strings identified in the v3.4.0 audit and additional short human-facing remnants exposed by the restored regression guard (including `Colonne introuvable`, `ont`, `doublons`, `Complets`, `canonique`, `incorrecte`, `bande nominale`, `OUI`, `finaux`, and `finale`).
- Restored `tests/test_source_language.py` as a regression guard, with word-boundary patterns that catch French remnants without flagging English identifiers such as `ambiguous_fake`.
- Scientific protocol, numerical registry, analysis logic, seeds, datasets, and manuscript-facing results are unchanged.

## 3.4.0-paper-faithful

- Reorganized the release around the analyses actually retained in the final manuscript.
- Removed superseded H3/H4 recovery engines, the historical master, duplicated figure-reference code, the release-bundle helper, and the unfinished exhaustive physical-audit engine from the active package.
- Replaced the synthetic A-to-Z orchestration with one canonical `paper-from-residuals` manuscript chain.
- Kept raw pairing/extraction scripts for reproducibility without implying they were rerun as one uninterrupted final pipeline.
- Kept the MLAAD negative-pair control as an explicit supplementary stage rather than part of the main-paper chain.
- Preserved `config/protocol_v3_2.json` and the frozen manuscript numerical registry unchanged.
