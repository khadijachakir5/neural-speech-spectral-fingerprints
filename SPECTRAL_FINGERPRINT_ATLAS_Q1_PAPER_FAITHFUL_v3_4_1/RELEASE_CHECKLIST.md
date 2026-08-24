# Release checklist — v3.4.1 paper-faithful

- [ ] `config/protocol_v3_2.json` remains the final manuscript protocol.
- [ ] `reference_results/manuscript_results_v1.json` matches final text/tables.
- [ ] Only one active implementation is mapped to each final manuscript endpoint.
- [ ] Superseded recovery/deep-dive/historical-master scripts are absent from the active package.
- [ ] `paper-from-residuals` does not silently run raw pairing/extraction or the incomplete physical audit.
- [ ] MLAAD negative pairing and RELAXED H4 sensitivity are explicit non-primary stages, not part of the main-paper chain.
- [ ] H3b remains exploratory: exactly five English-matched Pearson tests, Holm across five.
- [ ] H4 global verdict remains `INSUFFICIENT_EVIDENCE`.
- [ ] Family-specific neural-codec result remains explicitly exploratory.
- [ ] README retains the physical-audit qualification: 57,999/146,140 checked, 0 failures in checked subset, full audit incomplete.
- [ ] Main manuscript figures are exactly the selected PDFs under `paper_figures/main/`.
- [ ] `python -m compileall -q src run_pipeline.py` passes.
- [ ] `pytest -q` passes.
- [ ] `tests/test_source_language.py` passes and remains in the release.
- [ ] registry-only master validation passes.
- [ ] release validation passes.
- [ ] `sha256sum -c MANIFEST_SHA256.txt` passes.
