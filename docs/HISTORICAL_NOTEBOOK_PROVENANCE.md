# Historical notebook provenance

The final paper was developed through several large Colab notebooks. They are valuable for provenance but they are not a clean executable description of the final method.

The audited development notebooks contain repeated or superseded blocks, including:

- several complete H1→H4 master implementations in `Untitled15`;
- multiple versions of global fingerprint analysis and figure generation in `article_2_inchaelah`;
- repair/finalization cells for corrected LibriSeVoc manifests in `CORRECTIONFINGERPRINT`;
- successive MLAAD family, lineage, and sensitivity analyses in `fingerprint_mlaad`;
- LibriSeVoc pairing/freeze/final-audit cells in `fingerprintlibrisecvox`.

Some historical cells failed before later corrected cells were run; some later cells existed only to repair or audit earlier outputs. Consequently, publishing the notebooks verbatim as a single pipeline would falsely imply that every cell contributed to the final results.

The active source tree therefore keeps one implementation per final manuscript endpoint. Historical recovery masters, duplicated figure builders, and exploratory deep-dive scripts not retained as paper endpoints are excluded from the release.

This cleanup changes repository organization only. It does not change the final scientific protocol, final numerical registry, or inferential decisions.
