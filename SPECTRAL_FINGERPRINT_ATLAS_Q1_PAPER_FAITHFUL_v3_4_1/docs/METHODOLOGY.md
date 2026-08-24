# Methodology summary

The residual spectral fingerprint for pair `i` is

\[
r_i^{raw}(f)=S_i^{fake}(f)-S_i^{real}(f),
\]

where `S` is the median active-frame log-power spectrum. The broadband level term is removed by

\[
r_i(f)=r_i^{raw}(f)-\mathrm{median}_{80\le f\le7600}r_i^{raw}(f).
\]

The inferential unit is adapted to each question (content reconstruction, multilingual generator, checkpoint, or generator-level family analysis); raw recordings are not treated as independent technological realizations for generator/family claims. See `config/protocol_v3_2.json` for the frozen endpoint definitions, multiplicity families, bootstrap counts, and decision rules.
