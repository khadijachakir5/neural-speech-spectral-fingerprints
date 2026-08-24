# Purpose: Prevent silent regression of the MLAAD H1 population and same-language reference guardrails.

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO
    / "src"
    / "03_h1_controls"
    / "Q1_03_MLAAD_GENERATOR_STABILITY_v2_NEW_STORY.py"
)


def load_q103():
    module_dir = MODULE_PATH.parent
    sys.path.insert(0, str(module_dir))
    try:
        spec = importlib.util.spec_from_file_location("q103_guard_test", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if sys.path and sys.path[0] == str(module_dir):
            sys.path.pop(0)


def test_q103_version_and_canonical_constants():
    q = load_q103()
    assert q.VERSION == "Q1_03-MLAAD-GENERATOR-STABILITY-v2.0.0"
    assert q.EXPECTED_STRICT_PAIRS == 62079
    assert q.EXPECTED_STRICT_GENERATORS == 52
    assert q.EXPECTED_STRICT_LANGUAGES == 8


def test_q103_population_guard_rejects_noncanonical_input():
    q = load_q103()
    frame = pd.DataFrame(
        {
            "independent_generator_id": ["g1", "g2"],
            "language": ["en", "en"],
        }
    )
    with pytest.raises(RuntimeError, match="Canonical MLAAD STRICT population guard failed"):
        q.validate_canonical_population(frame)


def test_q103_forbids_global_language_fallback():
    q = load_q103()
    diagnostics = pd.DataFrame(
        {
            "reference_source": ["language", "global_fallback"],
            "n_reference_generators": [4, 51],
            "independent_generator_id": ["g1", "g2"],
            "language": ["en", "uk"],
        }
    )
    with pytest.raises(RuntimeError, match="forbids global language fallback"):
        q.assert_language_only_references(
            diagnostics,
            half="A",
            repeat=0,
            minimum_reference_generators=2,
        )


def test_q103_accepts_same_language_references_only():
    q = load_q103()
    diagnostics = pd.DataFrame(
        {
            "reference_source": ["language", "language"],
            "n_reference_generators": [4, 3],
            "independent_generator_id": ["g1", "g2"],
            "language": ["en", "fr"],
        }
    )
    q.assert_language_only_references(
        diagnostics,
        half="B",
        repeat=7,
        minimum_reference_generators=2,
    )
