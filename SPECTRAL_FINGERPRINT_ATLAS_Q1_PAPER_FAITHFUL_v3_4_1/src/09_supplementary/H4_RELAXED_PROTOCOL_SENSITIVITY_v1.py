#!/usr/bin/env python3
"""Optional MLAAD RELAXED protocol sensitivity using the final H4 engine.

This stage is not part of the primary global H4 decision. It reuses the exact
same H4 computations on the broader 64,625-pair RELAXED population, matching
the manuscript description of that population as sensitivity-only.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
H4_PATH = HERE.parents[1] / "04_h4/H4_GLOBAL_CONFIRMATORY_FINAL_v1.py"


def load_h4_module():
    spec = importlib.util.spec_from_file_location("h4_global_final", H4_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load H4 engine: {H4_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    h4 = load_h4_module()
    root = Path(os.environ.get("FINGERPRINT_OUTPUT_ROOT", "/content/drive/MyDrive/fingerprint_q1_outputs"))
    phase1a_candidates = [
        root / "phase1a/phase1a_mlaad_spectral_residuals_v1",
        root / "phase1a/phase1a_mlaad_spectral_residuals_v2_new_story",
    ]
    phase1a = next((p for p in phase1a_candidates if (p / "fingerprints_pair_level_relaxed.parquet").is_file()), None)
    if phase1a is None:
        raise FileNotFoundError("MLAAD RELAXED pair-level residual parquet not found")

    relaxed_input = phase1a / "fingerprints_pair_level_relaxed.parquet"
    axis_input = phase1a / "frequency_axis.csv"
    if not axis_input.is_file():
        raise FileNotFoundError(axis_input)

    # Repoint the shared engine's axis input before reading it.
    h4.FREQUENCY_AXIS_INPUT = axis_input
    frequency_axis = h4.read_frequency_axis()
    residual_columns = frequency_axis["column_name"].astype(str).tolist()
    analysis_mask = (
        (frequency_axis["frequency_hz"].to_numpy(float) >= h4.ANALYSIS_MIN_HZ)
        & (frequency_axis["frequency_hz"].to_numpy(float) <= h4.ANALYSIS_MAX_HZ)
    )
    if int(np.sum(analysis_mask)) != 481:
        raise RuntimeError(f"Expected 481 final inferential bins, found {int(np.sum(analysis_mask))}")

    out_summary = h4.OUTPUT_DIR / "relaxed" / "phase1b_protocol_summary.json"
    if out_summary.is_file() and not h4.FORCE_REBUILD:
        summary = json.loads(out_summary.read_text(encoding="utf-8"))
        if summary.get("status") != "COMPLETE" or str(summary.get("protocol")).lower() != "relaxed":
            raise RuntimeError("Existing RELAXED sensitivity summary is incomplete or incompatible")
        print("[RESUME] Existing RELAXED sensitivity result reused.")
    else:
        summary = h4.run_protocol(
            "relaxed",
            relaxed_input,
            64_625,
            residual_columns,
            analysis_mask,
            protocol_seed=h4.RANDOM_SEED + 1,
        )

    note = {
        "role": "SENSITIVITY_ONLY_NOT_PRIMARY_H4_DECISION",
        "summary": str(out_summary),
        "decision": summary["evidence_decision"]["status"],
        "pairs": 64_625,
    }
    h4.atomic_json_dump(note, h4.OUTPUT_DIR / "H4_RELAXED_SENSITIVITY_NOTE.json")
    print(json.dumps(note, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
