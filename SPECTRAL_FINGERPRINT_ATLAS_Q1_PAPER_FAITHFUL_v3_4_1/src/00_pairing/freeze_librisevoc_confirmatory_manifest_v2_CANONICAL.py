

# Purpose: Freeze the QC-valid six-vocoder balanced LibriSeVoc confirmatory manifest deterministically.

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


CONFIG = dict(
    input_manifest=Path(
        "/content/drive/MyDrive/fingerprint_q1_outputs/phase0_librisevoc_v2/librisevoc_manifest.parquet"
    ),
    output_dir=Path(
        "/content/drive/MyDrive/fingerprint_q1_outputs/phase0_librisevoc_final_v2"
    ),
    required_columns={
        "pair_id", "dataset", "independent_generator_id",
        "waveform_architecture", "waveform_family", "original_id",
        "fake_path", "real_path", "fake_sha256", "real_sha256", "qc_status", "exclusion_reason",
    },
    n_generators_expected=6,
    expected_source_pairs=79_206,
    expected_qc_ok_pairs=77_890,
    expected_confirmatory_originals=12_029,
    expected_confirmatory_pairs=72_174,
    
    quick_mode=False,
    quick_n_originals=200,   
    master_seed=20260711,    
)


def validate_schema(df: pd.DataFrame, required_columns: set) -> None:
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if len(df) == 0:
        raise RuntimeError("The manifest is empty.")


def validate_dataset_single(df: pd.DataFrame, expected: str = "librisevoc") -> None:
    dataset_values = sorted(df["dataset"].astype(str).str.lower().unique())
    if not all(v == expected for v in dataset_values):
        raise ValueError(f"Unexpected dataset: {dataset_values}")


def validate_no_duplicates(df: pd.DataFrame) -> None:
    if df["pair_id"].duplicated().any():
        raise RuntimeError("Duplicate pair_id values detected.")
    if df.duplicated(["independent_generator_id", "original_id"]).any():
        raise RuntimeError("Duplicate generator-original_id pairs detected.")


def validate_generator_count(ok: pd.DataFrame, expected: int) -> list:
    generator_ids = sorted(ok["independent_generator_id"].astype(str).unique())
    if len(generator_ids) != expected:
        raise RuntimeError(f"{expected} generators expected, found : {len(generator_ids)}")
    return generator_ids


def validate_taxonomy(ok: pd.DataFrame) -> pd.DataFrame:
    taxonomy = ok[
        ["independent_generator_id", "waveform_architecture", "waveform_family"]
    ].drop_duplicates()
    taxonomy_counts = taxonomy.groupby("independent_generator_id").size()
    if not taxonomy_counts.eq(1).all():
        raise RuntimeError("A generator has multiple taxonomy assignments.")
    if taxonomy[["waveform_architecture", "waveform_family"]].isna().any().any():
        raise RuntimeError("Taxonomy LibriSeVoc incomplete.")
    return taxonomy


def validate_hashes_present(ok: pd.DataFrame) -> None:

    missing_fake = ok["fake_sha256"].isna().sum()
    missing_real = ok["real_sha256"].isna().sum()
    if missing_fake or missing_real:
        raise RuntimeError(
            f"Missing hashes among QC=ok pairs : "
            f"{missing_fake} fake_sha256 and {missing_real} real_sha256 values are missing."
        )


def validate_real_consistency_per_original(ok: pd.DataFrame) -> None:


    path_variants = ok.groupby("original_id")["real_path"].nunique()
    hash_variants = ok.groupby("original_id")["real_sha256"].nunique()
    bad_paths = path_variants.loc[path_variants > 1]
    bad_hashes = hash_variants.loc[hash_variants > 1]
    if len(bad_paths) or len(bad_hashes):
        raise RuntimeError(
            f"Inconsistency real_path/real_sha256 par original_id : "
            f"{len(bad_paths)} original_id values with multiple real_path values, "
            f"{len(bad_hashes)} with multiple real_sha256 values."
        )


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:

    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def audit_hashes(ok: pd.DataFrame) -> tuple:

    fake_hash_rows = ok.dropna(subset=["fake_sha256"]).copy()
    duplicated_fake = fake_hash_rows.groupby("fake_sha256").agg(
        n_rows=("pair_id", "size"),
        n_originals=("original_id", "nunique"),
        n_generators=("independent_generator_id", "nunique"),
    )
    problematic_fake = duplicated_fake.loc[
        (duplicated_fake["n_originals"] > 1) | (duplicated_fake["n_generators"] > 1)
    ]
    if len(problematic_fake):
        raise RuntimeError(f"{len(problematic_fake)} ambiguous fake hashes detected.")

    real_hash_rows = ok.dropna(subset=["real_sha256"]).copy()
    real_conflicts = real_hash_rows.groupby("real_sha256")["original_id"].nunique()
    real_conflicts = real_conflicts.loc[real_conflicts > 1]
    if len(real_conflicts):
        raise RuntimeError(f"{len(real_conflicts)} real hashes associated with multiple original_id values.")

    fake_hash_set = set(fake_hash_rows["fake_sha256"].astype(str))
    real_hash_set = set(real_hash_rows["real_sha256"].astype(str))
    collisions = fake_hash_set & real_hash_set
    if collisions:
        raise RuntimeError(f"{len(collisions)} collisions fake–real detected.")

    return problematic_fake, real_conflicts, collisions


def build_balanced_set(ok: pd.DataFrame, n_generators: int) -> tuple:
    coverage = ok.groupby("original_id")["independent_generator_id"].nunique()
    complete_ids = coverage.loc[coverage.eq(n_generators)].index

    balanced = ok.loc[ok["original_id"].isin(complete_ids)].copy()
    balanced = balanced.sort_values(
        ["original_id", "independent_generator_id", "pair_id"]
    ).reset_index(drop=True)

    expected_pairs = len(complete_ids) * n_generators
    if len(balanced) != expected_pairs:
        raise RuntimeError(f"{len(balanced)} pairs observed, {expected_pairs} expected.")

    counts = balanced["independent_generator_id"].value_counts().sort_index()
    if counts.nunique() != 1:
        raise RuntimeError("The final manifest is not balanced.")

    n_incomplete = int((coverage < n_generators).sum())
    return balanced, complete_ids, counts, n_incomplete


def apply_quick_subsample(balanced: pd.DataFrame, n_originals: int, seed: int) -> pd.DataFrame:


    unique_originals = sorted(balanced["original_id"].unique())
    n = min(n_originals, len(unique_originals))
    rng = np.random.default_rng(seed)
    selected = rng.choice(unique_originals, size=n, replace=False)
    subset = balanced.loc[balanced["original_id"].isin(selected)].copy()
    return subset.sort_values(
        ["original_id", "independent_generator_id", "pair_id"]
    ).reset_index(drop=True)


def compute_selection_hash(balanced: pd.DataFrame) -> str:
    columns = ["pair_id", "original_id", "independent_generator_id", "fake_path", "real_path"]
    selection_text = "\n".join(
        balanced[columns].astype(str).agg("||".join, axis=1).tolist()
    )
    return hashlib.sha256(selection_text.encode("utf-8")).hexdigest()


def freeze_manifest():
    cfg = CONFIG
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)

    suffix = "_quick" if cfg["quick_mode"] else ""
    output_manifest = cfg["output_dir"] / f"librisevoc_manifest_confirmatory_balanced{suffix}.parquet"
    output_report = cfg["output_dir"] / f"librisevoc_manifest_confirmatory_report{suffix}.json"

    if not cfg["input_manifest"].exists():
        raise FileNotFoundError(f"Manifest not found: {cfg['input_manifest']}")

    df = pd.read_parquet(cfg["input_manifest"])
    validate_schema(df, cfg["required_columns"])
    validate_dataset_single(df)
    validate_no_duplicates(df)
    if not cfg["quick_mode"] and len(df) != cfg["expected_source_pairs"]:
        raise RuntimeError(f"Source LibriSeVoc: {len(df):,} rows; {cfg['expected_source_pairs']:,} expected.")
    contradiction = df["qc_status"].astype(str).str.lower().eq("ok") & df["exclusion_reason"].fillna("").astype(str).str.strip().ne("")
    if contradiction.any():
        raise RuntimeError(f"{int(contradiction.sum())} rows have qc_status='ok' but a non-empty exclusion_reason.")

    ok = df.loc[df["qc_status"].astype(str).str.lower().eq("ok")].copy()
    if not cfg["quick_mode"] and len(ok) != cfg["expected_qc_ok_pairs"]:
        raise RuntimeError(f"QC LibriSeVoc: {len(ok):,} pairs OK; {cfg['expected_qc_ok_pairs']:,} expected.")
    validate_generator_count(ok, cfg["n_generators_expected"])
    validate_hashes_present(ok)
    validate_real_consistency_per_original(ok)
    taxonomy = validate_taxonomy(ok)
    problematic_fake, real_conflicts, collisions = audit_hashes(ok)
    source_manifest_sha256 = sha256_file(cfg["input_manifest"])

    balanced, complete_ids, counts_by_generator, n_incomplete = build_balanced_set(
        ok, cfg["n_generators_expected"]
    )
    if not cfg["quick_mode"]:
        if len(complete_ids) != cfg["expected_confirmatory_originals"] or len(balanced) != cfg["expected_confirmatory_pairs"]:
            raise RuntimeError(
                f"Unexpected LibriSeVoc freeze: {len(complete_ids):,} originals/{len(balanced):,} pairs; "
                f"expected {cfg['expected_confirmatory_originals']:,}/{cfg['expected_confirmatory_pairs']:,}."
            )

    quick_applied = False
    if cfg["quick_mode"]:
        balanced = apply_quick_subsample(balanced, cfg["quick_n_originals"], cfg["master_seed"])
        counts_by_generator = balanced["independent_generator_id"].value_counts().sort_index()
        quick_applied = True

    selection_sha256 = compute_selection_hash(balanced)
    tmp_manifest = output_manifest.with_suffix(output_manifest.suffix + ".tmp")
    balanced.to_parquet(tmp_manifest, index=False)
    tmp_manifest.replace(output_manifest)

    report = dict(
        status="VALIDATED_AND_FROZEN",
        source_manifest=str(cfg["input_manifest"]),
        output_manifest=str(output_manifest),
        n_pairs_source=int(len(df)),
        n_pairs_qc_ok_before_balancing=int(len(ok)),
        n_pairs_confirmatory_balanced=int(len(balanced)),
        n_originals_confirmatory=int(balanced["original_id"].nunique()),
        n_generators=cfg["n_generators_expected"],
        pairs_per_generator={str(k): int(v) for k, v in counts_by_generator.items()},
        n_originals_with_incomplete_coverage_removed=n_incomplete,
        n_problematic_fake_hashes=int(len(problematic_fake)),
        n_real_hash_conflicts=int(len(real_conflicts)),
        n_fake_real_hash_collisions=int(len(collisions)),
        source_manifest_sha256=source_manifest_sha256,
        selection_sha256=selection_sha256,
        governance_note=(
            "source_manifest_sha256 is local to this script. Check consistency "
            "with taxonomy_hash.txt / PROTOCOL_VERSION.txt from the canonical protocol "
            "if these files exist, to avoid two sources of truth."
        ),
        quick_mode_applied=quick_applied,
        quick_mode_seed=cfg["master_seed"] if quick_applied else None,
        deterministic_note=(
            "Full balancing is a deterministic set intersection; no bootstrap or "
            "randomness involved. Seed only applies to the optional --quick subsample."
        ),
        taxonomy=taxonomy.sort_values("independent_generator_id").to_dict(orient="records"),
    )
    tmp_report = output_report.with_suffix(output_report.suffix + ".tmp")
    with tmp_report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    tmp_report.replace(output_report)

    print("=" * 78)
    print(f"LIBRISEVOC — FROZEN CONFIRMATORY MANIFEST{' (QUICK)' if quick_applied else ''}")
    print("=" * 78)
    print("QC-valid pairs before balance   :", len(ok))
    print("Originals shared across all 6 generators :", len(complete_ids))
    print("Final confirmatory pairs          :", len(balanced))
    print("Pairs by generator                :", counts_by_generator.to_dict())
    print("Originals incomplete removed          :", n_incomplete)
    if quick_applied:
        print(f"Mode quick : {cfg['quick_n_originals']} originals drawn (seed={cfg['master_seed']})")
    print("Hash selection                       :", selection_sha256)
    print("Final manifest:", output_manifest)
    print("Final report  :", output_report)

    return balanced, report


if __name__ == "__main__":
    librisevoc_balanced_df, librisevoc_freeze_report = freeze_manifest()