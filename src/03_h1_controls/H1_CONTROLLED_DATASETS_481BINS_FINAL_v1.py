# ======================================================================================
# Q1 — H1 FINAL HARMONIZED
# 481 BINS + 50 CONTENT BLOCKS + 200 SPLIT-HALF
#
# CORPORA:
#   - WaveFake-LJSpeech
#   - WaveFake-JSUT
#   - LibriSeVoc v2
#
# MLAAD: EXCLUDED — already handled separately
#
# NONE OF THE FOLLOWING:
#   - WAV
#   - pairing
#   - taxonomy
#   - extraction
#   - bootstrap H1
#
# The script:
#   1) validates the final populations;
#   2) loads the 50 actual content blocks already produced by Q1_01;
#   3) verifies that they cover 100% of the pairs;
#   4) reproduces the previous 513-bin H1 EXACTLY;
#   5) only if this self-check passes, selects the 481 bins;
#   6) executes the same 200 splits;
#   7) checkpoints after each repeat;
#   8) produces table + JSON + ZIP.
# ======================================================================================

from pathlib import Path
import hashlib
import json
import os
import shutil
import gc

try:
    from google.colab import drive, files
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive", force_remount=False)
except Exception:
    drive = None
    files = None
    print("[INFO] Google Colab not detected or Drive already unavailable; using existing filesystem paths.")

# ======================================================================================
# DEPENDENCIES
# ======================================================================================

try:
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
except Exception:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy", "pandas", "pyarrow"])
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq


# ======================================================================================
# FROZEN CONFIGURATION
# ======================================================================================

VERSION = "Q1-H1-481BINS-50BLOCKS-v1.2.0-A2Z"

MASTER_SEED = 20260711

FMIN = 80.0
FMAX = 7600.0

EXPECTED_FULL_BINS = 513
EXPECTED_ANALYSIS_BINS = 481

CONTENT_BLOCKS = 50
SPLIT_REPEATS = 200
CONFIDENCE = 0.95

PRINT_EVERY = 5

# False = automatic resume
# True = delete only the NEW 481-bin outputs and restart
FORCE_RESTART = False

AUTO_DOWNLOAD_ZIP = os.environ.get("H1_AUTO_DOWNLOAD_ZIP", "0") == "1"

MYDRIVE = Path(os.environ.get("FINGERPRINT_MYDRIVE", "/content/drive/MyDrive"))
ROOT = Path(os.environ.get("FINGERPRINT_OUTPUT_ROOT", str(MYDRIVE / "fingerprint_q1_outputs")))

OUTPUT_ROOT = (
    ROOT
    / "H1_481BINS_50BLOCKS_FINAL_v1"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================================================================
# INPUT DISCOVERY
# ======================================================================================

def first_existing_dir(candidates, label):
    candidates = [Path(p) for p in candidates]
    for p in candidates:
        if (p / "fingerprints_pair_level_harmonized.parquet").is_file():
            return p
    # If running from raw, the v3_new_story directory may not exist until extraction.
    # Return the first canonical target so the error message remains deterministic.
    print(f"[INFO] {label}: no completed harmonized directory found yet; canonical target is {candidates[0]}")
    return candidates[0]

# ======================================================================================
# DATASETS — MLAAD INTENTIONALLY EXCLUDED
# ======================================================================================

DATASETS = {

    "wavefake_ljspeech": {
        "label": "WaveFake-LJSpeech",

        "full_dir": first_existing_dir([
            ROOT / "q1_harmonized/v3_new_story/wavefake_ljspeech/full",
            ROOT / "q1_harmonized/v1/wavefake_ljspeech/full",
        ], "WaveFake-LJSpeech"),

        "expected_pairs": 91700,
        "expected_originals": 13100,
        "expected_generators": 7,
    },

    "wavefake_jsut": {
        "label": "WaveFake-JSUT",

        "full_dir": first_existing_dir([
            ROOT / "q1_harmonized/v3_new_story/wavefake_jsut/full",
            ROOT / "q1_harmonized/v1/wavefake_jsut/full",
        ], "WaveFake-JSUT"),

        "expected_pairs": 10000,
        "expected_originals": 5000,
        "expected_generators": 2,
    },

    "librisevoc": {
        "label": "LibriSeVoc-v2",

        "full_dir": first_existing_dir([
            ROOT / "q1_harmonized/v3_new_story/librisevoc/full",
            ROOT / "q1_harmonized/v2/librisevoc/full",
        ], "LibriSeVoc-v2"),

        "expected_pairs": 72174,
        "expected_originals": 12029,
        "expected_generators": 6,
    },
}


# ======================================================================================
# UTILITIES
# ======================================================================================

def banner(text):
    print("\n" + "=" * 118)
    print(text)
    print("=" * 118, flush=True)


def atomic_json(obj, path):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
            default=str
        ),
        encoding="utf-8"
    )

    os.replace(
        tmp,
        path
    )


def atomic_csv(df, path):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    df.to_csv(
        tmp,
        index=False
    )

    os.replace(
        tmp,
        path
    )


def stable_int_hash(text, modulo):

    digest = hashlib.sha256(
        str(text).encode("utf-8")
    ).digest()

    return (
        int.from_bytes(
            digest[:8],
            byteorder="little",
            signed=False
        )
        % modulo
    )


def pearson_correlation(first, second):

    a = np.asarray(
        first,
        dtype=np.float64
    )

    b = np.asarray(
        second,
        dtype=np.float64
    )

    a = a - a.mean()
    b = b - b.mean()

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator <= 1e-15:
        return 0.0

    return float(
        np.dot(a, b)
        / denominator
    )


def percentile_ci(
    values,
    confidence=0.95
):

    x = np.asarray(
        values,
        dtype=np.float64
    )

    x = x[
        np.isfinite(x)
    ]

    if x.size == 0:

        return [
            float("nan"),
            float("nan")
        ]

    alpha = (
        1.0
        - confidence
    )

    low, high = np.quantile(
        x,
        [
            alpha / 2.0,
            1.0 - alpha / 2.0
        ]
    )

    return [
        float(low),
        float(high)
    ]


def append_jsonl(
    path,
    obj
):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    line = (
        json.dumps(
            obj,
            ensure_ascii=False,
            default=str
        )
        + "\n"
    )

    with path.open(
        "a",
        encoding="utf-8"
    ) as f:

        f.write(line)

        f.flush()

        try:
            os.fsync(
                f.fileno()
            )
        except Exception:
            pass


def read_jsonl(path):

    path = Path(path)

    rows = {}

    if not path.is_file():
        return rows

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = (
                line.strip()
            )

            if not line:
                continue

            try:
                obj = json.loads(
                    line
                )

            except Exception:
                # last line may have been interrupted
                continue

            if (
                isinstance(obj, dict)
                and "repeat" in obj
            ):

                rows[
                    int(obj["repeat"])
                ] = obj

    return rows


def detect_column(
    columns,
    candidates,
    required=True
):

    for candidate in candidates:

        if candidate in columns:
            return candidate

    if required:

        raise RuntimeError(
            "Column not found: "
            f"{candidates}"
        )

    return None


# ======================================================================================
# EXACT Q1_01 SPLIT SEED
# ======================================================================================

SPLIT_SEED = (
    stable_int_hash(
        "split-block-reproducibility",
        2**32 - 1
    )
    + MASTER_SEED
)

print(
    "MASTER_SEED =",
    MASTER_SEED
)

print(
    "SPLIT_SEED  =",
    SPLIT_SEED
)


# ======================================================================================
# SPLIT-HALF ENGINE
# ======================================================================================

def one_split(
    block_vectors,
    permutation
):

    n_generators = (
        block_vectors.shape[0]
    )

    n_blocks = (
        block_vectors.shape[1]
    )

    split_point = (
        n_blocks // 2
    )

    first_blocks = (
        permutation[
            :split_point
        ]
    )

    second_blocks = (
        permutation[
            split_point:
        ]
    )

    if (
        len(first_blocks) == 0
        or
        len(second_blocks) == 0
    ):

        raise RuntimeError(
            "Split vide."
        )

    first_fp = np.median(
        block_vectors[
            :,
            first_blocks,
            :
        ],
        axis=1
    )

    second_fp = np.median(
        block_vectors[
            :,
            second_blocks,
            :
        ],
        axis=1
    )

    same = []

    for i in range(
        n_generators
    ):

        same.append(
            pearson_correlation(
                first_fp[i],
                second_fp[i]
            )
        )

    different = []

    for i in range(
        n_generators
    ):

        for j in range(
            n_generators
        ):

            if i == j:
                continue

            different.append(
                pearson_correlation(
                    first_fp[i],
                    second_fp[j]
                )
            )

    return {

        "n_blocks_first":
            int(
                len(first_blocks)
            ),

        "n_blocks_second":
            int(
                len(second_blocks)
            ),

        "mean_same_generator_correlation":
            float(
                np.mean(same)
            ),

        "mean_different_generator_correlation":
            float(
                np.mean(different)
            ),

        "generator_specificity_delta":
            float(
                np.mean(same)
                - np.mean(different)
            ),

        "minimum_same_generator_correlation":
            float(
                np.min(same)
            ),
    }


def compute_200_splits_no_checkpoint(
    block_vectors
):

    rng = np.random.default_rng(
        SPLIT_SEED
    )

    rows = []

    for repeat in range(
        SPLIT_REPEATS
    ):

        permutation = (
            rng.permutation(
                block_vectors.shape[1]
            )
        )

        rec = one_split(
            block_vectors,
            permutation
        )

        rec[
            "repeat"
        ] = repeat

        rows.append(
            rec
        )

    return pd.DataFrame(
        rows
    )


# ======================================================================================
# AUDIT + HISTORICAL SELF-CHECK
# ======================================================================================

def prepare_dataset(
    key,
    spec
):

    label = (
        spec["label"]
    )

    full_dir = (
        spec["full_dir"]
    )

    pair_file = (
        full_dir
        / "fingerprints_pair_level_harmonized.parquet"
    )

    axis_file = (
        full_dir
        / "frequency_axis.csv"
    )

    old_npz = (
        full_dir
        / "stability"
        / "content_block_fingerprints.npz"
    )

    old_summary = (
        full_dir
        / "stability"
        / "stability_summary.json"
    )

    banner(
        f"AUDIT — {label}"
    )

    required_files = {

        "pair parquet":
            pair_file,

        "frequency axis":
            axis_file,

        "content block fingerprints":
            old_npz,

        "historical stability summary":
            old_summary,
    }

    for name, path in (
        required_files.items()
    ):

        if not path.is_file():

            raise FileNotFoundError(
                f"{label} — "
                f"{name} missing:\n"
                f"{path}"
            )

        print(
            f"[PASS] {name}"
        )


    # ==================================================================================
    # POPULATION
    # ==================================================================================

    pf = pq.ParquetFile(
        pair_file
    )

    n_pairs = int(
        pf.metadata.num_rows
    )

    if (
        n_pairs
        != spec["expected_pairs"]
    ):

        raise RuntimeError(
            f"{label}: "
            f"{n_pairs:,} pairs instead of "
            f"{spec['expected_pairs']:,}"
        )

    schema = (
        pf.schema.names
    )

    gen_col = detect_column(
        schema,
        [
            "independent_generator_id",
            "generator_id",
            "generator"
        ]
    )

    orig_col = detect_column(
        schema,
        [
            "original_id",
            "real_id",
            "source_id"
        ]
    )

    pair_col = detect_column(
        schema,
        [
            "pair_id",
            "analysis_id"
        ],
        required=False
    )

    status_col = detect_column(
        schema,
        [
            "status",
            "status_harmonized"
        ],
        required=False
    )

    meta_cols = [
        gen_col,
        orig_col
    ]

    if pair_col:
        meta_cols.append(
            pair_col
        )

    if status_col:
        meta_cols.append(
            status_col
        )

    meta = pd.read_parquet(
        pair_file,
        columns=meta_cols
    )

    meta[gen_col] = (
        meta[gen_col]
        .astype(str)
    )

    meta[orig_col] = (
        meta[orig_col]
        .astype(str)
    )

    n_generators = int(
        meta[
            gen_col
        ].nunique()
    )

    n_originals = int(
        meta[
            orig_col
        ].nunique()
    )

    if (
        n_generators
        != spec[
            "expected_generators"
        ]
    ):

        raise RuntimeError(
            f"{label}: incorrect number "
            "of generators."
        )

    if (
        n_originals
        != spec[
            "expected_originals"
        ]
    ):

        raise RuntimeError(
            f"{label}: incorrect number "
            "of originals."
        )


    # at most one row per generator × original
    duplicates = int(
        meta.duplicated(
            subset=[
                gen_col,
                orig_col
            ]
        ).sum()
    )

    if duplicates != 0:

        raise RuntimeError(
            f"{label}: "
            f"{duplicates} duplicate rows "
            "for generator × original_id."
        )


    if pair_col:

        pair_duplicates = int(
            meta[
                pair_col
            ]
            .astype(str)
            .duplicated()
            .sum()
        )

        if pair_duplicates != 0:

            raise RuntimeError(
                f"{label}: "
                f"{pair_duplicates} "
                "duplicate pair_id values."
            )


    # each original must appear for every generator
    per_original = (
        meta.groupby(
            orig_col
        )[
            gen_col
        ]
        .nunique()
    )

    bad_originals = int(
        (
            per_original
            != spec[
                "expected_generators"
            ]
        ).sum()
    )

    if bad_originals != 0:

        raise RuntimeError(
            f"{label}: "
            f"{bad_originals} originals "
            "without complete coverage."
        )


    if status_col:

        status = (
            meta[
                status_col
            ]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        bad_status = int(
            (
                ~status.eq("ok")
            ).sum()
        )

        if bad_status != 0:

            raise RuntimeError(
                f"{label}: "
                f"{bad_status} rows "
                "status != OK."
            )


    generators_meta = sorted(
        meta[
            gen_col
        ]
        .astype(str)
        .unique()
    )

    print(
        f"[PASS] pairs          : "
        f"{n_pairs:,}/"
        f"{spec['expected_pairs']:,}"
    )

    print(
        f"[PASS] originals   : "
        f"{n_originals:,}/"
        f"{spec['expected_originals']:,}"
    )

    print(
        f"[PASS] generators     : "
        f"{n_generators}/"
        f"{spec['expected_generators']}"
    )

    print(
        "[PASS] population perfectly balanced"
    )


    # ==================================================================================
    # FREQUENCY AXIS
    # ==================================================================================

    axis = pd.read_csv(
        axis_file
    )

    required_axis_columns = {
        "bin_index",
        "column_name",
        "frequency_hz"
    }

    if not (
        required_axis_columns
        .issubset(axis.columns)
    ):

        raise RuntimeError(
            f"{label}: "
            "frequency_axis.csv is invalid."
        )

    if (
        len(axis)
        != EXPECTED_FULL_BINS
    ):

        raise RuntimeError(
            f"{label}: "
            f"{len(axis)} bins instead of 513."
        )


    # CRITICAL verification:
    # bin_index must match exactly
    # the actual position 0..512.
    expected_index = np.arange(
        EXPECTED_FULL_BINS,
        dtype=int
    )

    actual_index = (
        axis[
            "bin_index"
        ]
        .astype(int)
        .to_numpy()
    )

    if not np.array_equal(
        actual_index,
        expected_index
    ):

        raise RuntimeError(
            f"{label}: "
            "bin_index does not match "
            "positions 0..512."
        )

    print(
        "[PASS] bin_index == positions 0..512"
    )


    frequencies_full = (
        axis[
            "frequency_hz"
        ]
        .astype(float)
        .to_numpy()
    )

    band_mask = (
        (
            frequencies_full
            >= FMIN
        )
        &
        (
            frequencies_full
            <= FMAX
        )
    )

    axis481 = (
        axis.loc[
            band_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if (
        len(axis481)
        != EXPECTED_ANALYSIS_BINS
    ):

        raise RuntimeError(
            f"{label}: "
            f"{len(axis481)} bins "
            "within 80–7600 Hz instead of 481."
        )

    selected_indices = (
        axis481[
            "bin_index"
        ]
        .astype(int)
        .to_numpy()
    )

    frequencies_481 = (
        axis481[
            "frequency_hz"
        ]
        .astype(float)
        .to_numpy()
    )

    if not np.isclose(
        frequencies_481[0],
        93.75
    ):

        raise RuntimeError(
            f"{label}: "
            "unexpected first bin."
        )

    if not np.isclose(
        frequencies_481[-1],
        7593.75
    ):

        raise RuntimeError(
            f"{label}: "
            "unexpected last bin."
        )

    print(
        "[PASS] nominal band: "
        "80–7600 Hz"
    )

    print(
        "[PASS] FFT centers      : "
        f"{frequencies_481[0]:.2f}"
        "–"
        f"{frequencies_481[-1]:.2f} Hz"
    )

    print(
        "[PASS] number of bins   : "
        f"{len(frequencies_481)}"
    )


    # ==================================================================================
    # ACTUAL 50 HISTORICAL BLOCKS
    # ==================================================================================

    z = np.load(
        old_npz,
        allow_pickle=True
    )

    required_npz = {
        "generator_ids",
        "block_ids",
        "block_vectors",
        "block_counts"
    }

    if not required_npz.issubset(
        z.files
    ):

        raise RuntimeError(
            f"{label}: "
            "content_block_fingerprints.npz is incomplete."
        )


    generator_ids = (
        np.asarray(
            z["generator_ids"]
        )
        .astype(str)
        .tolist()
    )

    block_ids = np.asarray(
        z["block_ids"],
        dtype=np.int32
    )

    block_vectors_513 = np.asarray(
        z["block_vectors"],
        dtype=np.float32
    )

    block_counts = np.asarray(
        z["block_counts"],
        dtype=np.int64
    )


    expected_shape = (
        spec[
            "expected_generators"
        ],
        CONTENT_BLOCKS,
        EXPECTED_FULL_BINS
    )

    if (
        block_vectors_513.shape
        != expected_shape
    ):

        raise RuntimeError(
            f"{label}: "
            f"block shape = "
            f"{block_vectors_513.shape}, "
            f"expected {expected_shape}."
        )


    if (
        block_counts.shape
        != (
            spec[
                "expected_generators"
            ],
            CONTENT_BLOCKS
        )
    ):

        raise RuntimeError(
            f"{label}: "
            "block_counts shape is incorrect."
        )


    if (
        len(block_ids)
        != CONTENT_BLOCKS
    ):

        raise RuntimeError(
            f"{label}: "
            f"{len(block_ids)} blocks "
            "instead of 50."
        )


    if not np.all(
        block_counts > 0
    ):

        raise RuntimeError(
            f"{label}: "
            "at least one block is empty."
        )


    if not np.isfinite(
        block_vectors_513
    ).all():

        raise RuntimeError(
            f"{label}: "
            "NaN/Inf found in the blocks."
        )


    if (
        sorted(
            generator_ids
        )
        != generators_meta
    ):

        raise RuntimeError(
            f"{label}: "
            "NPZ generators != Parquet generators."
        )


    # ==================================================================================
    # PAIR COVERAGE
    # ==================================================================================

    total_pairs_in_blocks = int(
        block_counts.sum()
    )

    if (
        total_pairs_in_blocks
        != spec[
            "expected_pairs"
        ]
    ):

        raise RuntimeError(
            f"{label}: "
            f"{total_pairs_in_blocks:,} "
            "pairs in the blocks instead of "
            f"{spec['expected_pairs']:,}."
        )


    per_generator_counts = (
        block_counts.sum(
            axis=1
        )
    )

    if not np.all(
        per_generator_counts
        == spec[
            "expected_originals"
        ]
    ):

        raise RuntimeError(
            f"{label}: "
            "block/generator coverage is incorrect."
        )


    print(
        "[PASS] content blocks : 50"
    )

    print(
        f"[PASS] H1 coverage      : "
        f"{total_pairs_in_blocks:,}/"
        f"{spec['expected_pairs']:,}"
        " = 100 %"
    )


    # ==================================================================================
    # HISTORICAL 513-BIN SELF-CHECK
    # ==================================================================================

    banner(
        f"SELF-CHECK 513 BINS — {label}"
    )

    historical = json.loads(
        old_summary.read_text(
            encoding="utf-8"
        )
    )

    old_block = (
        historical[
            "split_block_reproducibility"
        ]
    )

    check513 = (
        compute_200_splits_no_checkpoint(
            block_vectors_513
        )
    )

    calc_same = float(
        check513[
            "mean_same_generator_correlation"
        ].mean()
    )

    calc_diff = float(
        check513[
            "mean_different_generator_correlation"
        ].mean()
    )

    calc_delta = float(
        check513[
            "generator_specificity_delta"
        ].mean()
    )


    ref_same = float(
        old_block[
            "same_generator_correlation_mean"
        ]
    )

    ref_diff = float(
        old_block[
            "different_generator_correlation_mean"
        ]
    )

    ref_delta = float(
        old_block[
            "generator_specificity_delta_mean"
        ]
    )


    print(
        f"same : recomputed={calc_same:.12f} | "
        f"historical={ref_same:.12f} | "
        f"Δ={calc_same-ref_same:+.3e}"
    )

    print(
        f"diff : recomputed={calc_diff:.12f} | "
        f"historical={ref_diff:.12f} | "
        f"Δ={calc_diff-ref_diff:+.3e}"
    )

    print(
        f"Δgen : recomputed={calc_delta:.12f} | "
        f"historical={ref_delta:.12f} | "
        f"Δ={calc_delta-ref_delta:+.3e}"
    )


    SELF_CHECK_TOLERANCE = 1e-8

    self_check_ok = (
        abs(
            calc_same - ref_same
        )
        <= SELF_CHECK_TOLERANCE

        and

        abs(
            calc_diff - ref_diff
        )
        <= SELF_CHECK_TOLERANCE

        and

        abs(
            calc_delta - ref_delta
        )
        <= SELF_CHECK_TOLERANCE
    )


    if not self_check_ok:

        raise RuntimeError(
            f"{label}: "
            "HISTORICAL 513-BIN SELF-CHECK FAILED.\n"
            "The 481-bin H1 WILL NOT be run."
        )


    print(
        "[PASS] moteur H1 reproduit "
        "the previous 513-bin result exactly."
    )


    # ==================================================================================
    # SWITCH TO 481 BINS
    # ==================================================================================

    block_vectors_481 = (
        block_vectors_513[
            :,
            :,
            selected_indices
        ]
        .astype(
            np.float32,
            copy=True
        )
    )


    expected_481_shape = (
        spec[
            "expected_generators"
        ],
        CONTENT_BLOCKS,
        EXPECTED_ANALYSIS_BINS
    )


    if (
        block_vectors_481.shape
        != expected_481_shape
    ):

        raise RuntimeError(
            f"{label}: "
            "481-bin shape is incorrect."
        )


    print(
        "[PASS] 481-bin selection ready."
    )


    return {

        "pair_file":
            pair_file,

        "axis_file":
            axis_file,

        "old_npz":
            old_npz,

        "old_summary":
            old_summary,

        "generator_ids":
            generator_ids,

        "block_ids":
            block_ids,

        "block_counts":
            block_counts,

        "block_vectors_481":
            block_vectors_481,

        "axis481":
            axis481,

        "frequencies":
            frequencies_481,

        "historical_513": {

            "same":
                ref_same,

            "different":
                ref_diff,

            "delta":
                ref_delta,
        },

        "population": {

            "n_pairs":
                n_pairs,

            "n_originals":
                n_originals,

            "n_generators":
                n_generators,

            "pairs_in_content_blocks":
                total_pairs_in_blocks,

            "all_pairs_covered":
                True,
        },
    }


# ======================================================================================
# H1 481 BINS WITH RESUME
# ======================================================================================

def run_h1_481(
    key,
    spec,
    prepared
):

    label = (
        spec["label"]
    )

    out = (
        OUTPUT_ROOT
        / key
    )


    if (
        FORCE_RESTART
        and out.exists()
    ):

        print(
            f"[FORCE] suppression : "
            f"{out}"
        )

        shutil.rmtree(
            out
        )


    out.mkdir(
        parents=True,
        exist_ok=True
    )


    summary_file = (
        out
        / "stability_summary_481bins.json"
    )

    complete_file = (
        out
        / ".H1_481_COMPLETE.json"
    )


    # ==================================================================================
    # ALREADY COMPLETE
    # ==================================================================================

    if (
        summary_file.is_file()
        and complete_file.is_file()
        and not FORCE_RESTART
    ):

        previous = json.loads(
            summary_file.read_text(
                encoding="utf-8"
            )
        )

        if (
            previous.get(
                "status"
            )
            == "COMPLETE"
        ):

            banner(
                f"{label} — ALREADY COMPLETE"
            )

            block = (
                previous[
                    "split_block_reproducibility"
                ]
            )

            print(
                "[SKIP] 200/200 already computed."
            )

            print(
                f"Δgen = "
                f"{block['generator_specificity_delta_mean']:.9f}"
            )

            return previous


    # ==================================================================================
    # CONFIG LOCK
    # ==================================================================================

    config = {

        "version":
            VERSION,

        "dataset":
            key,

        "pair_file":
            str(
                prepared[
                    "pair_file"
                ]
            ),

        "pair_file_size":
            int(
                prepared[
                    "pair_file"
                ].stat().st_size
            ),

        "old_content_block_npz":
            str(
                prepared[
                    "old_npz"
                ]
            ),

        "master_seed":
            MASTER_SEED,

        "split_seed":
            int(
                SPLIT_SEED
            ),

        "fmin_nominal":
            FMIN,

        "fmax_nominal":
            FMAX,

        "actual_first_bin_hz":
            float(
                prepared[
                    "frequencies"
                ][0]
            ),

        "actual_last_bin_hz":
            float(
                prepared[
                    "frequencies"
                ][-1]
            ),

        "analysis_bins":
            EXPECTED_ANALYSIS_BINS,

        "content_blocks":
            CONTENT_BLOCKS,

        "split_repeats":
            SPLIT_REPEATS,

        "bootstrap":
            False,
    }


    lock_file = (
        out
        / "H1_481_CONFIG_LOCK.json"
    )


    if lock_file.is_file():

        previous_config = json.loads(
            lock_file.read_text(
                encoding="utf-8"
            )
        )

        if (
            previous_config
            != config
        ):

            raise RuntimeError(
                f"{label}: "
                "configuration differs "
                "from a previous run."
            )

    else:

        atomic_json(
            config,
            lock_file
        )


    # ==================================================================================
    # SAVE 481-BIN AXIS + BLOCKS
    # ==================================================================================

    atomic_csv(
        prepared[
            "axis481"
        ],
        out
        / "frequency_axis_481bins.csv"
    )


    np.savez_compressed(

        out
        / "content_block_fingerprints_481bins.npz",

        generator_ids=np.asarray(
            prepared[
                "generator_ids"
            ],
            dtype="U"
        ),

        block_ids=np.asarray(
            prepared[
                "block_ids"
            ],
            dtype=np.int32
        ),

        block_vectors=np.asarray(
            prepared[
                "block_vectors_481"
            ],
            dtype=np.float32
        ),

        block_counts=np.asarray(
            prepared[
                "block_counts"
            ],
            dtype=np.int32
        ),
    )


    # ==================================================================================
    # RESUME
    # ==================================================================================

    checkpoint = (
        out
        / "split_repeats_481bins_checkpoint.jsonl"
    )

    completed = read_jsonl(
        checkpoint
    )


    banner(
        f"H1 481 BINS — {label}"
    )

    print(
        f"Resume           : "
        f"{len(completed)}/200"
    )

    print(
        f"Pairs            : "
        f"{spec['expected_pairs']:,}"
    )

    print(
        "Coverage          : 100 %"
    )

    print(
        "Content blocks    : 50"
    )

    print(
        "Split             : "
        f"{CONTENT_BLOCKS // 2} vs "
        f"{CONTENT_BLOCKS - CONTENT_BLOCKS // 2}"
    )

    print(
        "Analysis bins     : 481"
    )

    print(
        "Nominal band      : 80–7600 Hz"
    )

    print(
        "FFT centers       : "
        f"{prepared['frequencies'][0]:.2f}"
        "–"
        f"{prepared['frequencies'][-1]:.2f} Hz"
    )

    print(
        f"MASTER_SEED       : "
        f"{MASTER_SEED}"
    )

    print(
        "H1 bootstrap      : NO"
    )


    vectors = (
        prepared[
            "block_vectors_481"
        ]
    )


    # Recreating the RNG from the beginning guarantees
    # an EXACT resume of the sequence.
    rng = np.random.default_rng(
        SPLIT_SEED
    )


    for repeat in range(
        SPLIT_REPEATS
    ):

        permutation = (
            rng.permutation(
                CONTENT_BLOCKS
            )
        )


        # Important :
        # permutation is consumed even if the repeat was already computed.
        if repeat in completed:
            continue


        rec = one_split(
            vectors,
            permutation
        )

        rec[
            "repeat"
        ] = int(
            repeat
        )


        append_jsonl(
            checkpoint,
            rec
        )


        completed[
            repeat
        ] = rec


        done = len(
            completed
        )


        if (
            done == 1
            or
            done % PRINT_EVERY == 0
            or
            done == SPLIT_REPEATS
        ):

            pct = (
                100.0
                * done
                / SPLIT_REPEATS
            )

            print(
                f"[{label}] "
                f"{done:3d}/200 "
                f"({pct:5.1f}%) | "
                f"same="
                f"{rec['mean_same_generator_correlation']:.6f} | "
                f"diff="
                f"{rec['mean_different_generator_correlation']:.6f} | "
                f"Δ="
                f"{rec['generator_specificity_delta']:.6f} | "
                f"checkpoint=OK",
                flush=True
            )


    if (
        len(completed)
        != SPLIT_REPEATS
    ):

        raise RuntimeError(
            f"{label}: "
            f"{len(completed)}/200 repeats."
        )


    # ==================================================================================
    # FINALIZATION
    # ==================================================================================

    rows = [

        completed[i]

        for i in range(
            SPLIT_REPEATS
        )
    ]


    df = pd.DataFrame(
        rows
    ).sort_values(
        "repeat"
    )


    df = df[
        [
            "repeat",
            "n_blocks_first",
            "n_blocks_second",
            "mean_same_generator_correlation",
            "mean_different_generator_correlation",
            "generator_specificity_delta",
            "minimum_same_generator_correlation",
        ]
    ]


    atomic_csv(
        df,
        out
        / "split_block_reproducibility_481bins.csv"
    )


    same_ci = percentile_ci(
        df[
            "mean_same_generator_correlation"
        ],
        CONFIDENCE
    )

    different_ci = percentile_ci(
        df[
            "mean_different_generator_correlation"
        ],
        CONFIDENCE
    )

    delta_ci = percentile_ci(
        df[
            "generator_specificity_delta"
        ],
        CONFIDENCE
    )


    same_mean = float(
        df[
            "mean_same_generator_correlation"
        ].mean()
    )

    different_mean = float(
        df[
            "mean_different_generator_correlation"
        ].mean()
    )

    delta_mean = float(
        df[
            "generator_specificity_delta"
        ].mean()
    )


    supported = bool(
        delta_ci[0] > 0
    )


    summary = {

        "version":
            VERSION,

        "status":
            "COMPLETE",

        "dataset_key":
            key,

        "dataset_label":
            label,

        "population": {

            **prepared[
                "population"
            ],

            "expected_pairs":
                spec[
                    "expected_pairs"
                ],

            "expected_originals":
                spec[
                    "expected_originals"
                ],

            "expected_generators":
                spec[
                    "expected_generators"
                ],
        },

        "representation": {

            "stored_spectral_bins":
                513,

            "inferential_bins":
                481,

            "nominal_analysis_band_hz":
                [
                    80.0,
                    7600.0
                ],

            "actual_fft_centers_hz":
                [
                    float(
                        prepared[
                            "frequencies"
                        ][0]
                    ),
                    float(
                        prepared[
                            "frequencies"
                        ][-1]
                    )
                ],
        },

        "protocol": {

            "master_seed":
                MASTER_SEED,

            "split_seed":
                int(
                    SPLIT_SEED
                ),

            "content_blocks":
                CONTENT_BLOCKS,

            "split_repetitions":
                SPLIT_REPEATS,

            "bootstrap":
                False,

            "block_source":
                str(
                    prepared[
                        "old_npz"
                    ]
                ),

            "split_rule":
                (
                    f"{CONTENT_BLOCKS // 2} "
                    "content blocks vs "
                    f"{CONTENT_BLOCKS - CONTENT_BLOCKS // 2} "
                    "content blocks"
                ),

            "aggregation":
                "median of content-block fingerprints",

            "similarity":
                "Pearson correlation",

            "different_generator_baseline":
                (
                    "all ordered cross-generator comparisons "
                    "i != j"
                ),
        },

        "coverage": {

            "pairs_available":
                int(
                    spec[
                        "expected_pairs"
                    ]
                ),

            "pairs_represented_in_blocks":
                int(
                    prepared[
                        "block_counts"
                    ].sum()
                ),

            "coverage_fraction":
                1.0,

            "all_50_blocks_used_each_repeat":
                True,
        },

        "historical_513_self_check": {

            "status":
                "PASS",

            "same_generator_correlation_mean":
                prepared[
                    "historical_513"
                ][
                    "same"
                ],

            "different_generator_correlation_mean":
                prepared[
                    "historical_513"
                ][
                    "different"
                ],

            "generator_specificity_delta_mean":
                prepared[
                    "historical_513"
                ][
                    "delta"
                ],
        },

        "split_block_reproducibility": {

            "n_repeats":
                SPLIT_REPEATS,

            "same_generator_correlation_mean":
                same_mean,

            "same_generator_correlation_ci":
                same_ci,

            "different_generator_correlation_mean":
                different_mean,

            "different_generator_correlation_ci":
                different_ci,

            "generator_specificity_delta_mean":
                delta_mean,

            "generator_specificity_delta_ci":
                delta_ci,

            "minimum_same_generator_correlation":
                float(
                    df[
                        "minimum_same_generator_correlation"
                    ].min()
                ),

            "generator_specificity_supported":
                supported,

            "scope_note":
                (
                    "Intervals are split-resampling stability intervals "
                    "under content variation, not population confidence intervals."
                ),
        },
    }


    atomic_json(
        summary,
        summary_file
    )


    atomic_json(

        {
            "status":
                "COMPLETE",

            "dataset":
                key,

            "summary":
                str(
                    summary_file
                ),
        },

        complete_file
    )


    banner(
        f"FINAL RESULT — {label}"
    )

    print(
        f"same      = "
        f"{same_mean:.9f}"
    )

    print(
        f"same CI95 = "
        f"[{same_ci[0]:.9f}, "
        f"{same_ci[1]:.9f}]"
    )

    print(
        f"different = "
        f"{different_mean:.9f}"
    )

    print(
        f"diff CI95 = "
        f"[{different_ci[0]:.9f}, "
        f"{different_ci[1]:.9f}]"
    )

    print(
        f"Δgen      = "
        f"{delta_mean:.9f}"
    )

    print(
        f"Δgen CI95 = "
        f"[{delta_ci[0]:.9f}, "
        f"{delta_ci[1]:.9f}]"
    )

    print(
        "H1        =",
        (
            "REPRODUCIBLE"
            if supported
            else
            "INSUFFICIENT_EVIDENCE"
        )
    )

    return summary


# ======================================================================================
# EXECUTION
# ======================================================================================

banner(
    "Q1 — H1 FINAL 481 BINS / 50 BLOCKS / 200 SPLITS"
)

print(
    "WaveFake-LJSpeech : YES"
)

print(
    "WaveFake-JSUT     : YES"
)

print(
    "LibriSeVoc-v2     : YES"
)

print(
    "MLAAD             : NO — excluded"
)

print(
    "WAV               : NO"
)

print(
    "Extraction        : NO"
)

print(
    "Pairing           : NO"
)

print(
    "H1 bootstrap      : NO"
)

print(
    "Resume           : YES"
)

print(
    "Outputs:",
    OUTPUT_ROOT
)


all_results = {}


for key, spec in (
    DATASETS.items()
):

    try:

        prepared = prepare_dataset(
            key,
            spec
        )

        result = run_h1_481(
            key,
            spec,
            prepared
        )

        all_results[
            key
        ] = result

        del prepared

        gc.collect()


    except KeyboardInterrupt:

        print(
            "\n[INTERRUPTION]"
        )

        print(
            "Checkpoints have been preserved."
        )

        print(
            "Rerun this exact same cell."
        )

        raise


# ======================================================================================
# FINAL TABLE
# ======================================================================================

rows = []


for key, result in (
    all_results.items()
):

    block = (
        result[
            "split_block_reproducibility"
        ]
    )

    population = (
        result[
            "population"
        ]
    )


    rows.append({

        "dataset":
            result[
                "dataset_label"
            ],

        "n_pairs":
            population[
                "n_pairs"
            ],

        "pairs_in_blocks":
            population[
                "pairs_in_content_blocks"
            ],

        "coverage_percent":
            100.0,

        "n_originals":
            population[
                "n_originals"
            ],

        "n_generators":
            population[
                "n_generators"
            ],

        "analysis_bins":
            481,

        "band_hz":
            "93.75-7593.75",

        "content_blocks":
            50,

        "split_repeats":
            200,

        "same_mean":
            block[
                "same_generator_correlation_mean"
            ],

        "same_ci95_low":
            block[
                "same_generator_correlation_ci"
            ][0],

        "same_ci95_high":
            block[
                "same_generator_correlation_ci"
            ][1],

        "different_mean":
            block[
                "different_generator_correlation_mean"
            ],

        "different_ci95_low":
            block[
                "different_generator_correlation_ci"
            ][0],

        "different_ci95_high":
            block[
                "different_generator_correlation_ci"
            ][1],

        "delta_gen":
            block[
                "generator_specificity_delta_mean"
            ],

        "delta_ci95_low":
            block[
                "generator_specificity_delta_ci"
            ][0],

        "delta_ci95_high":
            block[
                "generator_specificity_delta_ci"
            ][1],

        "H1":
            (
                "REPRODUCIBLE"
                if block[
                    "generator_specificity_supported"
                ]
                else
                "INSUFFICIENT_EVIDENCE"
            ),

        "historical_delta_513":
            result[
                "historical_513_self_check"
            ][
                "generator_specificity_delta_mean"
            ],
    })


final_table = pd.DataFrame(
    rows
)


TABLE_PATH = (
    OUTPUT_ROOT
    / "H1_481BINS_50BLOCKS_FINAL_TABLE.csv"
)


atomic_csv(
    final_table,
    TABLE_PATH
)


SUMMARY_PATH = (
    OUTPUT_ROOT
    / "H1_481BINS_50BLOCKS_FINAL_SUMMARY.json"
)


atomic_json(

    {
        "version":
            VERSION,

        "status":
            "COMPLETE",

        "mlaad":
            "EXCLUDED_ALREADY_RUNNING_SEPARATELY",

        "stored_bins":
            513,

        "inferential_bins":
            481,

        "nominal_band_hz":
            [
                80.0,
                7600.0
            ],

        "actual_fft_centers_hz":
            [
                93.75,
                7593.75
            ],

        "content_blocks":
            50,

        "split_repeats":
            200,

        "bootstrap_h1":
            False,

        "master_seed":
            MASTER_SEED,

        "datasets":
            all_results,
    },

    SUMMARY_PATH
)


banner(
    "H1 481 BINS — COMPLETE"
)


print(
    final_table[
        [
            "dataset",
            "n_pairs",
            "coverage_percent",
            "analysis_bins",
            "content_blocks",
            "split_repeats",
            "same_mean",
            "different_mean",
            "delta_gen",
            "delta_ci95_low",
            "delta_ci95_high",
            "H1",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nFinal table:",
    TABLE_PATH
)

print(
    "JSON summary:",
    SUMMARY_PATH
)


# ======================================================================================
# FINAL ZIP
# ======================================================================================

ZIP_BASE = Path(
    "/content/H1_481BINS_50BLOCKS_FINAL_v1"
)


ZIP_PATH = Path(
    shutil.make_archive(
        str(
            ZIP_BASE
        ),
        "zip",
        root_dir=str(
            OUTPUT_ROOT
        )
    )
)


print(
    "\nZIP :",
    ZIP_PATH
)

print(
    "Size:",
    f"{ZIP_PATH.stat().st_size / 1024**2:.2f} MB"
)


if AUTO_DOWNLOAD_ZIP:

    print(
        "\nDownloading..."
    )

    files.download(
        str(
            ZIP_PATH
        )
    )