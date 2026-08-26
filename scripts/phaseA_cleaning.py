#!/usr/bin/env python3
"""
Phase A - Data assembly and cleaning for:
"Comparative Genomic and Network Analysis of Co-occurring Drug-Resistance
Mutations Across African Mycobacterium tuberculosis Lineages"

Splits the raw TB-Profiler dr_variants.csv (14,734 rows) into four explicit,
documented buckets. No bucket is inferred implicitly by a confidence
.isin() filter without the others being accounted for - every row in the
raw file ends up in exactly one of the four outputs, and this is asserted
at the end.

Authoritative sample-level metadata file: y_labels.csv (NOT labels.csv -
they differ in 41 rows, both in main_lineage/sub_lineage, where
y_labels.csv has "Unknown" and labels.csv has NaN for the same samples;
y_labels.csv is the standardized source everywhere in this pipeline).

Buckets produced:
  1. core_set            - confidence in {"Assoc w R", "Assoc w R - Interim"}
                             Primary analysis set for Objectives 1-4.
                             Verified duplicate-free; drop_duplicates() is
                             still applied defensively and asserted to be a
                             no-op.
  2. sensitivity_set     - confidence == "Uncertain significance", correctly
                             deduplicated with drop_duplicates() (default
                             keep='first'). Raw = 3,633 rows in exactly 1,163
                             duplicate PAIRS (no larger clusters) + 1,307
                             singletons -> 2,470 unique rows retained.
                             NEVER filtered with duplicated(keep=False),
                             which would incorrectly drop 2,326 rows instead
                             of the true 1,163 excess copies and undercount
                             this set by 47%.
  3. excluded_set        - confidence in {"Not assoc w R",
                             "Not assoc w R - Interim"}. Excluded per
                             TB-Profiler's own annotation, not analyzed
                             further, but kept on disk (not silently
                             dropped) for auditability.
  4. unlabeled_confidence - confidence is NaN (257 rows, 1.7% of raw file).
                             Genuinely unlabeled by TB-Profiler, not the same
                             as "Uncertain significance". Kept as its own
                             bucket rather than merged into any other set or
                             silently dropped by a confidence .isin() filter.

Also produces:
  - core_set_with_metadata.csv: core_set joined to y_labels.csv sample-level
    fields (drtype, main_lineage, sub_lineage, MDR/pre_XDR/XDR, per-drug
    phenotype + binary columns) for downstream Objective 1-4 use.
  - lineage_stratification_flags.csv: one row per sample_id in y_labels.csv,
    flagging the 49 samples (41 "Unknown" + 8 compound e.g.
    "lineage2;lineage4") that are NOT a single clean lineage label. Does NOT
    silently drop or default these - Objective 4 must decide explicitly
    whether to exclude them from per-lineage stratification or bucket them
    separately; that decision is deferred to the Objective 4 script, not
    made here.

Inputs expected (all in -indir, untouched raw data):
    dr_variants.csv     TB-Profiler per-sample resistance mutation calls
    y_labels.csv         authoritative sample-level metadata (lineage, drtype,
                          MDR/pre_XDR/XDR, per-drug phenotype)
    sample_ids.txt        full 1,858-sample cohort list

Outputs written to -outdir (results/phaseA_cleaning/, NOT data/ - data/
holds only untouched raw input; every step's output lives under results/
in its own step-specific subfolder):
    core_set.csv
    sensitivity_set.csv
    excluded_set.csv
    unlabeled_confidence_set.csv
    core_set_with_metadata.csv
    lineage_stratification_flags.csv
"""
import argparse
import os

import pandas as pd

RAW_ROW_COUNT_EXPECTED = 14734
CORE_ROW_COUNT_EXPECTED = 10199
EXCLUDED_ROW_COUNT_EXPECTED = 645
UNCERTAIN_RAW_EXPECTED = 3633
UNCERTAIN_DEDUP_EXPECTED = 2470
NAN_CONF_EXPECTED = 257
TOTAL_AFTER_DEDUP_EXPECTED = 13571  # 10199 + 645 + 2470 + 257

CORE_LABELS = ["Assoc w R", "Assoc w R - Interim"]
EXCLUDED_LABELS = ["Not assoc w R", "Not assoc w R - Interim"]
UNCERTAIN_LABEL = "Uncertain significance"


def load_raw(indir):
    df = pd.read_csv(os.path.join(indir, "dr_variants.csv"))
    assert len(df) == RAW_ROW_COUNT_EXPECTED, (
        f"Raw dr_variants.csv row count changed: expected "
        f"{RAW_ROW_COUNT_EXPECTED}, got {len(df)}. Stop - re-verify before "
        f"trusting downstream numbers."
    )
    return df


def split_buckets(df: pd.DataFrame):
    core = df[df["confidence"].isin(CORE_LABELS)].copy()
    excluded = df[df["confidence"].isin(EXCLUDED_LABELS)].copy()
    uncertain_raw = df[df["confidence"] == UNCERTAIN_LABEL].copy()
    unlabeled = df[df["confidence"].isna()].copy()

    # Defensive: assert every raw row lands in exactly one bucket.
    total_bucketed = len(core) + len(excluded) + len(uncertain_raw) + len(unlabeled)
    assert total_bucketed == len(df), (
        f"Bucket split doesn't cover all rows: {total_bucketed} bucketed vs "
        f"{len(df)} raw rows. There is a confidence value not accounted for."
    )

    # Core set: verify it's already duplicate-free, then dedup defensively
    # (should be a no-op - assert that it is).
    core_deduped = core.drop_duplicates()
    assert len(core_deduped) == len(core), (
        "Core set was assumed duplicate-free but drop_duplicates() removed "
        f"rows ({len(core)} -> {len(core_deduped)}). This assumption no "
        "longer holds - investigate before proceeding."
    )
    core = core_deduped

    # Sensitivity set: CORRECT dedup (keep='first'), never duplicated(keep=False).
    sensitivity = uncertain_raw.drop_duplicates()

    return core, sensitivity, excluded, unlabeled, uncertain_raw


def validate_counts(core, sensitivity, excluded, unlabeled, uncertain_raw):
    checks = {
        "core_set": (len(core), CORE_ROW_COUNT_EXPECTED),
        "excluded_set": (len(excluded), EXCLUDED_ROW_COUNT_EXPECTED),
        "uncertain_raw (pre-dedup)": (len(uncertain_raw), UNCERTAIN_RAW_EXPECTED),
        "sensitivity_set (post-dedup)": (len(sensitivity), UNCERTAIN_DEDUP_EXPECTED),
        "unlabeled_confidence": (len(unlabeled), NAN_CONF_EXPECTED),
    }
    failures = []
    for name, (actual, expected) in checks.items():
        status = "OK" if actual == expected else "MISMATCH"
        print(f"  [{status}] {name}: {actual} (expected {expected})")
        if actual != expected:
            failures.append(name)

    total_after_dedup = len(core) + len(excluded) + len(sensitivity) + len(unlabeled)
    status = "OK" if total_after_dedup == TOTAL_AFTER_DEDUP_EXPECTED else "MISMATCH"
    print(
        f"  [{status}] total rows after correct dedup: {total_after_dedup} "
        f"(expected {TOTAL_AFTER_DEDUP_EXPECTED})"
    )
    if total_after_dedup != TOTAL_AFTER_DEDUP_EXPECTED:
        failures.append("total_after_dedup")

    if failures:
        raise AssertionError(f"Count validation failed for: {failures}")


def load_metadata(indir):
    y = pd.read_csv(os.path.join(indir, "y_labels.csv"))
    sample_ids = set(
        l.strip() for l in open(os.path.join(indir, "sample_ids.txt")) if l.strip()
    )
    assert len(sample_ids) == 1858, f"sample_ids.txt count changed: {len(sample_ids)}"
    assert set(y["sample_id"]) == sample_ids, (
        "y_labels.csv sample_id set does not match sample_ids.txt - "
        "cohort mismatch, stop and investigate."
    )
    return y


def flag_lineage_stratification(y: pd.DataFrame) -> pd.DataFrame:
    """
    Flags samples that are NOT usable as a single, clean lineage label for
    Objective 4 stratification. Does not make the exclude/bucket decision -
    that belongs in the Objective 4 script - only surfaces it explicitly.
    """
    is_unknown = y["main_lineage"] == "Unknown"
    is_compound = y["main_lineage"].str.contains(";", na=False)
    flags = y[["sample_id", "main_lineage", "sub_lineage"]].copy()
    flags["is_unknown_lineage"] = is_unknown
    flags["is_compound_lineage"] = is_compound
    flags["stratifiable_single_lineage"] = ~(is_unknown | is_compound)

    n_flagged = (~flags["stratifiable_single_lineage"]).sum()
    print(
        f"  Lineage stratification flags: {n_flagged} samples NOT a single "
        f"clean lineage ({is_unknown.sum()} Unknown + {is_compound.sum()} "
        f"compound). Objective 4 must decide explicitly how to handle these."
    )
    assert n_flagged == 49, f"Expected 49 flagged samples, got {n_flagged}"
    return flags


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "-indir", default="../data/raw",
        help="directory containing dr_variants.csv, y_labels.csv, sample_ids.txt "
             "(default: ../data/raw, i.e. run from the scripts/ folder)",
    )
    ap.add_argument(
        "-outdir", default="../results/phaseA_cleaning",
        help="directory to write the six cleaned output files to "
             "(default: ../results/phaseA_cleaning, i.e. run from the "
             "scripts/ folder - data/ is raw input only, every step's "
             "output lives under results/ in its own subfolder)",
    )
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Reading from: {os.path.abspath(args.indir)}")
    print(f"Writing to:   {os.path.abspath(args.outdir)}")
    print()

    print("Loading raw dr_variants.csv ...")
    df = load_raw(args.indir)

    print("Splitting into 4 buckets ...")
    core, sensitivity, excluded, unlabeled, uncertain_raw = split_buckets(df)

    print("Validating counts against pre-verified numbers ...")
    validate_counts(core, sensitivity, excluded, unlabeled, uncertain_raw)

    print("Loading and validating y_labels.csv (authoritative metadata) ...")
    y = load_metadata(args.indir)

    print("Joining core set to sample metadata ...")
    core_with_meta = core.merge(y, on="sample_id", how="left", validate="m:1")
    assert core_with_meta["drtype"].isna().sum() == 0, (
        "Some core-set rows failed to join to y_labels.csv metadata - "
        "unexpected sample_id mismatch."
    )

    print("Flagging lineage stratification eligibility ...")
    lineage_flags = flag_lineage_stratification(y)

    print(f"Writing outputs to {args.outdir} ...")
    core.to_csv(os.path.join(args.outdir, "core_set.csv"), index=False)
    sensitivity.to_csv(os.path.join(args.outdir, "sensitivity_set.csv"), index=False)
    excluded.to_csv(os.path.join(args.outdir, "excluded_set.csv"), index=False)
    unlabeled.to_csv(os.path.join(args.outdir, "unlabeled_confidence_set.csv"), index=False)
    core_with_meta.to_csv(os.path.join(args.outdir, "core_set_with_metadata.csv"), index=False)
    lineage_flags.to_csv(os.path.join(args.outdir, "lineage_stratification_flags.csv"), index=False)

    print()
    print("Phase A complete. Row counts:")
    print(f"  core_set.csv                    {len(core):>6}")
    print(f"  sensitivity_set.csv              {len(sensitivity):>6}")
    print(f"  excluded_set.csv                 {len(excluded):>6}")
    print(f"  unlabeled_confidence_set.csv     {len(unlabeled):>6}")
    print(f"  core_set_with_metadata.csv       {len(core_with_meta):>6}")
    print(f"  lineage_stratification_flags.csv {len(lineage_flags):>6}")


if __name__ == "__main__":
    main()
