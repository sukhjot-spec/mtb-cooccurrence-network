#!/usr/bin/env python3
"""
Objective 1 - Identify genomic mutations associated with antimicrobial drug resistance among African Mycobacterium tuberculosis isolates.

FORWARD-COMPATIBILITY NOTE :
This script is the ONE place mutation_id gets assigned. Every later step (Objective 2's co-occurrence testing, Objective 3's network nodes,
Objective 4's per-lineage repeats) must read mutation_id from this script's own output (core_variants_with_mutation_id.csv / mutation_node_table.csv)
rather than recomputing gene+change concatenation itself. mutation_id is built once, here, as f"{gene}|{change}" (verified against the real data:
neither column contains a "|" character, so this is a safe, stable, human-readable separator) - if any later script builds its own id from
gene+change with a different separator or formatting, joins across steps will silently fail to match. Treat mutation_id as an immutable contract.

*** CRITICAL METHODOLOGICAL FINDING - READ BEFORE INTERPRETING obj1_fisher_results_core.csv ***
Testing core-set (Assoc w R) mutations against y_labels.csv's per-drug_binary columns is CIRCULAR, not confirmatory, and this was confirmed
directly against the real data before this script was finalised: across all 10,199 core mutation-drug rows, with ZERO exceptions, every sample
carrying a core mutation for drug D is marked resistant to D. TB-Profiler's per-drug binary/categorical phenotype columns are themselves generated
from the same Assoc-w-R catalogue tier being tested against them - they are a genotype-derived PREDICTION, not an independent phenotypic (culture
DST) measurement. Running Fisher's exact test between a core mutation and its own drug's binary column therefore recovers the catalogue's own
resistance-calling rule for this cohort (near-infinite odds ratios, near-zero p-values, for essentially every mutation) rather than providing
new evidence. This is expected and is not a bug in this script.

The genuinely meaningful confirmatory test runs on the SENSITIVITY set ("Uncertain significance" tier) instead: checked directly, only 64.9% of
sensitivity-set mutation-drug carriers are marked resistant to that drug (35.1% are marked susceptible despite carrying the variant) - real,
non-circular variance, because the binary phenotype columns are NOT derived from Uncertain-significance evidence. A significant, strong
association here is genuine cohort-specific evidence that could support promoting a specific Uncertain-significance variant, which is exactly
the kind of finding "not just trusting catalog membership, actually demonstrating the association in this cohort" (per the original project
plan) is asking for. Both tests are run and saved, clearly labelled and separated, so this distinction is never lost in a later step: obj1_fisher_
results_core.csv is background/consistency-check output only, obj1_fisher_results_sensitivity.csv is the real confirmatory analysis.

WHAT THIS SCRIPT DOES, IN ORDER:
  1. Loads core_set_with_metadata.csv (Phase A's output - 10,199 rows, confidence-confirmed resistance mutations only) and assigns mutation_id.
  2. Builds mutation_node_table.csv - ONE row per unique (gene, change), with all associated drugs listed as a single attribute (not one row
     per drug - this is the same node-identity decision already verified for the sibling-drug pairs, e.g. rifampicin/rifapentine share an
     identical carrier set, so a mutation must not be split into two nodes just because it confers resistance to two drugs). This table is the
     canonical node list Objective 3's network will be built from directly.
  3. Builds mutation_drug_frequency.csv - the same information in long format (one row per mutation-drug pair), for drug-specific reporting.
  4. Runs the confirmatory Fisher's exact test TWICE - once on the core set (background/consistency check only, see the circularity note
     above) and once on the sensitivity set (the real confirmatory analysis) - BH-FDR corrected separately within each, since they are
     two different families of hypotheses, not one combined test.

THE QC EXCLUSION, AND WHY IT'S HANDLED HERE, ONCE, EXPLICITLY:
Section 4 of the Phase A QC exploration established that 53 samples' drtype/phenotype columns cannot be trusted as ground truth (36 are
QC-outlier samples with implausible total_variants_qc; 42 have a drtype-implied resistance classification not supported by any
confidence-confirmed mutation - all 42 traced to Uncertain-significance or unlabelled evidence, never to a confirmed mutation). This script excludes
those 53 samples from BOTH sides of every confirmatory Fisher's test (neither their genotype nor their phenotype is used in that specific test)
- not just their phenotype label alone, because a 2x2 co-occurrence test needs both axes to come from a population whose evidence is trusted. This
applies uniformly to both the core-set and sensitivity-set tests. Mutation carriers remain in the node/frequency tables for both tiers regardless;
only the confirmatory PHENOTYPE tests exclude these 53 samples.

EDGE CASES CONFIRMED IN THE REAL DATA AND HANDLED EXPLICITLY (do not remove these checks - all fire on the real, current dataset, not just in theory):
  - rpoB p.Ser431Thr's only carrier (SRR33383939) is one of the 53 excluded samples, leaving zero carriers in the test population for that specific
    core-set mutation. Tests with zero carriers after exclusion are skipped with an explicit "insufficient_data" status, never silently run on a
    degenerate 2x2 table.
  - para-aminosalicylic_acid and cycloserine have zero core-set mutations at all, so no core-set (mutation, drug) test is ever generated for
    them. Expected, not a bug - logged explicitly.
  - Core-set odds ratios are mathematically infinite whenever the "mutation present but susceptible" cell is zero, which (per the circularity
    finding above) is EVERY core-set test. scipy.stats.fisher_exact still returns a valid p-value in this situation via the exact hypergeometric
    calculation; only the odds-ratio point estimate is undefined/infinite. This is left as inf rather than clipped to a large finite number, so
    it is never mistaken for a genuine finite effect size downstream.

Inputs expected:
    -cleaning_dir/core_set_with_metadata.csv   Phase A cleaning output
    -cleaning_dir/sensitivity_set.csv          Phase A cleaning output
    -qc_dir/qc_flagged_samples.csv             Phase A QC exploration output
    -raw_dir/y_labels.csv                      raw cohort metadata

Outputs written to -outdir:
    mutation_node_table.csv                one row per (gene, change) node
    core_variants_with_mutation_id.csv       core_set_with_metadata.csv + mutation_id
    mutation_drug_frequency.csv            long format, one row per mutation-drug pair
    obj1_fisher_results_core.csv           background/consistency-check only (circular, see above)
    obj1_fisher_results_sensitivity.csv    the real confirmatory analysis, BH-FDR corrected
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

CORE_ROW_COUNT_EXPECTED = 10199
SENSITIVITY_ROW_COUNT_EXPECTED = 2470
COHORT_SIZE_EXPECTED = 1858
LOW_POWER_THRESHOLD = 5  # carriers below this count are flagged, not suppressed


def build_mutation_id(gene, change):
    return f"{gene}|{change}"


def load_inputs(cleaning_dir, qc_dir, raw_dir):
    core = pd.read_csv(os.path.join(cleaning_dir, "core_set_with_metadata.csv"))
    assert len(core) == CORE_ROW_COUNT_EXPECTED, (
        f"core_set_with_metadata.csv row count changed: expected "
        f"{CORE_ROW_COUNT_EXPECTED}, got {len(core)}. Stop - re-verify "
        f"Phase A's output before trusting anything downstream."
    )

    dupe_check = core.duplicated(subset=["sample_id", "gene", "change", "drug"]).sum()
    assert dupe_check == 0, (
        f"{dupe_check} duplicate (sample_id, gene, change, drug) tuples found "
        f"in core_set_with_metadata.csv - this was verified as 0 during "
        f"development; investigate before proceeding."
    )

    sensitivity = pd.read_csv(os.path.join(cleaning_dir, "sensitivity_set.csv"))
    assert len(sensitivity) == SENSITIVITY_ROW_COUNT_EXPECTED, (
        f"sensitivity_set.csv row count changed: expected "
        f"{SENSITIVITY_ROW_COUNT_EXPECTED}, got {len(sensitivity)}."
    )

    qc = pd.read_csv(os.path.join(qc_dir, "qc_flagged_samples.csv"))
    assert len(qc) == COHORT_SIZE_EXPECTED, (
        f"qc_flagged_samples.csv row count changed: expected "
        f"{COHORT_SIZE_EXPECTED}, got {len(qc)}."
    )

    y = pd.read_csv(os.path.join(raw_dir, "y_labels.csv"))
    assert len(y) == COHORT_SIZE_EXPECTED, (
        f"y_labels.csv row count changed: expected {COHORT_SIZE_EXPECTED}, "
        f"got {len(y)}."
    )
    assert set(core["sample_id"]).issubset(set(y["sample_id"])), (
        "core_set_with_metadata.csv contains a sample_id not present in "
        "y_labels.csv - cohort mismatch, stop and investigate."
    )
    assert set(sensitivity["sample_id"]).issubset(set(y["sample_id"])), (
        "sensitivity_set.csv contains a sample_id not present in "
        "y_labels.csv - cohort mismatch, stop and investigate."
    )

    return core, sensitivity, qc, y


def add_mutation_id(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = df.copy()
    df["mutation_id"] = [build_mutation_id(g, c) for g, c in zip(df["gene"], df["change"])]
    n_unique_pairs = df[["gene", "change"]].drop_duplicates().shape[0]
    n_unique_ids = df["mutation_id"].nunique()
    assert n_unique_pairs == n_unique_ids, (
        f"[{label}] mutation_id is not a 1:1 stand-in for (gene, change): "
        f"{n_unique_pairs} unique (gene, change) pairs but {n_unique_ids} "
        f"unique mutation_id values. The '|' separator has likely collided "
        f"with a character inside gene or change - investigate before "
        f"trusting any downstream join on mutation_id."
    )
    return df


def build_mutation_node_table(core: pd.DataFrame) -> pd.DataFrame:
    n_cohort_with_core_mutation = core["sample_id"].nunique()
    rows = []
    for (mutation_id, gene, change), grp in core.groupby(["mutation_id", "gene", "change"]):
        drugs = sorted(grp["drug"].unique())
        n_samples = grp["sample_id"].nunique()
        rows.append({
            "mutation_id": mutation_id,
            "gene": gene,
            "change": change,
            "drugs": ";".join(drugs),
            "n_drugs": len(drugs),
            "n_samples": n_samples,
            "pct_of_cohort": round(100 * n_samples / COHORT_SIZE_EXPECTED, 3),
        })
    node_table = pd.DataFrame(rows).sort_values("n_samples", ascending=False).reset_index(drop=True)
    print(f"  mutation_node_table: {len(node_table)} unique (gene, change) nodes")
    print(f"  ({n_cohort_with_core_mutation} of {COHORT_SIZE_EXPECTED} cohort samples carry at least one)")
    return node_table


def build_mutation_drug_frequency(df: pd.DataFrame, label: str) -> pd.DataFrame:
    freq = (
        df.groupby(["mutation_id", "gene", "change", "drug"])["sample_id"]
        .nunique()
        .reset_index(name="n_samples")
        .sort_values("n_samples", ascending=False)
        .reset_index(drop=True)
    )
    print(f"  [{label}] mutation_drug_frequency: {len(freq)} (mutation, drug) pairs")
    return freq


def run_confirmatory_tests(mutation_df, freq, qc, y, label):
    """
    Generic confirmatory Fisher's exact test runner. Used for BOTH the core
    set (background/consistency check - see the circularity note in the
    module docstring) and the sensitivity set (the real confirmatory
    analysis). mutation_df supplies carrier sets; freq supplies the
    (mutation_id, gene, change, drug) pairs to test; y supplies phenotype.
    """
    exclude_ids = set(qc.loc[qc["recommended_exclude_from_phenotype_ground_truth"], "sample_id"])
    test_pop_ids = set(y["sample_id"]) - exclude_ids
    print(f"  [{label}] Confirmatory test population: {len(test_pop_ids)} of {COHORT_SIZE_EXPECTED} "
          f"samples ({len(exclude_ids)} excluded per Phase A QC exploration)")

    y_indexed = y.set_index("sample_id")
    available_binary_cols = set(c for c in y.columns if c.endswith("_binary"))

    carriers_by_mutation = mutation_df.groupby("mutation_id")["sample_id"].apply(set).to_dict()

    results = []
    skipped = []
    for _, row in freq.iterrows():
        mutation_id, gene, change, drug = row["mutation_id"], row["gene"], row["change"], row["drug"]
        binary_col = f"{drug}_binary"

        if binary_col not in available_binary_cols:
            skipped.append((mutation_id, drug, "no_phenotype_column"))
            continue

        carriers = carriers_by_mutation[mutation_id]
        carriers_in_test = carriers & test_pop_ids
        n_excluded = len(carriers) - len(carriers_in_test)

        if len(carriers_in_test) == 0:
            skipped.append((mutation_id, drug, "insufficient_data_zero_carriers_after_qc_exclusion"))
            continue

        non_carriers_in_test = test_pop_ids - carriers

        resistant_ids = set(y_indexed.index[y_indexed[binary_col] == 1])

        a = len(carriers_in_test & resistant_ids)          # mutation + resistant
        b = len(carriers_in_test - resistant_ids)           # mutation + susceptible
        c = len(non_carriers_in_test & resistant_ids)        # no mutation + resistant
        d = len(non_carriers_in_test - resistant_ids)        # no mutation + susceptible

        if (a + c) == 0 or (b + d) == 0:
            skipped.append((mutation_id, drug, "insufficient_data_single_class_phenotype"))
            continue

        try:
            odds_ratio, pvalue = fisher_exact([[a, b], [c, d]])
        except Exception as e:
            skipped.append((mutation_id, drug, f"fisher_exact_error: {e}"))
            continue

        results.append({
            "mutation_id": mutation_id, "gene": gene, "change": change, "drug": drug,
            "a_mut_resistant": a, "b_mut_susceptible": b,
            "c_nomut_resistant": c, "d_nomut_susceptible": d,
            "n_carriers_total": len(carriers), "n_carriers_tested": len(carriers_in_test),
            "n_excluded_qc": n_excluded,
            "odds_ratio": odds_ratio, "pvalue": pvalue,
            "low_power_flag": len(carriers_in_test) < LOW_POWER_THRESHOLD,
        })

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        reject, padj, _, _ = multipletests(results_df["pvalue"], method="fdr_bh")
        results_df["padj_BH"] = padj
        results_df["significant"] = reject
        results_df = results_df.sort_values("padj_BH").reset_index(drop=True)

    print(f"  [{label}] Confirmatory tests run: {len(results_df)} valid, {len(skipped)} skipped")
    skip_reasons = pd.Series([s[2] for s in skipped]).value_counts() if skipped else pd.Series(dtype=int)
    for reason, n in skip_reasons.items():
        print(f"    skipped ({n}): {reason}")
        if "insufficient_data_zero_carriers" in reason:
            examples = [f"{m}/{d}" for m, d, r in skipped if r == reason][:5]
            print(f"      e.g. {examples}")

    if len(results_df) > 0:
        n_circular = int((results_df["b_mut_susceptible"] == 0).sum())
        print(f"  [{label}] Tests with zero 'mutation+susceptible' carriers (odds ratio = inf): "
              f"{n_circular} / {len(results_df)} "
              f"({'expected - this IS the circular background check' if label == 'core' else 'check if unexpectedly high for the sensitivity set'})")

    return results_df, skipped


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-cleaning_dir", default="../results/phaseA_cleaning",
                     help="directory containing core_set_with_metadata.csv and sensitivity_set.csv")
    ap.add_argument("-qc_dir", default="../results/phaseA_qc_exploration",
                     help="directory containing qc_flagged_samples.csv")
    ap.add_argument("-raw_dir", default="../data/raw",
                     help="directory containing y_labels.csv")
    ap.add_argument("-outdir", default="../results/objective1",
                     help="directory to write this objective's output files to")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Reading Phase A cleaning from: {os.path.abspath(args.cleaning_dir)}")
    print(f"Reading QC exploration from:   {os.path.abspath(args.qc_dir)}")
    print(f"Reading raw metadata from:     {os.path.abspath(args.raw_dir)}")
    print(f"Writing outputs to:            {os.path.abspath(args.outdir)}")
    print()

    print("Loading and validating inputs ...")
    core, sensitivity, qc, y = load_inputs(args.cleaning_dir, args.qc_dir, args.raw_dir)

    print("Assigning mutation_id (canonical for all downstream objectives) ...")
    core = add_mutation_id(core, "core")
    sensitivity = add_mutation_id(sensitivity, "sensitivity")

    print("Building mutation_node_table.csv (core set only - this is the network node list) ...")
    node_table = build_mutation_node_table(core)

    print("Building mutation_drug_frequency.csv (core set only) ...")
    freq_core = build_mutation_drug_frequency(core, "core")

    print("Building the sensitivity-set frequency table (for the real confirmatory test) ...")
    freq_sensitivity = build_mutation_drug_frequency(sensitivity, "sensitivity")

    print()
    print("Running confirmatory Fisher's exact tests ...")
    print("  - CORE SET: background/consistency check only, expected near-total circularity, see docstring -")
    fisher_core, skipped_core = run_confirmatory_tests(core, freq_core, qc, y, "core")

    print("  - SENSITIVITY SET: the real confirmatory analysis -")
    fisher_sensitivity, skipped_sensitivity = run_confirmatory_tests(sensitivity, freq_sensitivity, qc, y, "sensitivity")

    print()
    print(f"Writing outputs to {args.outdir} ...")
    node_table.to_csv(os.path.join(args.outdir, "mutation_node_table.csv"), index=False)
    core.to_csv(os.path.join(args.outdir, "core_variants_with_mutation_id.csv"), index=False)
    freq_core.to_csv(os.path.join(args.outdir, "mutation_drug_frequency.csv"), index=False)
    fisher_core.to_csv(os.path.join(args.outdir, "obj1_fisher_results_core.csv"), index=False)
    fisher_sensitivity.to_csv(os.path.join(args.outdir, "obj1_fisher_results_sensitivity.csv"), index=False)

    print()
    print("Objective 1 complete. Row counts:")
    print(f"  mutation_node_table.csv                  {len(node_table):>6}")
    print(f"  core_variants_with_mutation_id.csv        {len(core):>6}")
    print(f"  mutation_drug_frequency.csv               {len(freq_core):>6}")
    print(f"  obj1_fisher_results_core.csv              {len(fisher_core):>6}  (background check)")
    print(f"  obj1_fisher_results_sensitivity.csv       {len(fisher_sensitivity):>6}  (real confirmatory test)")
    if len(fisher_sensitivity) > 0:
        n_sig = int(fisher_sensitivity["significant"].sum())
        print(f"    of which significant (padj_BH < 0.05): {n_sig}")


if __name__ == "__main__":
    main()
