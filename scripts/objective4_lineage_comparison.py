#!/usr/bin/env python3
"""
Objective 4 - Compare the distribution and network patterns of co-occurring
drug-resistance mutations across Mycobacterium tuberculosis lineages.

WHY THIS OBJECTIVE EXISTS, IN ONE SENTENCE:
Objective 2's report found that 296 of 441 pooled significant co-occurrence
pairs (67%) have >=90% of their joint carriers concentrated in a single
lineage - the signature of shared clonal ancestry, not necessarily a real
pairwise interaction - and Objective 3's report found the same pattern
holds for 66.5% of edges within the giant component specifically. This
objective is the step that actually resolves that question for each
flagged pair, rather than leaving it as an open caveat: does the pair
remain statistically significant when the co-occurrence test is re-run
using ONLY samples from its own dominant lineage? If yes, the pair
survives a real test of independence from population structure. If no,
the pooled signal was very plausibly an artefact of clonal inheritance.

THE ELIGIBLE-LINEAGE DECISION, MADE EXPLICITLY HERE (Phase A deliberately
left this open for this objective to decide):
Of the cohort's 1,858 samples, 49 have no single clean main_lineage label
(41 "Unknown" + 8 compound values like "lineage2;lineage4") and are
excluded from every per-lineage stratum entirely - there is no sense in
which a compound-lineage or unknown-lineage sample belongs to exactly one
"within-lineage" population. Of the remaining single-lineage samples
(after also removing the 36 high-QC samples, exactly as Objective 2 does
for the pooled analysis), six single-lineage values exist in this cohort:
lineage4 (1,255), lineage2 (360), lineage3 (117), lineage1 (63), lineage7
(2), lineage5 (1). A minimum stratum size of MIN_LINEAGE_SIZE (50) is
applied before running any per-lineage analysis at all - lineage7 (2
samples) and lineage5 (1 sample) are excluded on this basis, not analysed
with a token, statistically meaningless network. This is a pragmatic
cutoff, not a formally derived one: 63 (lineage1, the smallest analysed
stratum) is comfortably above it, while 1-2 samples could not support any
meaningful pairwise test regardless of where the threshold were set.

HOW THIS SCRIPT REUSES OBJECTIVES 2 AND 3 RATHER THAN DUPLICATING THEM:
objective2_cooccurrence.py's -lineage argument and objective3_network.py's
existing -obj2_dir/-outdir arguments were built for exactly this reuse.
This script calls both, once per eligible lineage, as subprocesses -
identical code paths to running them by hand, so the pooled and per-
lineage results are guaranteed to come from the same tested logic and
cannot silently drift apart the way a second, separately-written copy of
the same logic could.

A REAL BUG FOUND AND FIXED WHILE BUILDING THIS OBJECTIVE:
objective3_network.py originally asserted cooccurrence_significant_edges.csv
has EXACTLY 441 rows - correct for the pooled cohort, but wrong by
construction for every per-lineage rerun (which produced 300 / 113 / 23 / 55
edges for lineage4 / lineage2 / lineage3 / lineage1 respectively, all
valid). This broke immediately, exactly as a real assertion should, the
first time it was tested against a real lineage run rather than assumed to
generalise. It has been replaced with a sanity range (more than 0, no more
than the 240-node maximum possible) rather than a fixed count, since this
script is now legitimately called with very different edge-list sizes.

THE LINEAGE-CONCENTRATION DIAGNOSTIC IS NOT REUSABLE INSIDE A SINGLE-
LINEAGE RUN - WHY THE RESOLUTION LOGIC BELOW DOES NOT USE IT:
Objective 2's dominant_lineage_among_both_carriers / pct_both_carriers_in_
dominant_lineage columns become close to tautological once the analysis
population is already restricted to one lineage (every pair's joint
carriers are, trivially, almost entirely that one lineage, because there
is effectively no other lineage left in the population to dilute it -
confirmed directly: lineage4's own per-lineage run still flags 269 of its
300 significant pairs as ">=90% one lineage", which no longer means what
it meant in the pooled analysis). The real resolution test implemented
below is therefore a direct comparison instead: does this specific pair
appear in its OWN dominant lineage's independently-run significant-edges
list, at all, with its own freshly-computed p-value from that smaller
population - not a reuse of the pooled run's concentration diagnostic.

WHAT THIS SCRIPT DOES, IN ORDER:
  1. Determines eligible lineages and their post-exclusion sample sizes.
  2. Runs objective2_cooccurrence.py and objective3_network.py once per
     eligible lineage (subprocess calls, real reuse, not duplicated logic).
  3. Builds a cross-lineage topology comparison table (nodes, edges,
     density, components, modularity per lineage).
  4. Resolves every one of the pooled Objective 2 run's 441 significant
     pairs against its own dominant lineage's independent rerun, and
     records whether it was confirmed, not significant, or untestable
     there (node absent from that lineage, or dominant lineage not
     analysed at all).

Inputs expected:
    -obj1_dir/mutation_node_table.csv, core_variants_with_mutation_id.csv
    -obj2_pooled_dir/cooccurrence_significant_edges.csv   the pooled Objective 2 run
    -qc_dir/qc_flagged_samples.csv
    -raw_dir/y_labels.csv
    -lineage_flags_dir/lineage_stratification_flags.csv
    -scripts_dir      directory containing objective2_cooccurrence.py and
                       objective3_network.py (defaults to this script's own directory)

Outputs written to -outdir:
    results/objective4/<lineage>/objective2/...   full Objective 2 rerun, per lineage
    results/objective4/<lineage>/objective3/...   full Objective 3 rerun, per lineage
    cross_lineage_comparison.csv    one row per analysed lineage, topology side by side
    lineage_confound_resolution.csv one row per pooled significant pair, resolved
    objective4_summary.csv          headline counts
"""
import argparse
import os
import subprocess
import sys

import pandas as pd

MIN_LINEAGE_SIZE = 50
COHORT_SIZE_EXPECTED = 1858


def determine_eligible_lineages(y, qc, lineage_flags):
    high_qc_ids = set(qc.loc[qc["high_variant_qc"], "sample_id"])
    non_single_ids = set(lineage_flags.loc[~lineage_flags["stratifiable_single_lineage"], "sample_id"])
    usable = y[~y["sample_id"].isin(high_qc_ids) & ~y["sample_id"].isin(non_single_ids)]

    counts = usable["main_lineage"].value_counts()
    print("Per-lineage sample counts after excluding high-QC and non-single-lineage samples:")
    for lineage, n in counts.items():
        status = "ELIGIBLE" if n >= MIN_LINEAGE_SIZE else f"excluded (< {MIN_LINEAGE_SIZE})"
        print(f"  {lineage:12s} {n:>5} samples  [{status}]")

    eligible = counts[counts >= MIN_LINEAGE_SIZE].index.tolist()
    return eligible, counts


def run_lineage_stratum(lineage, scripts_dir, obj1_dir, qc_dir, raw_dir, outdir):
    lineage_outdir = os.path.join(outdir, lineage)
    obj2_dir = os.path.join(lineage_outdir, "objective2")
    obj3_dir = os.path.join(lineage_outdir, "objective3")
    os.makedirs(obj2_dir, exist_ok=True)
    os.makedirs(obj3_dir, exist_ok=True)

    print(f"\n--- Running Objective 2 for {lineage} ---")
    subprocess.run(
        [sys.executable, os.path.join(scripts_dir, "objective2_cooccurrence.py"),
         "-obj1_dir", obj1_dir, "-qc_dir", qc_dir, "-raw_dir", raw_dir,
         "-outdir", obj2_dir, "-lineage", lineage],
        check=True,
    )

    print(f"\n--- Running Objective 3 for {lineage} ---")
    subprocess.run(
        [sys.executable, os.path.join(scripts_dir, "objective3_network.py"),
         "-obj1_dir", obj1_dir, "-obj2_dir", obj2_dir, "-outdir", obj3_dir],
        check=True,
    )

    return obj2_dir, obj3_dir


def build_cross_lineage_comparison(lineage_dirs):
    rows = []
    for lineage, (obj2_dir, obj3_dir) in lineage_dirs.items():
        summary = pd.read_csv(os.path.join(obj3_dir, "network_summary.csv")).iloc[0]
        n = int(summary["n_nodes_connected"])
        e = int(summary["n_edges"])
        max_possible = n * (n - 1) / 2 if n > 1 else 1
        rows.append({
            "lineage": lineage,
            "n_nodes_total_from_obj1": int(summary["n_nodes_total_from_obj1"]),
            "n_nodes_connected": n,
            "n_edges": e,
            "density": round(e / max_possible, 4) if max_possible > 0 else 0.0,
            "n_connected_components": int(summary["n_connected_components"]),
            "giant_component_size": int(summary["giant_component_size"]),
            "n_louvain_communities": int(summary["n_louvain_communities"]),
            "modularity": float(summary["modularity"]),
        })
    return pd.DataFrame(rows)


def resolve_pooled_pairs(pooled_sig, lineage_dirs, eligible_lineages):
    """
    For every pooled-significant pair, check whether it is confirmed within
    its own dominant lineage's independently-run stratified analysis.
    """
    # Load each eligible lineage's FULL pairs table (not just significant) and
    # usable-node set, so we can distinguish "tested and not significant"
    # from "could not be tested there at all" (node absent from that lineage).
    lineage_full_pairs = {}
    lineage_usable_nodes = {}
    for lineage in eligible_lineages:
        obj2_dir, _ = lineage_dirs[lineage]
        full = pd.read_csv(os.path.join(obj2_dir, "cooccurrence_pairs.csv"))
        full = full.set_index(["mutation_id_1", "mutation_id_2"])
        lineage_full_pairs[lineage] = full
        lineage_usable_nodes[lineage] = set(full.index.get_level_values(0)) | set(full.index.get_level_values(1))

    results = []
    for _, row in pooled_sig.iterrows():
        mid1, mid2 = row["mutation_id_1"], row["mutation_id_2"]
        target_lineage = row["dominant_lineage_among_both_carriers"]

        result = {
            "mutation_id_1": mid1, "mutation_id_2": mid2,
            "gene_1": row["gene_1"], "change_1": row["change_1"],
            "gene_2": row["gene_2"], "change_2": row["change_2"],
            "class": row["class"],
            "pooled_odds_ratio": row["odds_ratio"], "pooled_padj_BH": row["padj_BH"],
            "target_lineage": target_lineage,
            "pct_both_carriers_in_target_lineage_pooled": row["pct_both_carriers_in_dominant_lineage"],
        }

        if target_lineage not in eligible_lineages:
            result["resolution"] = "target_lineage_not_analysed"
            result["lineage_odds_ratio"] = None
            result["lineage_padj_BH"] = None
            results.append(result)
            continue

        usable = lineage_usable_nodes[target_lineage]
        if mid1 not in usable or mid2 not in usable:
            result["resolution"] = "node_absent_in_lineage"
            result["lineage_odds_ratio"] = None
            result["lineage_padj_BH"] = None
            results.append(result)
            continue

        full = lineage_full_pairs[target_lineage]
        key = (mid1, mid2) if (mid1, mid2) in full.index else (mid2, mid1)
        if key not in full.index:
            result["resolution"] = "pair_not_tested_in_lineage"
            result["lineage_odds_ratio"] = None
            result["lineage_padj_BH"] = None
            results.append(result)
            continue

        lineage_row = full.loc[key]
        result["lineage_odds_ratio"] = lineage_row["odds_ratio"]
        result["lineage_padj_BH"] = lineage_row["padj_BH"]
        result["resolution"] = "confirmed_within_lineage" if lineage_row["significant"] else "not_significant_within_lineage"
        results.append(result)

    return pd.DataFrame(results)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-obj1_dir", default="../results/objective1")
    ap.add_argument("-obj2_pooled_dir", default="../results/objective2")
    ap.add_argument("-qc_dir", default="../results/phaseA_qc_exploration")
    ap.add_argument("-raw_dir", default="../data/raw")
    ap.add_argument("-cleaning_dir", default="../results/phaseA_cleaning")
    ap.add_argument("-scripts_dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("-outdir", default="../results/objective4")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Reading Objective 1 output from: {os.path.abspath(args.obj1_dir)}")
    print(f"Reading pooled Objective 2 from:  {os.path.abspath(args.obj2_pooled_dir)}")
    print(f"Writing outputs to:               {os.path.abspath(args.outdir)}")
    print()

    y = pd.read_csv(os.path.join(args.raw_dir, "y_labels.csv"))
    assert len(y) == COHORT_SIZE_EXPECTED
    qc = pd.read_csv(os.path.join(args.qc_dir, "qc_flagged_samples.csv"))
    lineage_flags = pd.read_csv(os.path.join(args.cleaning_dir, "lineage_stratification_flags.csv"))

    print("Determining eligible lineages ...")
    eligible_lineages, counts = determine_eligible_lineages(y, qc, lineage_flags)
    print(f"\nEligible lineages for per-lineage analysis: {eligible_lineages}")

    lineage_dirs = {}
    for lineage in eligible_lineages:
        obj2_dir, obj3_dir = run_lineage_stratum(
            lineage, args.scripts_dir, args.obj1_dir, args.qc_dir, args.raw_dir, args.outdir
        )
        lineage_dirs[lineage] = (obj2_dir, obj3_dir)

    print("\nBuilding cross-lineage topology comparison ...")
    comparison_df = build_cross_lineage_comparison(lineage_dirs)
    print(comparison_df.to_string(index=False))

    print("\nResolving pooled significant pairs against their own lineage's rerun ...")
    pooled_sig = pd.read_csv(os.path.join(args.obj2_pooled_dir, "cooccurrence_significant_edges.csv"))
    resolution_df = resolve_pooled_pairs(pooled_sig, lineage_dirs, eligible_lineages)

    resolution_counts = resolution_df["resolution"].value_counts()
    print("\nResolution outcome counts (of", len(resolution_df), "pooled significant pairs):")
    print(resolution_counts.to_string())

    summary_df = pd.DataFrame([{
        "n_lineages_analysed": len(eligible_lineages),
        "lineages_analysed": ";".join(eligible_lineages),
        "n_pooled_significant_pairs": len(pooled_sig),
        **{f"resolution_{k}": v for k, v in resolution_counts.to_dict().items()},
    }])

    comparison_df.to_csv(os.path.join(args.outdir, "cross_lineage_comparison.csv"), index=False)
    resolution_df.to_csv(os.path.join(args.outdir, "lineage_confound_resolution.csv"), index=False)
    summary_df.to_csv(os.path.join(args.outdir, "objective4_summary.csv"), index=False)

    print(f"\nObjective 4 complete. Outputs written to {args.outdir}:")
    print(f"  cross_lineage_comparison.csv       {len(comparison_df):>4} rows")
    print(f"  lineage_confound_resolution.csv    {len(resolution_df):>4} rows")
    print(f"  objective4_summary.csv                1 row")


if __name__ == "__main__":
    main()
