#!/usr/bin/env python3
"""
Objective 2 - Determine patterns of co-occurrence among drug-resistance-associated mutations in the study isolates.

FORWARD-COMPATIBILITY NOTE:
This script reads mutation_id and the node list directly from Objective 1's output (mutation_node_table.csv, core_variants_with_mutation_id.csv) 
rather than recomputing anything. Objective 3 will read this script's cooccurrence_significant_edges.csv directly as its network edge list - do

*** OBJECTIVE 4 REUSE: THE -lineage ARGUMENT ***
Objective 4 calls this exact script once per lineage stratum via the -lineage argument, rather than duplicating this file's logic into a
second, separately-maintained copy. Passing -lineage lineage4 restricts the analysis population to samples whose main_lineage is EXACTLY that
string (see build_analysis_population) before anything else runs; every downstream step (usable-node filtering, pairwise testing, BH-FDR
correction, the lineage-concentration diagnostic) then operates on that restricted population using identical code to the pooled run. This
guarantees the pooled and per-lineage results are produced by the same tested logic and cannot silently drift apart the way two independently
written copies could. not change that file's column names without checking Objective 3's script.

*** A DIFFERENT QC EXCLUSION THAN OBJECTIVE 1 - READ BEFORE REUSING EITHER SCRIPT AS A TEMPLATE FOR THE OTHER ***
Objective 1 excluded 53 samples (qc_flagged_samples.csv's recommended_exclude_from_phenotype_ground_truth) from its confirmatory
test, because that test compared genotype against phenotype, and 42 of those 53 samples specifically have an untrustworthy PHENOTYPE label (their
drtype rests on Uncertain-significance evidence, not a confirmed mutation). This script tests genotype against genotype - mutation vs mutation, not
mutation vs phenotype - so a sample's phenotype trustworthiness is irrelevant here. Only the 36 samples flagged high_variant_qc (implausible
total_variants_qc, i.e. likely contaminated/misassembled sequencing) are excluded, because THAT flag calls the sample's variant CALLS themselves
into question, genotype included. The other 17 of the 53 (normal QC, merely phenotype-inconsistent) are kept in the analysis population here -
excluding them would have no statistical justification for a genotype-only test and would needlessly shrink the population. Using the Objective 1
exclusion set here, or vice versa, would be a real methodological error, not a stylistic difference - re-derive the correct exclusion for whichever
kind of test is being run, do not copy one number across scripts blindly.

CONSEQUENCE OF THIS EXCLUSION, CONFIRMED AGAINST THE REAL DATA: dropping the 36 high-QC samples leaves the analysis population at 1,822 of 1,858
samples, and reduces rpoB|p.Ser431Thr's carrier count from 1 to 0 (its only carrier was one of the 36) - this node is therefore dropped entirely
from pairwise testing (239 usable nodes, not 240; it remains untouched in Objective 1's own mutation_node_table.csv, which is cohort-wide and not
QC-filtered). Several other mutations' carrier counts shift down by 1-3 samples for the same reason - this script recomputes every mutation's
carrier count freshly WITHIN the 1,822-sample analysis population rather than reusing Objective 1's cohort-wide n_samples column, specifically so
the a/b/c/d counts in every contingency table stay internally consistent.

WHAT "WITHIN-DRUG-CLASS" AND "CROSS-DRUG-CLASS" MEAN HERE:
A pair of mutation nodes is classified using their drugs sets from mutation_node_table.csv (semicolon-joined, e.g. "rifampicin;rifapentine").
If the two nodes' drug sets share at least one drug in common, the pair is "within_class" (e.g. two different isoniazid-associated mutations). If the
two drug sets are completely disjoint, the pair is "cross_class" (e.g. a rifampicin-associated mutation and an isoniazid-associated mutation). This
directly implements the distinction the project plan calls for: within- class co-occurrence is the expected, lower-priority category; cross-class
is the analytically interesting one - though see the module-level docstring's note on the MDR-definition confound (Section 4.3 of the
literature review) before treating any single cross-class hit as conclusive: because MDR/pre-XDR/XDR-TB is clinically DEFINED as multi-drug
resistant, a general tendency for rifampicin- and isoniazid-associated mutations to co-occur is partly expected by clinical definition, not solely
by biological interaction. The genuinely interesting signal is which SPECIFIC mutation pairs stand out beyond that general background rate, not
the mere existence of cross-class significant pairs.

A SECOND CONFOUND, FOUND DIRECTLY IN THIS COHORT'S REAL RESULTS, IS FLAGGED THE SAME WAY: population structure (Section 6.4 of the literature review).
The single strongest within-class pair in this cohort's real results, rpoB|p.Asp435Gly + rpoB|p.Leu452Pro (odds ratio ~2720, 191 joint carriers),
was checked directly against sample lineage and found to be carried by lineage4 samples ONLY - 100% concentration, against a ~64% lineage4
baseline in the core-mutation-carrying population. That is exactly the signature a shared, clonally-inherited genetic background produces, not
necessarily a real biological interaction between the two mutations. This script therefore computes, for every pair, what fraction of its joint
carriers fall into a single dominant lineage (dominant_lineage_among_both_ carriers, pct_both_carriers_in_dominant_lineage) as a cheap early-warning
diagnostic - NOT a substitute for Objective 4's proper per-lineage stratified re-analysis, which is the only way to actually resolve whether
a flagged pair holds up within a single lineage or disappears entirely.

WHAT THIS SCRIPT DOES, IN ORDER:
  1. Loads Objective 1's node table and mutation-tagged core variants.
  2. Builds the QC-filtered analysis population (1,822 samples) and recomputes each mutation's carrier set within it, dropping any
     mutation left with zero carriers.
  3. Tests every unique pair of the remaining 239 nodes with Fisher's exact test (2x2: both / mut1-only / mut2-only / neither), one
     combined Benjamini-Hochberg correction across all pairs together (unlike Objective 1, within- and cross-class pairs are NOT separate
     test families - they are the same kind of test, categorised only for reporting purposes after the fact).
  4. Splits the results by within/cross-drug-class for reporting, and writes a separate, ready-to-use significant-edges file for Objective
     3's network construction.

Inputs expected:
    -obj1_dir/mutation_node_table.csv              Objective 1 output
    -obj1_dir/core_variants_with_mutation_id.csv    Objective 1 output
    -qc_dir/qc_flagged_samples.csv                  Phase A QC exploration output
    -raw_dir/y_labels.csv                            raw cohort metadata (population size only)

Outputs written to -outdir:
    cooccurrence_pairs.csv              every valid pair tested, BH-FDR corrected
    cooccurrence_significant_edges.csv  significant pairs only - Objective 3's edge list
    cooccurrence_summary.csv            within/cross-class significant counts
"""
import argparse
import itertools
import os

import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

NODE_COUNT_EXPECTED = 240
CORE_ROW_COUNT_EXPECTED = 10199
COHORT_SIZE_EXPECTED = 1858
HIGH_QC_COUNT_EXPECTED = 36
LOW_POWER_THRESHOLD = 5


def load_inputs(obj1_dir, qc_dir, raw_dir):
    node = pd.read_csv(os.path.join(obj1_dir, "mutation_node_table.csv"))
    assert len(node) == NODE_COUNT_EXPECTED, (
        f"mutation_node_table.csv row count changed: expected "
        f"{NODE_COUNT_EXPECTED}, got {len(node)}. Stop - re-verify "
        f"Objective 1's output before trusting anything downstream."
    )

    core_id = pd.read_csv(os.path.join(obj1_dir, "core_variants_with_mutation_id.csv"))
    assert len(core_id) == CORE_ROW_COUNT_EXPECTED, (
        f"core_variants_with_mutation_id.csv row count changed: expected "
        f"{CORE_ROW_COUNT_EXPECTED}, got {len(core_id)}."
    )
    assert "mutation_id" in core_id.columns, (
        "core_variants_with_mutation_id.csv has no mutation_id column - "
        "was this accidentally the pre-Objective-1 core_set_with_metadata.csv?"
    )

    qc = pd.read_csv(os.path.join(qc_dir, "qc_flagged_samples.csv"))
    assert len(qc) == COHORT_SIZE_EXPECTED, (
        f"qc_flagged_samples.csv row count changed: expected "
        f"{COHORT_SIZE_EXPECTED}, got {len(qc)}."
    )
    n_high_qc = int(qc["high_variant_qc"].sum())
    assert n_high_qc == HIGH_QC_COUNT_EXPECTED, (
        f"Number of high_variant_qc samples changed: expected "
        f"{HIGH_QC_COUNT_EXPECTED}, got {n_high_qc}. This changes the "
        f"analysis population size - re-verify before proceeding."
    )

    y = pd.read_csv(os.path.join(raw_dir, "y_labels.csv"))
    assert len(y) == COHORT_SIZE_EXPECTED, (
        f"y_labels.csv row count changed: expected {COHORT_SIZE_EXPECTED}, "
        f"got {len(y)}."
    )

    return node, core_id, qc, y


def build_analysis_population(core_id, qc, y, lineage=None):
    high_qc_ids = set(qc.loc[qc["high_variant_qc"], "sample_id"])
    pop_ids = set(y["sample_id"]) - high_qc_ids

    if lineage is not None:
        # Objective 4 reuse: restrict to samples whose main_lineage is EXACTLY this value (an exact string match against y_labels.csv's main_lineage
        # column) - this naturally excludes compound-lineage samples like "lineage2;lineage4" from every single-lineage stratum, since neither
        # "lineage2" nor "lineage4" equals that compound string. Combined with the high-QC exclusion above, this reproduces the same two exclusion
        # categories Phase A's lineage_stratification_flags.csv already identified, without needing to re-derive that logic here.
        lineage_ids = set(y.loc[y["main_lineage"] == lineage, "sample_id"])
        pop_ids = pop_ids & lineage_ids
        print(f"  Lineage filter active: '{lineage}' - population restricted to "
              f"{len(pop_ids)} samples (from {len(lineage_ids)} total '{lineage}' "
              f"samples, minus high-QC exclusions)")
    else:
        print(f"  Analysis population: {len(pop_ids)} of {COHORT_SIZE_EXPECTED} samples "
              f"({len(high_qc_ids)} high-QC samples excluded - genotype calls only, "
              f"NOT the same 53-sample exclusion Objective 1 used)")

    pop_core = core_id[core_id["sample_id"].isin(pop_ids)]
    carriers_by_mutation = pop_core.groupby("mutation_id")["sample_id"].apply(set).to_dict()
    return pop_ids, carriers_by_mutation


def drug_sets_from_node_table(node: pd.DataFrame) -> dict:
    return {
        row["mutation_id"]: set(row["drugs"].split(";"))
        for _, row in node.iterrows()
    }


def run_pairwise_tests(usable_mutation_ids, carriers_by_mutation, drug_sets, pop_size, sample_lineage):
    pairs = list(itertools.combinations(sorted(usable_mutation_ids), 2))
    print(f"  Testing {len(pairs)} unique pairs across {len(usable_mutation_ids)} usable nodes ...")

    results = []
    for mid1, mid2 in pairs:
        c1 = carriers_by_mutation[mid1]
        c2 = carriers_by_mutation[mid2]

        a = len(c1 & c2)
        b = len(c1 - c2)
        c = len(c2 - c1)
        d = pop_size - a - b - c

        try:
            odds_ratio, pvalue = fisher_exact([[a, b], [c, d]])
        except Exception as e:
            print(f"    fisher_exact_error for {mid1} / {mid2}: {e}")
            continue

        drugs1, drugs2 = drug_sets[mid1], drug_sets[mid2]
        pair_class = "within_class" if (drugs1 & drugs2) else "cross_class"

        # Early-warning lineage-concentration diagnostic: a strong co-occurrence signal driven almost entirely by one lineage is a candidate 
        # population-structure confound (a lineage marker masquerading as co-occurrence), exactly the pattern Objective 4's stratified 
        # re-analysis is designed to test properly. This is a cheap flag here, not a substitute for that re-analysis - it exists so a pair 
        # like this is never mistaken for a settled finding before Objective 4 has actually checked it.
        both_carriers = c1 & c2
        if len(both_carriers) > 0:
            lineages_of_both = pd.Series([sample_lineage.get(s, "Unknown") for s in both_carriers])
            dominant_lineage = lineages_of_both.value_counts().idxmax()
            dominant_lineage_pct = round(100 * lineages_of_both.value_counts().max() / len(both_carriers), 1)
        else:
            dominant_lineage, dominant_lineage_pct = None, None

        n1, n2 = len(c1), len(c2)
        results.append({
            "mutation_id_1": mid1, "mutation_id_2": mid2,
            "gene_1": mid1.split("|")[0], "change_1": mid1.split("|", 1)[1],
            "gene_2": mid2.split("|")[0], "change_2": mid2.split("|", 1)[1],
            "drugs_1": ";".join(sorted(drugs1)), "drugs_2": ";".join(sorted(drugs2)),
            "class": pair_class,
            "n1_in_population": n1, "n2_in_population": n2,
            "a_both": a, "b_mut1_only": b, "c_mut2_only": c, "d_neither": d,
            "odds_ratio": odds_ratio, "pvalue": pvalue,
            "low_power_flag": min(n1, n2) < LOW_POWER_THRESHOLD,
            "dominant_lineage_among_both_carriers": dominant_lineage,
            "pct_both_carriers_in_dominant_lineage": dominant_lineage_pct,
        })

    results_df = pd.DataFrame(results)
    reject, padj, _, _ = multipletests(results_df["pvalue"], method="fdr_bh")
    results_df["padj_BH"] = padj
    results_df["significant"] = reject
    results_df = results_df.sort_values("padj_BH").reset_index(drop=True)
    return results_df


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-obj1_dir", default="../results/objective1",
                     help="directory containing mutation_node_table.csv and "
                          "core_variants_with_mutation_id.csv")
    ap.add_argument("-qc_dir", default="../results/phaseA_qc_exploration",
                     help="directory containing qc_flagged_samples.csv")
    ap.add_argument("-raw_dir", default="../data/raw",
                     help="directory containing y_labels.csv")
    ap.add_argument("-outdir", default="../results/objective2",
                     help="directory to write this objective's output files to")
    ap.add_argument("-lineage", default=None,
                     help="Objective 4 reuse only: restrict the analysis population to "
                          "samples whose main_lineage exactly equals this value (e.g. "
                          "'lineage4'). Compound values ('lineage2;lineage4') and "
                          "'Unknown' are never matched by this filter, by construction. "
                          "Omit for the pooled, cohort-wide analysis (default).")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Reading Objective 1 output from: {os.path.abspath(args.obj1_dir)}")
    print(f"Reading QC exploration from:     {os.path.abspath(args.qc_dir)}")
    print(f"Reading raw metadata from:       {os.path.abspath(args.raw_dir)}")
    print(f"Writing outputs to:              {os.path.abspath(args.outdir)}")
    if args.lineage:
        print(f"Lineage stratum:                 {args.lineage}")
    print()

    print("Loading and validating inputs ...")
    node, core_id, qc, y = load_inputs(args.obj1_dir, args.qc_dir, args.raw_dir)

    print("Building QC-filtered analysis population and recomputing carrier sets ...")
    pop_ids, carriers_by_mutation = build_analysis_population(core_id, qc, y, lineage=args.lineage)

    all_node_ids = set(node["mutation_id"])
    usable_ids = set(carriers_by_mutation.keys())
    dropped_ids = all_node_ids - usable_ids
    print(f"  Nodes with >=1 carrier remaining in the analysis population: {len(usable_ids)} of {len(all_node_ids)}")
    if dropped_ids:
        print(f"  Dropped entirely from pairwise testing (zero carriers after QC exclusion): {sorted(dropped_ids)}")

    drug_sets = drug_sets_from_node_table(node)

    print("Building sample -> lineage lookup (for the lineage-concentration diagnostic) ...")
    sample_lineage = core_id.drop_duplicates("sample_id").set_index("sample_id")["main_lineage"].to_dict()

    print("Running pairwise Fisher's exact tests (one combined BH-FDR correction) ...")
    results_df = run_pairwise_tests(usable_ids, carriers_by_mutation, drug_sets, len(pop_ids), sample_lineage)

    n_sig = int(results_df["significant"].sum())
    sig = results_df[results_df["significant"]]
    n_within_sig = int((sig["class"] == "within_class").sum())
    n_cross_sig = int((sig["class"] == "cross_class").sum())
    n_lineage_suspect = int((sig["pct_both_carriers_in_dominant_lineage"] >= 90).sum())
    print(f"  {len(results_df)} pairs tested, {n_sig} significant "
          f"({n_within_sig} within-class, {n_cross_sig} cross-class)")
    print(f"  Of the significant pairs, {n_lineage_suspect} have >=90% of their joint carriers "
          f"concentrated in a single lineage - flagged as population-structure-confound "
          f"candidates for Objective 4 to check directly, not yet resolved here.")

    summary = pd.DataFrame([
        {"class": "within_class", "n_tested": int((results_df["class"] == "within_class").sum()),
         "n_significant": n_within_sig},
        {"class": "cross_class", "n_tested": int((results_df["class"] == "cross_class").sum()),
         "n_significant": n_cross_sig},
        {"class": "ALL (lineage-suspect subset)", "n_tested": n_sig,
         "n_significant": n_lineage_suspect},
    ])

    print(f"Writing outputs to {args.outdir} ...")
    results_df.to_csv(os.path.join(args.outdir, "cooccurrence_pairs.csv"), index=False)
    sig.to_csv(os.path.join(args.outdir, "cooccurrence_significant_edges.csv"), index=False)
    summary.to_csv(os.path.join(args.outdir, "cooccurrence_summary.csv"), index=False)

    print()
    print("Objective 2 complete. Row counts:")
    print(f"  cooccurrence_pairs.csv               {len(results_df):>6}")
    print(f"  cooccurrence_significant_edges.csv   {len(sig):>6}")
    print(f"  cooccurrence_summary.csv             {len(summary):>6}")


if __name__ == "__main__":
    main()
