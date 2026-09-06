# Comparative Genomic and Network Analysis of Co-occurring Drug-Resistance Mutations Across African *Mycobacterium tuberculosis* Lineages

This repository identifies drug-resistance mutations in a cohort of 1,858 African *M. tuberculosis* isolates, tests pairs of mutations for statistically significant co-occurrence, constructs and characterises the resulting co-occurrence network, and compares that network's structure across four phylogenetic lineages to determine which findings are robust to population structure and which are not.

Supervised by Hussaini Yohanna. Project head: Prof. A. I. Akyla. Global Health and Infectious Disease Control Institute, Nasarawa State University, Keffi.

---

## 1. Directory Structure

```
coocc_network_pipeline/
├── data/
│   └── raw/                              untouched input data - never edit these files
│       ├── dr_variants.csv               raw per-variant calls, all confidence tiers
│       ├── y_labels.csv                  per-sample metadata (lineage, MDR status, per-drug phenotype)
│       └── sample_ids.txt                full list of the 1,858 sample IDs
├── scripts/                              deterministic pipeline scripts (argparse -indir/-outdir)
│   ├── phaseA_cleaning.py
│   ├── objective1_mutation_id.py
│   ├── objective2_cooccurrence.py
│   ├── objective3_network.py
│   └── objective4_lineage_comparison.py
├── notebooks/                            exploratory / visual analysis (not part of the deterministic chain)
│   ├── phaseA2_qc_exploration.ipynb
│   ├── objective3_visualization.ipynb
│   └── objective3_community_lineage_check.ipynb
├── reports/                              completion reports and the literature review (.docx)
│   ├── Literature_Review_Theoretical_Framework.docx
│   ├── PhaseA_Cleaning_QC_Report.docx
│   ├── Objective1_Mutation_ID_Report.docx
│   ├── PhaseC-Objective2_Report.docx
│   ├── PhaseD-Objective3_Report.docx
│   ├── PhaseE-Objective4_Report.docx
│   └── Final_Project_Synthesis.docx      <- start here for the overall narrative
└── results/                              every script/notebook writes here; nothing here is hand-edited
    ├── phaseA_cleaning/
    ├── phaseA_qc_exploration/
    ├── objective1/
    ├── objective2/
    ├── objective3/
    ├── objective3_visualization/
    ├── objective3_community_check/
    └── objective4/
        ├── lineage4/objective2/ , lineage4/objective3/
        ├── lineage2/objective2/ , lineage2/objective3/
        ├── lineage3/objective2/ , lineage3/objective3/
        └── lineage1/objective2/ , lineage1/objective3/
```

**Convention:** every script defaults to reading from `../results/<previous-stage>/` and writing to `../results/<this-stage>/`, assuming it is run from inside `scripts/`. Every script also accepts explicit `-indir`/`-outdir`-style arguments (see each stage below) so it can be pointed anywhere if the default layout is not being used.

---

## 2. Environment Setup

```bash
mamba create -n coocc_network python=3.12
mamba activate coocc_network
pip install pandas numpy scipy statsmodels networkx matplotlib adjustText jupyter nbclient nbformat ipykernel
```

Versions this pipeline was built and verified against: `networkx==3.6.1`, `matplotlib==3.10.8`, `adjustText==1.4.0`. Later versions of `matplotlib`/`adjustText` should work - the one place a version difference caused a real (cosmetic, not scientific) problem is documented in Section 6 below.

Report generation (the `.docx` files in `reports/`) was done separately from this analysis environment and is not something you need to reproduce the pipeline itself.

---

## 3. Running the Pipeline, Start to Finish

All commands below assume your working directory is `scripts/`. Run the stages in this order - each one reads the previous stage's output.

### Phase A - Data Cleaning and Quality Control

```bash
python3 phaseA_cleaning.py
```
Reads `data/raw/`, writes 6 files to `results/phaseA_cleaning/` (`core_set.csv`, `sensitivity_set.csv`, `excluded_set.csv`, `unlabeled_confidence_set.csv`, `core_set_with_metadata.csv`, `lineage_stratification_flags.csv`).

Then open and run `notebooks/phaseA2_qc_exploration.ipynb` (Jupyter). Writes `results/phaseA_qc_exploration/qc_flagged_samples.csv` and a QC distribution figure. This step identifies the 53 samples whose phenotype cannot be trusted as ground truth (36 sequencing-quality outliers, 42 phenotype-inconsistent, 25 overlapping).

Full detail: `reports/PhaseA_Cleaning_QC_Report.docx`.

### Objective 1 - Mutation Identification

```bash
python3 objective1_mutation_id.py
```
Reads Phase A's output, writes 5 files to `results/objective1/`: `mutation_node_table.csv` (240 confirmed mutations - the canonical node list every later stage reads from), `core_variants_with_mutation_id.csv`, `mutation_drug_frequency.csv`, `obj1_fisher_results_core.csv` (a circularity background-check, not a finding - see the report), and `obj1_fisher_results_sensitivity.csv` (the real confirmatory test, 26 significant candidates).

Full detail: `reports/Objective1_Mutation_ID_Report.docx`.

### Objective 2 - Pairwise Co-occurrence Testing

```bash
python3 objective2_cooccurrence.py
```
Reads Objective 1's output, writes 3 files to `results/objective2/`: `cooccurrence_pairs.csv` (all 28,441 pairs tested), `cooccurrence_significant_edges.csv` (441 significant pairs - this is Objective 3's edge list), `cooccurrence_summary.csv`.

The `-lineage <value>` argument restricts the analysis population to one phylogenetic lineage instead of the pooled cohort - this is how Objective 4 reruns this exact script per lineage rather than duplicating its logic (see Objective 4 below). Leave it unset for the pooled, cohort-wide run.

Full detail: `reports/PhaseC-Objective2_Report.docx`.

### Objective 3 - Network Construction and Visualization

```bash
python3 objective3_network.py
```
Reads Objective 2's output, writes 5 files to `results/objective3/`: `network_nodes.csv`, `network_edges.csv`, `isolated_nodes.csv`, `network_summary.csv`, and `network.graphml` (self-contained, usable directly in Gephi).

Then open and run `notebooks/objective3_visualization.ipynb`. Writes 4 figures to `results/objective3_visualization/` (the giant-component network, the lineage-confound overlay, the satellite components, and centrality distributions).

Optionally, run `notebooks/objective3_community_lineage_check.ipynb` - a follow-up check on whether the network's largest Louvain community corresponds to a single lineage (see Section 5 below). Writes to `results/objective3_community_check/`.

Full detail: `reports/PhaseD-Objective3_Report.docx`.

### Objective 4 - Cross-Lineage Comparison

```bash
python3 objective4_lineage_comparison.py
```
This is an orchestrator: it determines which lineages have ≥50 samples (after excluding high-QC and non-single-lineage samples), then calls `objective2_cooccurrence.py -lineage <X>` and `objective3_network.py` once per qualifying lineage as subprocesses, then resolves every one of Objective 2's 441 pooled significant pairs against its own dominant lineage's independent result.

Writes `results/objective4/cross_lineage_comparison.csv`, `results/objective4/lineage_confound_resolution.csv`, `results/objective4/objective4_summary.csv`, plus a full `objective2/` and `objective3/` output pair under `results/objective4/<lineage>/` for each of the four analysed lineages (lineage4, lineage2, lineage3, lineage1).

Full detail: `reports/PhaseE-Objective4_Report.docx`.

**Runtime:** the full pipeline (Phase A through Objective 4, including all four lineage reruns) completes in well under a minute on a standard laptop - none of these steps are computationally heavy.

---

## 4. Key Outputs at a Glance

| File | Where | What It Is |
|---|---|---|
| `mutation_node_table.csv` | `results/objective1/` | The 240 confirmed resistance mutations - the canonical node list for everything downstream |
| `cooccurrence_significant_edges.csv` | `results/objective2/` | The 441 significant co-occurrence pairs - the network's edge list |
| `network.graphml` | `results/objective3/` | The pooled network, self-contained, for Gephi or further analysis |
| `lineage_confound_resolution.csv` | `results/objective4/` | The 441 pairs resolved against their own lineage - the project's central result |

---

## 5. Key Findings (Summary - See `Final_Project_Synthesis.docx` for the Full Narrative)

- **240 confirmed resistance mutations** across 21 genes; the five most frequent all match established literature hotspots (`katG p.Ser315Thr`, `rpoB p.Ser450Leu`, `embB p.Met306Val`, etc.), an independent sanity check that the pipeline behaves correctly.
- **A circularity was found and corrected in Objective 1**: TB-Profiler's phenotype columns for catalogue-confirmed mutations are derived from the same catalogue rule being tested, making a naive confirmatory test tautological. Confirmatory testing was redirected to the WHO catalogue's "Uncertain significance" tier instead, yielding 26 statistically significant candidates for reclassification.
- **441 significant co-occurrence pairs** form a 130-node network (105-node giant component, 19 Louvain communities, modularity 0.4478).
- **A population-structure confound was found and then resolved**: two-thirds of pooled significant pairs are concentrated in a single lineage. Retesting each pair within its own dominant lineage (Objective 4) showed 332 (75.3%) remain significant, and of the 51 that do not, systematic classification shows only 6 (1.4% of all 441) show clear evidence of being a genuine artefact - most of the rest are simply underpowered in a smaller lineage, not falsified.
- **The single best-supported individual finding**: `rpoB p.Asp435Gly` + `p.Leu452Pro` - strengthens, rather than weakens, when retested within lineage4 alone.
- **A follow-up check** (`objective3_community_lineage_check.ipynb`) found that the network's largest community (31 nodes) shows the same pattern at a different scale: its loosely-connected periphery spans multiple lineages, but the samples actually driving its dense internal structure are ~90-100% lineage4, including an exact 4-mutation combination recurring identically across 8 unrelated samples - consistent with a clonally-spread, multidrug-resistant lineage4 sub-population.

---

## 6. Reproducibility Notes

- Every stage of this pipeline has been independently run and verified on two separate machines, and the underlying data and statistical results were confirmed identical in every case.
- **One cross-machine difference was found and is not a bug**: the exact pixel layout of the force-directed network figures (`objective3_visualization.ipynb`) can differ slightly between machines due to `networkx`/`matplotlib`/`scipy` version differences affecting `spring_layout`'s convergence path. This affects only where nodes visually sit on the page, never the underlying data, edges, or statistics.
- **One real rendering bug was found and fixed**: an earlier version of the visualization notebook used `adjustText`'s built-in connector-arrow drawing, which hit a version-sensitive fallback path on at least one real environment and drew a line through a label's text. The fix draws connector lines manually instead, which cannot hit that fallback. If you are working from a version of this notebook older than that fix, re-pull the current one.
- **`objective2_cooccurrence.py`'s `-lineage` argument and `objective3_network.py`'s edge-count check**: an early version of `objective3_network.py` asserted its input has exactly 441 edges, which is only true for the pooled run. This broke immediately and correctly the first time it was run against a lineage-restricted edge list. The current version uses a sanity range instead, since it is legitimately called with very different edge-list sizes.
- The minimum-lineage-size threshold (50 samples) used in Objective 4 is a pragmatic cutoff, not a formally derived one - see `PhaseE-Objective4_Report.docx` Section 2.2 for the reasoning.

---

## 7. Documentation Index

| Document | Covers |
|---|---|
| `Literature_Review_Theoretical_Framework.docx` | Background theory and literature underpinning the whole project |
| `PhaseA_Cleaning_QC_Report.docx` | Data cleaning, confidence-tier splitting, the QC investigation |
| `Objective1_Mutation_ID_Report.docx` | Mutation identification, the circularity finding, confirmatory testing |
| `PhaseC-Objective2_Report.docx` | Pairwise co-occurrence testing, the population-structure finding |
| `PhaseD-Objective3_Report.docx` | Network construction, topology, community detection, visualization |
| `PhaseE-Objective4_Report.docx` | Cross-lineage stratification, the confound resolution, the 51-pair breakdown |
| `Final_Project_Synthesis.docx` | The complete project told once, in order, as a single scientific narrative - **read this first** if you only read one document |

---

## 8. Authors and Acknowledgments

Analysis pipeline built and verified by Sukhjot Singh. Supervised by Hussaini Yohanna and Prof. A. I. Akyla, Global Health and Infectious Disease Control Institute, Nasarawa State University, Keffi.
