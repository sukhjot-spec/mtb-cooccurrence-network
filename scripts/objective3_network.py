#!/usr/bin/env python3
"""
Objective 3 - Construct and characterise networks of co-occurring drug-
resistance mutations in African Mycobacterium tuberculosis isolates.

FORWARD-COMPATIBILITY NOTE:
This script reads mutation_id and the node list from Objective 1
(mutation_node_table.csv, unchanged) and its edge list from Objective 2
(cooccurrence_significant_edges.csv). Objective 4 will re-run this same
construction and metrics logic once per lineage stratum on that lineage's
own significant-edges file - the functions here (build_graph,
compute_node_metrics) are written to accept any edges dataframe with the
same column names for exactly that reason. Do not hardcode anything
specific to the pooled cohort inside those functions.

*** EDGE WEIGHT DESIGN DECISION - READ BEFORE USING odds_ratio DIRECTLY ***
Checked directly against the real data before choosing a weight: 136 of
441 significant edges (31%) have odds_ratio == inf (Objective 2's own
report already explains why - perfect or near-perfect separation is
common and expected for real co-occurrence signal, not an error). Most
graph algorithms - centrality measures, community detection - require
finite numeric edge weights, so odds_ratio cannot be used as the weight
directly without either dropping 31% of edges or silently producing NaN/
inf propagation through the algorithms. Instead, -log10(padj_BH) is used
as the edge weight for every algorithm in this script: it is always
finite (checked directly - zero edges have padj_BH exactly 0 in this
cohort's results), strictly positive, and monotonically increasing with
statistical significance, so stronger evidence still produces a larger
weight. The raw odds_ratio and padj_BH are both kept as edge attributes
on every edge for reporting and interpretation - only the numeric
algorithms use the transformed weight.

*** EIGENVECTOR CENTRALITY IS ONLY MEANINGFUL WITHIN THE GIANT COMPONENT -
    CONFIRMED DIRECTLY, NOT ASSUMED ***
The pooled cohort's significant-edges graph has 130 nodes in 12 connected
components: one giant component of 105 nodes, and 11 small satellite
components of 2-3 nodes each. Running eigenvector centrality on the graph
as a whole was checked directly: every one of the 25 satellite-component
nodes receives a value 8 to 10 orders of magnitude smaller than the giant
component's nodes (~1e-10 to 1e-12, versus ~0.2-0.3) - this is numerical
noise from power iteration on a disconnected system, not a real, if small,
centrality ranking among those 25 nodes. This script therefore computes
eigenvector centrality ONLY within the giant component's own induced
subgraph, and records it as null/blank for every node outside that
component, rather than reporting a number that looks meaningful but isn't.
Betweenness centrality, degree, and clustering coefficient do not have
this problem (they are well-defined for disconnected graphs) and are
reported for every node without qualification.

ISOLATED NODES (ZERO SIGNIFICANT CO-OCCURRENCE) ARE KEPT, NOT DROPPED:
Only 130 of Objective 1's 240 mutation nodes appear in at least one
significant co-occurrence edge. The other 110 are not included in the
NetworkX graph object itself (a graph algorithm has nothing meaningful to
compute for a node with no edges), but are written to their own explicit
output file (isolated_nodes.csv) rather than silently disappearing between
Objective 1's node count and this objective's network size.

COMMUNITY DETECTION REPRODUCIBILITY:
Louvain community detection has a stochastic tie-breaking step. A fixed
seed (42) is passed explicitly on every run so the partition is identical
across runs and across machines - checked directly: this cohort's real
graph happens to produce an identical partition and modularity score
(0.448) even across different seeds, but the seed is still pinned
explicitly rather than relying on that stability holding for every future
re-run (e.g. Objective 4's per-lineage graphs, which are smaller and could
plausibly be less stable).

WHAT THIS SCRIPT DOES, IN ORDER:
  1. Loads Objective 1's node table and Objective 2's significant edges.
  2. Builds the NetworkX graph, identifies isolated nodes, and separates
     connected components.
  3. Computes degree, betweenness centrality, and clustering coefficient
     for every connected node; eigenvector centrality for giant-component
     nodes only.
  4. Runs Louvain community detection (weighted, fixed seed) and reports
     the resulting modularity score.
  5. Writes a node table, an edge table, an isolated-nodes table, a
     summary table, and a .graphml file for reuse in Gephi or a
     visualization notebook.

Inputs expected:
    -obj1_dir/mutation_node_table.csv                Objective 1 output
    -obj2_dir/cooccurrence_significant_edges.csv      Objective 2 output

Outputs written to -outdir:
    network_nodes.csv      one row per connected node, all topology metrics
    network_edges.csv      one row per edge, weights and attributes
    isolated_nodes.csv     the 110 nodes with zero significant co-occurrence
    network_summary.csv    headline network-level statistics
    network.graphml        the graph itself, for Gephi or further analysis
"""
import argparse
import os

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community import louvain_communities, modularity

NODE_COUNT_EXPECTED = 240
MAX_POSSIBLE_EDGES = NODE_COUNT_EXPECTED * (NODE_COUNT_EXPECTED - 1) // 2  # 240 choose 2
LOUVAIN_SEED = 42


def load_inputs(obj1_dir, obj2_dir):
    node = pd.read_csv(os.path.join(obj1_dir, "mutation_node_table.csv"))
    assert len(node) == NODE_COUNT_EXPECTED, (
        f"mutation_node_table.csv row count changed: expected "
        f"{NODE_COUNT_EXPECTED}, got {len(node)}. Stop -- re-verify "
        f"Objective 1's output before trusting anything downstream. This "
        f"file is always Objective 1's cohort-wide node table, unchanged "
        f"regardless of whether obj2_dir holds a pooled or lineage-"
        f"restricted edge list, so this assertion stays a strict equality "
        f"check even when the edges below do not."
    )

    edges = pd.read_csv(os.path.join(obj2_dir, "cooccurrence_significant_edges.csv"))
    # NOT a fixed expected count: this script is reused for both the pooled
    # cohort-wide run (441 edges) and Objective 4's per-lineage reruns (each
    # producing a different, smaller edge count from a smaller population --
    # 300 / 113 / 23 / 55 for lineage4 / lineage2 / lineage3 / lineage1
    # respectively, all confirmed valid during development). A fixed
    # expected-count assertion here was tried first and correctly broke
    # every lineage run immediately, which is exactly why it was replaced
    # with this generic sanity range instead of silently loosened.
    assert 0 < len(edges) <= MAX_POSSIBLE_EDGES, (
        f"cooccurrence_significant_edges.csv has {len(edges)} rows, outside "
        f"the sane range (0, {MAX_POSSIBLE_EDGES}]. Either no significant "
        f"pairs were found for this population (0 rows -- check whether "
        f"this stratum is simply too small) or something is structurally "
        f"wrong with the file (more rows than 240 nodes could ever produce)."
    )
    print(f"  Loaded {len(edges)} significant edges from {obj2_dir}")

    n_zero_padj = (edges["padj_BH"] <= 0).sum()
    assert n_zero_padj == 0, (
        f"{n_zero_padj} edges have padj_BH <= 0 -- this breaks the "
        f"-log10(padj_BH) edge weight (verified 0 during development). "
        f"Investigate before trusting the resulting weights."
    )

    return node, edges


def build_graph(edges: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for _, row in edges.iterrows():
        G.add_edge(
            row["mutation_id_1"], row["mutation_id_2"],
            weight=-np.log10(row["padj_BH"]),
            odds_ratio=row["odds_ratio"],
            padj_BH=row["padj_BH"],
            pair_class=row["class"],
            dominant_lineage=row.get("dominant_lineage_among_both_carriers"),
            pct_dominant_lineage=row.get("pct_both_carriers_in_dominant_lineage"),
        )
    return G


def compute_node_metrics(G: nx.Graph):
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    giant = components[0]

    comp_id_of = {}
    comp_size_of = {}
    for i, comp in enumerate(components):
        for n in comp:
            comp_id_of[n] = i
            comp_size_of[n] = len(comp)

    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G, weight="weight")
    clustering = nx.clustering(G, weight="weight")

    giant_subgraph = G.subgraph(giant)
    eigen_giant = nx.eigenvector_centrality(giant_subgraph, weight="weight", max_iter=1000)
    eigenvector = {n: eigen_giant.get(n) for n in G.nodes()}  # None for non-giant nodes

    communities = louvain_communities(G, weight="weight", seed=LOUVAIN_SEED)
    community_of = {}
    for i, comm in enumerate(communities):
        for n in comm:
            community_of[n] = i
    mod_score = modularity(G, communities, weight="weight")

    return {
        "degree": degree, "betweenness": betweenness, "clustering": clustering,
        "eigenvector": eigenvector, "comp_id": comp_id_of, "comp_size": comp_size_of,
        "community": community_of, "n_communities": len(communities),
        "modularity": mod_score, "giant_component_size": len(giant),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-obj1_dir", default="../results/objective1",
                     help="directory containing mutation_node_table.csv")
    ap.add_argument("-obj2_dir", default="../results/objective2",
                     help="directory containing cooccurrence_significant_edges.csv")
    ap.add_argument("-outdir", default="../results/objective3",
                     help="directory to write this objective's output files to")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Reading Objective 1 output from: {os.path.abspath(args.obj1_dir)}")
    print(f"Reading Objective 2 output from: {os.path.abspath(args.obj2_dir)}")
    print(f"Writing outputs to:              {os.path.abspath(args.outdir)}")
    print()

    print("Loading and validating inputs ...")
    node, edges = load_inputs(args.obj1_dir, args.obj2_dir)

    print("Building the graph (edge weight = -log10(padj_BH), see docstring) ...")
    G = build_graph(edges)

    all_ids = set(node["mutation_id"])
    connected_ids = set(G.nodes())
    isolated_ids = all_ids - connected_ids
    print(f"  {len(connected_ids)} of {len(all_ids)} nodes have >=1 significant co-occurrence "
          f"({len(isolated_ids)} isolated, written to isolated_nodes.csv)")

    print("Computing topology metrics and running Louvain community detection ...")
    metrics = compute_node_metrics(G)
    print(f"  Connected components: {len(set(metrics['comp_id'].values()))} "
          f"(giant component: {metrics['giant_component_size']} nodes)")
    print(f"  Louvain communities: {metrics['n_communities']} (modularity = {metrics['modularity']:.4f})")
    n_eigen_null = sum(1 for v in metrics["eigenvector"].values() if v is None)
    print(f"  Eigenvector centrality computed for giant component only "
          f"({len(connected_ids) - n_eigen_null} nodes; {n_eigen_null} outside it left null)")

    node_lookup = node.set_index("mutation_id")
    node_rows = []
    for mid in sorted(connected_ids):
        row = node_lookup.loc[mid]
        node_rows.append({
            "mutation_id": mid, "gene": row["gene"], "change": row["change"],
            "drugs": row["drugs"], "n_drugs": row["n_drugs"],
            "n_samples_cohort_wide": row["n_samples"],
            "degree": metrics["degree"][mid],
            "betweenness_centrality": metrics["betweenness"][mid],
            "eigenvector_centrality": metrics["eigenvector"][mid],
            "clustering_coefficient": metrics["clustering"][mid],
            "connected_component_id": metrics["comp_id"][mid],
            "connected_component_size": metrics["comp_size"][mid],
            "is_giant_component": metrics["comp_id"][mid] == 0,
            "louvain_community_id": metrics["community"][mid],
        })
    node_df = pd.DataFrame(node_rows).sort_values("degree", ascending=False).reset_index(drop=True)

    edge_rows = []
    for u, v, data in G.edges(data=True):
        edge_rows.append({
            "mutation_id_1": u, "mutation_id_2": v,
            "edge_weight_neglog10_padj": data["weight"],
            "odds_ratio": data["odds_ratio"], "padj_BH": data["padj_BH"],
            "class": data["pair_class"],
            "dominant_lineage_among_both_carriers": data["dominant_lineage"],
            "pct_both_carriers_in_dominant_lineage": data["pct_dominant_lineage"],
            "community_id_1": metrics["community"][u], "community_id_2": metrics["community"][v],
            "same_community": metrics["community"][u] == metrics["community"][v],
        })
    edge_df = pd.DataFrame(edge_rows).sort_values("edge_weight_neglog10_padj", ascending=False).reset_index(drop=True)

    isolated_df = node[node["mutation_id"].isin(isolated_ids)].copy()

    print("Attaching node attributes to the graph itself (so network.graphml is "
          "self-contained -- usable directly in Gephi or elsewhere without also "
          "needing network_nodes.csv alongside it) ...")
    for _, row in node_df.iterrows():
        mid = row["mutation_id"]
        G.nodes[mid]["gene"] = row["gene"]
        G.nodes[mid]["change"] = row["change"]
        G.nodes[mid]["drugs"] = row["drugs"]
        G.nodes[mid]["n_drugs"] = int(row["n_drugs"])
        G.nodes[mid]["n_samples_cohort_wide"] = int(row["n_samples_cohort_wide"])
        G.nodes[mid]["degree"] = int(row["degree"])
        G.nodes[mid]["betweenness_centrality"] = float(row["betweenness_centrality"])
        # graphml has no native null -- eigenvector_centrality is legitimately
        # undefined (not zero) for the 25 non-giant-component nodes (see module
        # docstring), so it is omitted from the graph attributes entirely for
        # those nodes rather than written as a misleading 0.0 or a string "None"
        # that would silently become a real float on the next read.
        if pd.notna(row["eigenvector_centrality"]):
            G.nodes[mid]["eigenvector_centrality"] = float(row["eigenvector_centrality"])
        G.nodes[mid]["clustering_coefficient"] = float(row["clustering_coefficient"])
        G.nodes[mid]["connected_component_id"] = int(row["connected_component_id"])
        G.nodes[mid]["connected_component_size"] = int(row["connected_component_size"])
        G.nodes[mid]["is_giant_component"] = bool(row["is_giant_component"])
        G.nodes[mid]["louvain_community_id"] = int(row["louvain_community_id"])

    summary_df = pd.DataFrame([{
        "n_nodes_total_from_obj1": len(all_ids),
        "n_nodes_connected": len(connected_ids),
        "n_nodes_isolated": len(isolated_ids),
        "n_edges": G.number_of_edges(),
        "n_connected_components": len(set(metrics["comp_id"].values())),
        "giant_component_size": metrics["giant_component_size"],
        "n_louvain_communities": metrics["n_communities"],
        "modularity": round(metrics["modularity"], 4),
        "louvain_seed_used": LOUVAIN_SEED,
    }])

    print(f"Writing outputs to {args.outdir} ...")
    node_df.to_csv(os.path.join(args.outdir, "network_nodes.csv"), index=False)
    edge_df.to_csv(os.path.join(args.outdir, "network_edges.csv"), index=False)
    isolated_df.to_csv(os.path.join(args.outdir, "isolated_nodes.csv"), index=False)
    summary_df.to_csv(os.path.join(args.outdir, "network_summary.csv"), index=False)
    nx.write_graphml(G, os.path.join(args.outdir, "network.graphml"))

    print()
    print("Objective 3 complete. Row counts:")
    print(f"  network_nodes.csv       {len(node_df):>6}")
    print(f"  network_edges.csv       {len(edge_df):>6}")
    print(f"  isolated_nodes.csv      {len(isolated_df):>6}")
    print(f"  network_summary.csv     {len(summary_df):>6}")
    print(f"  network.graphml         ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")


if __name__ == "__main__":
    main()
