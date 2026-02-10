"""
Test baseline layouts (neato and sfdp) on all test graphs
"""
import networkx as nx
import torch
import sys
import csv
from pathlib import Path
from tqdm import tqdm

sys.path.append('/Users/wsq/Nextcloud/CSA/CS5180RL/project/RewardHunter-')
from xing import XingLoss

# Paths
PROJECT_ROOT = Path('/Users/wsq/Nextcloud/CSA/CS5180RL/project/RewardHunter-')
DATA_DIR = PROJECT_ROOT / 'data'
TEST_GRAPH_FILE = DATA_DIR / 'test_graph.txt'
OUTPUT_CSV = PROJECT_ROOT / 'results' / 'baseline_results.csv'

# Create results directory
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

def compute_crossings(G, layout_func, prog):
    """Compute edge crossings for a given layout algorithm."""
    try:
        pos = layout_func(G, prog=prog)
        coords = torch.tensor(
            [[pos[v][0], pos[v][1]] for v in G.nodes()],
            dtype=torch.float32
        )
        xing_loss = XingLoss(G, soft=False)
        crossings = xing_loss(coords)
        return int(crossings.item())
    except Exception as e:
        print(f"Error with {prog}: {e}")
        return None

def main():
    # Read test graph list
    with open(TEST_GRAPH_FILE, 'r') as f:
        graph_files = [line.strip() for line in f if line.strip()]

    print(f"Testing {len(graph_files)} graphs...")

    results = []

    for graph_file in tqdm(graph_files, desc="Processing graphs"):
        graph_path = DATA_DIR / graph_file

        # Extract graph ID from filename (e.g., grafo10000.38.graphml -> 10000)
        graph_name = graph_path.stem  # grafo10000.38
        graph_id = graph_name.split('.')[0].replace('grafo', '')

        try:
            G = nx.read_graphml(graph_path)
            G = nx.convert_node_labels_to_integers(G, ordering="sorted")

            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()

            # Compute crossings for each baseline
            neato_xing = compute_crossings(G, nx.nx_agraph.graphviz_layout, "neato")
            sfdp_xing = compute_crossings(G, nx.nx_agraph.graphviz_layout, "sfdp")

            results.append({
                'graph_id': graph_id,
                'graph_file': graph_file,
                'nodes': n_nodes,
                'edges': n_edges,
                'neato_xing': neato_xing,
                'sfdp_xing': sfdp_xing
            })

        except Exception as e:
            print(f"Error loading {graph_file}: {e}")
            continue

    # Write results to CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        fieldnames = ['graph_id', 'graph_file', 'nodes', 'edges', 'neato_xing', 'sfdp_xing']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {OUTPUT_CSV}")

    # Summary statistics
    valid_results = [r for r in results if r['neato_xing'] is not None and r['sfdp_xing'] is not None]

    if valid_results:
        avg_neato = sum(r['neato_xing'] for r in valid_results) / len(valid_results)
        avg_sfdp = sum(r['sfdp_xing'] for r in valid_results) / len(valid_results)

        neato_wins = sum(1 for r in valid_results if r['neato_xing'] < r['sfdp_xing'])
        sfdp_wins = sum(1 for r in valid_results if r['sfdp_xing'] < r['neato_xing'])
        ties = sum(1 for r in valid_results if r['neato_xing'] == r['sfdp_xing'])

        print(f"\n=== Summary ({len(valid_results)} graphs) ===")
        print(f"Average neato crossings: {avg_neato:.2f}")
        print(f"Average sfdp crossings:  {avg_sfdp:.2f}")
        print(f"neato wins: {neato_wins}, sfdp wins: {sfdp_wins}, ties: {ties}")

if __name__ == "__main__":
    main()
