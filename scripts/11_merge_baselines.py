"""
Merge baseline results with SmartGD metrics
"""
import csv
from pathlib import Path

PROJECT_ROOT = Path('/Users/wsq/Nextcloud/CSA/CS5180RL/project/RewardHunter-')
BASELINE_CSV = PROJECT_ROOT / 'results' / 'baseline_results.csv'
SMARTGD_CSV = PROJECT_ROOT / 'results' / 'smartgd_test.csv'
OUTPUT_CSV = PROJECT_ROOT / 'results' / 'all_baselines.csv'

def main():
    # Load baseline results
    baseline_data = {}
    with open(BASELINE_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            graph_id = row['graph_id']
            baseline_data[graph_id] = {
                'graph_file': row['graph_file'],
                'nodes': int(row['nodes']),
                'edges': int(row['edges']),
                'neato_xing': int(row['neato_xing']) if row['neato_xing'] else None,
                'sfdp_xing': int(row['sfdp_xing']) if row['sfdp_xing'] else None,
            }

    # Load SmartGD results
    smartgd_data = {}
    with open(SMARTGD_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # graph_id format: grafo10000.38 -> 10000
            graph_id = row['graph_id'].split('.')[0].replace('grafo', '')
            smartgd_data[graph_id] = int(float(row['xing']))

    # Merge data
    merged = []
    for graph_id, data in baseline_data.items():
        smartgd_xing = smartgd_data.get(graph_id)
        merged.append({
            'graph_id': graph_id,
            'graph_file': data['graph_file'],
            'nodes': data['nodes'],
            'edges': data['edges'],
            'neato_xing': data['neato_xing'],
            'sfdp_xing': data['sfdp_xing'],
            'smartgd_xing': smartgd_xing
        })

    # Sort by graph_id
    merged.sort(key=lambda x: int(x['graph_id']))

    # Write merged results
    with open(OUTPUT_CSV, 'w', newline='') as f:
        fieldnames = ['graph_id', 'graph_file', 'nodes', 'edges', 'neato_xing', 'sfdp_xing', 'smartgd_xing']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)

    print(f"Merged results saved to {OUTPUT_CSV}")

    # Summary statistics
    valid = [r for r in merged if r['neato_xing'] is not None and r['sfdp_xing'] is not None and r['smartgd_xing'] is not None]

    if valid:
        avg_neato = sum(r['neato_xing'] for r in valid) / len(valid)
        avg_sfdp = sum(r['sfdp_xing'] for r in valid) / len(valid)
        avg_smartgd = sum(r['smartgd_xing'] for r in valid) / len(valid)

        print(f"\n=== Summary ({len(valid)} graphs) ===")
        print(f"Average crossings:")
        print(f"  neato:   {avg_neato:.2f}")
        print(f"  sfdp:    {avg_sfdp:.2f}")
        print(f"  SmartGD: {avg_smartgd:.2f}")

        # Win counts
        def count_wins(results, key):
            wins = 0
            for r in results:
                if r[key] == min(r['neato_xing'], r['sfdp_xing'], r['smartgd_xing']):
                    wins += 1
            return wins

        # Best method for each graph
        best_counts = {'neato': 0, 'sfdp': 0, 'smartgd': 0}
        for r in valid:
            min_xing = min(r['neato_xing'], r['sfdp_xing'], r['smartgd_xing'])
            if r['neato_xing'] == min_xing:
                best_counts['neato'] += 1
            if r['sfdp_xing'] == min_xing:
                best_counts['sfdp'] += 1
            if r['smartgd_xing'] == min_xing:
                best_counts['smartgd'] += 1

        print(f"\nBest method counts (ties counted for all):")
        print(f"  neato:   {best_counts['neato']}")
        print(f"  sfdp:    {best_counts['sfdp']}")
        print(f"  SmartGD: {best_counts['smartgd']}")

if __name__ == "__main__":
    main()
