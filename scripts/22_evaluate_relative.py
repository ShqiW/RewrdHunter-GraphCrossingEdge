"""
Evaluate models using relative improvement metric.
Based on the formula: rel_imp = (model_xing - baseline_xing) / max(model_xing, baseline_xing)
Negative value = better than baseline
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BASELINE_CSV = PROJECT_ROOT / 'results' / 'all_baselines.csv'
RL_CSV = PROJECT_ROOT / 'results' / 'rl_results.csv'
OUTPUT_DIR = PROJECT_ROOT / 'results'
FIGURE_DIR = OUTPUT_DIR / 'figures'
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load baseline and RL results."""
    # Load baselines
    baselines = {}
    with open(BASELINE_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            graph_id = row['graph_id']
            baselines[graph_id] = {
                'neato': int(row['neato_xing']),
                'sfdp': int(row['sfdp_xing']),
                'smartgd': int(row['smartgd_xing']),
            }

    # Load RL results (if exists)
    rl_data = {}
    if RL_CSV.exists():
        with open(RL_CSV, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                graph_id = row['graph_id']
                rl_data[graph_id] = int(row['final_xing'])

    return baselines, rl_data


def compute_relative_improvement(model_xing, baseline_xing):
    """
    Compute relative improvement.
    rel_imp = (model_xing - baseline_xing) / max(model_xing, baseline_xing)
    Negative = model is better than baseline
    """
    if max(model_xing, baseline_xing) == 0:
        return 0.0
    return (model_xing - baseline_xing) / max(model_xing, baseline_xing)


def evaluate_relative(baselines, rl_data, baseline_name='neato'):
    """
    Compute relative improvement for all models against a baseline.

    Args:
        baselines: Dict of baseline results
        rl_data: Dict of RL results
        baseline_name: Which baseline to use as reference ('neato', 'sfdp', 'smartgd')

    Returns:
        results: List of dicts with per-graph results
        summary: Dict with average improvements
    """
    results = []

    for graph_id in baselines:
        baseline_xing = baselines[graph_id][baseline_name]

        row = {
            'graph_id': graph_id,
            'baseline_xing': baseline_xing,
        }

        # Compute relative improvement for each model
        for model_name in ['neato', 'sfdp', 'smartgd']:
            if model_name != baseline_name:
                model_xing = baselines[graph_id][model_name]
                rel_imp = compute_relative_improvement(model_xing, baseline_xing)
                row[f'{model_name}_xing'] = model_xing
                row[f'{model_name}_rel_imp'] = rel_imp

        # RL model
        if graph_id in rl_data:
            rl_xing = rl_data[graph_id]
            rel_imp = compute_relative_improvement(rl_xing, baseline_xing)
            row['rl_xing'] = rl_xing
            row['rl_rel_imp'] = rel_imp

        results.append(row)

    # Compute summary statistics
    summary = {}

    for model_name in ['neato', 'sfdp', 'smartgd']:
        if model_name != baseline_name:
            key = f'{model_name}_rel_imp'
            values = [r[key] for r in results if key in r]
            summary[model_name] = {
                'mean_rel_imp': np.mean(values),
                'std_rel_imp': np.std(values),
                'count': len(values),
            }

    # RL summary
    if rl_data:
        values = [r['rl_rel_imp'] for r in results if 'rl_rel_imp' in r]
        summary['rl'] = {
            'mean_rel_imp': np.mean(values),
            'std_rel_imp': np.std(values),
            'count': len(values),
        }

    return results, summary


def save_results(results, summary, baseline_name):
    """Save results to CSV."""
    output_csv = OUTPUT_DIR / f'relative_improvement_{baseline_name}.csv'

    # Determine fieldnames based on what's in results
    fieldnames = ['graph_id', 'baseline_xing']
    for model in ['neato', 'sfdp', 'smartgd']:
        if model != baseline_name:
            fieldnames.extend([f'{model}_xing', f'{model}_rel_imp'])
    if 'rl_xing' in results[0]:
        fieldnames.extend(['rl_xing', 'rl_rel_imp'])

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {output_csv}")
    return output_csv


def plot_bar_chart(summary, baseline_name):
    """Plot bar chart of average relative improvements."""
    models = []
    means = []
    stds = []

    for model_name in ['sfdp', 'smartgd', 'rl']:
        if model_name in summary:
            models.append(model_name if model_name != 'rl' else 'RL')
            means.append(summary[model_name]['mean_rel_imp'] * 100)  # Convert to percentage
            stds.append(summary[model_name]['std_rel_imp'] * 100)

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['#55A868' if m < 0 else '#C44E52' for m in means]
    bars = ax.bar(models, means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Relative Improvement (%)', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_title(f'Relative Improvement vs {baseline_name} (negative = better)', fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.annotate(f'{mean:.2f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3 if height >= 0 else -15),
                   textcoords="offset points",
                   ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=11, fontweight='bold')

    plt.tight_layout()
    output_path = FIGURE_DIR / f'relative_improvement_bar_{baseline_name}.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Bar chart saved to {output_path}")


def plot_heatmap(results, baseline_name):
    """Plot heatmap of relative improvements for each graph."""
    from matplotlib.colors import TwoSlopeNorm

    # Prepare data
    graph_ids = [r['graph_id'] for r in results]

    models = []
    for model in ['sfdp', 'smartgd', 'rl']:
        key = f'{model}_rel_imp'
        if key in results[0]:
            models.append(model)

    data = []
    for model in models:
        key = f'{model}_rel_imp'
        values = [r[key] * 100 for r in results]  # Convert to percentage
        data.append(values)

    data = np.array(data).T  # Shape: (n_graphs, n_models)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(6, 20))

    # Use diverging colormap centered at 0
    vmax = max(abs(data.min()), abs(data.max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto', norm=norm)

    # Labels
    model_labels = [m.upper() if m == 'rl' else m for m in models]
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(model_labels, fontsize=11)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Graph ID', fontsize=12)
    ax.set_title(f'Relative Improvement vs {baseline_name} (%)\n(Green=Better, Red=Worse)', fontsize=14)

    # Y-axis: show every 10th graph
    n_graphs = len(graph_ids)
    ytick_positions = list(range(0, n_graphs, 10))
    ytick_labels = [graph_ids[i] for i in ytick_positions]
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels, fontsize=9)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.3)
    cbar.set_label('Relative Improvement (%)', fontsize=11)

    plt.tight_layout()
    output_path = FIGURE_DIR / f'relative_improvement_heatmap_{baseline_name}.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Heatmap saved to {output_path}")


def print_summary(summary, baseline_name):
    """Print summary table."""
    print(f"\n{'=' * 60}")
    print(f"Relative Improvement vs {baseline_name}")
    print(f"{'=' * 60}")
    print(f"{'Model':<12} {'Mean Rel. Imp.':<18} {'Interpretation'}")
    print(f"{'-' * 60}")

    for model_name in ['sfdp', 'smartgd', 'rl']:
        if model_name in summary:
            mean_imp = summary[model_name]['mean_rel_imp'] * 100
            std_imp = summary[model_name]['std_rel_imp'] * 100

            if mean_imp < 0:
                interp = f"Better than {baseline_name} by {abs(mean_imp):.2f}%"
            else:
                interp = f"Worse than {baseline_name} by {mean_imp:.2f}%"

            display_name = model_name.upper() if model_name == 'rl' else model_name
            print(f"{display_name:<12} {mean_imp:>+.2f}% ± {std_imp:.2f}%   {interp}")

    print(f"{'=' * 60}")


def main():
    print("Loading data...")
    baselines, rl_data = load_data()

    if not rl_data:
        print("Warning: No RL results found. Run training and evaluation first.")

    baseline_name = 'neato'  # Use neato as baseline

    print(f"\nComputing relative improvement against {baseline_name}...")
    results, summary = evaluate_relative(baselines, rl_data, baseline_name)

    # Save results
    save_results(results, summary, baseline_name)

    # Print summary
    print_summary(summary, baseline_name)

    # Plot visualizations
    print("\nGenerating visualizations...")
    plot_bar_chart(summary, baseline_name)
    plot_heatmap(results, baseline_name)

    print("\nDone!")


if __name__ == "__main__":
    main()
