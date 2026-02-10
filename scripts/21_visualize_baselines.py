"""
Visualize baseline results: boxplot, scatter plots, pie chart
"""
import csv
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path('/Users/wsq/Nextcloud/CSA/CS5180RL/project/RewardHunter-')
INPUT_CSV = PROJECT_ROOT / 'results' / 'all_baselines.csv'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load baseline results from CSV."""
    data = {'neato': [], 'sfdp': [], 'smartgd': []}
    with open(INPUT_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['neato_xing'] and row['sfdp_xing'] and row['smartgd_xing']:
                data['neato'].append(int(row['neato_xing']))
                data['sfdp'].append(int(row['sfdp_xing']))
                data['smartgd'].append(int(row['smartgd_xing']))
    return data

def plot_boxplot(data):
    """Plot boxplot comparing distributions."""
    fig, ax = plt.subplots(figsize=(8, 6))

    box_data = [data['neato'], data['sfdp'], data['smartgd']]
    labels = ['neato', 'sfdp', 'SmartGD']
    colors = ['#4C72B0', '#55A868', '#C44E52']

    bp = ax.boxplot(box_data, labels=labels, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Edge Crossings', fontsize=12)
    ax.set_title('Distribution of Edge Crossings by Baseline Method', fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    # Add mean markers
    means = [sum(d)/len(d) for d in box_data]
    ax.scatter([1, 2, 3], means, color='white', marker='D', s=50, zorder=3, edgecolor='black')

    # Add mean values as text
    for i, mean in enumerate(means):
        ax.annotate(f'μ={mean:.1f}', (i+1, mean), textcoords="offset points",
                   xytext=(25, 0), ha='left', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'boxplot_baselines.png', dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'boxplot_baselines.png'}")

def plot_scatter_matrix(data):
    """Plot scatter plots comparing pairs of methods."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    pairs = [
        ('neato', 'sfdp', 'neato vs sfdp'),
        ('neato', 'smartgd', 'neato vs SmartGD'),
        ('sfdp', 'smartgd', 'sfdp vs SmartGD')
    ]

    for ax, (x_key, y_key, title) in zip(axes, pairs):
        x = data[x_key]
        y = data[y_key]

        ax.scatter(x, y, alpha=0.6, edgecolor='black', linewidth=0.5)

        # Diagonal line (equal performance)
        max_val = max(max(x), max(y))
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Equal')

        ax.set_xlabel(x_key, fontsize=11)
        ax.set_ylabel(y_key, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

        # Count wins
        x_wins = sum(1 for xi, yi in zip(x, y) if xi < yi)
        y_wins = sum(1 for xi, yi in zip(x, y) if yi < xi)
        ties = sum(1 for xi, yi in zip(x, y) if xi == yi)
        ax.annotate(f'{x_key}: {x_wins} wins\n{y_key}: {y_wins} wins\nTies: {ties}',
                   xy=(0.05, 0.95), xycoords='axes fraction',
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'scatter_comparisons.png', dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'scatter_comparisons.png'}")

def plot_pie_chart(data):
    """Plot pie chart showing best method distribution."""
    n = len(data['neato'])

    # Count exclusive wins (no ties)
    neato_best = 0
    sfdp_best = 0
    smartgd_best = 0
    ties = 0

    for i in range(n):
        vals = [data['neato'][i], data['sfdp'][i], data['smartgd'][i]]
        min_val = min(vals)
        winners = [j for j, v in enumerate(vals) if v == min_val]

        if len(winners) == 1:
            if winners[0] == 0:
                neato_best += 1
            elif winners[0] == 1:
                sfdp_best += 1
            else:
                smartgd_best += 1
        else:
            ties += 1

    fig, ax = plt.subplots(figsize=(8, 8))

    sizes = [neato_best, sfdp_best, smartgd_best, ties]
    labels = [f'neato\n({neato_best})', f'sfdp\n({sfdp_best})',
              f'SmartGD\n({smartgd_best})', f'Ties\n({ties})']
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3']
    explode = (0, 0.05, 0, 0)  # Highlight sfdp (best performer)

    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                       explode=explode, autopct='%1.1f%%',
                                       startangle=90, pctdistance=0.6)

    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')

    ax.set_title('Best Method Distribution (99 Test Graphs)', fontsize=14)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'pie_best_method.png', dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'pie_best_method.png'}")

def plot_rank_heatmap(data):
    """Plot rank heatmap: rows=graphs, cols=methods, color=rank."""
    import numpy as np
    from matplotlib.colors import ListedColormap

    n = len(data['neato'])

    # Compute ranks for each graph (1=best, 3=worst)
    ranks = []
    graph_ids = []
    for i in range(n):
        vals = [data['neato'][i], data['sfdp'][i], data['smartgd'][i]]
        # Rank: lower crossing = better rank
        sorted_indices = sorted(range(3), key=lambda x: vals[x])
        rank = [0, 0, 0]
        current_rank = 1
        j = 0
        while j < 3:
            # Handle ties
            tie_group = [sorted_indices[j]]
            while j + 1 < 3 and vals[sorted_indices[j+1]] == vals[sorted_indices[j]]:
                j += 1
                tie_group.append(sorted_indices[j])
            for idx in tie_group:
                rank[idx] = current_rank
            current_rank += len(tie_group)
            j += 1
        ranks.append(rank)
        graph_ids.append(10000 + i)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(6, 20))

    rank_array = np.array(ranks)

    # Custom colormap: 1=green (best), 2=yellow, 3=red (worst)
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']  # green, yellow, red
    cmap = ListedColormap(colors)

    im = ax.imshow(rank_array, cmap=cmap, aspect='auto', vmin=1, vmax=3)

    # Labels
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['neato', 'sfdp', 'SmartGD'], fontsize=11)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Graph ID', fontsize=12)
    ax.set_title('Ranking Heatmap (1=Best, 3=Worst)', fontsize=14)

    # Y-axis: show every 10th graph
    ytick_positions = list(range(0, n, 10))
    ytick_labels = [str(graph_ids[i]) for i in ytick_positions]
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels, fontsize=9)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[1, 2, 3], shrink=0.3)
    cbar.set_label('Rank', fontsize=11)
    cbar.ax.set_yticklabels(['1 (Best)', '2', '3 (Worst)'])

    # Add rank counts as summary
    rank1_counts = [sum(1 for r in ranks if r[i] == 1) for i in range(3)]
    summary = f"Rank 1 counts: neato={rank1_counts[0]}, sfdp={rank1_counts[1]}, SmartGD={rank1_counts[2]}"
    ax.text(0.5, -0.02, summary, transform=ax.transAxes, ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'rank_heatmap.png', dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'rank_heatmap.png'}")


def main():
    data = load_data()
    print(f"Loaded {len(data['neato'])} graphs")

    plot_boxplot(data)
    plot_scatter_matrix(data)
    plot_pie_chart(data)
    plot_rank_heatmap(data)

    print(f"\nAll figures saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
