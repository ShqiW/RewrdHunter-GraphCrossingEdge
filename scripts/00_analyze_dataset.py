#!/usr/bin/env python3
"""
Analyze Rome dataset and split into train/test sets.
"""
import os
import glob
import networkx as nx
from collections import defaultdict


def analyze_dataset(data_dir="rome"):
    """Analyze the Rome dataset distribution."""
    pattern = os.path.join(data_dir, "grafo*.graphml")
    files = glob.glob(pattern)

    nums = []
    file_map = {}

    for fpath in files:
        fname = os.path.basename(fpath)
        # Extract number from filename like grafo1000.14.graphml
        num = int(fname.split('grafo')[1].split('.')[0])
        nums.append(num)
        file_map[num] = fpath

    nums.sort()

    print("=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"Total files: {len(nums)}")
    print(f"Min graph number: {min(nums)}")
    print(f"Max graph number: {max(nums)}")
    print(f"Graph numbers ≤9999: {sum(1 for n in nums if n <= 9999)}")
    print(f"Graph numbers 10000-10100: {sum(1 for n in nums if 10000 <= n <= 10100)}")
    print(f"Graph numbers >10100: {sum(1 for n in nums if n > 10100)}")
    print("=" * 60)

    return nums, file_map


def analyze_node_distribution(train_files, test_files, bin_size=10):
    """统计 train/test 集中各图的节点数分布，按区间分组输出。"""

    def count_bins(fpaths):
        bins = defaultdict(int)
        counts = []
        for fpath in fpaths:
            G = nx.read_graphml(fpath)
            n = G.number_of_nodes()
            counts.append(n)
            lo = (n - 1) // bin_size * bin_size + 1
            bins[lo] += 1
        return bins, counts

    print("Counting train set node distribution...")
    train_bins, train_counts = count_bins(train_files)
    print("Counting test set node distribution...")
    test_bins, test_counts = count_bins(test_files)

    all_los = sorted(set(train_bins) | set(test_bins))

    print()
    print("=" * 50)
    print("Node Count Distribution")
    print("=" * 50)
    print(f"{'Range':<12} {'Train':>8} {'Test':>8}")
    print("-" * 30)
    for lo in all_los:
        hi = lo + bin_size - 1
        print(f"{lo:>3} - {hi:<4}   {train_bins[lo]:>8} {test_bins[lo]:>8}")
    print("-" * 30)
    print(f"{'Total':<12} {len(train_counts):>8} {len(test_counts):>8}")
    print(f"{'Min nodes':<12} {min(train_counts):>8} {min(test_counts):>8}")
    print(f"{'Max nodes':<12} {max(train_counts):>8} {max(test_counts):>8}")
    print(f"{'Mean nodes':<12} {sum(train_counts)/len(train_counts):>8.1f} {sum(test_counts)/len(test_counts):>8.1f}")
    print("=" * 50)


def split_dataset(nums, file_map, train_cutoff=9999, test_start=10000, test_end=10100):
    """
    Split dataset into train and test sets.

    Args:
        nums: List of graph numbers
        file_map: Dict mapping graph number to file path
        train_cutoff: Max number for training set (inclusive)
        test_start: Min number for test set (inclusive)
        test_end: Max number for test set (inclusive)
    """
    train_files = []
    test_files = []

    for num in nums:
        if num <= train_cutoff:
            train_files.append(file_map[num])
        elif test_start <= num <= test_end:
            test_files.append(file_map[num])

    # Sort by graph number
    train_files.sort()
    test_files.sort()

    # Write to files
    with open("train_graph.txt", "w") as f:
        for fpath in train_files:
            f.write(fpath + "\n")

    with open("test_graph.txt", "w") as f:
        for fpath in test_files:
            f.write(fpath + "\n")

    print(f"\nSplit Results:")
    print(f"  Train set: {len(train_files)} graphs (numbers ≤{train_cutoff})")
    print(f"  Test set: {len(test_files)} graphs (numbers {test_start}-{test_end})")
    print(f"  Unused: {len(nums) - len(train_files) - len(test_files)} graphs")
    print(f"\nFiles saved:")
    print(f"  train_graph.txt")
    print(f"  test_graph.txt")

    return train_files, test_files


if __name__ == "__main__":
    # Analyze dataset
    nums, file_map = analyze_dataset()

    # Split into train (≤9999) and test (10000-10100)
    train_files, test_files = split_dataset(
        nums, file_map,
        train_cutoff=9999,
        test_start=10000,
        test_end=10100
    )

    # Node count distribution
    analyze_node_distribution(train_files, test_files)
