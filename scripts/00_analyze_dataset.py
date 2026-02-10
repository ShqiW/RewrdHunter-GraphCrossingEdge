#!/usr/bin/env python3
"""
Analyze Rome dataset and split into train/test sets.
"""
import os
import glob


def analyze_dataset(data_dir="rome/rome"):
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
