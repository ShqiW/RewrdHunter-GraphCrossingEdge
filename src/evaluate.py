"""
Evaluate trained model on test set
"""
import torch
import numpy as np
import csv
from pathlib import Path
from tqdm import tqdm

from src.env import GraphLayoutEnv
from src.models import GNNPolicy


def evaluate_model(
    policy: GNNPolicy,
    test_graph_file: str,
    data_dir: str,
    device: str = "cpu",
    max_steps: int = 100,
    deterministic: bool = True,
    output_csv: str = None,
):
    """
    Evaluate model on test graphs.

    Args:
        policy: Trained GNN policy
        test_graph_file: Path to file with test graph list
        data_dir: Base directory for graph files
        device: Device to run on
        max_steps: Max steps per episode
        deterministic: Use deterministic actions
        output_csv: Path to save results CSV

    Returns:
        results: List of dicts with evaluation results
    """
    policy = policy.to(device)
    policy.eval()

    data_dir = Path(data_dir)

    # Load test graph list
    with open(test_graph_file, 'r') as f:
        graph_files = [line.strip() for line in f if line.strip()]

    print(f"Evaluating on {len(graph_files)} test graphs...")

    results = []

    for graph_file in tqdm(graph_files, desc="Evaluating"):
        graph_path = data_dir / graph_file

        # Extract graph ID
        graph_name = Path(graph_path).stem
        graph_id = graph_name.split('.')[0].replace('grafo', '')

        try:
            env = GraphLayoutEnv(
                graph_path=str(graph_path),
                max_steps=max_steps,
                patience=20,
            )

            obs, info = env.reset()
            initial_crossings = info["crossings"]

            graph_data = env.get_graph_data()
            edge_index = graph_data["edge_index"].to(device)

            total_reward = 0
            done = False

            while not done:
                coords = torch.tensor(obs, dtype=torch.float32, device=device)

                with torch.no_grad():
                    action, _, _ = policy.get_action(
                        coords, edge_index, deterministic=deterministic
                    )

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated

            final_crossings = info["best_crossings"]
            improvement = initial_crossings - final_crossings

            results.append({
                "graph_id": graph_id,
                "graph_file": graph_file,
                "nodes": env.num_nodes,
                "edges": env.num_edges,
                "initial_xing": initial_crossings,
                "final_xing": final_crossings,
                "improvement": improvement,
                "total_reward": total_reward,
                "steps": info["steps"],
            })

        except Exception as e:
            print(f"Error evaluating {graph_file}: {e}")
            continue

    # Save results
    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            fieldnames = list(results[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"Results saved to {output_csv}")

    # Print summary
    if results:
        avg_initial = np.mean([r["initial_xing"] for r in results])
        avg_final = np.mean([r["final_xing"] for r in results])
        avg_improvement = np.mean([r["improvement"] for r in results])

        improved = sum(1 for r in results if r["improvement"] > 0)
        same = sum(1 for r in results if r["improvement"] == 0)
        worse = sum(1 for r in results if r["improvement"] < 0)

        print(f"\n=== Evaluation Summary ({len(results)} graphs) ===")
        print(f"Average initial crossings: {avg_initial:.2f}")
        print(f"Average final crossings:   {avg_final:.2f}")
        print(f"Average improvement:       {avg_improvement:.2f}")
        print(f"Improved: {improved}, Same: {same}, Worse: {worse}")

        # Relative improvement (as defined in project)
        rel_improvements = []
        for r in results:
            if max(r["initial_xing"], r["final_xing"]) > 0:
                rel_imp = (r["initial_xing"] - r["final_xing"]) / max(r["initial_xing"], r["final_xing"])
                rel_improvements.append(rel_imp)

        if rel_improvements:
            avg_rel_improvement = np.mean(rel_improvements)
            print(f"Average relative improvement: {avg_rel_improvement:.4f} ({avg_rel_improvement*100:.2f}%)")

    return results


def compare_with_baselines(
    rl_results_csv: str,
    baseline_csv: str,
    output_csv: str = None,
):
    """
    Compare RL results with baselines.

    Args:
        rl_results_csv: CSV with RL evaluation results
        baseline_csv: CSV with baseline results (neato, sfdp, smartgd)
        output_csv: Path to save comparison CSV
    """
    # Load RL results
    rl_data = {}
    with open(rl_results_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rl_data[row["graph_id"]] = int(row["final_xing"])

    # Load baseline results
    baseline_data = {}
    with open(baseline_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            baseline_data[row["graph_id"]] = {
                "neato": int(row["neato_xing"]),
                "sfdp": int(row["sfdp_xing"]),
                "smartgd": int(row["smartgd_xing"]),
            }

    # Compare
    comparison = []
    for graph_id in rl_data:
        if graph_id in baseline_data:
            rl_xing = rl_data[graph_id]
            baselines = baseline_data[graph_id]

            comparison.append({
                "graph_id": graph_id,
                "rl_xing": rl_xing,
                "neato_xing": baselines["neato"],
                "sfdp_xing": baselines["sfdp"],
                "smartgd_xing": baselines["smartgd"],
                "best_baseline": min(baselines.values()),
                "rl_vs_neato": baselines["neato"] - rl_xing,
                "rl_vs_sfdp": baselines["sfdp"] - rl_xing,
                "rl_vs_smartgd": baselines["smartgd"] - rl_xing,
                "rl_vs_best": min(baselines.values()) - rl_xing,
            })

    # Save comparison
    if output_csv and comparison:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=comparison[0].keys())
            writer.writeheader()
            writer.writerows(comparison)
        print(f"Comparison saved to {output_csv}")

    # Summary
    if comparison:
        avg_rl = np.mean([c["rl_xing"] for c in comparison])
        avg_neato = np.mean([c["neato_xing"] for c in comparison])
        avg_sfdp = np.mean([c["sfdp_xing"] for c in comparison])
        avg_smartgd = np.mean([c["smartgd_xing"] for c in comparison])

        rl_beats_neato = sum(1 for c in comparison if c["rl_xing"] < c["neato_xing"])
        rl_beats_sfdp = sum(1 for c in comparison if c["rl_xing"] < c["sfdp_xing"])
        rl_beats_smartgd = sum(1 for c in comparison if c["rl_xing"] < c["smartgd_xing"])
        rl_beats_all = sum(1 for c in comparison if c["rl_xing"] < c["best_baseline"])

        print(f"\n=== Comparison with Baselines ({len(comparison)} graphs) ===")
        print(f"Average crossings:")
        print(f"  RL:      {avg_rl:.2f}")
        print(f"  neato:   {avg_neato:.2f}")
        print(f"  sfdp:    {avg_sfdp:.2f}")
        print(f"  SmartGD: {avg_smartgd:.2f}")
        print(f"\nRL wins:")
        print(f"  vs neato:   {rl_beats_neato}/{len(comparison)}")
        print(f"  vs sfdp:    {rl_beats_sfdp}/{len(comparison)}")
        print(f"  vs SmartGD: {rl_beats_smartgd}/{len(comparison)}")
        print(f"  vs ALL:     {rl_beats_all}/{len(comparison)}")

    return comparison


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--test_graphs", type=str, default="data/test_graph.txt")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--baseline_csv", type=str, default="results/all_baselines.csv")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension of model")
    args = parser.parse_args()

    # Load model
    policy = GNNPolicy(input_dim=2, hidden_dim=args.hidden_dim, num_gnn_layers=3)
    policy.load_state_dict(torch.load(args.model, map_location=args.device))

    # Evaluate
    output_csv = Path(args.output_dir) / "rl_results.csv"
    results = evaluate_model(
        policy=policy,
        test_graph_file=args.test_graphs,
        data_dir=args.data_dir,
        device=args.device,
        output_csv=str(output_csv),
    )

    # Compare with baselines
    comparison_csv = Path(args.output_dir) / "rl_vs_baselines.csv"
    compare_with_baselines(
        rl_results_csv=str(output_csv),
        baseline_csv=args.baseline_csv,
        output_csv=str(comparison_csv),
    )
