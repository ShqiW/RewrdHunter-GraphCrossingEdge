"""
Rome数据集评估

在Rome数据集上评估RL模型，并与baseline方法比较:
- neato (Graphviz force-directed)
- sfdp (Graphviz scalable force-directed)
- random (随机布局作为下限)
"""
import torch
import numpy as np
import csv
import networkx as nx
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import warnings

from src.enviroment.discrete_env import DiscreteGraphEnv
from src.models.discrete_policy import DiscreteGNNPolicy
from xing import XingLoss


def load_rome_graph(graphml_path: str) -> nx.Graph:
    """加载Rome数据集中的图"""
    graph = nx.read_graphml(graphml_path)
    # 转换为无向图并重新映射节点ID为整数
    graph = nx.Graph(graph)
    graph = nx.convert_node_labels_to_integers(graph)
    return graph


def get_layout_crossings(graph: nx.Graph, layout: str, seed: int = 42) -> Tuple[int, np.ndarray]:
    """
    获取特定布局方法的crossing数

    Args:
        graph: NetworkX图
        layout: 布局方法 (neato, sfdp, spring, random)
        seed: 随机种子

    Returns:
        crossings: crossing数
        positions: 节点坐标 [num_nodes, 2]
    """
    if layout == "neato":
        try:
            pos = nx.nx_agraph.graphviz_layout(graph, prog="neato")
        except:
            # Fallback to spring if graphviz not available
            pos = nx.spring_layout(graph, seed=seed)
    elif layout == "sfdp":
        try:
            pos = nx.nx_agraph.graphviz_layout(graph, prog="sfdp")
        except:
            pos = nx.spring_layout(graph, seed=seed)
    elif layout == "spring":
        pos = nx.spring_layout(graph, seed=seed)
    elif layout == "random":
        np.random.seed(seed)
        pos = {i: np.random.rand(2) for i in range(graph.number_of_nodes())}
    else:
        raise ValueError(f"Unknown layout: {layout}")

    # 转换为numpy数组
    coords = np.array([pos[i] for i in range(graph.number_of_nodes())], dtype=np.float32)

    # 归一化到[0, 1]
    coords = coords - coords.min(axis=0)
    max_range = coords.max()
    if max_range > 0:
        coords = coords / max_range

    # 计算crossings
    edge_list = list(graph.edges())
    if len(edge_list) < 2:
        return 0, coords

    coords_tensor = torch.tensor(coords, dtype=torch.float32)

    # XingLoss需要Graph作为参数
    xing_loss = XingLoss(graph, soft=False)
    crossings = int(xing_loss(coords_tensor).item())

    return crossings, coords


def evaluate_rl_model(
    policy: DiscreteGNNPolicy,
    graph: nx.Graph,
    device: str = "cpu",
    max_steps: int = 100,
    move_scale: float = 0.05,
    initial_layout: str = "random",
) -> Dict:
    """
    使用RL模型评估单个图

    Returns:
        dict with initial_crossings, final_crossings, improvement
    """
    env = DiscreteGraphEnv(
        graph=graph,
        max_steps=max_steps,
        move_scale=move_scale,
        initial_layout=initial_layout,
    )

    obs, info = env.reset()
    initial_crossings = info["crossings"]
    graph_data = env.get_graph_data()

    done = False
    steps = 0

    while not done:
        coords = torch.tensor(obs, dtype=torch.float32, device=device)
        edge_index = graph_data["edge_index"].to(device)

        with torch.no_grad():
            action, _, _ = policy.get_action(coords, edge_index, deterministic=True)

        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

    final_crossings = info["crossings"]

    return {
        "initial_crossings": initial_crossings,
        "final_crossings": final_crossings,
        "improvement": initial_crossings - final_crossings,
        "steps": steps,
    }


def evaluate_rome_dataset(
    model_path: str,
    test_graph_file: str,
    data_dir: str,
    device: str = "cpu",
    max_graphs: int = None,
    output_csv: str = None,
    hidden_dim: int = 128,
) -> List[Dict]:
    """
    在Rome数据集上评估模型

    Args:
        model_path: 训练好的模型路径
        test_graph_file: 测试图列表文件
        data_dir: 数据目录
        device: 设备
        max_graphs: 最大评估图数 (None表示全部)
        output_csv: 输出CSV路径
        hidden_dim: 模型隐藏层维度

    Returns:
        评估结果列表
    """
    # 加载模型
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    policy = DiscreteGNNPolicy(
        input_dim=2,
        hidden_dim=hidden_dim,
        num_gnn_layers=3,
    ).to(device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()

    # 加载测试图列表
    data_dir = Path(data_dir)
    with open(test_graph_file, 'r') as f:
        graph_files = [line.strip() for line in f if line.strip()]

    if max_graphs:
        graph_files = graph_files[:max_graphs]

    print(f"Evaluating on {len(graph_files)} graphs...")

    results = []

    for graph_file in tqdm(graph_files, desc="Evaluating"):
        graph_path = data_dir / graph_file
        graph_name = Path(graph_path).stem

        # 提取图ID和节点数 (格式: grafo{id}.{nodes})
        parts = graph_name.split('.')
        graph_id = parts[0].replace('grafo', '')
        num_nodes_from_name = int(parts[1]) if len(parts) > 1 else 0

        try:
            graph = load_rome_graph(str(graph_path))
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()

            # 跳过太大的图 (curriculum训练最大到20-25节点)
            if num_nodes > 100:
                print(f"Skipping {graph_name}: {num_nodes} nodes (too large)")
                continue

            # 获取baseline crossings
            neato_crossings, _ = get_layout_crossings(graph, "neato")
            sfdp_crossings, _ = get_layout_crossings(graph, "sfdp")
            spring_crossings, _ = get_layout_crossings(graph, "spring")
            random_crossings, _ = get_layout_crossings(graph, "random")

            # RL模型评估 (从random布局开始，与训练一致)
            rl_result = evaluate_rl_model(
                policy, graph, device=device,
                max_steps=100, move_scale=0.05, initial_layout="random"
            )

            result = {
                "graph_id": graph_id,
                "graph_file": graph_file,
                "nodes": num_nodes,
                "edges": num_edges,
                "neato_xing": neato_crossings,
                "sfdp_xing": sfdp_crossings,
                "spring_xing": spring_crossings,
                "random_xing": random_crossings,
                "rl_initial_xing": rl_result["initial_crossings"],
                "rl_final_xing": rl_result["final_crossings"],
                "rl_improvement": rl_result["improvement"],
                "rl_steps": rl_result["steps"],
            }

            results.append(result)

        except Exception as e:
            print(f"Error processing {graph_file}: {e}")
            continue

    # 保存结果
    if output_csv and results:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        print(f"Results saved to {output_csv}")

    # 打印summary
    print_evaluation_summary(results)

    return results


def evaluate_rl_refinement(
    policy: DiscreteGNNPolicy,
    graph: nx.Graph,
    baseline_layout: str = "neato",
    device: str = "cpu",
    max_steps: int = 100,
    move_scale: float = 0.02,
) -> Dict:
    """
    使用RL模型从baseline布局开始refinement

    Args:
        policy: 训练好的策略
        graph: NetworkX图
        baseline_layout: 初始布局方法
        device: 设备
        max_steps: 最大步数
        move_scale: 移动比例（refinement用更小的步长）

    Returns:
        dict with baseline_crossings, rl_refined_crossings, improvement
    """
    # 获取baseline布局
    baseline_crossings, baseline_coords = get_layout_crossings(graph, baseline_layout)

    # 创建环境
    env = DiscreteGraphEnv(
        graph=graph,
        max_steps=max_steps,
        move_scale=move_scale,
        initial_layout="random",  # 会被覆盖
    )

    # 重置环境，然后手动设置初始坐标为baseline布局
    env.reset()
    env.coords = baseline_coords.copy()
    env.current_crossings = env._compute_crossings(env.coords)
    env.current_stress = env._compute_stress(env.coords)
    env.current_potential = env._compute_potential(env.coords)
    env.initial_crossings = env.current_crossings
    env.best_crossings = env.current_crossings

    obs = env.coords.copy()
    graph_data = env.get_graph_data()

    done = False
    steps = 0

    while not done:
        coords = torch.tensor(obs, dtype=torch.float32, device=device)
        edge_index = graph_data["edge_index"].to(device)

        with torch.no_grad():
            action, _, _ = policy.get_action(coords, edge_index, deterministic=True)

        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

    refined_crossings = info["crossings"]

    return {
        "baseline": baseline_layout,
        "baseline_crossings": baseline_crossings,
        "rl_refined_crossings": refined_crossings,
        "improvement": baseline_crossings - refined_crossings,
        "steps": steps,
    }


def print_evaluation_summary(results: List[Dict]):
    """打印评估摘要"""
    if not results:
        print("No results to summarize")
        return

    print("\n" + "=" * 70)
    print("  Evaluation Summary")
    print("=" * 70)

    n = len(results)
    print(f"\nTotal graphs evaluated: {n}")

    # 按节点数分组
    size_groups = {
        "small (≤20)": [r for r in results if r["nodes"] <= 20],
        "medium (21-50)": [r for r in results if 20 < r["nodes"] <= 50],
        "large (>50)": [r for r in results if r["nodes"] > 50],
    }

    # 总体统计
    print("\n--- Overall Statistics ---")

    avg_neato = np.mean([r["neato_xing"] for r in results])
    avg_sfdp = np.mean([r["sfdp_xing"] for r in results])
    avg_spring = np.mean([r["spring_xing"] for r in results])
    avg_random = np.mean([r["random_xing"] for r in results])
    avg_rl_final = np.mean([r["rl_final_xing"] for r in results])
    avg_rl_improvement = np.mean([r["rl_improvement"] for r in results])

    print(f"\nAverage crossings by method:")
    print(f"  Random (baseline): {avg_random:.1f}")
    print(f"  Spring:           {avg_spring:.1f}")
    print(f"  Neato:            {avg_neato:.1f}")
    print(f"  SFDP:             {avg_sfdp:.1f}")
    print(f"  RL (from random): {avg_rl_final:.1f}")

    print(f"\nRL improvement: {avg_rl_improvement:+.1f} crossings on average")

    # RL vs baselines
    print("\n--- RL vs Baselines ---")

    rl_beats_random = sum(1 for r in results if r["rl_final_xing"] < r["random_xing"])
    rl_beats_spring = sum(1 for r in results if r["rl_final_xing"] < r["spring_xing"])
    rl_beats_neato = sum(1 for r in results if r["rl_final_xing"] < r["neato_xing"])
    rl_beats_sfdp = sum(1 for r in results if r["rl_final_xing"] < r["sfdp_xing"])

    print(f"RL beats Random: {rl_beats_random}/{n} ({100*rl_beats_random/n:.1f}%)")
    print(f"RL beats Spring: {rl_beats_spring}/{n} ({100*rl_beats_spring/n:.1f}%)")
    print(f"RL beats Neato:  {rl_beats_neato}/{n} ({100*rl_beats_neato/n:.1f}%)")
    print(f"RL beats SFDP:   {rl_beats_sfdp}/{n} ({100*rl_beats_sfdp/n:.1f}%)")

    # RL improvement分析
    print("\n--- RL Improvement Analysis ---")

    rl_improved = sum(1 for r in results if r["rl_improvement"] > 0)
    rl_same = sum(1 for r in results if r["rl_improvement"] == 0)
    rl_worse = sum(1 for r in results if r["rl_improvement"] < 0)

    print(f"Improved:  {rl_improved}/{n} ({100*rl_improved/n:.1f}%)")
    print(f"Same:      {rl_same}/{n} ({100*rl_same/n:.1f}%)")
    print(f"Worse:     {rl_worse}/{n} ({100*rl_worse/n:.1f}%)")

    # 按大小分组
    print("\n--- Results by Graph Size ---")

    for group_name, group_results in size_groups.items():
        if not group_results:
            continue

        gn = len(group_results)
        avg_impr = np.mean([r["rl_improvement"] for r in group_results])
        avg_final = np.mean([r["rl_final_xing"] for r in group_results])
        beats_neato = sum(1 for r in group_results if r["rl_final_xing"] < r["neato_xing"])

        print(f"\n{group_name}: {gn} graphs")
        print(f"  Avg RL improvement: {avg_impr:+.1f}")
        print(f"  Avg RL crossings:   {avg_final:.1f}")
        print(f"  Beats neato:        {beats_neato}/{gn}")

    print("\n" + "=" * 70)


def evaluate_refinement(
    model_path: str,
    test_graph_file: str,
    data_dir: str,
    device: str = "cpu",
    max_graphs: int = None,
    hidden_dim: int = 128,
    baseline_layout: str = "neato",
) -> List[Dict]:
    """
    评估RL模型对baseline布局的refinement能力

    Args:
        model_path: 模型路径
        test_graph_file: 测试图列表
        data_dir: 数据目录
        device: 设备
        max_graphs: 最大评估图数
        hidden_dim: 模型隐藏层维度
        baseline_layout: baseline布局方法

    Returns:
        评估结果列表
    """
    # 加载模型
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    policy = DiscreteGNNPolicy(
        input_dim=2,
        hidden_dim=hidden_dim,
        num_gnn_layers=3,
    ).to(device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()

    # 加载测试图列表
    data_dir = Path(data_dir)
    with open(test_graph_file, 'r') as f:
        graph_files = [line.strip() for line in f if line.strip()]

    if max_graphs:
        graph_files = graph_files[:max_graphs]

    print(f"Evaluating refinement on {len(graph_files)} graphs...")
    print(f"Baseline layout: {baseline_layout}")

    results = []

    for graph_file in tqdm(graph_files, desc="Refinement"):
        graph_path = data_dir / graph_file
        graph_name = Path(graph_path).stem

        parts = graph_name.split('.')
        graph_id = parts[0].replace('grafo', '')

        try:
            graph = load_rome_graph(str(graph_path))
            num_nodes = graph.number_of_nodes()

            if num_nodes > 100:
                continue

            # RL refinement评估
            refine_result = evaluate_rl_refinement(
                policy, graph,
                baseline_layout=baseline_layout,
                device=device,
                max_steps=100,
                move_scale=0.02,  # 更小的步长用于refinement
            )

            results.append({
                "graph_id": graph_id,
                "nodes": num_nodes,
                "baseline": baseline_layout,
                "baseline_xing": refine_result["baseline_crossings"],
                "rl_refined_xing": refine_result["rl_refined_crossings"],
                "improvement": refine_result["improvement"],
            })

        except Exception as e:
            print(f"Error: {graph_file}: {e}")
            continue

    # 打印summary
    if results:
        print("\n" + "=" * 60)
        print(f"  Refinement Summary ({baseline_layout})")
        print("=" * 60)

        n = len(results)
        avg_baseline = np.mean([r["baseline_xing"] for r in results])
        avg_refined = np.mean([r["rl_refined_xing"] for r in results])
        avg_improvement = np.mean([r["improvement"] for r in results])

        improved = sum(1 for r in results if r["improvement"] > 0)
        same = sum(1 for r in results if r["improvement"] == 0)
        worse = sum(1 for r in results if r["improvement"] < 0)

        print(f"\nAvg {baseline_layout} crossings: {avg_baseline:.1f}")
        print(f"Avg RL refined crossings: {avg_refined:.1f}")
        print(f"Avg improvement: {avg_improvement:+.2f}")
        print(f"\nImproved: {improved}/{n} ({100*improved/n:.1f}%)")
        print(f"Same:     {same}/{n} ({100*same/n:.1f}%)")
        print(f"Worse:    {worse}/{n} ({100*worse/n:.1f}%)")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate on Rome Dataset")
    parser.add_argument("--model", type=str, default="checkpoints_curriculum/final_model.pt",
                        help="Path to trained model")
    parser.add_argument("--test-graphs", type=str, default="data/test_graph.txt",
                        help="Path to test graph list")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Data directory")
    parser.add_argument("--output-csv", type=str, default="results/rome_evaluation.csv",
                        help="Output CSV path")
    parser.add_argument("--max-graphs", type=int, default=None,
                        help="Maximum number of graphs to evaluate")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--mode", choices=["full", "refinement"], default="full",
                        help="Evaluation mode: full (from random) or refinement (from baseline)")
    parser.add_argument("--baseline", type=str, default="neato",
                        help="Baseline layout for refinement mode")

    args = parser.parse_args()

    if args.mode == "full":
        results = evaluate_rome_dataset(
            model_path=args.model,
            test_graph_file=args.test_graphs,
            data_dir=args.data_dir,
            device=args.device,
            max_graphs=args.max_graphs,
            output_csv=args.output_csv,
            hidden_dim=args.hidden_dim,
        )
    else:
        results = evaluate_refinement(
            model_path=args.model,
            test_graph_file=args.test_graphs,
            data_dir=args.data_dir,
            device=args.device,
            max_graphs=args.max_graphs,
            hidden_dim=args.hidden_dim,
            baseline_layout=args.baseline,
        )
