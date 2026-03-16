"""
Unified training module.

Handles:
- Environment creation based on args
- Model creation based on args
- PPO training loop (discrete and continuous action spaces)
"""
import torch
import networkx as nx
from pathlib import Path

from src.tasks.base import BaseArgs
from src.trainer.utils import create_env
# ──────────────────────────────────────────────────────────────────────────────


def create_graph(args: BaseArgs) -> nx.Graph:
    graph_config = args.graph
    if graph_config.graph_path:
        graph = nx.read_graphml(graph_config.graph_path)
        graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")
    else:
        graph = nx.erdos_renyi_graph(graph_config.num_nodes,
                                     graph_config.edge_prob,
                                     seed=args.seed)
        if not nx.is_connected(graph):
            components = list(nx.connected_components(graph))
            for i in range(len(components) - 1):
                u = list(components[i])[0]
                v = list(components[i + 1])[0]
                graph.add_edge(u, v)
    return graph


def create_model(args: BaseArgs, env):
    model_type = args.model.type
    match model_type:
        case "gnn":
            from src.models.gnn import DiscreteGNNPolicy
            return DiscreteGNNPolicy(config=args.model)
        case "transformer":
            from src.models.transformer_policy import TransformerPlacementPolicy
            return TransformerPlacementPolicy(config=args.model)
        case _:
            raise NotImplementedError(
                f"Model type '{model_type}' not implemented")


# ──────────────────────────────────────────────────────────────────────────────
# PPO Trainer — continuous (sequential placement)
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Unified train() entry point
# ──────────────────────────────────────────────────────────────────────────────


def train(args: BaseArgs):
    device = "cuda" if torch.cuda.is_available(
    ) and args.num_gpus > 0 else "cpu"
    print(f"Task: {args.name}")
    print(f"Seed: {args.seed}")

    dataset = None
    graph_data = None

    if args.graph.use_dataset:
        from src.data.rome import RomeDataset
        dataset = RomeDataset(root=args.graph.data_root,
                              split=args.graph.data_split)
        print(
            f"Dataset: {len(dataset)} graphs from {args.graph.data_split} split"
        )
        graph_data = dataset.sample()
        print(
            f"Initial graph: {graph_data.graph_name} ({graph_data.num_nodes} nodes)"
        )
        env = create_env(args, device=device, graph_data=graph_data)
    else:
        raise NotImplementedError(
            "Single-graph mode not supported; set graph.use_dataset=True")

    obs, info = env.reset()

    model = create_model(args, env)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model.type}, params: {n_params:,}")

    # Select trainer based on env type
    if args.env.type == "sequential":
        from src.trainer.SequentialPPOTrainer import SequentialPPOTrainer
        TrainerClass = SequentialPPOTrainer
    else:
        from src.trainer.PPOTrainer import PPOTrainer
        TrainerClass = PPOTrainer
    trainer = TrainerClass(
        policy=model,
        env=env,
        args=args,
        dataset=dataset,
        device=device,
    )

    # Resume from checkpoint if available
    save_path = Path(args.save_path)
    ckpt_path = save_path / "final_model.pt"
    if not ckpt_path.exists():
        checkpoints = sorted(save_path.glob("checkpoint_*.pt"))
        if checkpoints:
            ckpt_path = checkpoints[-1]
    if ckpt_path.exists():
        print(f"Loading from {ckpt_path}")
        trainer.load_checkpoint(ckpt_path)

    trained_policy = trainer.train()
    return trained_policy, env, trainer.history
