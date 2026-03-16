from src.tasks.base import BaseArgs


def create_env(
    args: BaseArgs,
    device,
    graph_data,
):
    env_type = args.env.type
    match env_type:
        case "discrete":
            from src.envs.discrete import DiscreteGraphEnv
            return DiscreteGraphEnv(
                graph_data=graph_data,
                device=device,
                config=args.env,
            )
        case "sequential":
            from src.envs.sequential import SequentialGraphEnv
            return SequentialGraphEnv(
                graph_data=graph_data,
                device=device,
                config=args.env,
            )
        case _:
            raise NotImplementedError(
                f"Environment type '{env_type}' not implemented")
