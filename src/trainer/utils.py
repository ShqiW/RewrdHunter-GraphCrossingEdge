from src.tasks.base import BaseArgs

from src.tasks.discrete_ppo import DiscretePPOArgs
from src.tasks.sequential_ppo import SequentialPPOArgs, SequentialRefinementArgs
from src.tasks.continuous_ppo import ContinuousPPOArgs


def create_env(
    args: BaseArgs,
    device,
    graph_data,
):
    match args:
        case DiscretePPOArgs():
            from src.envs.discrete import DiscreteGraphEnv
            return DiscreteGraphEnv(
                graph_data=graph_data,
                device=device,
                config=args.env,
            )
        case ContinuousPPOArgs():
            from src.envs.continuous import ContinuousGraphEnv
            return ContinuousGraphEnv(
                graph_data=graph_data,
                device=device,
                config=args.env,
            )
        case SequentialRefinementArgs():
            from src.envs.refinement import RefinementGraphEnv
            return RefinementGraphEnv(
                graph_data=graph_data,
                device=device,
                config=args.env,
            )
        case SequentialPPOArgs():
            from src.envs.sequential import SequentialGraphEnv
            return SequentialGraphEnv(
                graph_data=graph_data,
                device=device,
                config=args.env,
            )
        case _:
            raise NotImplementedError(
                f"Environment type '{args.name}' not implemented")
