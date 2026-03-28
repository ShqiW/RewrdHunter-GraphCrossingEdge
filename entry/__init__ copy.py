"""
Entry point configuration for investigation experiments.

This file should be created in your project's 'entry/' directory to define:
1. LIST_OF_ARGS: List of argument classes for your experiments
2. LIST_OF_CLUSTERS: List of cluster names (optional, for Slurm integration)
3. parameter2SlurmJob: Function to convert parameters to Slurm jobs

Example structure:
    entry/
    └── __init__.py  # This file

See examples/ctrl/entry/__init__.py for a complete example.
"""

from investigation.doeargs.args import ExampleArgs
from investigation import cwd
from typing import Type, List, Dict
from slurming.SlurmJob.job import SlurmJob
from slurming.Slurm.shellUtils import make_command
from pathlib import Path
import os
import environs
# Import Args classes
from src.tasks.discrete_ppo import DiscretePPOArgs
from src.tasks.sequential_ppo import SequentialPPOArgs

env = environs.Env()
env.read_env()

# host = "cardinal-cpu"
hostname = os.uname().nodename
LIST_OF_CLUSTERS: List[str] = ["example"]
for h in LIST_OF_CLUSTERS:
    if (h.startswith(hostname[0])):
        hostname = h
        break

# Switch this list when changing experiment type.
# Only ONE args class at a time (investigation requires uniform shared fields).
# Sequential placement (current): SequentialPPOArgs
# Discrete refinement (legacy):   DiscretePPOArgs
LIST_OF_ARGS: List[Type[ExampleArgs]] = [
    DiscretePPOArgs,
    SequentialPPOArgs,
]

CLUSTER_CONFIG = {
    "example-gpu": {
        "hostname": "example-cluster",
        "mem_per_cpu": 4,
        "min_cores": 1,
        "partition": ["example-gpu-partition"],
        "core_per_job": 1,
        "max_num_cpus": 48,
        "max_num_gpus": 1,
    },
    "example-cpu": {
        "hostname": "example-cluster",
        "mem_per_cpu": 4,
        "min_cores": 1,
        "partition": ["example-cpu-partition"],
        "core_per_job": 1,
        "max_num_cpus": 48,
        "max_num_gpus": 0,
    }
}


def parameter2SlurmJob(
    param_dict: Dict,
    num_cpus: int,
    num_gpus: int,
    script_path: str | Path,
) -> SlurmJob:
    """
    Convert a parameter dictionary to a SlurmJob instance.
    """
    # Determine cluster config based on hostname
    if hostname.startswith("e"):
        if num_gpus > 0:
            cluster_config = CLUSTER_CONFIG["example-gpu"]
            partition = cluster_config["partition"]
        else:
            cluster_config = CLUSTER_CONFIG["example-cpu"]
            partition = cluster_config["partition"]
    else:
        raise ValueError(
            f"Hostname {hostname} does not match any known cluster configurations."
        )

    entry_command_python = "poetry run python -m src.main"
    # Build training command
    train_command = make_command(
        entry_command_python,
        params1_dict={},
        params2_dict=param_dict | {
            "num_gpus": num_gpus,
            "num_cpus": num_cpus,
            "if_train": True,
        },
        connection=" ",
    )

    # Build training command
    plot_command = make_command(
        entry_command_python,
        params1_dict={},
        params2_dict=param_dict | {
            "num_gpus": num_gpus,
            "num_cpus": num_cpus,
            "if_plot": True,
        },
        connection=" ",
    )
    eval_command = make_command(
        entry_command_python,
        params1_dict={},
        params2_dict=param_dict | {
            "num_gpus": num_gpus,
            "num_cpus": num_cpus,
            "if_evaluate": True,
            "baseline_csv": cwd / "results/all_baselines.csv",
        },
        connection=" ",
    )
    job = SlurmJob(
        account=env.str("CLAUSTER_ACCOUNT"),
        content=[
            train_command,
            plot_command,
            eval_command,
        ],
        licenses={},
        modules=[],
        env_vars={},
        notify_email=[],
        aliases={},
        paths=[],
        sbatch_args={
            "partition": ",".join(partition),
        },
        interactive=False,
        output_storage=[
            "checkpoints/",
            "visualization/",
            "eval_results/",
            "history.csv",
        ],
        partition=",".join(partition),
        bashinit=[
            # "source $HOME/.bashenv",
        ],
    )

    return job
