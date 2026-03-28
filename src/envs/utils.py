import numpy as np
import networkx as nx


def get_initial_layout(initial_layout, num_nodes, neato_coords,
                       graph) -> np.ndarray:
    """Return raw (unnormalized) initial coordinates."""
    match initial_layout:
        case "random":
            return np.random.rand(num_nodes, 2).astype(np.float32)
        case "neato":
            return neato_coords
        case "sfdp":
            pos = nx.nx_agraph.graphviz_layout(graph, prog="sfdp")
            return np.array(
                [[pos[v][0], pos[v][1]] for v in graph.nodes()],
                dtype=np.float32,
            )
        case "spring":
            pos = nx.spring_layout(graph)
            return np.array(
                [[pos[v][0], pos[v][1]] for v in graph.nodes()],
                dtype=np.float32,
            )
        case _:
            assert False, f"Unknown initial layout: {initial_layout}"
