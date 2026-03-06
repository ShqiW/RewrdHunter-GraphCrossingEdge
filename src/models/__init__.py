# Models module

# New structure
from src.models.gnn import DiscreteGNNPolicy, GNNConfig

# Legacy imports (for backward compatibility)
try:
    from src.models.mlp_policy import MLPPolicy
    from src.models.gnn_policy import GNNPolicy
    from src.models.discrete_policy import DiscreteMLP
except ImportError:
    pass

__all__ = ["DiscreteGNNPolicy", "GNNConfig"]
