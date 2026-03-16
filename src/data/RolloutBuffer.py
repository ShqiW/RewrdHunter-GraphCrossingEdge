class RolloutBuffer:
    """Rollout buffer for discrete action spaces (original)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.node_features = []
        self.edge_indices = []
        self.edge_attrs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, node_features, edge_index, edge_attr, action, log_prob,
            reward, value, done):
        self.node_features.append(node_features)
        self.edge_indices.append(edge_index)
        self.edge_attrs.append(edge_attr)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def __len__(self):
        return len(self.rewards)
