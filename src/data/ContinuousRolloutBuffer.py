import torch


class ContinuousRolloutBuffer:
    """
    Rollout buffer for continuous action spaces (sequential placement).

    Pre-allocates fixed-size tensors for scalar fields; node_features remain
    a list because each step has a variable number of placed nodes.
    """

    def __init__(self, n_steps: int, device):
        self.n_steps = n_steps
        self.device = device
        self._alloc()

    def _alloc(self):
        self.node_features = [None
                              ] * self.n_steps  # list of torch.Tensor [ni, 3]
        self.actions = torch.zeros(self.n_steps,
                                   2,
                                   dtype=torch.float32,
                                   device=self.device)
        self.log_probs = torch.zeros(self.n_steps,
                                     dtype=torch.float32,
                                     device=self.device)
        self.rewards = torch.zeros(self.n_steps,
                                   dtype=torch.float32,
                                   device=self.device)
        self.values = torch.zeros(self.n_steps,
                                  dtype=torch.float32,
                                  device=self.device)
        self.dones = torch.zeros(self.n_steps,
                                 dtype=torch.bool,
                                 device=self.device)
        self.ptr = 0

    def reset(self):
        self.ptr = 0

    def add(self, node_features, action, log_prob, reward, value, done):
        i = self.ptr
        self.node_features[i] = node_features
        self.actions[i] = torch.as_tensor(action,
                                          dtype=torch.float32,
                                          device=self.device)
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.values[i] = value
        self.dones[i] = done
        self.ptr += 1

    def get_padded_features(self):
        """
        Pad all stored node_features into a single tensor (run once before PPO epochs).

        Returns:
            padded: [n, max_n, 3] — pre-padded features on device
            mask:   [n, max_n]    — True = padding position
        """
        n = self.ptr
        sizes = [self.node_features[i].shape[0] for i in range(n)]
        max_n = max(sizes) if sizes and max(sizes) > 0 else 1
        padded = torch.zeros(
            n,
            max_n,
            3,
            dtype=torch.float32,
            device=self.device,
        )
        mask = torch.ones(
            n,
            max_n,
            dtype=torch.bool,
            device=self.device,
        )
        for i in range(n):
            sz = sizes[i]
            if sz > 0:
                padded[i, :sz] = self.node_features[i]
                mask[i, :sz] = False
        return padded, mask

    def __len__(self):
        return self.ptr
