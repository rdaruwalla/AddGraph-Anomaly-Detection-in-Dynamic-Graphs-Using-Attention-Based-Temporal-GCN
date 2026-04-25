from dataclasses import dataclass


@dataclass
class AddGraphConfig:
    num_nodes: int
    hidden_dim: int = 64
    gcn_layers: int = 3
    window_size: int = 3
    beta: float = 1.0
    mu: float = 0.3
    margin: float = 0.6
    lr: float = 1e-3
    weight_decay: float = 5e-7
    dropout: float = 0.2
    filter_training_pairs: bool = True
    device: str = "cpu"
    seed: int = 42
