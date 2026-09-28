import os
import random

import numpy as np
import torch


# reproducibility
def seed_everything(init_seed: int) -> None:
    """Seed Python, numpy and torch (incl. MPS) RNGs and force deterministic algorithms for reproducibility."""
    random.seed(init_seed)
    np.random.seed(init_seed)
    torch.manual_seed(init_seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(init_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=False)

def seed_worker(_worker_id):
    """Seed numpy and Python RNGs inside a DataLoader worker from torch's per-worker seed."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_generator(seed: int) -> torch.Generator:
    """Return a torch.Generator seeded with `seed` for reproducible DataLoader shuffling."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
