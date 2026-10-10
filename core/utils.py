"""Seeding, content fingerprints and checkpoint provenance."""

import fcntl
import hashlib
import json
import os
import random
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


# reproducibility
def seed_everything(init_seed: int) -> None:
    """Seed Python, numpy and torch (incl. MPS) and force deterministic algorithms."""
    random.seed(init_seed)
    np.random.seed(init_seed)
    torch.manual_seed(init_seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(init_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=False)

def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g

# provenance
def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()

def file_digest(path: str | Path) -> str:
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def fingerprint(dataset_dir: str = 'dataset') -> dict:
    """Hashes of the classification recordings and of the code (core/, models/), independent of their location.

    :param dataset_dir: dataset directory with the class subfolders
    :return: {'dataset_sha256', 'code_sha256'}
    """
    root = Path(dataset_dir)
    files = {str(p.relative_to(root)): file_digest(p)
             for cls in ('bradypnea', 'eupnea', 'tachypnea') for p in sorted((root / cls).glob('*.dat'))}
    if not files:
        raise ValueError(f'No classification recordings in {root}')
    source = Path(__file__).resolve().parents[1]
    code = {str(p.relative_to(source)): file_digest(p)
            for folder in ('core', 'models') for p in sorted((source / folder).glob('*.py'))}
    return {'dataset_sha256': digest(files), 'code_sha256': digest(code)}

@contextmanager
def artifact_lock(path: str | Path) -> Iterator[None]:
    # advisory lock: parallel tasks never write the same artifact at once
    lock = Path(str(path) + '.lock')
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

def atomic_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

def reserve_training(path: str, config: dict, inputs: dict) -> dict:
    """Record the identity of a checkpoint (data, code, training settings) and reject reuse of its path for another.

    :param path: checkpoint path; the record is stored next to it as <path>.json
    :param config: training settings
    :param inputs: data and code hashes (fingerprint)
    :return: the identity record
    """
    settings = {k: v for k, v in config.items() if k not in ('dataset_dir', 'log_every')}
    record = {**inputs, 'training_config': settings}
    record['training_id'] = digest(record)
    sidecar = Path(path + '.json')
    with artifact_lock(path):
        if sidecar.exists():
            if json.loads(sidecar.read_text()) != record:
                raise ValueError(f'Checkpoint identity conflict at {path}; use a new run directory or hp_tag')
        elif Path(path).exists():
            raise ValueError(f'Unversioned checkpoint at {path}; use a new run directory')
        atomic_json(sidecar, record)
    return record
