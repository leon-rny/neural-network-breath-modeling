"""Content fingerprints and atomic artifact writes for reproducible experiments."""

import fcntl
import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def digest(value: object) -> str:
    """Hash JSON-compatible configuration data deterministically."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()

def file_digest(path: str | Path) -> str:
    """Hash a file without loading large artifacts into memory."""
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def fingerprint(dataset_dir: str = 'dataset') -> dict:
    """Identify classification inputs and scientific source independently of their location."""
    root = Path(dataset_dir)
    files = {str(p.relative_to(root)): file_digest(p)
             for cls in ('bradypnea', 'eupnea', 'tachypnea') for p in sorted((root / cls).glob('*.dat'))}
    if not files:
        raise ValueError(f'No classification recordings in {root}')
    source = Path(__file__).resolve().parents[1]
    code = {str(p.relative_to(source)): file_digest(p)
            for folder in ('core', 'models', 'ablations') for p in sorted((source / folder).glob('*.py'))}
    return {'dataset_sha256': digest(files), 'code_sha256': digest(code)}

@contextmanager
def artifact_lock(path: str | Path) -> Iterator[None]:
    """Serialize writers to one artifact using an advisory filesystem lock."""
    lock = Path(str(path) + '.lock')
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

def atomic_json(path: str | Path, value: object) -> None:
    """Replace a JSON artifact only after writing the complete payload."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

def reserve_training(path: str, config: dict, inputs: dict) -> dict:
    """Reject reuse of a checkpoint filename for different data, code or training settings."""
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
