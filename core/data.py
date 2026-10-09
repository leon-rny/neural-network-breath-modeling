import os
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CLASSES = ['bradypnea', 'eupnea', 'tachypnea']
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}
PARTICIPANTS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'p']
PARTICIPANT_TO_IDX = {p: i for i, p in enumerate(PARTICIPANTS)}
TARGET_LEN = 36

def _to_target_len(t: np.ndarray, h: np.ndarray, temp: np.ndarray, n: int = TARGET_LEN):
    """Resample a single trial to n points over its own time span (identity when already n samples)."""
    arrays = (t, h, temp)
    if any(np.asarray(a).ndim != 1 for a in arrays) or len(t) < 2 or not len(t) == len(h) == len(temp):
        raise ValueError('Each trial needs matching one-dimensional time, humidity and temperature arrays')
    if any(not np.isfinite(a).all() for a in arrays) or np.any(np.diff(t) <= 0):
        raise ValueError('Trial values must be finite and timestamps strictly increasing')
    if len(h) == n:
        return t, h, temp
    tg = np.linspace(t[0], t[-1], n)
    return tg, np.interp(tg, t, h), np.interp(tg, t, temp)

def load_dataset(dataset_dir: str = 'dataset') -> pd.DataFrame:
    """
    Load all .dat trials from the dataset directory into a DataFrame.

    :param dataset_dir: Path to the dataset directory containing class subfolders.
    :return: A pandas DataFrame with columns time, humidity, temperature, class,
        participant, region, trial_num, filename (one row per trial).
    """
    records = []
    for cls in CLASSES:
        folder = os.path.join(dataset_dir, cls)
        for fname in sorted(os.listdir(folder)):
            if not fname.endswith('.dat'):
                continue
            m = re.match(r'^(([abcdefgp])_)?(mouth|nose)_trial_(\d+)\.dat$', fname)
            if m is None:
                continue
            df = pd.read_csv(os.path.join(folder, fname))
            t, h, temp = _to_target_len(df['Time'].values / 1000, df['Humidity'].values, df['Temperature'].values)
            records.append({'time': t,
                            'humidity': h,
                            'temperature': temp,
                            'class': cls,
                            'participant': m.group(2) if m.group(2) else 'a',
                            'region': m.group(3),
                            'trial_num': int(m.group(4)),
                            'filename': fname})
    return pd.DataFrame(records)

def nested_split(subjects: list[str], outer: int, inner: int = -1) -> tuple[list[str], str, str]:
    """Participants of one nested-LOSO task: (training participants, validation participant, test participant).

    Outer task (inner = -1): test = subjects[outer], validation = the next participant (cyclic), training = the rest.
    Inner task: the outer test participant is removed; among the remaining participants test = pool[inner],
    validation = the next one (cyclic), training = the rest. The outer test participant is never part of an inner task.

    :param subjects: sorted participant ids of the region.
    :param outer: 0-indexed outer test participant.
    :param inner: 0-indexed inner test participant among the other participants, -1 for the outer task.
    """
    if not 0 <= outer < len(subjects) or not -1 <= inner < len(subjects) - 1:
        raise ValueError(f'outer={outer}, inner={inner} invalid for {len(subjects)} participants')
    pool = subjects if inner < 0 else [s for s in subjects if s != subjects[outer]]
    k = outer if inner < 0 else inner
    test, val = pool[k], pool[(k + 1) % len(pool)]
    if len(pool) < 3:
        raise ValueError(f'a split needs >= 3 participants (train / val / test), got {len(pool)}')
    return [s for s in pool if s not in (test, val)], val, test

def split_frames(df: pd.DataFrame, train: list[str], val: str, test: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(train, val, test) trial frames of the given participants, each in the order of df."""
    pick = lambda m: df[m].reset_index(drop=True)  # noqa: E731
    return pick(df['participant'].isin(train)), pick(df['participant'] == val), pick(df['participant'] == test)

class BreathDataset(Dataset):
    """PyTorch Dataset of z-score normalised breath signals (humidity, temperature), the CVAE's training space."""

    def __init__(self, dataframe: pd.DataFrame, stats: dict | None = None, alpha: float = 0.0, n_copies: int = 1) -> None:
        """
        :param dataframe: DataFrame of trials as returned by load_dataset.
        :param stats: Normalisation stats dict, computed from this data if None (pass the
            train stats to val/test to avoid leakage).
        :param alpha: Std of Gaussian noise added to the signal (0 disables augmentation).
        :param n_copies: Number of times each trial is repeated (>=1) for noise augmentation.
        """
        self.records = dataframe.to_dict('records')
        self.stats = stats if stats is not None else self._compute_stats()
        self.alpha = alpha
        self.n_copies = max(n_copies, 1)

    def _compute_stats(self) -> dict:
        """Per-channel [humidity, temperature] mean and std ((2,) float32 arrays) over all samples."""
        h_all = np.concatenate([r['humidity'] for r in self.records])
        t_all = np.concatenate([r['temperature'] for r in self.records])
        return {'mean': np.array([h_all.mean(), t_all.mean()], dtype=np.float32),
                'std': np.maximum(np.array([h_all.std(), t_all.std()], dtype=np.float32), 1e-8)}

    def __len__(self) -> int:
        """Number of records times n_copies."""
        return len(self.records) * self.n_copies

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        """Return one trial, z-score normalised (optionally noise-augmented).

        :param idx: Sample index (wraps modulo the record count for n_copies > 1).
        :return: (signal, label, participant): (2, 36) float32 tensor [humidity, temperature] and the integer
            class / participant indices.
        """
        r = self.records[idx % len(self.records)]
        h = (r['humidity'] - self.stats['mean'][0]) / (self.stats['std'][0] + 1e-8)
        t = (r['temperature'] - self.stats['mean'][1]) / (self.stats['std'][1] + 1e-8)
        signal = torch.tensor(np.stack([h, t], axis=0), dtype=torch.float32)
        if self.alpha > 0:
            signal = signal + self.alpha * torch.randn_like(signal)
        return signal, CLASS_TO_IDX[r['class']], PARTICIPANT_TO_IDX[r['participant']]

class PhysicsInformedDataset(Dataset):
    """Dataset of the PI-CVAE: baseline-corrected signals (first 5 samples) scaled per channel by the baseline-corrected
    training SD (stdscale)."""

    def __init__(self, dataframe: pd.DataFrame, stats: dict | None = None, alpha: float = 0.0, n_copies: int = 1) -> None:
        """
        :param dataframe: DataFrame of trials as returned by load_dataset.
        :param stats: Scaling stats dict, computed from this data if None (pass the
            train stats to val/test to avoid leakage).
        :param alpha: Std of Gaussian noise added to the signal (0 disables augmentation).
        :param n_copies: Number of times each trial is repeated (>=1) for noise augmentation.
        """
        self.records = dataframe.to_dict('records')
        self.stats = stats if stats is not None else self._compute_stats()
        self.alpha = alpha
        self.n_copies = max(n_copies, 1)

    def _compute_stats(self) -> dict:
        """Per-channel SD of the baseline-corrected training signals (h_bcstd, t_bcstd)."""
        h_bc = [r['humidity'] - np.mean(r['humidity'][:5]) for r in self.records]
        t_bc = [r['temperature'] - np.mean(r['temperature'][:5]) for r in self.records]
        return {'h_bcstd': max(float(np.concatenate(h_bc).std()), 1e-8),
                't_bcstd': max(float(np.concatenate(t_bc).std()), 1e-8)}

    def __len__(self) -> int:
        """Number of records times n_copies."""
        return len(self.records) * self.n_copies

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        """Return one trial, baseline-corrected and scaled (optionally noise-augmented).

        :param idx: Sample index (wraps modulo the record count for n_copies > 1).
        :return: (signal, label, participant): (2, 36) float32 tensor [humidity, temperature] and the integer
            class / participant indices.
        """
        r = self.records[idx % len(self.records)]
        h = (r['humidity'] - np.mean(r['humidity'][:5])) / self.stats['h_bcstd']
        t = (r['temperature'] - np.mean(r['temperature'][:5])) / self.stats['t_bcstd']
        signal = torch.tensor(np.stack([h, t], axis=0), dtype=torch.float32)
        if self.alpha > 0:
            signal = signal + self.alpha * torch.randn_like(signal)
        return signal, CLASS_TO_IDX[r['class']], PARTICIPANT_TO_IDX[r['participant']]
