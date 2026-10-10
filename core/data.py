"""Dataset loading, nested-LOSO and k-fold splits, and the training spaces of the generators."""

import os
import re

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset

CLASSES = ['bradypnea', 'eupnea', 'tachypnea']
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}
PARTICIPANTS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'p']
PARTICIPANT_TO_IDX = {p: i for i, p in enumerate(PARTICIPANTS)}
TARGET_LEN = 36

def _to_target_len(t: np.ndarray, h: np.ndarray, temp: np.ndarray, n: int = TARGET_LEN) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # resample a trial to n points over its own time span (identity when it already has n samples)
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
    """Load all .dat trials of the class subfolders.

    :param dataset_dir: dataset directory with the class subfolders
    :return: one row per trial: time, humidity, temperature, class, participant, region, trial_num, filename
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
    """Participants of one nested-LOSO task. Outer task: test = subjects[outer], validation = the next participant
    (cyclic), training = the rest. Inner task: the same among the participants without the outer test participant.

    :param subjects: sorted participant ids of the region
    :param outer: 0-indexed outer test participant
    :param inner: 0-indexed inner test participant among the other participants, -1 = outer task
    :return: (training participants, validation participant, test participant)
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
    masks = (df['participant'].isin(train), df['participant'] == val, df['participant'] == test)
    return tuple(df[m].reset_index(drop=True) for m in masks)

def kfold_split(df: pd.DataFrame, fold: int, n_folds: int = 5, split_seed: int = 42, val_size: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Thesis k-fold protocol: class-stratified n_folds split of the trials, validation = val_size of the training folds
    (class-stratified). Training, validation and test trials come from the same participants.

    :param df: trials of one region
    :param fold: 0-indexed test fold
    :param split_seed: fixes both splits
    :return: (training, validation, test) trials
    """
    if not 0 <= fold < n_folds:
        raise ValueError(f'fold must be in [0, {n_folds}), got {fold}')
    df = df.reset_index(drop=True)
    train_idx, test_idx = list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=split_seed).split(df, df['class']))[fold]
    pool = df.iloc[train_idx].reset_index(drop=True)
    tr, va = train_test_split(pool, test_size=val_size, stratify=pool['class'], random_state=split_seed)
    return tr.reset_index(drop=True), va.reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)

class BreathDataset(Dataset):
    """z-scored trials (humidity, temperature), the CVAE's training space.

    :param stats: normalisation stats, computed from this data if None (pass the training stats to val / test)
    :param alpha: SD of the Gaussian jitter (0 = none)
    :param n_copies: repetitions of each trial
    """

    def __init__(self, dataframe: pd.DataFrame, stats: dict | None = None, alpha: float = 0.0, n_copies: int = 1) -> None:
        self.records = dataframe.to_dict('records')
        self.stats = stats if stats is not None else self._compute_stats()
        self.alpha = alpha
        self.n_copies = max(n_copies, 1)

    def _compute_stats(self) -> dict:
        h_all = np.concatenate([r['humidity'] for r in self.records])
        t_all = np.concatenate([r['temperature'] for r in self.records])
        return {'mean': np.array([h_all.mean(), t_all.mean()], dtype=np.float32),
                'std': np.maximum(np.array([h_all.std(), t_all.std()], dtype=np.float32), 1e-8)}

    def __len__(self) -> int:
        return len(self.records) * self.n_copies

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        # (signal (2, 36), class index, participant index); idx wraps around the records for n_copies > 1
        r = self.records[idx % len(self.records)]
        h = (r['humidity'] - self.stats['mean'][0]) / (self.stats['std'][0] + 1e-8)
        t = (r['temperature'] - self.stats['mean'][1]) / (self.stats['std'][1] + 1e-8)
        signal = torch.tensor(np.stack([h, t], axis=0), dtype=torch.float32)
        if self.alpha > 0:
            signal = signal + self.alpha * torch.randn_like(signal)
        return signal, CLASS_TO_IDX[r['class']], PARTICIPANT_TO_IDX[r['participant']]

class PhysicsInformedDataset(Dataset):
    """Baseline-corrected trials (first 5 samples) scaled per channel by the baseline-corrected training SD, the
    PI-CVAE's training space. Parameters as BreathDataset."""

    def __init__(self, dataframe: pd.DataFrame, stats: dict | None = None, alpha: float = 0.0, n_copies: int = 1) -> None:
        self.records = dataframe.to_dict('records')
        self.stats = stats if stats is not None else self._compute_stats()
        self.alpha = alpha
        self.n_copies = max(n_copies, 1)

    def _compute_stats(self) -> dict:
        h_bc = [r['humidity'] - np.mean(r['humidity'][:5]) for r in self.records]
        t_bc = [r['temperature'] - np.mean(r['temperature'][:5]) for r in self.records]
        return {'h_bcstd': max(float(np.concatenate(h_bc).std()), 1e-8),
                't_bcstd': max(float(np.concatenate(t_bc).std()), 1e-8)}

    def __len__(self) -> int:
        return len(self.records) * self.n_copies

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        # (signal (2, 36), class index, participant index); idx wraps around the records for n_copies > 1
        r = self.records[idx % len(self.records)]
        h = (r['humidity'] - np.mean(r['humidity'][:5])) / self.stats['h_bcstd']
        t = (r['temperature'] - np.mean(r['temperature'][:5])) / self.stats['t_bcstd']
        signal = torch.tensor(np.stack([h, t], axis=0), dtype=torch.float32)
        if self.alpha > 0:
            signal = signal + self.alpha * torch.randn_like(signal)
        return signal, CLASS_TO_IDX[r['class']], PARTICIPANT_TO_IDX[r['participant']]
