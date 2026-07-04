import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
from torch.utils.data import Dataset

CLASSES = ['bradypnea', 'eupnea', 'tachypnea']
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}
PARTICIPANTS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'p']
PARTICIPANT_TO_IDX = {p: i for i, p in enumerate(PARTICIPANTS)}
TARGET_LEN = 36

def _to_target_len(t: np.ndarray, h: np.ndarray, temp: np.ndarray, n: int = TARGET_LEN):
    """Resample a single trial to n points over its own time span (identity when already n samples)."""
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
            m = re.match(r'^(([bcdefgp])_)?(mouth|nose)_trial_(\d+)\.dat$', fname)
            if m is None:
                continue
            df = pd.read_csv(os.path.join(folder, fname))
            t, h, temp = _to_target_len(df['Time'].values/1000, df['Humidity'].values, df['Temperature'].values)
            records.append({'time': t,
                            'humidity': h,
                            'temperature': temp,
                            'class': cls,
                            'participant': m.group(2) if m.group(2) else 'a',
                            'region': m.group(3),
                            'trial_num': int(m.group(4)),
                            'filename': fname})
            
    return pd.DataFrame(records)

def split_dataset(df: pd.DataFrame, val_size: float = 0.1, test_size: float = 0.1, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/val/test split by class label.

    :param df: DataFrame containing the dataset.
    :param val_size: Proportion of the dataset to include in the validation split.
    :param test_size: Proportion of the dataset to include in the test split.
    :param random_state: Random seed for reproducibility.
    :return: A tuple of (train_df, val_df, test_df) DataFrames.
    """
    df_train_val, df_test = train_test_split(df, test_size=test_size, stratify=df['class'], random_state=random_state)
    
    val_relative = val_size/(1-test_size)
    df_train, df_val = train_test_split(df_train_val, test_size=val_relative, stratify=df_train_val['class'], random_state=random_state)
    
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)

def kfold_split_dataset(df: pd.DataFrame, split_seed: int, fold: int, n_folds: int = 5, val_size: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified k-fold split returning the (train, val, test) DataFrames for a given fold.

    The outer split (test) is determined by k-fold partitioning: each fold defines
    a unique test set, and the remaining samples form the training pool. The training
    pool is then further split into train and val for hyperparameter tuning and
    early stopping.

    :param df: DataFrame containing the dataset.
    :param split_seed: Random seed for both the outer k-fold and the inner train/val split. Should be FIXED across all configurations for paired comparisons.
    :param fold: Which fold to return (0..n_folds-1).
    :param n_folds: Number of outer folds.
    :param val_size: Proportion of the training pool to use as validation.
    :return: A tuple of (train_df, val_df, test_df) DataFrames.
    """
    if not 0 <= fold < n_folds:
        raise ValueError(f'fold must be in [0, {n_folds}), got {fold}')

    df = df.reset_index(drop=True)
    y = df['class'].values

    # outer k-fold defines test
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=split_seed)
    splits = list(skf.split(df, y))
    train_idx, test_idx = splits[fold]

    df_trainfull = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # inner split: carve val out of trainfull
    df_train, df_val = train_test_split(df_trainfull, test_size=val_size, stratify=df_trainfull['class'], random_state=split_seed)

    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test

def n_loso_folds(df: pd.DataFrame) -> int:
    """Number of LOSO folds = number of distinct subjects. Single source of truth so callers
    (and the SLURM array) derive the fold count from the data rather than a hardcoded value."""
    return df['participant'].nunique()

def loso_split_dataset(df: pd.DataFrame, fold: int, val_fold: int | None = None, val_size: float = 0.15, split_seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Leave-one-subject-out split returning (train, val, test) DataFrames.

    :param df: DataFrame containing the dataset.
    :param fold: 0-indexed test-subject position in sorted(participant), in [0, n_subjects).
    :param val_fold: 0-indexed val-subject position, defaults to the neighbour of `fold`.
    :param val_size: unused (kept for signature parity with kfold_split_dataset).
    :param split_seed: unused (subject partition is deterministic), kept for parity.
    :return: A tuple of (train_df, val_df, test_df) DataFrames.
    """
    subjects = sorted(df['participant'].unique())
    n = len(subjects)
    if n < 3:
        raise ValueError(f'LOSO needs >=3 subjects (train/val/test), got {n}')
    if not 0 <= fold < n:
        raise ValueError(f'fold must be in [0, {n}) for {n} subjects, got {fold}')
    test_subj = subjects[fold]
    val_subj = subjects[val_fold] if val_fold is not None else subjects[(fold + 1) % n]
    if val_subj == test_subj:
        raise ValueError(f'val subject must differ from test subject (both {test_subj!r})')

    df_test = df[df['participant'] == test_subj]
    df_val = df[df['participant'] == val_subj]
    df_train = df[~df['participant'].isin([test_subj, val_subj])]
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)

def loso_split_final(df: pd.DataFrame, test_subject_idx: int, exclude_subject_idxs: tuple = (), val_size: float = 0.15, split_seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Nested-LOSO building block returning (train, earlystop_val, test) DataFrames.

    :param df: DataFrame containing the dataset.
    :param test_subject_idx: 0-indexed test-subject position in sorted(participant).
    :param exclude_subject_idxs: 0-indexed subject positions to drop from the pool entirely.
    :param val_size: trial-level early-stop val fraction of the training pool.
    :param split_seed: seed for the trial-level train/val split.
    :return: (train_df, earlystop_val_df, test_df).
    """
    subjects = sorted(df['participant'].unique())
    n = len(subjects)
    if not 0 <= test_subject_idx < n:
        raise ValueError(f'test_subject_idx must be in [0, {n}) for {n} subjects, got {test_subject_idx}')
    test_subj = subjects[test_subject_idx]
    excl = {subjects[i] for i in exclude_subject_idxs}
    if test_subj in excl:
        raise ValueError(f'test subject {test_subj!r} cannot also be in exclude {sorted(excl)}')
    pool_subjects = [s for s in subjects if s != test_subj and s not in excl]
    if len(pool_subjects) < 2:
        raise ValueError(f'nested LOSO needs >=2 training-pool subjects (after removing test + exclude), '
                         f'got {len(pool_subjects)} from {n} total. This is a degenerate run, not a real '
                         f'result - record more subjects (>=4 total) before trusting nested-LOSO numbers.')

    df_test = df[df['participant'] == test_subj]
    df_pool = df[df['participant'].isin(pool_subjects)]
    df_train, df_val = train_test_split(df_pool, test_size=val_size, stratify=df_pool['class'], random_state=split_seed)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # leakage assertions
    train_subs, val_subs = set(df_train['participant']), set(df_val['participant'])
    assert val_subs <= set(pool_subjects), f'earlystop_val leaked outside training pool: {val_subs - set(pool_subjects)}'
    assert test_subj not in train_subs and test_subj not in val_subs, f'test subject {test_subj!r} leaked into train/val'
    assert not (excl & train_subs) and not (excl & val_subs), f'excluded subject leaked into train/val: {excl & (train_subs | val_subs)}'
    return df_train, df_val, df_test

def loso_path_tag(loso_trial_val: bool, exclude_subject_idxs: tuple = ()) -> str:
    """Path-only marker namespacing nested-LOSO artifacts so they never collide"""
    if not loso_trial_val:
        return ''
    tag = '_nested'
    if exclude_subject_idxs:
        tag += '_x' + '-'.join(str(i) for i in sorted(exclude_subject_idxs))
    return tag

def subset_tag(include_subjects: tuple = ()) -> str:
    """Path/dedup marker for a participant-subset run, empty -> '' so full-pool runs stay byte-identical."""
    if not include_subjects:
        return ''
    return '_sub' + ''.join(sorted(str(s) for s in include_subjects))

def cir_marker(cir_tag: str = '') -> str:
    """run_id/summary marker for an alternate fitted CIR channel (e.g. '300s'), empty = active default channel."""
    return '' if not cir_tag else f'_cir{cir_tag}'

def prep_marker(preprocessing: str = 'raw') -> str:
    """run_id/summary marker for a signal-preprocessing variant before feature extraction, raw = current default."""
    return '' if preprocessing in ('', 'raw') else f'_prep{preprocessing}'

def phys_prep_marker(phys_prep: str = 'peakscale') -> str:
    """run_id/summary marker for the physics-generator input normalization (PhysicsInformedDataset), peakscale = current default."""
    return '' if phys_prep in ('', 'peakscale') else f'_pp{phys_prep}'

def get_split(df: pd.DataFrame, cv_mode: str, fold: int, n_folds: int, split_seed: int = 42, val_size: float = 0.15, exclude_subjects: tuple = (), loso_trial_val: bool = False, include_subjects: tuple = ()) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Dispatch to the k-fold or LOSO splitter."""
    if include_subjects:
        df = df[df['participant'].isin(include_subjects)].reset_index(drop=True)
    if cv_mode == 'loso':
        if loso_trial_val:
            return loso_split_final(df, test_subject_idx=fold, exclude_subject_idxs=tuple(exclude_subjects), val_size=val_size, split_seed=split_seed)
        return loso_split_dataset(df, fold=fold, val_size=val_size, split_seed=split_seed)
    if cv_mode == 'kfold':
        return kfold_split_dataset(df, split_seed=split_seed, fold=fold, n_folds=n_folds, val_size=val_size)
    raise ValueError(f"cv_mode must be 'kfold' or 'loso', got {cv_mode!r}")

class BreathDataset(Dataset):
    """PyTorch Dataset of z-score normalised breath signals (humidity, temperature)."""

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
        """Compute per-channel [humidity, temperature] normalisation stats.

        :return: Dict with keys max, min, mean, std, each a (2,) float32 array.
        """
        h_all = np.concatenate([r['humidity'] for r in self.records])
        t_all = np.concatenate([r['temperature'] for r in self.records])
        return {'max': np.array([h_all.max(), t_all.max()], dtype=np.float32),
                'min': np.array([h_all.min(), t_all.min()], dtype=np.float32),
                'mean': np.array([h_all.mean(), t_all.mean()], dtype=np.float32),
                'std': np.array([h_all.std(), t_all.std()], dtype=np.float32)}

    def __len__(self) -> int:
        """Number of records times n_copies."""
        return len(self.records) * self.n_copies

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """Return one trial, z-score normalised (optionally noise-augmented).

        :param idx: Sample index (wraps modulo the record count for n_copies > 1).
        :return: (signal, time, label, participant) where signal is a (2, 36) float32
            tensor [humidity, temperature], time is a (36,) float32 tensor of seconds since
            onset, and label/participant are integer class/participant indices.
        """
        r = self.records[idx % len(self.records)]

        # z-score normalisation
        h = (r['humidity'] - self.stats['mean'][0]) / (self.stats['std'][0] + 1e-8)
        t = (r['temperature'] - self.stats['mean'][1]) / (self.stats['std'][1] + 1e-8)

        # get all infos
        signal = torch.tensor(np.stack([h, t], axis=0), dtype=torch.float32)
        if self.alpha > 0:
            signal = signal + self.alpha * torch.randn_like(signal)
        time = torch.tensor(r['time'] - r['time'][0],  dtype=torch.float32)
        label = CLASS_TO_IDX[r['class']]
        participant = PARTICIPANT_TO_IDX[r['participant']]

        return signal, time, label, participant

class PhysicsInformedDataset(Dataset):
    """Dataset for the physics generator: baseline-corrected, rise-from-zero signals with onset index."""

    def __init__(self, dataframe: pd.DataFrame, stats: dict | None = None, alpha: float = 0.0, n_copies: int = 1, phys_prep: str = 'peakscale') -> None:
        """
        :param dataframe: DataFrame of trials as returned by load_dataset.
        :param stats: Normalisation stats dict, computed from this data if None (pass the
            train stats to val/test to avoid leakage).
        :param alpha: Std of Gaussian noise added to the signal (0 disables augmentation).
        :param n_copies: Number of times each trial is repeated (>=1) for noise augmentation.
        :param phys_prep: Input-normalization mode, overridden by stats['phys_prep'] when stats are passed.
        """
        self.records = dataframe.to_dict('records')
        self.phys_prep = phys_prep   # input-normalization mode, overridden by stats['phys_prep'] when stats are passed
        self.stats = stats if stats is not None else self._compute_stats()
        self.alpha = alpha
        self.n_copies = max(n_copies, 1)

    def _compute_stats(self) -> dict:
        """Compute baseline-corrected scaling stats for the physics-prep variants.

        :return: Dict with mean and std ((2,) float32 arrays), the float peak-deviation
            scales h_scale/t_scale, the shared and std-based scales shared_scale/h_bcstd/
            t_bcstd, and the phys_prep mode string.
        """
        h_peak_devs, t_peak_devs, h_bc, t_bc = [], [], [], []
        for r in self.records:
            hc = r['humidity'] - np.mean(r['humidity'][:5])
            tc = r['temperature'] - np.mean(r['temperature'][:5])
            h_peak_devs.append(hc.max())
            t_peak_devs.append(tc.max())
            h_bc.append(hc)
            t_bc.append(tc)
        h_scale = float(np.max(h_peak_devs) * 1.2)
        t_scale = float(np.max(t_peak_devs) * 1.2)
        shared_scale = float(max(h_scale, t_scale))
        h_bcstd = float(np.concatenate(h_bc).std())
        t_bcstd = float(np.concatenate(t_bc).std())

        h_all = np.concatenate([r['humidity'] for r in self.records])
        t_all = np.concatenate([r['temperature'] for r in self.records])
        return {'mean': np.array([h_all.mean(), t_all.mean()], dtype=np.float32),
                'std': np.array([h_all.std(), t_all.std()], dtype=np.float32),
                'h_scale': h_scale, 't_scale': t_scale,
                'shared_scale': shared_scale, 'h_bcstd': h_bcstd, 't_bcstd': t_bcstd,
                'phys_prep': self.phys_prep}

    def __len__(self) -> int:
        """Number of records times n_copies."""
        return len(self.records) * self.n_copies

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
        """Return one trial, baseline-corrected and phys_prep-scaled (optionally noise-augmented).

        :param idx: Sample index (wraps modulo the record count for n_copies > 1).
        :return: (signal, time, label, participant, onset_idx) where signal is a (2, 36)
            float32 tensor [humidity, temperature], time is a (36,) float32 tensor of seconds
            since onset, label/participant are integer class/participant indices, and onset_idx
            is the integer humidity-rise onset sample index.
        """
        r = self.records[idx % len(self.records)]

        # baseline correct
        baseline_h = np.mean(r['humidity'][:5])
        h = r['humidity'] - baseline_h

        # onset detection
        baseline_std = np.std(r['humidity'][:5])
        threshold = baseline_h + 3.0 * baseline_std
        onset_idx = 5
        for i in range(len(r['humidity'])):
            if r['humidity'][i] > threshold:
                onset_idx = i
                break
        
        # normalize - both channels baseline-corrected
        mode = self.stats.get('phys_prep', 'peakscale')
        t_bc = r['temperature'] - np.mean(r['temperature'][:5])
        if mode == 'shared': # one Lewis-coupled scale for both channels
            sc = self.stats['shared_scale']
            h = h / sc
            t = t_bc / sc
        elif mode == 'stdscale': # per-channel baseline-corrected std
            h = h / self.stats['h_bcstd']
            t = t_bc / self.stats['t_bcstd']
        elif 't_scale' in self.stats: # peakscale (default): per-channel peak-dev scale
            h = h / self.stats['h_scale']
            t = t_bc / self.stats['t_scale']
        else: # backward-compat: pre-t_scale checkpoint
            h = h / self.stats['h_scale']
            t = (r['temperature'] - self.stats['mean'][1]) / self.stats['std'][1]
        
        # get all infos
        signal = torch.tensor(np.stack([h, t], axis=0), dtype=torch.float32)
        if self.alpha > 0:
            signal = signal + self.alpha * torch.randn_like(signal)
        time = torch.tensor(r['time'] - r['time'][0],  dtype=torch.float32)
        label = CLASS_TO_IDX[r['class']]
        participant = PARTICIPANT_TO_IDX[r['participant']]

        return signal, time, label, participant, onset_idx