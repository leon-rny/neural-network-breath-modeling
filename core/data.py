import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

CLASSES = ['bradypnea', 'eupnea', 'tachypnea']
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}
PARTICIPANTS = ['a', 'p', 's']
PARTICIPANT_TO_IDX = {p: i for i, p in enumerate(PARTICIPANTS)}

def load_dataset(dataset_dir: str = 'dataset') -> pd.DataFrame:
    """
    Loads all .dat files from speficied dataset directory and returns a DataFrame with columns.

    :param dataset_dir: Path to the dataset directory containing class subfolders.
    :return: A pandas DataFrame with the loaded data.
    """
    records = []
    for cls in CLASSES:
        folder = os.path.join(dataset_dir, cls)
        for fname in sorted(os.listdir(folder)):
            if not fname.endswith('.dat'):
                continue
            m = re.match(r'^(([ps])_)?(mouth|nose)_trial_(\d+)\.dat$', fname)
            if m is None:
                continue
            df = pd.read_csv(os.path.join(folder, fname))
            records.append({'time': df['Time'].values/1000,
                            'humidity': df['Humidity'].values,
                            'temperature': df['Temperature'].values,
                            'class': cls,
                            'participant': m.group(2) if m.group(2) else 'a',
                            'region':  m.group(3),
                            'trial_num': int(m.group(4)),
                            'filename': fname})
            
    return pd.DataFrame(records)

def split_dataset(df: pd.DataFrame, val_size: float = 0.1, test_size: float = 0.1, random_state: int = 42,) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

class BreathDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, stats: dict | None = None,
                 alpha: float = 0.0, n_copies: int = 1) -> None:
        self.records = dataframe.to_dict('records')
        self.stats = stats if stats is not None else self._compute_stats()
        self.alpha = alpha
        self.n_copies = max(n_copies, 1)

    def _compute_stats(self) -> dict:
        h_all = np.concatenate([r['humidity'] for r in self.records])
        t_all = np.concatenate([r['temperature'] for r in self.records])
        return {'max':  np.array([h_all.max(), t_all.max()], dtype=np.float32),
                'min':  np.array([h_all.min(), t_all.min()], dtype=np.float32),
                'mean': np.array([h_all.mean(), t_all.mean()], dtype=np.float32),
                'std':  np.array([h_all.std(), t_all.std()], dtype=np.float32)}

    def __len__(self) -> int:
        return len(self.records) * self.n_copies

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
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
    def __init__(self, dataframe: pd.DataFrame, stats: dict | None = None) -> None:
        self.records = dataframe.to_dict('records')
        self.stats = stats if stats is not None else self._compute_stats()

    def _compute_stats(self) -> dict:
        peak_devs = []
        for r in self.records:
            baseline = np.mean(r['humidity'][:5])
            peak_devs.append((r['humidity'] - baseline).max())
        h_scale = float(np.max(peak_devs) * 1.2)

        h_all = np.concatenate([r['humidity'] for r in self.records])
        t_all = np.concatenate([r['temperature'] for r in self.records])
        return {'mean': np.array([h_all.mean(), t_all.mean()], dtype=np.float32),
                'std': np.array([h_all.std(), t_all.std()], dtype=np.float32),
                'h_scale': h_scale}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
        r = self.records[idx]

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
        
        # normalize
        h = h / self.stats['h_scale']
        t = (r['temperature'] - self.stats['mean'][1]) / self.stats['std'][1]
        
        # get all infos
        signal = torch.tensor(np.stack([h, t], axis=0), dtype=torch.float32)
        time = torch.tensor(r['time'] - r['time'][0],  dtype=torch.float32)
        label = CLASS_TO_IDX[r['class']]
        participant = PARTICIPANT_TO_IDX[r['participant']]

        return signal, time, label, participant, onset_idx