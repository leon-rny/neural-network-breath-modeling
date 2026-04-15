import argparse
import csv
import json
import os
import pickle
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── Constants ────────────────────────────────────────────────────────────────

CLASSES          = ['bradypnea', 'eupnea', 'tachypnea']
CLASS_TO_IDX     = {cls: i for i, cls in enumerate(CLASSES)}
PARTICIPANTS     = ['a', 'p', 's']
PARTICIPANT_TO_IDX = {p: i for i, p in enumerate(PARTICIPANTS)}
FIXED_LEN        = 36
T_MAX            = 70.0   # seconds


# ── Data ─────────────────────────────────────────────────────────────────────

def _load_dataset(dataset_dir: str = 'dataset') -> pd.DataFrame:
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
            records.append({
                'time':        df['Time'].values / 1000,
                'humidity':    df['Humidity'].values,
                'temperature': df['Temperature'].values,
                'class':       cls,
                'participant': m.group(2) if m.group(2) else 'a',
                'region':      m.group(3),
                'trial_num':   int(m.group(4)),
                'filename':    fname,
            })
    return pd.DataFrame(records)


def _split_dataset(df, val_size=0.1, test_size=0.1, random_state=42):
    from sklearn.model_selection import train_test_split as _tts
    df_tv, df_test = _tts(df, test_size=test_size,
                          stratify=df['class'], random_state=random_state)
    val_rel = val_size / (1 - test_size)
    df_train, df_val = _tts(df_tv, test_size=val_rel,
                             stratify=df_tv['class'], random_state=random_state)
    return (df_train.reset_index(drop=True),
            df_val.reset_index(drop=True),
            df_test.reset_index(drop=True))


class PIDataset(Dataset):
    """
    Baseline-correct each sample (subtract mean of first 3 timesteps per channel),
    then min-max normalise to [0, 1] using per-channel training-set maximum.

    Returns: (signal (2, 36), time (36,), label int, participant int)
    """

    def __init__(self, dataframe: pd.DataFrame,
                 train_max: np.ndarray | None = None) -> None:
        self.records = dataframe.to_dict('records')
        if train_max is None:
            self.train_max = self._compute_train_max()   # (2,) float32
        else:
            self.train_max = train_max

    def _compute_train_max(self) -> np.ndarray:
        h_max = t_max = -np.inf
        for r in self.records:
            h = r['humidity']    - r['humidity'][:3].mean()
            t = r['temperature'] - r['temperature'][:3].mean()
            h_max = max(h_max, float(h.max()))
            t_max = max(t_max, float(t.max()))
        return np.array([h_max, t_max], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        h = r['humidity']    - r['humidity'][:3].mean()
        t = r['temperature'] - r['temperature'][:3].mean()
        h = h / (self.train_max[0] + 1e-8)
        t = t / (self.train_max[1] + 1e-8)
        signal      = torch.tensor(np.stack([h, t], axis=0), dtype=torch.float32)
        time        = torch.tensor(r['time'] - r['time'][0],  dtype=torch.float32)
        label       = CLASS_TO_IDX[r['class']]
        participant = PARTICIPANT_TO_IDX[r['participant']]
        return signal, time, label, participant


# ── Model ────────────────────────────────────────────────────────────────────

class PIEncoder(nn.Module):
    """
    Shared Conv1d backbone conditioned on class + participant labels.
    Two output heads:
      - Physics head  → (mu_P, logvar_P)  shape (B, 4)
        latent encodes (α_H, c_H, α_T, c_T) in unconstrained space
      - Abstract head → (mu_A, logvar_A)  shape (B, latent_dim)
    """

    def __init__(self, latent_dim: int = 16,
                 num_classes: int = 3,    embed_dim: int = 8,
                 num_participants: int = 3, part_embed_dim: int = 8) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2,  16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.part_embed  = nn.Embedding(num_participants, part_embed_dim)
        fc_in = 64 * FIXED_LEN + embed_dim + part_embed_dim
        self.fc = nn.Sequential(nn.Linear(fc_in, 128), nn.ReLU())

        # physics head (4 ODE parameters)
        self.mu_P     = nn.Linear(128, 4)
        self.logvar_P = nn.Linear(128, 4)

        # abstract head
        self.mu_A     = nn.Linear(128, latent_dim)
        self.logvar_A = nn.Linear(128, latent_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor,
                p: torch.Tensor):
        h = self.conv(x).flatten(1)
        h = self.fc(torch.cat([h, self.label_embed(y), self.part_embed(p)], dim=1))
        return self.mu_P(h), self.logvar_P(h), self.mu_A(h), self.logvar_A(h)


class PhysicsDecoder(nn.Module):
    """
    Analytical ODE decoder — zero learnable parameters.

    z_P (B, 4)  →  softplus  →  (α_H, c_H, α_T, c_T) > 0
                →  y_k(t) = (c_k / α_k) * (1 - exp(-α_k * t))
                →  divide by train_max  →  (B, 2, 36)  in [0,1] scale
    """

    def __init__(self, train_max: np.ndarray) -> None:
        super().__init__()
        self.register_buffer(
            'train_max',
            torch.tensor(train_max, dtype=torch.float32).view(1, 2, 1),
        )
        self.register_buffer(
            'time_vec',
            torch.linspace(0, T_MAX, FIXED_LEN),   # (36,)
        )

    def forward(self, z_P: torch.Tensor) -> torch.Tensor:
        params  = F.softplus(z_P)                              # (B, 4)
        alpha_H = params[:, 0].clamp(min=1e-4).unsqueeze(1)   # (B, 1)
        c_H     = params[:, 1].unsqueeze(1)
        alpha_T = params[:, 2].clamp(min=1e-4).unsqueeze(1)
        c_T     = params[:, 3].unsqueeze(1)
        t       = self.time_vec.unsqueeze(0)                   # (1, 36)
        y_H     = (c_H / alpha_H) * (1 - torch.exp(-alpha_H * t))  # (B, 36)
        y_T     = (c_T / alpha_T) * (1 - torch.exp(-alpha_T * t))
        out     = torch.stack([y_H, y_T], dim=1)               # (B, 2, 36)
        return out / self.train_max                             # → [0,1] scale


class NeuralDecoder(nn.Module):
    """
    Residual decoder: same architecture as ConditionalDecoder in models/vae.py.
    Conditioned on class label and participant.
    """

    def __init__(self, latent_dim: int = 16,
                 num_classes: int = 3,    embed_dim: int = 8,
                 num_participants: int = 3, part_embed_dim: int = 8) -> None:
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.part_embed  = nn.Embedding(num_participants, part_embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + embed_dim + part_embed_dim, 128), nn.ReLU(),
            nn.Linear(128, 64 * FIXED_LEN),
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(16,  2, kernel_size=3, padding=1),   # no activation
        )

    def forward(self, z_A: torch.Tensor, y: torch.Tensor,
                p: torch.Tensor) -> torch.Tensor:
        h = self.fc(torch.cat([z_A, self.label_embed(y), self.part_embed(p)], dim=1))
        h = h.view(h.size(0), 64, FIXED_LEN)
        return self.conv(h)   # (B, 2, 36)


class PICVAE2(nn.Module):
    """
    Physics-Integrated CVAE.

    Encoder → (z_P, z_A)
    x_hat = PhysicsDecoder(z_P) + NeuralDecoder(z_A, class, participant)
    """

    LATENT_DIM_P = 4   # always 4: (α_H, c_H, α_T, c_T)

    def __init__(self, latent_dim: int = 16,
                 num_classes: int = 3,    embed_dim: int = 8,
                 num_participants: int = 3, part_embed_dim: int = 8,
                 train_max: np.ndarray | None = None) -> None:
        super().__init__()
        if train_max is None:
            train_max = np.ones(2, dtype=np.float32)
        self.latent_dim_A   = latent_dim
        self.num_participants = num_participants
        self.encoder     = PIEncoder(latent_dim, num_classes, embed_dim,
                                     num_participants, part_embed_dim)
        self.physics_dec = PhysicsDecoder(train_max)
        self.neural_dec  = NeuralDecoder(latent_dim, num_classes, embed_dim,
                                         num_participants, part_embed_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor,
                p: torch.Tensor):
        mu_P, lv_P, mu_A, lv_A = self.encoder(x, y, p)
        z_P = _reparam(mu_P, lv_P, self.training)
        z_A = _reparam(mu_A, lv_A, self.training)
        x_phys  = self.physics_dec(z_P)          # (B, 2, 36)
        x_resid = self.neural_dec(z_A, y, p)     # (B, 2, 36)
        x_hat   = x_phys + x_resid
        return x_hat, x_phys, x_resid, mu_P, lv_P, mu_A, lv_A

    @torch.no_grad()
    def sample(self, n: int, y: torch.Tensor,
               device: torch.device) -> torch.Tensor:
        """
        Sample n signals conditioned on class labels y.
        Returns (n, 2, 36) in [0, 1] normalised space.
        Caller multiplies by train_max to recover physical units.
        """
        z_P = torch.randn(n, self.LATENT_DIM_P,  device=device)
        z_A = torch.randn(n, self.latent_dim_A,  device=device)
        if y.dim() == 0:
            y = y.expand(n)
        p = torch.randint(0, self.num_participants, (n,), device=device)
        self.eval()
        x_phys  = self.physics_dec(z_P)
        x_resid = self.neural_dec(z_A, y.to(device), p)
        return x_phys + x_resid


def _reparam(mu: torch.Tensor, logvar: torch.Tensor,
             training: bool) -> torch.Tensor:
    if training:
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)
    return mu


# ── Loss ─────────────────────────────────────────────────────────────────────

def _kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q || N(0,I)) averaged over batch and latent dimensions."""
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()


def pi_loss(x, x_hat, x_resid, mu_P, lv_P, mu_A, lv_A,
            beta: float, lam_res: float):
    recon       = F.mse_loss(x_hat, x)
    kl_P        = _kl(mu_P, lv_P)
    kl_A        = _kl(mu_A, lv_A)
    res_penalty = x_resid.pow(2).mean()
    total       = recon + beta * (kl_P + kl_A) + lam_res * res_penalty
    return total, recon, kl_P, kl_A, res_penalty


def beta_capped(epoch: int, total_epochs: int, beta_max: float = 0.1) -> float:
    warmup = total_epochs * 0.5
    return min(beta_max, (epoch / warmup) * beta_max)


# ── Training helpers ─────────────────────────────────────────────────────────

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=False)


def _seed_worker(_wid):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def _train_epoch(model, loader, optimizer, epoch, total_epochs,
                 device, beta_max, lam_res):
    model.train()
    beta   = beta_capped(epoch, total_epochs, beta_max)
    sums   = dict(loss=0.0, recon=0.0, kl_P=0.0, kl_A=0.0, res=0.0)
    n      = 0
    for signal, _time, label, participant in loader:
        signal      = signal.to(device)
        label       = label.long().to(device)
        participant = participant.long().to(device)
        x_hat, _, x_resid, mu_P, lv_P, mu_A, lv_A = model(signal, label, participant)
        loss, recon, kl_P, kl_A, res = pi_loss(
            signal, x_hat, x_resid, mu_P, lv_P, mu_A, lv_A, beta, lam_res)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sums['loss']  += loss.item()
        sums['recon'] += recon.item()
        sums['kl_P']  += kl_P.item()
        sums['kl_A']  += kl_A.item()
        sums['res']   += res.item()
        n += 1
    return {k: v / n for k, v in sums.items()}, beta


@torch.no_grad()
def _eval_epoch(model, loader, epoch, total_epochs, device, beta_max, lam_res):
    model.eval()
    beta = beta_capped(epoch, total_epochs, beta_max)
    sums = dict(loss=0.0, recon=0.0, kl_P=0.0, kl_A=0.0, res=0.0)
    n    = 0
    for signal, _time, label, participant in loader:
        signal      = signal.to(device)
        label       = label.long().to(device)
        participant = participant.long().to(device)
        x_hat, _, x_resid, mu_P, lv_P, mu_A, lv_A = model(signal, label, participant)
        loss, recon, kl_P, kl_A, res = pi_loss(
            signal, x_hat, x_resid, mu_P, lv_P, mu_A, lv_A, beta, lam_res)
        sums['loss']  += loss.item()
        sums['recon'] += recon.item()
        sums['kl_P']  += kl_P.item()
        sums['kl_A']  += kl_A.item()
        sums['res']   += res.item()
        n += 1
    return {k: v / n for k, v in sums.items()}


@torch.no_grad()
def _log_physics_params(model, dataset, device):
    """Return mean softplus(mu_P) per class → {class_name: {α_H,c_H,α_T,c_T}}."""
    model.eval()
    buckets = {i: [] for i in range(len(CLASSES))}
    for i in range(len(dataset)):
        signal, _, label, participant = dataset[i]
        signal      = signal.unsqueeze(0).to(device)
        y           = torch.tensor([label]).long().to(device)
        p           = torch.tensor([participant]).long().to(device)
        mu_P, _, _, _ = model.encoder(signal, y, p)
        params = F.softplus(mu_P).squeeze(0).cpu().numpy()
        buckets[label].append(params)
    out = {}
    for cls_idx, name in enumerate(CLASSES):
        if buckets[cls_idx]:
            arr = np.array(buckets[cls_idx]).mean(axis=0)
            out[name] = {k: float(v) for k, v in
                         zip(['alpha_H', 'c_H', 'alpha_T', 'c_T'], arr)}
    return out


# ── TSTR (inlined) ───────────────────────────────────────────────────────────

def _df_to_long(df):
    hum  = np.stack(df['humidity'].values)
    temp = np.stack(df['temperature'].values)
    n, T = hum.shape
    ids   = np.repeat(np.arange(n), T)
    times = np.tile(np.arange(T), n)
    df_long = pd.DataFrame({'id': ids, 'time': times,
                             'Humidity':    hum.ravel(),
                             'Temperature': temp.ravel()})
    y = pd.Series([CLASS_TO_IDX[c] for c in df['class']],
                  index=np.arange(n), name='target')
    return df_long, y


def _extract_fixed_features(df_long, top_features_raw, n_jobs=4):
    from tsfresh import extract_features
    from tsfresh.feature_extraction.settings import from_columns
    from tsfresh.utilities.dataframe_functions import impute
    kind_to_fc = from_columns(top_features_raw)
    X = extract_features(df_long, column_id='id', column_sort='time',
                         kind_to_fc_parameters=kind_to_fc, n_jobs=n_jobs)
    impute(X)
    for c in top_features_raw:
        if c not in X.columns:
            X[c] = 0.0
    return X[top_features_raw]


def _train_stacker(X_train, y_train, seed):
    from imblearn.over_sampling import SMOTE
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from sklearn.model_selection import StratifiedKFold

    min_class  = int(np.bincount(y_train).min())
    k          = min(5, min_class - 1)
    if k < 1:
        X_res, y_res = X_train, y_train
    else:
        X_res, y_res = SMOTE(random_state=seed, k_neighbors=k).fit_resample(X_train, y_train)

    xgb = XGBClassifier(eval_metric='mlogloss', random_state=seed,
                        max_depth=4, reg_alpha=0.5, reg_lambda=1.0,
                        subsample=0.8, colsample_bytree=0.8, n_estimators=300)
    cat = CatBoostClassifier(logging_level='Silent', random_state=seed,
                             iterations=300, depth=4, l2_leaf_reg=5.0,
                             random_strength=2.0, bagging_temperature=2.0,
                             od_type='Iter', od_wait=20, allow_writing_files=False)
    meta = RandomForestClassifier(n_estimators=150, max_depth=3,
                                  min_samples_leaf=5, min_samples_split=10,
                                  random_state=seed)
    stacker = StackingClassifier(
        estimators=[('xgb', xgb), ('cat', cat)],
        final_estimator=meta, passthrough=True,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
        n_jobs=-1,
    )
    stacker.fit(X_res, y_res)
    return stacker


def _eval_clf(clf, X_test, y_test):
    from sklearn.metrics import (accuracy_score, classification_report,
                                  f1_score, log_loss, roc_auc_score)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    try:
        roc = float(roc_auc_score(y_test, y_prob, multi_class='ovr'))
    except Exception:
        roc = float('nan')
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return {
        'accuracy':    float(accuracy_score(y_test, y_pred)),
        'f1_weighted': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
        'roc_auc_ovr': roc,
        'log_loss':    float(log_loss(y_test, y_prob)),
        'per_class_f1': {
            cls: float(report.get(str(i), {}).get('f1-score', float('nan')))
            for i, cls in enumerate(CLASSES)
        },
    }


def _build_trtr_cache(dataset_dir, region, seed, n_jobs):
    """
    Build (or load) the TRTR feature cache for this region/seed.
    Stores tsfresh top-20 features from real training data plus test labels.
    """
    from tsfresh import extract_relevant_features
    from tsfresh.utilities.dataframe_functions import impute
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
    import shap

    cache_path = f'results/pi_cvae/{region}_s{seed}_trtr_cache.pkl'
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    df = _load_dataset(dataset_dir)
    df = df[df['region'] == region].reset_index(drop=True)
    df_train, _df_val, df_test = _split_dataset(df, random_state=seed)

    df_long_train, y_train = _df_to_long(df_train)
    X_raw = extract_relevant_features(df_long_train, y_train,
                                      column_id='id', column_sort='time',
                                      n_jobs=n_jobs)
    impute(X_raw)

    raw_to_san = {c: re.sub(r'[^\w]', '_', c) for c in X_raw.columns}
    san_to_raw = {v: k for k, v in raw_to_san.items()}
    X_san = X_raw.rename(columns=raw_to_san)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_san, y_train, test_size=0.2, stratify=y_train, random_state=seed)
    param_grid = {'max_depth': [4, 6], 'reg_alpha': [0.1, 1.0],
                  'reg_lambda': [0.5, 1.0], 'colsample_bytree': [0.8, 1.0]}
    gs = GridSearchCV(
        LGBMClassifier(n_estimators=1000, learning_rate=0.05,
                       random_state=seed, verbose=-1),
        param_grid, cv=StratifiedKFold(3), scoring='accuracy', n_jobs=-1, verbose=0)
    gs.fit(X_tr, y_tr)
    best_lgbm = LGBMClassifier(**gs.best_params_, n_estimators=1000,
                                learning_rate=0.05, random_state=seed, verbose=-1)
    best_lgbm.fit(X_tr.values, y_tr.values,
                  eval_set=[(X_val.values, y_val.values)],
                  eval_metric='multi_logloss',
                  callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])

    explainer  = shap.TreeExplainer(best_lgbm)
    shap_vals  = explainer.shap_values(X_san.values)
    mean_abs   = np.abs(shap_vals).mean(axis=0).mean(axis=1)
    top_idx    = np.argsort(mean_abs)[::-1][:20]
    top_20_san = X_san.columns.to_numpy()[top_idx].tolist()
    top_20_raw = [san_to_raw[s] for s in top_20_san]
    X_train_top = X_san[top_20_san].values

    df_long_test, y_test = _df_to_long(df_test)
    X_test_top = _extract_fixed_features(df_long_test, top_20_raw, n_jobs)
    X_test_top = X_test_top.rename(columns={c: re.sub(r'[^\w]', '_', c)
                                             for c in X_test_top.columns})
    X_test_top = X_test_top[top_20_san].values

    clf          = _train_stacker(X_train_top, y_train.values, seed)
    trtr_metrics = _eval_clf(clf, X_test_top, y_test.values)

    cache = {
        'top_20_raw': top_20_raw, 'top_20_san': top_20_san,
        'X_train_top': X_train_top, 'X_test_top': X_test_top,
        'y_train': y_train.values, 'y_test': y_test.values,
        'n_train': len(df_train),
        'trtr_metrics': trtr_metrics,
    }
    os.makedirs('results/pi_cvae', exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)
    print(f'[TRTR] cache saved → {cache_path}  |  '
          f'accuracy={trtr_metrics["accuracy"]:.3f}')
    return cache


def _run_tstr(model, train_max, cache, seed, n_synthetic, n_jobs, device):
    """Generate synthetic signals and compute TSTR metrics."""
    n_per = n_synthetic // len(CLASSES)
    rem   = n_synthetic % len(CLASSES)
    counts = [n_per + (1 if i < rem else 0) for i in range(len(CLASSES))]

    train_max_t = torch.tensor(train_max, dtype=torch.float32).view(1, 2, 1).to(device)
    all_sig, all_lbl = [], []
    for cls_idx, count in enumerate(counts):
        y_cls  = torch.tensor(cls_idx, dtype=torch.long)
        synth  = model.sample(count, y_cls, device)          # (count,2,36) in [0,1]
        synth_phys = (synth * train_max_t).cpu().numpy()     # un-normalise
        all_sig.append(synth_phys)
        all_lbl.append(np.full(count, cls_idx))

    signals = np.concatenate(all_sig, axis=0)    # (N, 2, 36)
    labels  = np.concatenate(all_lbl, axis=0)

    n, _C, T = signals.shape
    df_long = pd.DataFrame({
        'id':          np.repeat(np.arange(n), T),
        'time':        np.tile(np.arange(T), n),
        'Humidity':    signals[:, 0, :].ravel(),
        'Temperature': signals[:, 1, :].ravel(),
    })
    X_raw = _extract_fixed_features(df_long, cache['top_20_raw'], n_jobs)
    X_san = X_raw.rename(columns={c: re.sub(r'[^\w]', '_', c) for c in X_raw.columns})
    X_top = X_san[cache['top_20_san']].values

    stacker = _train_stacker(X_top, labels, seed)
    return _eval_clf(stacker, cache['X_test_top'], cache['y_test'])


# ── Main ─────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description='Train PI-CVAE')
    p.add_argument('--region',       required=True, choices=['mouth', 'nose'])
    p.add_argument('--dataset_dir',  default='dataset')
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--epochs',       type=int,   default=500)
    p.add_argument('--batch_size',   type=int,   default=32)
    p.add_argument('--latent_dim',   type=int,   default=16)
    p.add_argument('--embed_dim',    type=int,   default=8)
    p.add_argument('--part_embed_dim', type=int, default=8)
    p.add_argument('--lr',           type=float, default=1e-3)
    p.add_argument('--beta_max',     type=float, default=0.1)
    p.add_argument('--lam_res',      type=float, default=0.01)
    p.add_argument('--log_every',    type=int,   default=25)
    p.add_argument('--n_jobs',       type=int,   default=4)
    p.add_argument('--skip_tstr',    action='store_true')
    return p.parse_args()


def main():
    args = _parse_args()
    os.makedirs('results/pi_cvae', exist_ok=True)

    tag          = (f'{args.region}_s{args.seed}_ld{args.latent_dim}'
                    f'_ed{args.embed_dim}_bmax{args.beta_max}_lr{args.lam_res}')
    ckpt_path    = f'results/pi_cvae/{tag}_checkpoint.pt'
    history_path = f'results/pi_cvae/{tag}_train_history.csv'
    tstr_path    = f'results/pi_cvae/{tag}_tstr.json'

    _seed_everything(args.seed)
    device = torch.device('cpu')
    print(f'[PI-CVAE] region={args.region}  seed={args.seed}  device={device}')

    # ── data ────────────────────────────────────────────────────────────────
    df       = _load_dataset(args.dataset_dir)
    df       = df[df['region'] == args.region].reset_index(drop=True)
    df_train, df_val, _df_test = _split_dataset(df, random_state=args.seed)

    train_ds = PIDataset(df_train)
    val_ds   = PIDataset(df_val, train_max=train_ds.train_max)
    print(f'[PI-CVAE] train={len(train_ds)}  val={len(val_ds)}  '
          f'train_max(H,T)=({train_ds.train_max[0]:.2f}, {train_ds.train_max[1]:.2f})')

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=False,
                              worker_init_fn=_seed_worker, generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    # ── model ────────────────────────────────────────────────────────────────
    model = PICVAE2(
        latent_dim    = args.latent_dim,
        embed_dim     = args.embed_dim,
        part_embed_dim= args.part_embed_dim,
        train_max     = train_ds.train_max,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    history: list[dict] = []

    # ── training loop ────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        tr, beta = _train_epoch(model, train_loader, optimizer, epoch,
                                args.epochs, device, args.beta_max, args.lam_res)
        vl       = _eval_epoch(model, val_loader, epoch,
                               args.epochs, device, args.beta_max, args.lam_res)
        scheduler.step()

        # save best checkpoint (only after warmup is complete)
        if vl['loss'] < best_val_loss and beta >= args.beta_max:
            best_val_loss = vl['loss']
            torch.save({
                'epoch':          epoch,
                'model_state':    model.state_dict(),
                'train_max':      train_ds.train_max,
                'latent_dim':     args.latent_dim,
                'embed_dim':      args.embed_dim,
                'part_embed_dim': args.part_embed_dim,
                'region':         args.region,
                'beta_max':       args.beta_max,
                'lam_res':        args.lam_res,
            }, ckpt_path)

        row = {
            'epoch': epoch, 'beta': beta,
            'train_loss': tr['loss'], 'train_recon': tr['recon'],
            'train_kl_P': tr['kl_P'], 'train_kl_A': tr['kl_A'],
            'train_res':  tr['res'],
            'val_loss':   vl['loss'],  'val_recon':  vl['recon'],
            'val_kl_P':   vl['kl_P'], 'val_kl_A':   vl['kl_A'],
            'val_res':    vl['res'],
        }
        history.append(row)

        if epoch % args.log_every == 0 or epoch == 1:
            phys = _log_physics_params(model, train_ds, device)
            phys_str = '  |  '.join(
                f"{name}: α_H={v['alpha_H']:.3f} c_H={v['c_H']:.3f}"
                f" α_T={v['alpha_T']:.3f} c_T={v['c_T']:.3f}"
                for name, v in phys.items()
            )
            print(
                f'[PI-CVAE] ep {epoch:4d}/{args.epochs}'
                f' β={beta:.3f}'
                f' | train loss={tr["loss"]:.4f}'
                f' (recon={tr["recon"]:.4f}'
                f' KL_P={tr["kl_P"]:.4f}'
                f' KL_A={tr["kl_A"]:.4f}'
                f' res={tr["res"]:.4f})'
                f' | val={vl["loss"]:.4f}'
                f' | {phys_str}'
            )

    print(f'[PI-CVAE] best val loss={best_val_loss:.4f} | ckpt → {ckpt_path}')

    with open(history_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f'[PI-CVAE] history → {history_path}')

    # ── TSTR evaluation ──────────────────────────────────────────────────────
    if args.skip_tstr:
        return

    print('[PI-CVAE] Running TSTR ...')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    cache       = _build_trtr_cache(args.dataset_dir, args.region, args.seed, args.n_jobs)
    n_synthetic = cache['n_train']
    metrics     = _run_tstr(model, train_ds.train_max, cache,
                             args.seed, n_synthetic, args.n_jobs, device)

    print(f'[PI-CVAE] TSTR accuracy={metrics["accuracy"]:.3f}  '
          f'f1={metrics["f1_weighted"]:.3f}  '
          f'roc_auc={metrics["roc_auc_ovr"]:.3f}')
    print(f'[TRTR]    accuracy={cache["trtr_metrics"]["accuracy"]:.3f}')

    def _safe(obj):
        if isinstance(obj, float) and (obj != obj):
            return None
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_safe(v) for v in obj]
        return obj

    result = {
        'model':         'pi_cvae',
        'region':        args.region,
        'seed':          args.seed,
        'latent_dim':    args.latent_dim,
        'embed_dim':     args.embed_dim,
        'beta_max':      args.beta_max,
        'lam_res':       args.lam_res,
        'n_synthetic':   n_synthetic,
        'n_train_real':  cache['n_train'],
        'metrics':       metrics,
        'trtr_metrics':  cache['trtr_metrics'],
    }
    with open(tstr_path, 'w') as f:
        json.dump(_safe(result), f, indent=2)
    print(f'[PI-CVAE] TSTR results → {tstr_path}')


if __name__ == '__main__':
    main()
