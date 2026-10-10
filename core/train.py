"""Generator training on one split: CVAE and PI-CVAE (models/), conditioned on class and participant with participant
dropout (null token for unseen participants); checkpoint = lowest validation ELBO once beta has reached BETA_MAX."""

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.data import (
    CLASS_TO_IDX,
    PARTICIPANT_TO_IDX,
    PARTICIPANTS,
    BreathDataset,
    PhysicsInformedDataset,
)
from core.utils import (
    file_digest,
    fingerprint,
    make_generator,
    reserve_training,
    seed_everything,
    seed_worker,
)
from models.pinn import SharedTransportPINN
from models.vae import CVAE, elbo_loss

# cvae: z-scored signals, tenfold with jitter; picvae: baseline-corrected stdscale signals, tenfold without jitter, +
# shape term (shape_lambda x MSE of the first differences), calibrated transport + conv decoder as residual
GENERATORS = {'cvae': {'alpha': 0.05, 'n_copies': 10, 'shape_lambda': 0.0},
              'picvae': {'alpha': 0.0, 'n_copies': 10, 'shape_lambda': 3.0}}
DIMS = {'latent_dim': 16, 'embed_dim': 8, 'part_embed_dim': 8}
BETA_MAX, PART_DROPOUT, LR, BATCH_SIZE, EPOCHS = 0.01, 0.1, 1e-3, 32, 500
T_GRID = np.arange(36) * 2.0

def beta_capped(epoch: int, warmup_epochs: int, beta_max: float = BETA_MAX) -> float:
    # linear KL warmup from 0 to beta_max, then hold
    if warmup_epochs < 0:
        raise ValueError('warmup_epochs must be nonnegative')
    return beta_max if warmup_epochs == 0 else min(beta_max, (epoch / warmup_epochs) * beta_max)

def run_epoch(model: torch.nn.Module, loader: DataLoader, beta: float, shape_lambda: float = 0.0, optimizer: torch.optim.Optimizer | None = None) -> tuple[float, float]:
    # training pass with an optimizer, else evaluation without gradients; returns the epoch-mean (reconstruction, KL)
    model.train(optimizer is not None)
    recon_sum = kl_sum = 0.0
    with torch.set_grad_enabled(optimizer is not None):
        for signal, label, participant in loader:
            x_hat, mu, logvar = model(signal, label, participant)
            loss, recon, kl = elbo_loss(signal, x_hat, mu, logvar, beta)
            if optimizer is not None:
                if shape_lambda > 0:
                    loss = loss + shape_lambda * F.mse_loss(x_hat[:, :, 1:] - x_hat[:, :, :-1], signal[:, :, 1:] - signal[:, :, :-1])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            recon_sum += recon.item() * signal.size(0)
            kl_sum += kl.item() * signal.size(0)
    n = len(loader.dataset)
    return recon_sum / n, kl_sum / n

class Generator:
    """A trained CVAE / PI-CVAE with the transform between its training space and signals."""

    def __init__(self, name: str, model: torch.nn.Module, stats: dict) -> None:
        self.name, self.model, self.stats = name, model.eval(), stats
        self.space = PhysicsInformedDataset if name == 'picvae' else BreathDataset

    def inputs(self, df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(signals in the training space, class labels, participant labels) of the trials in df."""
        ds = self.space(df, stats=self.stats)
        x = torch.stack([ds[i][0] for i in range(len(ds))])
        y = torch.tensor([CLASS_TO_IDX[c] for c in df['class']], dtype=torch.long)
        p = torch.tensor([PARTICIPANT_TO_IDX[q] for q in df['participant']], dtype=torch.long)
        return x, y, p

    def signals(self, x_hat: torch.Tensor) -> np.ndarray:
        """Decoder output (n, 2, T) -> baseline-corrected signals in physical units."""
        x = np.asarray(x_hat.numpy(), dtype=float)
        if self.name == 'picvae':
            return x * np.array([self.stats['h_bcstd'], self.stats['t_bcstd']]).reshape(1, 2, 1)
        # cvae: de-standardize, then subtract the mean of the first 5 samples
        s = x * self.stats['std'].reshape(1, 2, 1) + self.stats['mean'].reshape(1, 2, 1)
        return s - s[:, :, :5].mean(2, keepdims=True)

def build(name: str, calibration: np.ndarray | None) -> torch.nn.Module:
    # untrained generator, consumes the global torch RNG for the initial weights
    kwargs = dict(**DIMS, num_classes=3, condition_on_participant=True, num_participants=len(PARTICIPANTS), part_dropout=PART_DROPOUT)
    if name == 'picvae':
        return SharedTransportPINN(cir_params_init=calibration, t_grid=T_GRID, residual=True, **kwargs)
    return CVAE(**kwargs)

def train(name: str, region: str, df_train: pd.DataFrame, df_val: pd.DataFrame, seed: int, path: str, split: dict, dataset_dir: str = 'dataset', calibration_dir: str = 'results/pinn', epochs: int = EPOCHS) -> Generator:
    """Train generator `name` on df_train (checkpoint selection on df_val) and save it to path; a checkpoint with the
    same data, code and settings is loaded instead (core.utils.reserve_training).

    :param name: 'cvae' or 'picvae'
    :param split: identification of the split recorded with the checkpoint (task, participants or fold)
    :param epochs: training epochs (lower only for smoke tests)
    :return: the trained generator
    """
    # provenance
    calibration_path = f'{calibration_dir}/params_{region}.npy'
    config = {'generator': name, 'region': region, 'seed': seed, 'epochs': epochs, **split, **DIMS, **GENERATORS[name],
              'beta_max': BETA_MAX, 'part_dropout': PART_DROPOUT, 'lr': LR, 'batch_size': BATCH_SIZE}
    inputs = fingerprint(dataset_dir)
    if name == 'picvae':
        inputs['calibration_sha256'] = file_digest(calibration_path)
    provenance = reserve_training(path, config, inputs)
    calibration = np.load(calibration_path) if name == 'picvae' else None
    trained_participants = sorted(PARTICIPANT_TO_IDX[p] for p in df_train['participant'].unique())
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    except FileNotFoundError:
        ckpt = None
    # train
    if ckpt is None:
        hp = GENERATORS[name]
        seed_everything(seed)
        g = make_generator(seed)
        space = PhysicsInformedDataset if name == 'picvae' else BreathDataset
        train_ds = space(df_train, alpha=hp['alpha'], n_copies=hp['n_copies'])
        val_ds = space(df_val, stats=train_ds.stats)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, worker_init_fn=seed_worker, generator=g)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        model = build(name, calibration)
        model.trained_participants = trained_participants
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        warmup = epochs // 2
        best, state, history = torch.inf, None, []
        for epoch in range(1, epochs + 1):
            beta = beta_capped(epoch, warmup)
            train_recon, train_kl = run_epoch(model, train_loader, beta, hp['shape_lambda'], optimizer)
            val_recon, val_kl = run_epoch(model, val_loader, beta)
            scheduler.step()
            val_elbo = val_recon + beta * val_kl
            if val_elbo < best and beta >= BETA_MAX:
                best, state = val_elbo, {'epoch': epoch, 'model_state': copy.deepcopy(model.state_dict())}
            history.append({'epoch': epoch, 'beta': beta, 'train_recon': train_recon, 'train_kl': train_kl, 'val_recon': val_recon, 'val_kl': val_kl})
        ckpt = {**state, 'stats': train_ds.stats, 'trained_participants': trained_participants, 'calibration': calibration,
                'history': history, 'provenance': provenance}
        torch.save(ckpt, path)
    # load
    if ckpt['provenance'] != provenance:
        raise ValueError(f'Checkpoint {path} was trained on other data, code or settings')
    model = build(name, calibration)
    model.load_state_dict(ckpt['model_state'])
    model.trained_participants = ckpt['trained_participants']
    return Generator(name, model, ckpt['stats'])
