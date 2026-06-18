import argparse
import csv
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.data import BreathDataset, load_dataset, get_split, loso_path_tag, kfold_split_dataset
from core.train import active_dims, beta_capped, evaluate, train_vae_one_epoch
from core.tstr import evaluate_classifier, extract_fixed_features, generate_synthetic_signals, load_cache, train_stacking_classifier, trtr
from core.utils import make_generator, seed_everything, seed_worker
from models.vae import elbo_loss
from ablations.cvae_models import VARIANT_MAP

MODES: dict[str, dict] = {
    "dynamics": {"results_dir": "results/ablation_cvae_training_dynamics", "ckpt": "f_checkpoint", "family": "ablation"},
    "jittering": {"results_dir": "results/ablation_cvae_jittering", "ckpt": "stem", "family": "ablation"},
    "architecture": {"results_dir": "results/ablation_cvae_architecture", "ckpt": "stem", "family": "architecture"}
}
CONFIGS: dict[str, dict] = {
    # beta_max sweep
    "beta_cap_0.001": {"beta_max": 0.001, "lag_n": 0, "lag_phase": 0},
    "beta_cap_0.01": {"beta_max": 0.01, "lag_n": 0, "lag_phase": 0},
    "beta_cap_0.03": {"beta_max": 0.03, "lag_n": 0, "lag_phase": 0},
    "beta_cap_0.05": {"beta_max": 0.05, "lag_n": 0, "lag_phase": 0},
    "beta_cap_0.1": {"beta_max": 0.1, "lag_n": 0, "lag_phase": 0},
    "beta_cap_0.5": {"beta_max": 0.5, "lag_n": 0, "lag_phase": 0},
    "beta_cap_1.0": {"beta_max": 1.0, "lag_n": 0, "lag_phase": 0}, # baseline
    # free-bits sweep
    "fb_0.1_b1.0": {"beta_max": 1.0, "lag_n": 0, "lag_phase": 0, "free_bits": 0.1},
    "fb_0.5_b1.0": {"beta_max": 1.0, "lag_n": 0, "lag_phase": 0, "free_bits": 0.5},
    "fb_1.0_b1.0": {"beta_max": 1.0, "lag_n": 0, "lag_phase": 0, "free_bits": 1.0},
    "fb_2.0_b1.0": {"beta_max": 1.0, "lag_n": 0, "lag_phase": 0, "free_bits": 2.0},
    # lagging inference sweep
    "lag_5_100": {"beta_max": 1.0, "lag_n": 5, "lag_phase": 100},
    "lag_5_250": {"beta_max": 1.0, "lag_n": 5, "lag_phase": 250},
    "lag_10_100": {"beta_max": 1.0, "lag_n": 10, "lag_phase": 100},
    "lag_10_250": {"beta_max": 1.0, "lag_n": 10, "lag_phase": 250},
    # warmup-fraction sweep
    "warmup_0.25": {"beta_max": 1.0, "lag_n": 0, "lag_phase": 0, "warmup_frac": 0.25},
    "warmup_0.75": {"beta_max": 1.0, "lag_n": 0, "lag_phase": 0, "warmup_frac": 0.75},
    # joint architecture x beta sweep
    "joint": {"beta_max": 0.1, "lag_n": 0, "lag_phase": 0},
    # free-bits x lagging-inference grid at fixed beta_max with jitter
    "fb0_off": {"beta_max": 0.01, "lag_n": 0, "lag_phase": 0, "free_bits": 0.0, "alpha": 0.05, "n_copies": 10},
    "fb0.1_off": {"beta_max": 0.01, "lag_n": 0, "lag_phase": 0, "free_bits": 0.1, "alpha": 0.05, "n_copies": 10},
    "fb0.5_off": {"beta_max": 0.01, "lag_n": 0, "lag_phase": 0, "free_bits": 0.5, "alpha": 0.05, "n_copies": 10},
    "fb0_lag5_250": {"beta_max": 0.01, "lag_n": 5, "lag_phase": 250, "free_bits": 0.0, "alpha": 0.05, "n_copies": 10},
    "fb0.1_lag5_250": {"beta_max": 0.01, "lag_n": 5, "lag_phase": 250, "free_bits": 0.1, "alpha": 0.05, "n_copies": 10},
    "fb0.5_lag5_250": {"beta_max": 0.01, "lag_n": 5, "lag_phase": 250, "free_bits": 0.5, "alpha": 0.05, "n_copies": 10},
    # jittering augmentation
    "baseline": {"alpha": 0.0, "n_copies": 0},
    "a0.01_n2": {"alpha": 0.01, "n_copies": 2},
    "a0.01_n5": {"alpha": 0.01, "n_copies": 5},
    "a0.01_n10": {"alpha": 0.01, "n_copies": 10},
    "a0.05_n2": {"alpha": 0.05, "n_copies": 2},
    "a0.05_n5": {"alpha": 0.05, "n_copies": 5},
    "a0.05_n10": {"alpha": 0.05, "n_copies": 10},
    "a0.1_n2": {"alpha": 0.1, "n_copies": 2},
    "a0.1_n5": {"alpha": 0.1, "n_copies": 5},
    "a0.1_n10": {"alpha": 0.1, "n_copies": 10},
    "a0.025_n5": {"alpha": 0.025, "n_copies": 5},
    "a0.025_n10": {"alpha": 0.025, "n_copies": 10},
}

# region-default beta_max
BETA_MAX = {"mouth": 0.01, "nose": 0.01}

# fixed parameters
LATENT_DIM = 16
EMBED_DIM = 8
PART_EMBED_DIM = 8
BATCH_SIZE = 32
LOG_EVERY = 25

# architecture-mode
ARCH_BETA_MAX = 1.0
ARCH_WARMUP_FRAC = 0.5
ABLATION_SUBSET = ["variant", "beta_max", "config", "region", "cv_mode", "loso_tag", "init_seed", "split_seed", "fold", "n_folds", "part_dropout"]
ARCH_SUBSET = ["variant", "region", "init_seed", "split_seed", "fold", "n_folds", "condition_on_participant"]

# modes
RESULTS_DIR = MODES["dynamics"]["results_dir"]
CKPT_SCHEME = MODES["dynamics"]["ckpt"]
FAMILY = MODES["dynamics"]["family"]
TAG = "DYNAMICS"

# cli
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Unified CVAE ablation (dynamics | jittering | architecture)")
    p.add_argument("--mode", default="dynamics", choices=list(MODES))
    p.add_argument("--configs", default=",".join(CONFIGS))
    p.add_argument("--init_seeds", default="0,1,7,42,123")
    p.add_argument("--folds", default="1,2,3,4,5")
    p.add_argument("--regions", default="mouth,nose")
    # single-combo overrides
    p.add_argument("--config", default=None)
    p.add_argument("--init_seed", type=int, default=None)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--region", default=None)
    # jittering single-combo synthesis (builds an "a{alpha}_n{n_copies}" config)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--n_copies", type=int, default=None)
    # architecture x beta
    p.add_argument("--variant", default=None)
    p.add_argument("--beta_max", type=float, default=None)
    # plural sweep axes (aggregate mode, and architecture variant sweep)
    p.add_argument("--variants", default=None)
    p.add_argument("--beta_maxes", default=None)
    # architecture: condition-on-participant axis
    p.add_argument("--cond_parts", default="true,false")
    p.add_argument("--cond_part", default=None)
    # protocol-level constants
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--dataset_dir", default="dataset")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--n_jobs", type=int, default=4)
    p.add_argument("--part_dropout", type=float, default=0.0)
    p.add_argument("--cv_mode", choices=["kfold", "loso"], default="kfold")
    p.add_argument("--loso_trial_val", action="store_true")
    p.add_argument("--loso_exclude", default="")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no_summary", action="store_true")
    p.add_argument("--aggregate", action="store_true")
    return p.parse_args()

def _parse_bool(s: str) -> bool:
    """Parse a truthy/falsy string into a bool."""
    if s.lower() in ("true", "1", "yes"):
        return True
    if s.lower() in ("false", "0", "no"):
        return False
    raise ValueError(f"Invalid boolean: {s!r}")

# config knob accessor
def resolve_beta(cfg: dict, region: str, override: float | None) -> float:
    """Resolve beta_max from override, config, then region default."""
    if override is not None:
        return override
    return cfg.get("beta_max", BETA_MAX[region])

## paths
def _pd_marker(part_dropout: float) -> str:
    """Filename marker for participant dropout (empty if disabled)."""
    return "" if part_dropout == 0.0 else f"_pd{part_dropout}"

def _cv_marker(cv_mode: str) -> str:
    """Filename marker for the cv mode (empty for kfold)."""
    return "" if cv_mode == "kfold" else f"_{cv_mode}"

def _stem(variant: str, beta_max: float, config: str, region: str, init_seed: int, split_seed: int, fold: int, n_folds: int, part_dropout: float = 0.0, cv_mode: str = "kfold", loso_tag: str = "") -> str:
    """Build the ablation artifact filename stem."""
    return f"{variant}_b{beta_max}_{config}_{region}_is{init_seed}_ss{split_seed}{_pd_marker(part_dropout)}{_cv_marker(cv_mode)}{loso_tag}_fold{fold}of{n_folds}"

def ckpt_path(variant: str, beta_max: float, config: str, region: str, init_seed: int, split_seed: int, fold: int, n_folds: int, part_dropout: float = 0.0, cv_mode: str = "kfold", loso_tag: str = "") -> str:
    """Path to the ablation checkpoint file."""
    if CKPT_SCHEME == "f_checkpoint":
        return f"{RESULTS_DIR}/{variant}_b{beta_max}_{config}_{region}_is{init_seed}_ss{split_seed}{_pd_marker(part_dropout)}{_cv_marker(cv_mode)}{loso_tag}_f{fold}of{n_folds}_checkpoint.pt"
    return f"{RESULTS_DIR}/{_stem(variant, beta_max, config, region, init_seed, split_seed, fold, n_folds, part_dropout, cv_mode, loso_tag)}.pt"

def hist_path(variant: str, beta_max: float, config: str, region: str, init_seed: int, split_seed: int, fold: int, n_folds: int, part_dropout: float = 0.0, cv_mode: str = "kfold", loso_tag: str = "") -> str:
    """Path to the ablation training-history CSV."""
    return f"{RESULTS_DIR}/{_stem(variant, beta_max, config, region, init_seed, split_seed, fold, n_folds, part_dropout, cv_mode, loso_tag)}_history.csv"

def result_path(variant: str, beta_max: float, config: str, region: str, init_seed: int, split_seed: int, fold: int, n_folds: int, part_dropout: float = 0.0, cv_mode: str = "kfold", loso_tag: str = "") -> str:
    """Path to the ablation per-combo result JSON."""
    return f"{RESULTS_DIR}/{_stem(variant, beta_max, config, region, init_seed, split_seed, fold, n_folds, part_dropout, cv_mode, loso_tag)}_result.json"

def _arch_stem(variant: str, region: str, init_seed: int, split_seed: int, fold: int, n_folds: int, cond_part: bool) -> str:
    """Build the architecture artifact filename stem."""
    cp_marker = "" if cond_part else "_nocp"
    return f"{variant}_{region}_is{init_seed}_ss{split_seed}{cp_marker}_fold{fold}of{n_folds}"

def arch_ckpt_path(variant: str, region: str, init_seed: int, split_seed: int, fold: int, n_folds: int, cond_part: bool) -> str:
    """Path to the architecture checkpoint file."""
    return f"{RESULTS_DIR}/{_arch_stem(variant, region, init_seed, split_seed, fold, n_folds, cond_part)}.pt"

def arch_hist_path(variant: str, region: str, init_seed: int, split_seed: int, fold: int, n_folds: int, cond_part: bool) -> str:
    """Path to the architecture training-history CSV."""
    return f"{RESULTS_DIR}/{_arch_stem(variant, region, init_seed, split_seed, fold, n_folds, cond_part)}_history.csv"

def arch_result_path(variant: str, region: str, init_seed: int, split_seed: int, fold: int, n_folds: int, cond_part: bool) -> str:
    """Path to the architecture per-combo result JSON."""
    return f"{RESULTS_DIR}/{_arch_stem(variant, region, init_seed, split_seed, fold, n_folds, cond_part)}_result.json"

# tstr scoring
def _tstr_score(model, stats: dict, cache: dict, device: torch.device, n_jobs: int, init_seed: int, label: str, cv_mode: str = "kfold") -> dict:
    """Generate synthetic signals from a trained model, extract the cached top-20
    tsfresh features, train a stacking classifier on synthetic, and score on the
    real test set (TSTR). Returns the evaluate_classifier metrics dict."""
    n_synthetic = cache["n_train"]
    # under LOSO the test subject is unseen -> generate from the learned null token
    participant_idx = model.null_part_idx if (cv_mode == "loso" and getattr(model, "_cond_part", False)) else None
    synth_signals, synth_labels = generate_synthetic_signals(model, "cvae_part", n_synthetic, stats, device, init_seed, participant_idx=participant_idx)
    print(f"  [{TAG}] {label} | Generated {n_synthetic} synthetic signals")

    n, _C, T = synth_signals.shape
    df_long = pd.DataFrame({"id": np.repeat(np.arange(n), T),
                            "time": np.tile(np.arange(T), n),
                            "Humidity": synth_signals[:, 0, :].ravel(),
                            "Temperature": synth_signals[:, 1, :].ravel()})
    X_raw = extract_fixed_features(df_long, cache["top_20_features_raw"], n_jobs)
    X_san = X_raw.copy()
    X_san.columns = [re.sub(r"[^\w]", "_", c) for c in X_san.columns]
    X_synth = X_san[cache["top_20_features_sanitized"]].values

    stacker = train_stacking_classifier(X_synth, synth_labels, init_seed, n_jobs=n_jobs)
    return evaluate_classifier(stacker, cache["X_test_top"], cache["y_test"])

def _null_gen_stats(model, device: torch.device, n: int = 60) -> dict:
    """Sanity check (check 3): decode with the null participant token and confirm the
    output is finite and not collapsed to a constant."""
    model.eval()
    z = torch.randn(n, model.latent_dim, device=device)
    y = (torch.arange(n, device=device) % model.num_classes).long()
    p = torch.full((n,), model.null_part_idx, dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.decoder(z, y, p)
    return {"null_gen_finite": bool(torch.isfinite(out).all().item()),
            "null_gen_std": float(out.std().item())}

def _final_kl_active(hp: str) -> tuple[float, int]:
    """Read final train_kl and active_dims from a history CSV (nan/0 if missing)."""
    kl_final, active_dims_final = float("nan"), 0
    if os.path.exists(hp):
        hist_df = pd.read_csv(hp)
        kl_final = float(hist_df["train_kl"].iloc[-1])
        active_dims_final = int(hist_df["active_dims"].iloc[-1])
    return kl_final, active_dims_final

## ablation family
# training
def train_config(config: str, region: str, init_seed: int, split_seed: int, fold: int, n_folds: int, dataset_dir: str, device: torch.device, epochs: int, lr: float, latent_dim: int, variant: str = "conv_baseline", beta_max_override: float | None = None, part_dropout: float = 0.0, cv_mode: str = "kfold", exclude_subjects: tuple = (), loso_trial_val: bool = False, verbose: bool = False) -> None:
    """Train one CVAE for a CONFIGS entry, checkpointing the best-val model and history CSV.

    Resolves the config knobs (beta_max, lag, free_bits, jitter, warmup), loads the region
    data, applies optional jitter augmentation, trains the selected variant with optional
    lagging-inference encoder updates, and saves the best-val checkpoint and full history.

    :param config: CONFIGS key selecting the ablation knobs.
    :param region: 'mouth' or 'nose'.
    :param init_seed: model/optimisation seed.
    :param split_seed: data-split seed.
    :param fold: 1-indexed fold.
    :param n_folds: number of cv folds.
    :param dataset_dir: dataset root.
    :param device: torch device.
    :param epochs: number of training epochs.
    :param lr: Adam learning rate.
    :param latent_dim: latent dimensionality.
    :param variant: VARIANT_MAP key selecting the model architecture.
    :param beta_max_override: beta_max overriding config/region default if not None.
    :param part_dropout: probability of dropping the participant token to the null token.
    :param cv_mode: 'kfold' or 'loso'.
    :param exclude_subjects: subjects held out under nested LOSO.
    :param loso_trial_val: under LOSO, draw the val split from held-out trials.
    :param verbose: show the per-epoch progress bar.
    :return: None.
    """
    loso_tag = loso_path_tag(loso_trial_val, exclude_subjects)
    cfg = CONFIGS[config]
    beta_max = resolve_beta(cfg, region, beta_max_override)
    lag_n = cfg.get("lag_n", 0)
    lag_phase = cfg.get("lag_phase", 0)
    free_bits = cfg.get("free_bits", 0.0)
    alpha = cfg.get("alpha", 0.0)
    n_copies = cfg.get("n_copies", 1)
    warmup_frac = cfg.get("warmup_frac", 0.5)
    warmup_epochs = int(epochs * warmup_frac)

    # reproducibility
    seed_everything(init_seed)
    g = make_generator(init_seed)

    # data
    df = load_dataset(dataset_dir)
    df = df[df["region"] == region].reset_index(drop=True)
    num_participants = df["participant"].nunique()  # embedding table covers all subjects; derive before split
    df_train, df_val, _ = get_split(df, cv_mode=cv_mode, fold=fold - 1, n_folds=n_folds, split_seed=split_seed, exclude_subjects=exclude_subjects, loso_trial_val=loso_trial_val)
    train_ds = BreathDataset(df_train, alpha=alpha, n_copies=n_copies)
    train_ds_clean = BreathDataset(df_train, stats=train_ds.stats)
    val_ds = BreathDataset(df_val, stats=train_ds.stats)

    if alpha > 0 and n_copies > 1:
        print(f"[{TAG}] Jitter: {len(train_ds_clean)} -> {len(train_ds)} samples (alpha={alpha}, n_copies={n_copies})")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # model (selected variant)
    model_cls = VARIANT_MAP[variant]
    model = model_cls(latent_dim=latent_dim, embed_dim=EMBED_DIM, condition_on_participant=True, num_participants=num_participants, part_embed_dim=PART_EMBED_DIM, part_dropout=part_dropout).to(device)

    opt_full = torch.optim.Adam(model.parameters(), lr=lr)
    opt_enc = torch.optim.Adam(model.encoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_full, T_max=epochs)

    best_val_loss = float("inf")
    history: list[dict] = []

    epoch_bar = tqdm(range(1, epochs + 1), desc=f"[{TAG}] {config}|{region}|is{init_seed}|f{fold}", unit="epoch", disable=not verbose)
    for epoch in epoch_bar:
        beta = beta_capped(epoch, warmup_epochs, beta_max)
        in_aggressive_phase = lag_n > 0 and epoch <= lag_phase

        if in_aggressive_phase:
            model.train()
            t_loss = t_recon = t_kl = 0.0
            n_batches = 0
            for signal, _time, label, participant in train_loader:
                signal = signal.to(device)
                label = label.long().to(device)
                participant = participant.long().to(device)

                for _ in range(lag_n):
                    opt_enc.zero_grad()
                    x_hat, mu, logvar = model(signal, label, participant)
                    loss_enc, _, _ = elbo_loss(signal, x_hat, mu, logvar, beta, free_bits)
                    loss_enc.backward()
                    opt_enc.step()

                # one full update per batch
                opt_full.zero_grad()
                x_hat, mu, logvar = model(signal, label, participant)
                loss, recon, kl = elbo_loss(signal, x_hat, mu, logvar, beta, free_bits)
                loss.backward()
                opt_full.step()

                t_loss += loss.item()
                t_recon += recon.item()
                t_kl += kl.item()
                n_batches += 1
            t_loss /= n_batches
            t_recon /= n_batches
            t_kl /= n_batches
        else:
            t_loss, t_recon, t_kl = train_vae_one_epoch(model, train_loader, opt_full, epoch, warmup_epochs, device, free_bits=free_bits, conditional=True, use_participant=True, beta_max=beta_max)

        # validation
        v_loss, v_recon, v_kl = evaluate(model, val_loader, epoch, warmup_epochs, device, free_bits=free_bits, conditional=True, use_participant=True, beta_max=beta_max)
        scheduler.step()

        if v_loss < best_val_loss and beta >= beta_max:
            best_val_loss = v_loss
            os.makedirs(RESULTS_DIR, exist_ok=True)
            torch.save({"config": config,
                        "region": region,
                        "cv_mode": cv_mode,
                        "init_seed": init_seed,
                        "split_seed": split_seed,
                        "fold": fold,
                        "n_folds": n_folds,
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "stats": train_ds.stats,
                        "variant": variant,
                        "latent_dim": latent_dim,
                        "embed_dim": EMBED_DIM,
                        "part_embed_dim": PART_EMBED_DIM,
                        "num_participants": num_participants,
                        "beta_max": beta_max,
                        "lag_n": lag_n,
                        "lag_phase": lag_phase,
                        "free_bits": free_bits,
                        "alpha": alpha,
                        "n_copies": n_copies,
                        "warmup_frac": warmup_frac,
                        "part_dropout": part_dropout,
                        "null_fire_frac": model.null_fire_frac()},
                       ckpt_path(variant, beta_max, config, region, init_seed, split_seed, fold, n_folds, part_dropout, cv_mode, loso_tag))

        # active dims
        if epoch % LOG_EVERY == 0 or epoch == 1:
            n_active = active_dims(model, train_ds_clean, device, conditional=True, use_participant=True)
        else:
            n_active = history[-1]["active_dims"] if history else 0

        history.append({"epoch": epoch,
                        "beta": beta,
                        "train_loss": t_loss, "train_recon": t_recon, "train_kl": t_kl,
                        "val_loss": v_loss, "val_recon": v_recon, "val_kl": v_kl,
                        "active_dims": n_active})

        epoch_bar.set_postfix({"beta": f"{beta:.5f}",
                               "lag": lag_n if in_aggressive_phase else 0,
                               "train": f"{t_loss:.4f}",
                               "val": f"{v_loss:.4f}",
                               "active": n_active})

    # persist full training history
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(hist_path(variant, beta_max, config, region, init_seed, split_seed, fold, n_folds, part_dropout, cv_mode, loso_tag), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    if part_dropout > 0.0:
        print(f"  [{TAG}] null-token fired {model._null_steps}/{model._total_steps} steps = {model.null_fire_frac():.3f} (target {part_dropout})")
    print(f"  [{TAG}] {config}|{region}|is{init_seed}|f{fold}|cv={cv_mode}{loso_tag} | Done. best_val={best_val_loss:.4f} | ckpt: {ckpt_path(variant, beta_max, config, region, init_seed, split_seed, fold, n_folds, part_dropout, cv_mode, loso_tag)}")

# tstr evaluation
def eval_config(config: str, region: str, init_seed: int, split_seed: int, fold: int, n_folds: int, cache: dict, device: torch.device, n_jobs: int, variant: str = "conv_baseline", beta_max_override: float | None = None, part_dropout: float = 0.0, cv_mode: str = "kfold", exclude_subjects: tuple = (), loso_trial_val: bool = False) -> dict:
    """Load an ablation checkpoint, run TSTR, and build the result row.

    Reloads the trained variant, scores it with TSTR against the real test fold, runs the
    null-token sanity check when participant dropout was used, and reads the final KL and
    active-dim counts from the history CSV.

    :param config: CONFIGS key selecting the ablation knobs.
    :param region: 'mouth' or 'nose'.
    :param init_seed: model/sampling seed.
    :param split_seed: data-split seed.
    :param fold: 1-indexed fold.
    :param n_folds: number of cv folds.
    :param cache: trtr cache providing top features, the real test set, and n_train.
    :param device: torch device.
    :param n_jobs: parallel workers.
    :param variant: VARIANT_MAP key selecting the model architecture.
    :param beta_max_override: beta_max overriding config/region default if not None.
    :param part_dropout: participant dropout used at train time (affects the path).
    :param cv_mode: 'kfold' or 'loso'.
    :param exclude_subjects: subjects held out under nested LOSO.
    :param loso_trial_val: under LOSO, whether the val split used held-out trials.
    :return: result row dict with config/seed/hyperparameter fields, the TSTR metrics
        (accuracy, f1_weighted, roc_auc_ovr, log_loss, per-class f1), null-gen diagnostics,
        and kl_final/active_dims.
    """
    loso_tag = loso_path_tag(loso_trial_val, exclude_subjects)
    cfg = CONFIGS[config]
    beta_max = resolve_beta(cfg, region, beta_max_override)
    path = ckpt_path(variant, beta_max, config, region, init_seed, split_seed, fold, n_folds, part_dropout, cv_mode, loso_tag)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    variant = ckpt.get("variant", "conv_baseline") # backward compat with old ckpts
    pd_eff = ckpt.get("part_dropout", 0.0) # rebuild with the same embedding size that was trained
    model_cls = VARIANT_MAP[variant]
    model = model_cls(latent_dim=ckpt["latent_dim"],
                      embed_dim=ckpt["embed_dim"],
                      condition_on_participant=True,
                      num_participants=ckpt.get("num_participants", 3),
                      part_embed_dim=ckpt["part_embed_dim"],
                      part_dropout=pd_eff).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    label = f"{config}|{region}|is{init_seed}|f{fold}|cv={cv_mode}"
    metrics = _tstr_score(model, ckpt["stats"], cache, device, n_jobs, init_seed, label, cv_mode=cv_mode)
    null_gen = _null_gen_stats(model, device) if pd_eff > 0.0 else {"null_gen_finite": None, "null_gen_std": None}
    kl_final, active_dims_final = _final_kl_active(hist_path(variant, beta_max, config, region, init_seed, split_seed, fold, n_folds, part_dropout, cv_mode, loso_tag))

    return {"config": config,
            "region": region,
            "cv_mode": cv_mode,
            "loso_tag": loso_tag,
            "init_seed": init_seed,
            "split_seed": split_seed,
            "fold": fold,
            "n_folds": n_folds,
            # broken-out hyperparameters
            "variant": variant,
            "beta_max": beta_max,
            "lag_n": cfg.get("lag_n", 0),
            "lag_phase": cfg.get("lag_phase", 0),
            "free_bits": cfg.get("free_bits", 0.0),
            "alpha": cfg.get("alpha", 0.0),
            "n_copies": cfg.get("n_copies", 1),
            "warmup_frac": cfg.get("warmup_frac", 0.5),
            # constants for this ablation
            "latent_dim": ckpt["latent_dim"],
            "embed_dim": ckpt["embed_dim"],
            "part_embed_dim": ckpt["part_embed_dim"],
            "condition_on_participant": True,
            "part_dropout": pd_eff,
            "null_fire_frac": ckpt.get("null_fire_frac"),
            "null_gen_finite": null_gen["null_gen_finite"],
            "null_gen_std": null_gen["null_gen_std"],
            "architecture": variant,
            "accuracy": metrics["accuracy"],
            "f1_weighted": metrics["f1_weighted"],
            "roc_auc_ovr": metrics["roc_auc_ovr"],
            "log_loss": metrics["log_loss"],
            "f1_bradypnea": metrics["per_class_f1"]["bradypnea"],
            "f1_eupnea": metrics["per_class_f1"]["eupnea"],
            "f1_tachypnea": metrics["per_class_f1"]["tachypnea"],
            "kl_final": kl_final,
            "active_dims": active_dims_final}

## architecture family
# training
def train_variant(variant: str, region: str, init_seed: int, split_seed: int, fold: int, n_folds: int, cond_part: bool, dataset_dir: str, device: torch.device, epochs: int, lr: float, verbose: bool = False) -> None:
    """Train one architecture variant under fixed hyperparameters, saving best-val and history.

    Trains the selected variant on a k-fold split at the architecture-sweep constants
    (ARCH_BETA_MAX, ARCH_WARMUP_FRAC, no free bits), checkpoints the best-val model once beta
    has fully warmed up, and writes the full training history.

    :param variant: VARIANT_MAP key selecting the model architecture.
    :param region: 'mouth' or 'nose'.
    :param init_seed: model/optimisation seed.
    :param split_seed: data-split seed.
    :param fold: 1-indexed fold.
    :param n_folds: number of cv folds.
    :param cond_part: condition the model on the participant token.
    :param dataset_dir: dataset root.
    :param device: torch device.
    :param epochs: number of training epochs.
    :param lr: Adam learning rate.
    :param verbose: show the per-epoch progress bar.
    :return: None.
    """
    warmup_epochs = int(epochs * ARCH_WARMUP_FRAC)

    # reproducibility
    seed_everything(init_seed)
    g = make_generator(init_seed)

    # data
    df = load_dataset(dataset_dir)
    df = df[df["region"] == region].reset_index(drop=True)
    df_train, df_val, _ = kfold_split_dataset(df, fold=fold - 1, n_folds=n_folds, split_seed=split_seed)
    train_ds = BreathDataset(df_train)
    val_ds = BreathDataset(df_val, stats=train_ds.stats)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # model + optimiser
    model_cls = VARIANT_MAP[variant]
    model = model_cls(latent_dim=LATENT_DIM, embed_dim=EMBED_DIM, condition_on_participant=cond_part, part_embed_dim=PART_EMBED_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    history: list[dict] = []

    epoch_bar = tqdm(range(1, epochs + 1), desc=f"[{TAG}] {variant}|{region}|is{init_seed}|f{fold}|cp={cond_part}", unit="epoch", disable=not verbose)
    for epoch in epoch_bar:
        # train and validate
        t_loss, t_recon, t_kl = train_vae_one_epoch(model, train_loader, optimizer, epoch, warmup_epochs, device, free_bits=0.0, conditional=True, use_participant=cond_part, beta_max=ARCH_BETA_MAX)
        v_loss, v_recon, v_kl = evaluate(model, val_loader, epoch, warmup_epochs, device, free_bits=0.0, conditional=True, use_participant=cond_part, beta_max=ARCH_BETA_MAX)
        scheduler.step()
        beta = beta_capped(epoch, warmup_epochs, ARCH_BETA_MAX)

        # checkpoint only once beta has fully warmed up
        if v_loss < best_val_loss and beta >= ARCH_BETA_MAX:
            best_val_loss = v_loss
            os.makedirs(RESULTS_DIR, exist_ok=True)
            torch.save({"variant": variant,
                        "region": region,
                        "init_seed": init_seed,
                        "split_seed": split_seed,
                        "fold": fold,
                        "n_folds": n_folds,
                        "condition_on_participant": cond_part,
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "stats": train_ds.stats,
                        "latent_dim": LATENT_DIM,
                        "embed_dim": EMBED_DIM,
                        "part_embed_dim": PART_EMBED_DIM},
                       arch_ckpt_path(variant, region, init_seed, split_seed, fold, n_folds, cond_part))

        # active dims
        if epoch % LOG_EVERY == 0 or epoch == 1:
            n_active = active_dims(model, train_ds, device, conditional=True, use_participant=cond_part)
        else:
            n_active = history[-1]["active_dims"] if history else 0

        history.append({"epoch": epoch, "beta": beta,
                        "train_loss": t_loss, "train_recon": t_recon, "train_kl": t_kl,
                        "val_loss": v_loss, "val_recon": v_recon, "val_kl": v_kl,
                        "active_dims": n_active})

        epoch_bar.set_postfix({"beta": f"{beta:.2f}",
                               "train": f"{t_loss:.4f}",
                               "val": f"{v_loss:.4f}",
                               "active": n_active})

    # persist history
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(arch_hist_path(variant, region, init_seed, split_seed, fold, n_folds, cond_part), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    print(f"  [{TAG}] {variant}|{region}|is{init_seed}|f{fold}|cp={cond_part} | Done. best_val={best_val_loss:.4f} | ckpt: {arch_ckpt_path(variant, region, init_seed, split_seed, fold, n_folds, cond_part)}")

# tstr evaluation
def eval_variant(variant: str, region: str, init_seed: int, split_seed: int, fold: int, n_folds: int, cond_part: bool, cache: dict, device: torch.device, n_jobs: int) -> dict:
    """Load an architecture checkpoint, run TSTR, and build the result row.

    Reloads the trained variant, scores it with TSTR against the real test fold, and reads
    the final KL and active-dim counts from the history CSV.

    :param variant: VARIANT_MAP key selecting the model architecture.
    :param region: 'mouth' or 'nose'.
    :param init_seed: model/sampling seed.
    :param split_seed: data-split seed.
    :param fold: 1-indexed fold.
    :param n_folds: number of cv folds.
    :param cond_part: whether the model was conditioned on the participant token.
    :param cache: trtr cache providing top features, the real test set, and n_train.
    :param device: torch device.
    :param n_jobs: parallel workers.
    :return: result row dict with variant/seed/cond_part fields, the TSTR metrics (accuracy,
        f1_weighted, roc_auc_ovr, log_loss, per-class f1), and kl_final/active_dims.
    """
    path = arch_ckpt_path(variant, region, init_seed, split_seed, fold, n_folds, cond_part)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model_cls = VARIANT_MAP[variant]
    model = model_cls(latent_dim=ckpt["latent_dim"],
                      embed_dim=ckpt["embed_dim"],
                      condition_on_participant=cond_part,
                      part_embed_dim=ckpt["part_embed_dim"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    label = f"{variant}|{region}|is{init_seed}|f{fold}|cp={cond_part}"
    metrics = _tstr_score(model, ckpt["stats"], cache, device, n_jobs, init_seed, label)
    kl_final, active_dims_final = _final_kl_active(arch_hist_path(variant, region, init_seed, split_seed, fold, n_folds, cond_part))

    return {"variant": variant,
            "region": region,
            "init_seed": init_seed,
            "split_seed": split_seed,
            "fold": fold,
            "n_folds": n_folds,
            "condition_on_participant": cond_part,
            "accuracy": metrics["accuracy"],
            "f1_weighted": metrics["f1_weighted"],
            "roc_auc_ovr": metrics["roc_auc_ovr"],
            "log_loss": metrics["log_loss"],
            "f1_bradypnea": metrics["per_class_f1"]["bradypnea"],
            "f1_eupnea": metrics["per_class_f1"]["eupnea"],
            "f1_tachypnea": metrics["per_class_f1"]["tachypnea"],
            "kl_final": kl_final,
            "active_dims": active_dims_final}

# persistence
def _json_safe(obj):
    """Recursively convert numpy/NaN values into JSON-serialisable Python types."""
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj

def save_result(row: dict) -> None:
    """Write one result row to its per-combo JSON, picking the path by FAMILY.

    :param row: result row dict from eval_config/eval_variant.
    :return: None.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if FAMILY == "architecture":
        path = arch_result_path(row["variant"], row["region"], row["init_seed"], row["split_seed"], row["fold"], row["n_folds"], row["condition_on_participant"])
    else:
        path = result_path(row["variant"], row["beta_max"], row["config"], row["region"], row["init_seed"], row["split_seed"], row["fold"], row["n_folds"], row.get("part_dropout", 0.0), row.get("cv_mode", "kfold"), row.get("loso_tag", ""))
    with open(path, "w") as f:
        json.dump(_json_safe(row), f, indent=2)

def save_summary(rows: list[dict]) -> None:
    """Merge result rows into the summary CSV, de-duplicating on the family key subset.

    Appends to an existing summary.csv (backfilling columns added after earlier runs) and
    keeps the last row per combo.

    :param rows: result row dicts to merge.
    :return: None.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = f"{RESULTS_DIR}/summary.csv"
    subset = ARCH_SUBSET if FAMILY == "architecture" else ABLATION_SUBSET
    df_new = pd.DataFrame(rows)
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        if FAMILY == "architecture" and "condition_on_participant" not in df_old.columns:
            df_old["condition_on_participant"] = True
        if FAMILY != "architecture" and "part_dropout" not in df_old.columns:
            df_old["part_dropout"] = 0.0
        if FAMILY != "architecture" and "cv_mode" not in df_old.columns:
            df_old["cv_mode"] = "kfold"  # rows predating LOSO are all k-fold
        if FAMILY != "architecture" and "loso_tag" not in df_old.columns:
            df_old["loso_tag"] = ""
        df_new = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(subset=subset, keep="last")
    df_new.to_csv(csv_path, index=False)
    print(f"[{TAG}] Saved {len(df_new)} rows -> {csv_path}")

# flow: ablation
def run_ablation(args: argparse.Namespace, device: torch.device) -> None:
    """Drive the ablation family: resolve the sweep axes, then train/eval/aggregate each combo.

    Resolves the config/seed/fold/region lists from args (including jitter single-combo
    synthesis and nested-LOSO excludes). In aggregate mode it only collects existing
    result.json files into the summary; otherwise it builds the TRTR cache, trains (unless
    skip_existing finds a checkpoint), runs TSTR, and writes results plus the summary.

    :param args: parsed command-line arguments.
    :param device: torch device.
    :return: None.
    """
    variant = args.variant if args.variant else "conv_baseline"
    if variant not in VARIANT_MAP:
        print(f'Unknown variant "{variant}". Valid: {list(VARIANT_MAP)}')
        sys.exit(1)

    # nested-LOSO: 0-indexed excludes from the 1-indexed CLI; tag namespaces nested artifacts
    exclude_subjects = tuple(int(x) - 1 for x in args.loso_exclude.split(",") if x.strip()) if args.loso_exclude else ()
    loso_tag = loso_path_tag(args.loso_trial_val, exclude_subjects)

    # resolve config list
    if args.alpha is not None and args.n_copies is not None:
        if args.alpha == 0:
            configs = ["baseline"]
        else:
            name = f"a{args.alpha}_n{args.n_copies}"
            if name not in CONFIGS:
                CONFIGS[name] = {"alpha": args.alpha, "n_copies": args.n_copies}
            configs = [name]
    elif args.config is not None:
        configs = [args.config]
    else:
        configs = args.configs.split(",")

    init_seeds = [args.init_seed] if args.init_seed is not None else [int(s) for s in args.init_seeds.split(",")]
    folds = [args.fold] if args.fold is not None else [int(s) for s in args.folds.split(",")]
    regions = [args.region] if args.region else args.regions.split(",")

    for c in configs:
        if c not in CONFIGS:
            print(f'Unknown config "{c}". Valid: {list(CONFIGS)}')
            sys.exit(1)
    for r in regions:
        if r not in ("mouth", "nose"):
            print(f'Unknown region "{r}". Valid: mouth, nose')
            sys.exit(1)

    # aggregate mode: sweep over variant/beta_max axes and collect result.json files
    if args.aggregate:
        variants = [v.strip() for v in args.variants.split(",")] if args.variants else [variant]
        for v in variants:
            if v not in VARIANT_MAP:
                print(f'Unknown variant "{v}". Valid: {list(VARIANT_MAP)}')
                sys.exit(1)
        beta_axis = ([float(b) for b in args.beta_maxes.split(",")] if args.beta_maxes
                     else [args.beta_max] if args.beta_max is not None else None)
        rows = []
        for v in variants:
            for region in regions:
                for config in configs:
                    beta_maxes = beta_axis if beta_axis is not None else [resolve_beta(CONFIGS[config], region, None)]
                    for beta_max in beta_maxes:
                        for init_seed in init_seeds:
                            for fold in folds:
                                rp = result_path(v, beta_max, config, region, init_seed, args.split_seed, fold, args.n_folds, args.part_dropout, args.cv_mode, loso_tag)
                                if not os.path.exists(rp):
                                    print(f"  [AGGREGATE] missing {rp}, skipping")
                                    continue
                                with open(rp) as f:
                                    rows.append(json.load(f))
        if rows:
            save_summary(rows)
        return

    # normal mode
    print(f"[{TAG}] device={device} | variant={variant} | beta_max={args.beta_max} | configs={configs} | init_seeds={init_seeds} | folds={folds} | regions={regions}")
    all_results: list[dict] = []

    for region in regions:
        for config in configs:
            for init_seed in init_seeds:
                for fold in folds:
                    cache = load_cache(region, init_seed, args.split_seed, fold, args.n_folds, channel="both", cv_mode=args.cv_mode, loso_tag=loso_tag)
                    if cache is None:
                        print(f"[{TAG}] Building TRTR cache region={region}, init_seed={init_seed}, fold={fold}, cv={args.cv_mode}{loso_tag} ...")
                        cache = trtr(args.dataset_dir, region, args.n_jobs, init_seed, args.split_seed, fold, args.n_folds, channel="both", cv_mode=args.cv_mode, exclude_subjects=exclude_subjects, loso_trial_val=args.loso_trial_val)

                    beta_max = resolve_beta(CONFIGS[config], region, args.beta_max)
                    print(f"[{TAG}] {config} | {region} | is={init_seed} | f={fold} | cv={args.cv_mode}{loso_tag} | beta_max={beta_max}")

                    path = ckpt_path(variant, beta_max, config, region, init_seed, args.split_seed, fold, args.n_folds, args.part_dropout, args.cv_mode, loso_tag)
                    if args.skip_existing and os.path.exists(path):
                        print("  checkpoint exists, skipping training")
                    else:
                        train_config(config, region, init_seed, args.split_seed, fold, args.n_folds,
                                     args.dataset_dir, device, args.epochs, args.lr, args.latent_dim,
                                     variant=variant, beta_max_override=args.beta_max, part_dropout=args.part_dropout, cv_mode=args.cv_mode, exclude_subjects=exclude_subjects, loso_trial_val=args.loso_trial_val, verbose=args.verbose)

                    row = eval_config(config, region, init_seed, args.split_seed, fold, args.n_folds, cache, device, args.n_jobs,
                                      variant=variant, beta_max_override=args.beta_max, part_dropout=args.part_dropout, cv_mode=args.cv_mode, exclude_subjects=exclude_subjects, loso_trial_val=args.loso_trial_val)
                    save_result(row)
                    all_results.append(row)
                    print(f"  TSTR: acc={row['accuracy']:.3f}, f1={row['f1_weighted']:.3f}, roc={row['roc_auc_ovr']:.3f}, log_loss={row['log_loss']:.3f}, kl={row['kl_final']:.4f}, active_dims={row['active_dims']}")

    if all_results and not args.no_summary:
        save_summary(all_results)

## flow: architecture
def run_architecture(args: argparse.Namespace, device: torch.device) -> None:
    """Drive the architecture family: sweep variants x cond_part, then train/eval/aggregate.

    Resolves the variant/seed/fold/region/cond_part lists from args. In aggregate mode it
    only collects existing result.json files into the summary; otherwise it builds the TRTR
    cache, trains each variant (unless skip_existing finds a checkpoint), runs TSTR, and
    writes results plus the summary.

    :param args: parsed command-line arguments.
    :param device: torch device.
    :return: None.
    """
    variants = ([args.variant] if args.variant
                else args.variants.split(",") if args.variants
                else list(VARIANT_MAP))
    init_seeds = [args.init_seed] if args.init_seed is not None else [int(s) for s in args.init_seeds.split(",")]
    folds = [args.fold] if args.fold is not None else [int(s) for s in args.folds.split(",")]
    regions = [args.region] if args.region else args.regions.split(",")
    cond_parts = [_parse_bool(args.cond_part)] if args.cond_part is not None \
                 else [_parse_bool(s) for s in args.cond_parts.split(",")]

    for v in variants:
        if v not in VARIANT_MAP:
            print(f'Unknown variant "{v}". Valid: {list(VARIANT_MAP)}')
            sys.exit(1)
    for r in regions:
        if r not in ("mouth", "nose"):
            print(f'Unknown region "{r}". Valid: mouth, nose')
            sys.exit(1)

    # aggregate mode: collect existing per-combo result jsons
    if args.aggregate:
        rows = []
        for region in regions:
            for variant in variants:
                for init_seed in init_seeds:
                    for fold in folds:
                        for cp in cond_parts:
                            rp = arch_result_path(variant, region, init_seed, args.split_seed, fold, args.n_folds, cp)
                            if not os.path.exists(rp):
                                print(f"  [AGGREGATE] missing {rp}, skipping")
                                continue
                            with open(rp) as f:
                                rows.append(json.load(f))
        if rows:
            save_summary(rows)
        return

    # normal mode: train and evaluate
    print(f"[{TAG}] device={device} | variants={variants} | init_seeds={init_seeds} | folds={folds} | regions={regions} | cond_parts={cond_parts}")
    all_results: list[dict] = []

    for region in regions:
        for variant in variants:
            for init_seed in init_seeds:
                for fold in folds:
                    cache = load_cache(region, init_seed, args.split_seed, fold, args.n_folds, channel="both")
                    if cache is None:
                        print(f"[{TAG}] Building TRTR cache region={region}, init_seed={init_seed}, fold={fold} ...")
                        cache = trtr(args.dataset_dir, region, args.n_jobs, init_seed, args.split_seed, fold, args.n_folds, channel="both")

                    for cp in cond_parts:
                        print(f"\n[{TAG}] {variant} | {region} | is={init_seed} | f={fold}/{args.n_folds} | cp={cp}")

                        path = arch_ckpt_path(variant, region, init_seed, args.split_seed, fold, args.n_folds, cp)
                        if args.skip_existing and os.path.exists(path):
                            print("  checkpoint exists, skipping training")
                        else:
                            train_variant(variant, region, init_seed, args.split_seed, fold, args.n_folds, cp,
                                          args.dataset_dir, device, args.epochs, args.lr, verbose=args.verbose)

                        row = eval_variant(variant, region, init_seed, args.split_seed, fold, args.n_folds, cp, cache, device, args.n_jobs)
                        save_result(row)
                        all_results.append(row)
                        print(f"  TSTR: acc={row['accuracy']:.3f}, f1={row['f1_weighted']:.3f}, roc={row['roc_auc_ovr']:.3f}, log_loss={row['log_loss']:.3f}, kl={row['kl_final']:.3f}, active_dims={row['active_dims']}")

    if all_results and not args.no_summary:
        save_summary(all_results)

# main
def main() -> None:
    """Parse args, set the mode globals, and dispatch to the architecture or ablation flow."""
    args = parse_args()

    # modes
    global RESULTS_DIR, CKPT_SCHEME, FAMILY, TAG
    RESULTS_DIR = MODES[args.mode]["results_dir"]
    CKPT_SCHEME = MODES[args.mode]["ckpt"]
    FAMILY = MODES[args.mode]["family"]
    TAG = "ARCH" if FAMILY == "architecture" else args.mode.upper()
    RESULTS_DIR = os.environ.get("CVAE_RESULTS_DIR", RESULTS_DIR)

    device = torch.device("cpu")

    if FAMILY == "architecture":
        run_architecture(args, device)
    else:
        run_ablation(args, device)

if __name__ == "__main__":
    main()
