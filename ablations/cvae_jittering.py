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

from core.data import BreathDataset, load_dataset, kfold_split_dataset
from core.train import active_dims, beta_capped, evaluate, train_vae_one_epoch
from core.tstr import (evaluate_classifier, extract_fixed_features, generate_synthetic_signals,
                       load_cache, train_stacking_classifier, trtr)
from core.utils import make_generator, seed_everything, seed_worker
from models.vae import CVAE

CONFIGS: dict[str, dict] = {"baseline": {"alpha": 0.0, "n_copies": 0},
                            "a0.01_n2": {"alpha": 0.01, "n_copies": 2},
                            "a0.01_n5": {"alpha": 0.01, "n_copies": 5},
                            "a0.01_n10": {"alpha": 0.01, "n_copies": 10},
                            "a0.05_n2": {"alpha": 0.05, "n_copies": 2},
                            "a0.05_n5": {"alpha": 0.05, "n_copies": 5},
                            "a0.05_n10": {"alpha": 0.05, "n_copies": 10},
                            "a0.1_n2": {"alpha": 0.1, "n_copies": 2},
                            "a0.1_n5": {"alpha": 0.1, "n_copies": 5},
                            "a0.1_n10": {"alpha": 0.1, "n_copies": 10},
                            "a0.025_n5":  {"alpha": 0.025, "n_copies": 5},
                            "a0.025_n10": {"alpha": 0.025, "n_copies": 10},}

# fix parameters
BETA_MAX = {"mouth": 0.1, "nose": 0.1} # adjust according to results of training dynamics ablation
LATENT_DIM = 16
EMBED_DIM = 8
PART_EMBED_DIM = 8
BATCH_SIZE = 32
LOG_EVERY = 25
WARMUP_FRAC = 0.5
RESULTS_DIR = "results/ablation_cvae_jittering"

# cli
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CVAE jittering augmentation ablation")
    p.add_argument("--configs", default=",".join(CONFIGS))
    p.add_argument("--init_seeds", default="0,1,7,42,123")
    p.add_argument("--folds", default="1,2,3,4,5")
    p.add_argument("--regions", default="mouth,nose")
    # single-combo overrides
    p.add_argument("--config", default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--n_copies", type=int, default=None)
    p.add_argument("--init_seed", type=int, default=None)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--region", default=None)
    # protocol-level constants
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--dataset_dir", default="dataset")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n_jobs", type=int, default=4)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no_summary", action="store_true")
    p.add_argument("--aggregate", action="store_true")
    return p.parse_args()

# paths
def _stem(config: str, region: str, init_seed: int, split_seed: int, fold: int, n_folds: int) -> str:
    return f"{config}_{region}_is{init_seed}_ss{split_seed}_fold{fold}of{n_folds}"

def ckpt_path(config: str, region: str, init_seed: int, split_seed: int, fold: int, n_folds: int) -> str:
    return f"{RESULTS_DIR}/{_stem(config, region, init_seed, split_seed, fold, n_folds)}.pt"

def hist_path(config: str, region: str, init_seed: int, split_seed: int, fold: int, n_folds: int) -> str:
    return f"{RESULTS_DIR}/{_stem(config, region, init_seed, split_seed, fold, n_folds)}_history.csv"

def result_path(config: str, region: str, init_seed: int, split_seed: int, fold: int, n_folds: int) -> str:
    return f"{RESULTS_DIR}/{_stem(config, region, init_seed, split_seed, fold, n_folds)}_result.json"

# training
def train_config(config: str, region: str, init_seed: int, split_seed: int, fold: int, n_folds: int, dataset_dir: str, device: torch.device, epochs: int, lr: float, verbose: bool = False) -> None:
    cfg = CONFIGS[config]
    alpha = cfg["alpha"]
    n_copies = cfg["n_copies"]
    beta_max = BETA_MAX[region]
    warmup_epochs = int(epochs * WARMUP_FRAC)

    # reproducibility
    seed_everything(init_seed)
    g = make_generator(init_seed)

    # data
    df = load_dataset(dataset_dir)
    df = df[df["region"] == region].reset_index(drop=True)
    df_train, df_val, _ = kfold_split_dataset(df, fold=fold - 1, n_folds=n_folds, split_seed=split_seed)
    train_ds = BreathDataset(df_train, alpha=alpha, n_copies=n_copies)
    train_ds_clean = BreathDataset(df_train, stats=train_ds.stats)
    val_ds = BreathDataset(df_val, stats=train_ds.stats)

    if alpha > 0 and n_copies > 1:
        print(f"[JITTER] Jitter: {len(train_ds_clean)} → {len(train_ds)} samples (alpha={alpha}, n_copies={n_copies})")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # model
    model = CVAE(latent_dim=LATENT_DIM, embed_dim=EMBED_DIM, condition_on_participant=True, part_embed_dim=PART_EMBED_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    history: list[dict] = []

    epoch_bar = tqdm(range(1, epochs + 1), desc=f"[JITTER] {config}|{region}|is{init_seed}|f{fold}", unit="epoch", disable=not verbose)
    for epoch in epoch_bar:
        # train and validate
        t_loss, t_recon, t_kl = train_vae_one_epoch(model, train_loader, optimizer, epoch, warmup_epochs, device, free_bits=0.0, conditional=True, use_participant=True, beta_max=beta_max)
        v_loss, v_recon, v_kl = evaluate(model, val_loader, epoch, warmup_epochs, device, free_bits=0.0, conditional=True, use_participant=True, beta_max=beta_max)
        scheduler.step()
        beta = beta_capped(epoch, warmup_epochs, beta_max)

        # checkpoint: best val once beta has fully warmed up
        if v_loss < best_val_loss and beta >= beta_max:
            best_val_loss = v_loss
            os.makedirs(RESULTS_DIR, exist_ok=True)
            torch.save({"config": config,
                        "region": region,
                        "init_seed": init_seed,
                        "split_seed": split_seed,
                        "fold": fold,
                        "n_folds": n_folds,
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "stats": train_ds.stats,
                        "latent_dim": LATENT_DIM,
                        "embed_dim": EMBED_DIM,
                        "part_embed_dim": PART_EMBED_DIM,
                        "beta_max": beta_max,
                        "alpha": alpha,
                        "n_copies": n_copies},
                       ckpt_path(config, region, init_seed, split_seed, fold, n_folds))

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
                               "train": f"{t_loss:.4f}",
                               "val": f"{v_loss:.4f}",
                               "active": n_active})

    # persist full training history
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(hist_path(config, region, init_seed, split_seed, fold, n_folds), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    print(f"  [JITTER] {config}|{region}|is{init_seed}|f{fold} | Done. best_val={best_val_loss:.4f} | ckpt: {ckpt_path(config, region, init_seed, split_seed, fold, n_folds)}")

# tstr evaluation
def eval_config(config: str, region: str, init_seed: int, split_seed: int, fold: int, n_folds: int, cache: dict, device: torch.device, n_jobs: int) -> dict:
    path = ckpt_path(config, region, init_seed, split_seed, fold, n_folds)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = CVAE(latent_dim=ckpt["latent_dim"], embed_dim=ckpt["embed_dim"], condition_on_participant=True, part_embed_dim=ckpt["part_embed_dim"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    stats = ckpt["stats"]

    # generate synthetic signals
    n_synthetic = cache["n_train"]
    synth_signals, synth_labels = generate_synthetic_signals(model, "cvae_part", n_synthetic, stats, device, init_seed)
    print(f"  [JITTER] {config}|{region}|is{init_seed}|f{fold} | Generated {n_synthetic} synthetic signals")

    # tsfresh features using TRTR cache top-20
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
    metrics = evaluate_classifier(stacker, cache["X_test_top"], cache["y_test"])

    kl_final = float("nan")
    active_dims_final = 0
    hp = hist_path(config, region, init_seed, split_seed, fold, n_folds)
    if os.path.exists(hp):
        hist_df = pd.read_csv(hp)
        kl_final = float(hist_df["train_kl"].iloc[-1])
        active_dims_final = int(hist_df["active_dims"].iloc[-1])

    return {"config": config,
            "alpha": CONFIGS[config]["alpha"],
            "n_copies": CONFIGS[config]["n_copies"],
            "region": region,
            "init_seed": init_seed,
            "split_seed": split_seed,
            "fold": fold,
            "n_folds": n_folds,
            "accuracy": metrics["accuracy"],
            "f1_weighted": metrics["f1_weighted"],
            "roc_auc_ovr": metrics["roc_auc_ovr"],
            "log_loss": metrics["log_loss"],
            "f1_bradypnea": metrics["per_class_f1"]["bradypnea"],
            "f1_eupnea": metrics["per_class_f1"]["eupnea"],
            "f1_tachypnea": metrics["per_class_f1"]["tachypnea"],
            "active_dims": active_dims_final,
            "kl_final": kl_final}

# persistence
def _json_safe(obj):
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
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = result_path(row["config"], row["region"], row["init_seed"], row["split_seed"], row["fold"], row["n_folds"])
    with open(path, "w") as f:
        json.dump(_json_safe(row), f, indent=2)

def save_summary(rows: list[dict]) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = f"{RESULTS_DIR}/summary.csv"
    df_new = pd.DataFrame(rows)
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df_new = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(
            subset=["config", "region", "init_seed", "split_seed", "fold", "n_folds"], keep="last")
    df_new.to_csv(csv_path, index=False)
    print(f"[JITTER] Saved {len(df_new)} rows → {csv_path}")

# main
def main() -> None:
    args = parse_args()

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

    device = torch.device("cpu")

    # aggregate mode
    if args.aggregate:
        rows = []
        for region in regions:
            for config in configs:
                for init_seed in init_seeds:
                    for fold in folds:
                        rp = result_path(config, region, init_seed, args.split_seed, fold, args.n_folds)
                        if not os.path.exists(rp):
                            print(f"  [AGGREGATE] missing {rp}, skipping")
                            continue
                        with open(rp) as f:
                            rows.append(json.load(f))
        if rows:
            save_summary(rows)
        return

    # normal mode
    print(f"[JITTER] device={device} | configs={configs} | init_seeds={init_seeds} | folds={folds} | regions={regions}")
    all_results: list[dict] = []

    for region in regions:
        for config in configs:
            for init_seed in init_seeds:
                for fold in folds:
                    cache = load_cache(region, init_seed, args.split_seed, fold, args.n_folds, channel="both")
                    if cache is None:
                        print(f"[JITTER] Building TRTR cache region={region}, init_seed={init_seed}, fold={fold} ...")
                        cache = trtr(args.dataset_dir, region, args.n_jobs, init_seed, args.split_seed, fold, args.n_folds, channel="both")

                    print(f"\n[JITTER] {config} | {region} | is={init_seed} | f={fold}/{args.n_folds}")

                    path = ckpt_path(config, region, init_seed, args.split_seed, fold, args.n_folds)
                    if args.skip_existing and os.path.exists(path):
                        print("  checkpoint exists, skipping training")
                    else:
                        train_config(config, region, init_seed, args.split_seed, fold, args.n_folds, args.dataset_dir, device, args.epochs, args.lr, verbose=args.verbose)

                    row = eval_config(config, region, init_seed, args.split_seed, fold, args.n_folds, cache, device, args.n_jobs)
                    save_result(row)
                    all_results.append(row)
                    print(f"  TSTR: acc={row['accuracy']:.3f}, f1={row['f1_weighted']:.3f}, roc={row['roc_auc_ovr']:.3f}, log_loss={row['log_loss']:.3f}, kl={row['kl_final']:.4f}, active_dims={row['active_dims']}")

    if all_results and not args.no_summary:
        save_summary(all_results)

if __name__ == "__main__":
    main()
