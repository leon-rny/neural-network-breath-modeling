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
from core.tstr import evaluate_classifier, extract_fixed_features, generate_synthetic_signals, load_cache, train_stacking_classifier, trtr
from core.utils import make_generator, seed_everything, seed_worker
from models.vae import elbo_loss
from ablations.cvae_architecture_models import VARIANT_MAP

CONFIGS: dict[str, dict] = {
    # beta-cap sweep
    "beta_cap_0.001": {"beta_max": 0.001, "lag_n": 0, "lag_phase": 0},
    "beta_cap_0.01": {"beta_max": 0.01, "lag_n": 0, "lag_phase": 0},
    "beta_cap_0.1": {"beta_max": 0.1, "lag_n": 0, "lag_phase": 0},
    "beta_cap_0.5": {"beta_max": 0.5, "lag_n": 0, "lag_phase": 0},
    "beta_cap_1.0": {"beta_max": 1.0, "lag_n": 0, "lag_phase": 0}, # baseline
    # free-bits sweep at best performing beta_max=0.1
    "fb_0.1_b0.1": {"beta_max": 0.1, "lag_n": 0, "lag_phase": 0, "free_bits": 0.1},
    "fb_0.5_b0.1": {"beta_max": 0.1, "lag_n": 0, "lag_phase": 0, "free_bits": 0.5},
    "fb_1.0_b0.1": {"beta_max": 0.1, "lag_n": 0, "lag_phase": 0, "free_bits": 1.0},
    "fb_2.0_b0.1": {"beta_max": 0.1, "lag_n": 0, "lag_phase": 0, "free_bits": 2.0},
    # lagging inference with beta_max=1.0
    "lag_5_100": {"beta_max": 1.0, "lag_n": 5, "lag_phase": 100},
    "lag_5_250": {"beta_max": 1.0, "lag_n": 5, "lag_phase": 250},
    "lag_10_100": {"beta_max": 1.0, "lag_n": 10, "lag_phase": 100},
    "lag_10_250": {"beta_max": 1.0, "lag_n": 10, "lag_phase": 250},
    # warmup-fraction sweep at beta_max=0.1
    "warmup_0.25": {"beta_max": 0.1, "lag_n": 0, "lag_phase": 0, "warmup_frac": 0.25},
    "warmup_0.75": {"beta_max": 0.1, "lag_n": 0, "lag_phase": 0, "warmup_frac": 0.75},
    # joint architecture x beta sweep (beta_max overridden by --beta_max)
    "joint": {"beta_max": 0.1, "lag_n": 0, "lag_phase": 0},
}
# fix parameters
LATENT_DIM = 16
EMBED_DIM = 8
PART_EMBED_DIM = 8
BATCH_SIZE = 32
LOG_EVERY = 25
RESULTS_DIR = "results/ablation_cvae_training_dynamics"

# cli
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CVAE training-dynamics ablation")
    p.add_argument("--configs", default=",".join(CONFIGS))
    p.add_argument("--init_seeds", default="0,1,7,42,123")
    p.add_argument("--folds", default="1,2,3,4,5")
    p.add_argument("--regions", default="mouth,nose")
    # single-combo overrides
    p.add_argument("--config", default=None)
    p.add_argument("--init_seed", type=int, default=None)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--region", default=None)
    # joint architecture x beta sweep
    p.add_argument("--variant", default="conv_baseline", choices=list(VARIANT_MAP))
    p.add_argument("--beta_max", type=float, default=None)
    # plural sweep axes (aggregate mode only)
    p.add_argument("--variants", default=None)
    p.add_argument("--beta_maxes", default=None)
    # protocol-level constants
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--dataset_dir", default="dataset")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--n_jobs", type=int, default=4)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no_summary", action="store_true")
    p.add_argument("--aggregate", action="store_true")
    return p.parse_args()

# paths
def _stem(variant: str, beta_max: float, config: str, region: str, init_seed: int, split_seed: int, fold: int, n_folds: int) -> str:
    return f"{variant}_b{beta_max}_{config}_{region}_is{init_seed}_ss{split_seed}_fold{fold}of{n_folds}"

def ckpt_path(variant: str, beta_max: float, config: str, region: str, init_seed: int, split_seed: int, fold: int, n_folds: int) -> str:
    return f"{RESULTS_DIR}/{variant}_b{beta_max}_{config}_{region}_is{init_seed}_ss{split_seed}_f{fold}of{n_folds}_checkpoint.pt"

def hist_path(variant: str, beta_max: float, config: str, region: str, init_seed: int, split_seed: int, fold: int, n_folds: int) -> str:
    return f"{RESULTS_DIR}/{_stem(variant, beta_max, config, region, init_seed, split_seed, fold, n_folds)}_history.csv"

def result_path(variant: str, beta_max: float, config: str, region: str, init_seed: int, split_seed: int, fold: int, n_folds: int) -> str:
    return f"{RESULTS_DIR}/{_stem(variant, beta_max, config, region, init_seed, split_seed, fold, n_folds)}_result.json"

# training
def train_config(config: str, region: str, init_seed: int, split_seed: int, fold: int, n_folds: int, dataset_dir: str, device: torch.device, epochs: int, lr: float, latent_dim: int, variant: str = "conv_baseline", beta_max_override: float | None = None, verbose: bool = False) -> None:
    cfg = CONFIGS[config]
    beta_max = beta_max_override if beta_max_override is not None else cfg["beta_max"]
    lag_n = cfg["lag_n"]
    lag_phase = cfg["lag_phase"]
    free_bits = cfg.get("free_bits", 0.0)
    warmup_frac = cfg.get("warmup_frac", 0.5)
    warmup_epochs = int(epochs * warmup_frac)

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

    # model (selected variant)
    model_cls = VARIANT_MAP[variant]
    model = model_cls(latent_dim=latent_dim, embed_dim=EMBED_DIM, condition_on_participant=True, part_embed_dim=PART_EMBED_DIM).to(device)

    opt_full = torch.optim.Adam(model.parameters(), lr=lr)
    opt_enc = torch.optim.Adam(model.encoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_full, T_max=epochs)

    best_val_loss = float("inf")
    history: list[dict] = []

    epoch_bar = tqdm(range(1, epochs + 1), desc=f"[DYNAMICS] {config}|{region}|is{init_seed}|f{fold}", unit="epoch", disable=not verbose)
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
                        "beta_max": beta_max,
                        "lag_n": lag_n,
                        "lag_phase": lag_phase,
                        "free_bits": free_bits,
                        "warmup_frac": warmup_frac},
                       ckpt_path(variant, beta_max, config, region, init_seed, split_seed, fold, n_folds))

        # active dims
        if epoch % LOG_EVERY == 0 or epoch == 1:
            n_active = active_dims(model, train_ds, device, conditional=True, use_participant=True)
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
    with open(hist_path(variant, beta_max, config, region, init_seed, split_seed, fold, n_folds), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    print(f"  [DYNAMICS] {config}|{region}|is{init_seed}|f{fold} | Done. best_val={best_val_loss:.4f} | ckpt: {ckpt_path(variant, beta_max, config, region, init_seed, split_seed, fold, n_folds)}")

# tstr evaluation
def eval_config(config: str, region: str, init_seed: int, split_seed: int, fold: int, n_folds: int, cache: dict, device: torch.device, n_jobs: int, variant: str = "conv_baseline", beta_max_override: float | None = None) -> dict:
    cfg = CONFIGS[config]
    beta_max = beta_max_override if beta_max_override is not None else cfg["beta_max"]
    path = ckpt_path(variant, beta_max, config, region, init_seed, split_seed, fold, n_folds)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    variant = ckpt.get("variant", "conv_baseline") # backward compat with old ckpts
    model_cls = VARIANT_MAP[variant]
    model = model_cls(latent_dim=ckpt["latent_dim"],
                      embed_dim=ckpt["embed_dim"],
                      condition_on_participant=True,
                      part_embed_dim=ckpt["part_embed_dim"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    stats = ckpt["stats"]

    # generate synthetic
    n_synthetic = cache["n_train"]
    synth_signals, synth_labels = generate_synthetic_signals(model, "cvae_part", n_synthetic, stats, device, init_seed)
    print(f"  [DYNAMICS] {config}|{region}|is{init_seed}|f{fold} | Generated {n_synthetic} synthetic signals")

    # tsfresh feature extraction using the top-20 features from the TRTR cache
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
    hp = hist_path(variant, beta_max, config, region, init_seed, split_seed, fold, n_folds)
    if os.path.exists(hp):
        hist_df = pd.read_csv(hp)
        kl_final = float(hist_df["train_kl"].iloc[-1])
        active_dims_final = int(hist_df["active_dims"].iloc[-1])

    return {"config": config,
            "region": region,
            "init_seed": init_seed,
            "split_seed": split_seed,
            "fold": fold,
            "n_folds": n_folds,
            # broken-out hyperparameters (varied across this ablation)
            "variant": variant,
            "beta_max": beta_max,
            "lag_n": cfg["lag_n"],
            "lag_phase": cfg["lag_phase"],
            "free_bits": cfg.get("free_bits", 0.0),
            "warmup_frac": cfg.get("warmup_frac", 0.5),
            # constants for this ablation (logged so columns align across ablations)
            "latent_dim": ckpt["latent_dim"],
            "embed_dim": ckpt["embed_dim"],
            "part_embed_dim": ckpt["part_embed_dim"],
            "condition_on_participant": True,
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
    path = result_path(row["variant"], row["beta_max"], row["config"], row["region"], row["init_seed"], row["split_seed"], row["fold"], row["n_folds"])
    with open(path, "w") as f:
        json.dump(_json_safe(row), f, indent=2)

def save_summary(rows: list[dict]) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = f"{RESULTS_DIR}/summary.csv"
    df_new = pd.DataFrame(rows)
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df_new = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(
            subset=["variant", "beta_max", "config", "region", "init_seed", "split_seed", "fold", "n_folds"], keep="last")
    df_new.to_csv(csv_path, index=False)
    print(f"[DYNAMICS] Saved {len(df_new)} rows → {csv_path}")

# main
def main() -> None:
    args = parse_args()

    configs = [args.config] if args.config else args.configs.split(",")
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
        variants = args.variants.split(",") if args.variants else [args.variant]
        beta_maxes = [float(b) for b in args.beta_maxes.split(",")] if args.beta_maxes else [args.beta_max]
        rows = []
        for variant in variants:
            for region in regions:
                for config in configs:
                    for bm in beta_maxes:
                        beta_max = bm if bm is not None else CONFIGS[config]["beta_max"]
                        for init_seed in init_seeds:
                            for fold in folds:
                                rp = result_path(variant, beta_max, config, region, init_seed, args.split_seed, fold, args.n_folds)
                                if not os.path.exists(rp):
                                    print(f"  [AGGREGATE] missing {rp}, skipping")
                                    continue
                                with open(rp) as f:
                                    rows.append(json.load(f))
        if rows:
            save_summary(rows)
        return

    # normal mode
    print(f"[DYNAMICS] device={device} | configs={configs} | init_seeds={init_seeds} | folds={folds} | regions={regions}")
    all_results: list[dict] = []

    for region in regions:
        for config in configs:
            for init_seed in init_seeds:
                for fold in folds:
                    cache = load_cache(region, init_seed, args.split_seed, fold, args.n_folds, channel="both")
                    if cache is None:
                        print(f"[DYNAMICS] Building TRTR cache region={region}, init_seed={init_seed}, fold={fold} ...")
                        cache = trtr(args.dataset_dir, region, args.n_jobs, init_seed, args.split_seed, fold, args.n_folds, channel="both")

                    print(f"[DYNAMICS] {config} | {region} | is={init_seed} | f={fold}/{args.n_folds}")

                    beta_max = args.beta_max if args.beta_max is not None else CONFIGS[config]["beta_max"]
                    path = ckpt_path(args.variant, beta_max, config, region, init_seed, args.split_seed, fold, args.n_folds)
                    if args.skip_existing and os.path.exists(path):
                        print("  checkpoint exists, skipping training")
                    else:
                        train_config(config, region, init_seed, args.split_seed, fold, args.n_folds,
                                     args.dataset_dir, device, args.epochs, args.lr, args.latent_dim,
                                     variant=args.variant, beta_max_override=args.beta_max, verbose=args.verbose)

                    row = eval_config(config, region, init_seed, args.split_seed, fold, args.n_folds, cache, device, args.n_jobs,
                                      variant=args.variant, beta_max_override=args.beta_max)
                    save_result(row)
                    all_results.append(row)
                    print(f"  TSTR: acc={row['accuracy']:.3f}, f1={row['f1_weighted']:.3f}, roc={row['roc_auc_ovr']:.3f}, log_loss={row['log_loss']:.3f}, kl={row['kl_final']:.4f}, active_dims={row['active_dims']}")

    if all_results and not args.no_summary:
        save_summary(all_results)

if __name__ == "__main__":
    main()
