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
from ablations.cvae_architecture_models import VARIANT_MAP

# fix parameters
LATENT_DIM = 16
EMBED_DIM = 8
PART_EMBED_DIM = 8
EPOCHS = 500
BATCH_SIZE = 32
LR = 1e-3
LOG_EVERY = 25
BETA_MAX = 1.0
WARMUP_FRAC = 0.5
RESULTS_DIR = "results/ablation_cvae_architecture"

# cli
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CVAE architecture ablation study")
    p.add_argument("--variants", default=",".join(VARIANT_MAP))
    p.add_argument("--init_seeds", default="0,1,7,42,123")
    p.add_argument("--folds", default="1,2,3,4,5")
    p.add_argument("--regions", default="mouth,nose")
    p.add_argument("--cond_parts", default="true,false")
    # single-combo overrides (used by the parallel shell scripts)
    p.add_argument("--variant", default=None)
    p.add_argument("--init_seed", type=int, default=None)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--region", default=None)
    p.add_argument("--cond_part", default=None)
    # protocol-level constants
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--dataset_dir", default="dataset")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--n_jobs", type=int, default=4)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no_summary", action="store_true")
    p.add_argument("--aggregate", action="store_true")
    return p.parse_args()

def _parse_bool(s: str) -> bool:
    if s.lower() in ("true", "1", "yes"):
        return True
    if s.lower() in ("false", "0", "no"):
        return False
    raise ValueError(f"Invalid boolean: {s!r}")

def _stem(variant: str, region: str, init_seed: int, split_seed: int, fold: int, n_folds: int, cond_part: bool) -> str:
    cp_marker = "" if cond_part else "_nocp"
    return f"{variant}_{region}_is{init_seed}_ss{split_seed}{cp_marker}_fold{fold}of{n_folds}"

def ckpt_path(variant: str, region: str, init_seed: int, split_seed: int, fold: int, n_folds: int, cond_part: bool) -> str:
    return f"{RESULTS_DIR}/{_stem(variant, region, init_seed, split_seed, fold, n_folds, cond_part)}.pt"

def hist_path(variant: str, region: str, init_seed: int, split_seed: int, fold: int, n_folds: int, cond_part: bool) -> str:
    return f"{RESULTS_DIR}/{_stem(variant, region, init_seed, split_seed, fold, n_folds, cond_part)}_history.csv"

def result_path(variant: str, region: str, init_seed: int, split_seed: int, fold: int, n_folds: int, cond_part: bool) -> str:
    return f"{RESULTS_DIR}/{_stem(variant, region, init_seed, split_seed, fold, n_folds, cond_part)}_result.json"

# training
def train_variant(variant: str, region: str, init_seed: int, split_seed: int, fold: int, n_folds: int, cond_part: bool, dataset_dir: str, device: torch.device, epochs: int = EPOCHS, verbose: bool = False) -> None:
    warmup_epochs = int(epochs * WARMUP_FRAC)

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
    ModelClass = VARIANT_MAP[variant]
    model = ModelClass(latent_dim=LATENT_DIM, embed_dim=EMBED_DIM, condition_on_participant=cond_part, part_embed_dim=PART_EMBED_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    history = []

    epoch_bar = tqdm(range(1, epochs + 1), desc=f"[ARCH] {variant}|{region}|is{init_seed}|f{fold}|cp={cond_part}", unit="epoch", disable=not verbose)
    for epoch in epoch_bar:
        # train and validate
        t_loss, t_recon, t_kl = train_vae_one_epoch(model, train_loader, optimizer, epoch, warmup_epochs, device, free_bits=0.0, conditional=True, use_participant=cond_part, beta_max=BETA_MAX)
        v_loss, v_recon, v_kl = evaluate(model, val_loader, epoch, warmup_epochs, device, free_bits=0.0, conditional=True, use_participant=cond_part, beta_max=BETA_MAX)
        scheduler.step()
        beta = beta_capped(epoch, warmup_epochs, BETA_MAX)

        # checkpoint only once beta has fully warmed up
        if v_loss < best_val_loss and beta >= BETA_MAX:
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
                       ckpt_path(variant, region, init_seed, split_seed, fold, n_folds, cond_part))

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
    with open(hist_path(variant, region, init_seed, split_seed, fold, n_folds, cond_part), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    print(f"  [ARCH] {variant}|{region}|is{init_seed}|f{fold}|cp={cond_part} | Done. best_val={best_val_loss:.4f} | ckpt: {ckpt_path(variant, region, init_seed, split_seed, fold, n_folds, cond_part)}")

# tstr evaluation
def eval_variant(variant: str, region: str, init_seed: int, split_seed: int, fold: int, n_folds: int,
                 cond_part: bool, cache: dict, device: torch.device, n_jobs: int) -> dict:
    path = ckpt_path(variant, region, init_seed, split_seed, fold, n_folds, cond_part)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    ModelClass = VARIANT_MAP[variant]
    model = ModelClass(latent_dim=ckpt["latent_dim"],
                       embed_dim=ckpt["embed_dim"],
                       condition_on_participant=cond_part,
                       part_embed_dim=ckpt["part_embed_dim"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    stats = ckpt["stats"]

    # generate synthetic signals
    n_synthetic = cache["n_train"]
    synth_signals, synth_labels = generate_synthetic_signals(model, "cvae_part", n_synthetic, stats, device, init_seed)
    print(f"  [ARCH] {variant}|{region}|is{init_seed}|f{fold}|cp={cond_part} | Generated {n_synthetic} synthetic signals")

    # tsfresh feature extraction on synthetic data
    n, _C, T = synth_signals.shape
    df_long = pd.DataFrame({"id": np.repeat(np.arange(n), T),
                            "time": np.tile(np.arange(T), n),
                            "Humidity": synth_signals[:, 0, :].ravel(),
                            "Temperature": synth_signals[:, 1, :].ravel()})
    X_raw = extract_fixed_features(df_long, cache["top_20_features_raw"], n_jobs)
    X_san = X_raw.copy()
    X_san.columns = [re.sub(r"[^\w]", "_", c) for c in X_san.columns]
    X_synth = X_san[cache["top_20_features_sanitized"]].values

    # train stacker on synthetic, test on real
    stacker = train_stacking_classifier(X_synth, synth_labels, init_seed, n_jobs=n_jobs)
    metrics = evaluate_classifier(stacker, cache["X_test_top"], cache["y_test"])

    # pull final-epoch kl/active_dims
    kl_final = float("nan")
    active_dims_final = 0
    hp = hist_path(variant, region, init_seed, split_seed, fold, n_folds, cond_part)
    if os.path.exists(hp):
        hist_df = pd.read_csv(hp)
        kl_final = float(hist_df["train_kl"].iloc[-1])
        active_dims_final = int(hist_df["active_dims"].iloc[-1])

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
    path = result_path(row["variant"], row["region"], row["init_seed"], row["split_seed"], row["fold"], row["n_folds"], row["condition_on_participant"])
    with open(path, "w") as f:
        json.dump(_json_safe(row), f, indent=2)

def save_summary(rows: list[dict]) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = f"{RESULTS_DIR}/summary.csv"
    df_new = pd.DataFrame(rows)
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        if "condition_on_participant" not in df_old.columns:
            df_old["condition_on_participant"] = True
        df_new = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(
            subset=["variant", "region", "init_seed", "split_seed", "fold", "n_folds", "condition_on_participant"], keep="last")
    df_new.to_csv(csv_path, index=False)
    print(f"[ARCH] Saved {len(df_new)} rows → {csv_path}")

# main
def main() -> None:
    args = parse_args()

    variants = [args.variant] if args.variant else args.variants.split(",")
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

    device = torch.device("cpu")

    # aggregate mode: just collect existing per-combo result jsons
    if args.aggregate:
        rows = []
        for region in regions:
            for variant in variants:
                for init_seed in init_seeds:
                    for fold in folds:
                        for cp in cond_parts:
                            rp = result_path(variant, region, init_seed, args.split_seed, fold, args.n_folds, cp)
                            if not os.path.exists(rp):
                                print(f"  [AGGREGATE] missing {rp}, skipping")
                                continue
                            with open(rp) as f:
                                rows.append(json.load(f))
        if rows:
            save_summary(rows)
        return

    # normal mode: train and evaluate
    print(f"[ARCH] device={device} | variants={variants} | init_seeds={init_seeds} | folds={folds} | regions={regions} | cond_parts={cond_parts}")
    all_results: list[dict] = []

    for region in regions:
        # TRTR cache
        for variant in variants:
            for init_seed in init_seeds:
                for fold in folds:
                    cache = load_cache(region, init_seed, args.split_seed, fold, args.n_folds, channel="both")
                    if cache is None:
                        print(f"[ARCH] Building TRTR cache region={region}, init_seed={init_seed}, fold={fold} ...")
                        cache = trtr(args.dataset_dir, region, args.n_jobs, init_seed, args.split_seed, fold, args.n_folds, channel="both")

                    for cp in cond_parts:
                        print(f"\n[ARCH] {variant} | {region} | is={init_seed} | f={fold}/{args.n_folds} | cp={cp}")

                        # train
                        path = ckpt_path(variant, region, init_seed, args.split_seed, fold, args.n_folds, cp)
                        if args.skip_existing and os.path.exists(path):
                            print("  checkpoint exists, skipping training")
                        else:
                            train_variant(variant, region, init_seed, args.split_seed, fold, args.n_folds, cp, args.dataset_dir, device, epochs=args.epochs, verbose=args.verbose)

                        # evaluate
                        row = eval_variant(variant, region, init_seed, args.split_seed, fold, args.n_folds, cp, cache, device, args.n_jobs)
                        save_result(row)
                        all_results.append(row)
                        print(f"  TSTR: acc={row['accuracy']:.3f}, f1={row['f1_weighted']:.3f}, roc={row['roc_auc_ovr']:.3f}, log_loss={row['log_loss']:.3f}, kl={row['kl_final']:.3f}, active_dims={row['active_dims']}")

    if all_results and not args.no_summary:
        save_summary(all_results)

if __name__ == "__main__":
    main()
