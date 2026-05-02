import argparse
import csv
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from core.data import CLASSES, BreathDataset, load_dataset, split_dataset
from core.train import active_dims
from core.tstr import evaluate_classifier, extract_fixed_features, load_cache, train_stacking_classifier, trtr
from models.vae import CVAE, elbo_loss

CONFIGS: dict[str, dict] = {
    # beta-cap sweep
    "beta_cap_0.001": {"beta_max": 0.001, "lag_n": 0, "lag_phase": 0},
    "beta_cap_0.01": {"beta_max": 0.01, "lag_n": 0, "lag_phase": 0},
    "beta_cap_0.1": {"beta_max": 0.1, "lag_n": 0, "lag_phase": 0},
    "beta_cap_0.5": {"beta_max": 0.5, "lag_n": 0, "lag_phase": 0},
    "beta_cap_1.0": {"beta_max": 1.0, "lag_n": 0, "lag_phase": 0}, # baseline
    # lagging inference with beta_max=1.0
    "lag_5_100": {"beta_max": 1.0, "lag_n": 5, "lag_phase": 100},
    "lag_5_250": {"beta_max": 1.0, "lag_n": 5, "lag_phase": 250},
    "lag_10_100": {"beta_max": 1.0, "lag_n": 10, "lag_phase": 100},
    "lag_10_250": {"beta_max": 1.0, "lag_n": 10, "lag_phase": 250},
    # lagging + beta-cap
    "lag_5_250_beta_0.1": {"beta_max": 0.1, "lag_n": 5, "lag_phase": 250},
    "lag_5_250_beta_0.01": {"beta_max": 0.01, "lag_n": 5, "lag_phase": 250},
}

# training dynamics vary
LATENT_DIM = 16
EMBED_DIM = 8
PART_EMBED_DIM = 8
BATCH_SIZE = 32
LOG_EVERY = 25
SAVE_WINDOW_START = 400

# cli
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default=",".join(CONFIGS))
    p.add_argument("--seeds", default="0,1,7,42,123")
    p.add_argument("--regions", default="mouth,nose")
    p.add_argument("--config", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--region", default=None)
    p.add_argument("--dataset_dir", default="dataset")
    p.add_argument("--epochs", type=int,   default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--n_jobs", type=int, default=4)
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()

def beta_capped(epoch: int, total_epochs: int, beta_max: float) -> float:
    """Linear warmup from 0 → beta_max over the first half of training, then hold."""
    warmup_epochs = total_epochs * 0.5
    return min(beta_max, (epoch / warmup_epochs) * beta_max)

# paths
def ckpt_path(config: str, region: str, seed: int) -> str:
    return f"results/ablation_cvae_training_dynamics/{config}_{region}_s{seed}.pt"

def hist_path(config: str, region: str, seed: int) -> str:
    return f"results/ablation_cvae_training_dynamics/{config}_{region}_s{seed}_history.csv"

# training
def train_config(config: str, region: str, seed: int, dataset_dir: str, device: torch.device, epochs: int, lr: float, latent_dim: int) -> None:
    cfg= CONFIGS[config]
    beta_max = cfg["beta_max"]
    lag_n = cfg["lag_n"]
    lag_phase = cfg["lag_phase"]

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # data
    df = load_dataset(dataset_dir)
    df = df[df["region"] == region].reset_index(drop=True)
    df_train, df_val, _ = split_dataset(df, random_state=seed)
    train_ds = BreathDataset(df_train)
    val_ds = BreathDataset(df_val, stats=train_ds.stats)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    val_loader = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # model (baseline)
    model = CVAE(latent_dim=latent_dim, embed_dim=EMBED_DIM, condition_on_participant=True, part_embed_dim=PART_EMBED_DIM).to(device)

    opt_full = torch.optim.Adam(model.parameters(), lr=lr)
    opt_enc = torch.optim.Adam(model.encoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_full, T_max=epochs)

    best_val_loss = float("inf")
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        beta = beta_capped(epoch, epochs, beta_max)
        in_aggressive_phase = lag_n > 0 and epoch <= lag_phase

        # training
        model.train()
        t_loss = t_recon = t_kl = 0.0
        n_batches = 0

        for signal, _time, label, participant in train_loader:
            signal = signal.to(device)
            label = label.long().to(device)
            participant = participant.long().to(device)

            if in_aggressive_phase:
                for _ in range(lag_n):
                    opt_enc.zero_grad()
                    x_hat, mu, logvar = model(signal, label, participant)
                    loss_enc, _, _ = elbo_loss(signal, x_hat, mu, logvar, beta)
                    loss_enc.backward()
                    opt_enc.step()

            # One full update for every batch
            opt_full.zero_grad()
            x_hat, mu, logvar = model(signal, label, participant)
            loss, recon, kl = elbo_loss(signal, x_hat, mu, logvar, beta)
            loss.backward()
            opt_full.step()

            t_loss += loss.item()
            t_recon += recon.item()
            t_kl += kl.item()
            n_batches += 1

        t_loss /= n_batches
        t_recon /= n_batches
        t_kl /= n_batches

        # validation
        model.eval()
        v_loss = v_recon = v_kl = 0.0
        with torch.no_grad():
            for signal, _time, label, participant in val_loader:
                signal = signal.to(device)
                label = label.long().to(device)
                participant = participant.long().to(device)
                x_hat, mu, logvar = model(signal, label, participant)
                loss, recon, kl = elbo_loss(signal, x_hat, mu, logvar, beta)
                v_loss += loss.item()
                v_recon += recon.item()
                v_kl += kl.item()
        n_val = len(val_loader)
        v_loss /= n_val
        v_recon /= n_val
        v_kl /= n_val

        scheduler.step()

        if epoch >= SAVE_WINDOW_START and v_loss < best_val_loss:
            best_val_loss = v_loss
            os.makedirs("results/ablation_cvae_training_dynamics", exist_ok=True)
            torch.save({"config": config,
                        "region": region,
                        "seed": seed,
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "stats": train_ds.stats,
                        "latent_dim": latent_dim,
                        "embed_dim": EMBED_DIM,
                        "part_embed_dim": PART_EMBED_DIM,
                        "beta_max": beta_max,}, ckpt_path(config, region, seed))

        # active dims
        if epoch % LOG_EVERY == 0 or epoch == 1:
            n_active = active_dims(model, train_ds, device, conditional=True, use_participant=True)
        else:
            n_active = history[-1]["active_dims"] if history else 0

        history.append({"epoch": epoch,
                        "beta": beta,
                        "train_loss": t_loss,  "train_recon": t_recon, "train_kl": t_kl,
                        "val_loss": v_loss,  "val_recon":   v_recon, "val_kl":   v_kl,
                        "active_dims": n_active})

        if epoch % LOG_EVERY == 0 or epoch == 1:
            lag_tag = f" [lag×{lag_n}]" if in_aggressive_phase else ""
            print(f" [{config}|{region}|s{seed}] epoch {epoch:4d}/{epochs} | β={beta:.5f}{lag_tag} | train {t_loss:.4f} (r={t_recon:.4f}, kl={t_kl:.4f}) | val {v_loss:.4f} (r={v_recon:.4f}, kl={v_kl:.4f}) | active_dims={n_active}")

    # persist full training history
    os.makedirs("results/ablation_cvae_training_dynamics", exist_ok=True)
    with open(hist_path(config, region, seed), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    print(f"  [{config}|{region}|s{seed}] done. best_val={best_val_loss:.4f} | ckpt: {ckpt_path(config, region, seed)}")

# tstr evaluation
def eval_config(config: str, region: str, seed: int, cache: dict, device: torch.device, n_jobs: int) -> dict:
    path = ckpt_path(config, region, seed)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = CVAE(latent_dim=ckpt["latent_dim"],
                 embed_dim=ckpt["embed_dim"],
                 condition_on_participant=True,
                 part_embed_dim=ckpt["part_embed_dim"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    stats = ckpt["stats"]

    # generate synthetic signals
    n_synthetic = cache["n_train"]
    n_per_class = n_synthetic // len(CLASSES)
    remainder   = n_synthetic % len(CLASSES)
    counts = [n_per_class + (1 if i < remainder else 0) for i in range(len(CLASSES))]

    mean_t = torch.tensor(stats["mean"], dtype=torch.float32).view(1, 2, 1).to(device)
    std_t  = torch.tensor(stats["std"],  dtype=torch.float32).view(1, 2, 1).to(device)

    all_signals, all_labels = [], []
    for cls_idx, count in enumerate(counts):
        y_cls = torch.tensor(cls_idx, dtype=torch.long)
        sigs  = model.sample(count, y_cls, device)
        all_signals.append((sigs * std_t + mean_t).cpu().numpy())
        all_labels.append(np.full(count, cls_idx))
    synth_signals = np.concatenate(all_signals, axis=0)
    synth_labels  = np.concatenate(all_labels,  axis=0)
    print(f"  [{config}|{region}|s{seed}] generated {n_synthetic} synthetic signals")

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

    stacker = train_stacking_classifier(X_synth, synth_labels, seed)
    metrics = evaluate_classifier(stacker, cache["X_test_top"], cache["y_test"])

    # pull final-epoch diagnostics from history file
    kl_final = float("nan")
    active_dims_final = 0
    hp = hist_path(config, region, seed)
    if os.path.exists(hp):
        hist_df = pd.read_csv(hp)
        kl_final          = float(hist_df["train_kl"].iloc[-1])
        active_dims_final = int(hist_df["active_dims"].iloc[-1])

    return {"config": config,
            "region": region,
            "seed": seed,
            "accuracy": metrics["accuracy"],
            "f1_weighted": metrics["f1_weighted"],
            "roc_auc": metrics["roc_auc_ovr"],
            "kl_final": kl_final,
            "active_dims": active_dims_final,
            "f1_brady": metrics["per_class_f1"]["bradypnea"],
            "f1_eupnea": metrics["per_class_f1"]["eupnea"],
            "f1_tachy": metrics["per_class_f1"]["tachypnea"]}

# results
def save_results(rows: list[dict]) -> None:
    os.makedirs("results/ablation_cvae_training_dynamics", exist_ok=True)
    csv_path = "results/ablation_cvae_training_dynamics/summary.csv"
    df_new = pd.DataFrame(rows)
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df_new = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(
            subset=["config", "region", "seed"], keep="last")
    df_new.to_csv(csv_path, index=False)
    print(f"[DYNAMICS] Saved {len(df_new)} rows → {csv_path}")

def print_summary(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    config_order = list(CONFIGS)
    for region in df["region"].unique():
        rdf = df[df["region"] == region]
        mean_acc = rdf.groupby("config")["accuracy"].mean()
        best_config = mean_acc.idxmax()
        print(f"\nRegion: {region}")
        print(f"  {"Config":<26} {"Accuracy":>12} {"F1-W":>12} {"ROC-AUC":>12} {"KL":>8} {"ActDims":>8}")
        print("  " + "-" * 78)
        for cfg in config_order:
            vdf = rdf[rdf["config"] == cfg]
            if vdf.empty:
                continue
            acc = vdf["accuracy"].mean()
            acc_s = vdf["accuracy"].std()
            f1  = vdf["f1_weighted"].mean()
            f1_s  = vdf["f1_weighted"].std()
            roc = vdf["roc_auc"].mean()
            roc_s = vdf["roc_auc"].std()
            kl  = vdf["kl_final"].mean()
            ad  = vdf["active_dims"].mean()
            flag = " *" if cfg == best_config else ""
            print(f"  {cfg:<26} {acc:.3f}±{acc_s:.3f}  {f1:.3f}±{f1_s:.3f}  {roc:.3f}±{roc_s:.3f}  {kl:>8.4f}  {ad:>8.1f}{flag}")
    print("  * = best accuracy for that region")

def main() -> None:
    args = parse_args()

    # resolve run list
    configs = [args.config] if args.config else args.configs.split(",")
    seeds   = [args.seed] if args.seed is not None else [int(s) for s in args.seeds.split(",")]
    regions = [args.region] if args.region else args.regions.split(",")

    for c in configs:
        if c not in CONFIGS:
            print(f'Unknown config "{c}". Valid: {list(CONFIGS)}')
            sys.exit(1)
    for r in regions:
        if r not in ("mouth", "nose"):
            print(f'Unknown region "{r}". Valid: mouth, nose')
            sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"[DYNAMICS] device={device} | configs={configs} | seeds={seeds} | regions={regions}")

    all_results: list[dict] = []

    for region in regions:
        # build/load TRTR cache once per (region, seed)
        caches: dict[int, dict] = {}
        for seed in seeds:
            cache = load_cache(region, seed)
            if cache is None:
                print(f"[DYNAMICS] Building TRTR cache region={region}, seed={seed} ...")
                cache = trtr(args.dataset_dir, region, args.n_jobs, seed)
            caches[seed] = cache
            print(f"[DYNAMICS] TRTR cache ready: region={region}, seed={seed}, n_train={cache["n_train"]}")

        for config in configs:
            for seed in seeds:
                print(f"[DYNAMICS] {config} | {region} | seed={seed}")

                # train
                path = ckpt_path(config, region, seed)
                if args.skip_existing and os.path.exists(path):
                    print("checkpoint exists, skipping training")
                else:
                    train_config(config, region, seed, args.dataset_dir,
                                 device, args.epochs, args.lr, args.latent_dim)

                # evaluate
                row = eval_config(config, region, seed, caches[seed], device, args.n_jobs)
                all_results.append(row)
                print(f"  TSTR → acc={row["accuracy"]:.3f}, f1={row["f1_weighted"]:.3f}, roc={row["roc_auc"]:.3f}, kl={row["kl_final"]:.4f}, active_dims={row["active_dims"]}")

    if all_results:
        save_results(all_results)
        print_summary(all_results)

if __name__ == "__main__":
    main()
