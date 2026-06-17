from collections import Counter
import json
import glob
import os
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE
import torch

from models.vae import CVAE, VAE
from core.data import BreathDataset, load_dataset, split_dataset

SEEDS = [0, 1, 7, 42, 123]
REGIONS = ["mouth", "nose"]
CLASSES  = ["bradypnea", "eupnea", "tachypnea"]
PARTICIPANTS = ["a", "b", "c", "d", "e", "f", "g", "p", "s"]

# plot settings and colors
plt.rcParams.update({"legend.fontsize": 9,
                     "axes.titlesize": 10})
CLASS_COLORS = {"bradypnea": "tab:blue", "eupnea": "tab:green", "tachypnea": "tab:purple"}
PARTICIPANT_COLORS =  {"a": "tab:red", "p": "tab:orange", "s": "tab:cyan", "e": "tab:gray", "f": "tab:olive", "g": "tab:pink"}
WIDTH = 3
HEIGHT = 2

# cvae ablation constants
ARCHITECTURES = ["conv_baseline", "conv_slim", "mlp", "mlp_small", "conv_asym"]
ARCHITECTURES_COLORS = dict(zip(ARCHITECTURES, sns.color_palette("tab10", n_colors=len(ARCHITECTURES))))
BETA_CONFIGS = ["beta_cap_0.001", "beta_cap_0.01", "beta_cap_0.1", "beta_cap_0.5", "beta_cap_1.0"]
BETA_VALUES = {cfg: float(cfg.split("_")[-1]) for cfg in BETA_CONFIGS}
LAG_CONFIGS = ["lag_5_100", "lag_5_250", "lag_10_100", "lag_10_250"]
CONFIG_LABELS = {"beta_cap_0.001": "β-cap 0.001",
                 "beta_cap_0.01": "β-cap 0.01",
                 "beta_cap_0.1": "β-cap 0.1",
                 "beta_cap_0.5": "β-cap 0.5",
                 "beta_cap_1.0": "β-cap 1.0",
                 "lag_5_100": "Lag 5 / 100",
                 "lag_5_250": "Lag 5 / 250",
                 "lag_10_100": "Lag 10 / 100",
                 "lag_10_250": "Lag 10 / 250",
                 "lag_5_250_beta_0.01": "Lag 5 / 250 + β 0.01",
                 "lag_5_250_beta_0.1": "Lag 5 / 250 + β 0.1"}

# exploratory data analysis
def load_breath_pattern(classes, dataset_dir="../dataset"):
    dataset = []

    # loop through all breath patterns
    for cls in classes:
        folder = os.path.join(dataset_dir, cls)

        # loop through all .dat files
        for fname in sorted(os.listdir(folder)):
            if not fname.endswith(".dat"):
                continue

            # extract metadata (unprefixed files belong to participant "a";
            # p_/s_/e_/f_/g_ encode the other participants)
            m = re.match(r"^(([bcdpsefg])_)?(mouth|nose)_trial_(\d+)\.dat$", fname)
            if m is None:
                continue
        
            # load the data
            df = pd.read_csv(os.path.join(folder, fname))
            dataset.append({"class": cls,
                            "filename": fname,
                            "participant": m.group(2) if m.group(2) else "a",
                            "region": m.group(3),
                            "trial_num": int(m.group(4)),
                            "time": df["Time"].values / 1000,
                            "humidity": df["Humidity"].values,
                            "temperature": df["Temperature"].values,})

    return pd.DataFrame(dataset)

def resample_trials(trials_df, t_common, measurement_type):
    out = []
    for row in trials_df.to_dict("records"):
        t = row["time"]
        h = row[measurement_type]
        out.append(np.interp(t_common, t, h, left=np.nan, right=np.nan))
    return np.array(out)

def load_cir(dataset_dir="../dataset"):
    dataset = []
    folder = os.path.join(dataset_dir, "cir")

    # loop through all .dat files
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".dat"):
            continue

        # extract metadata
        m = re.match(r"^(mouth|nose)_trial_(\d+)\.dat$", fname)
        if m is None:
            continue
        
        # load the data
        df = pd.read_csv(os.path.join(folder, fname))
        dataset.append({"filename": fname,
                        "region": m.group(1),
                        "trial_num": int(m.group(2)),
                        "time": df["Time"].values/1000,
                        "humidity": df["Humidity"].values,
                        "temperature": df["Temperature"].values})

    return pd.DataFrame(dataset)

def cir_stats(region, cir_df):
    trials = cir_df[cir_df["region"] == region]
    humidity_mat = np.stack(trials["humidity"].values)
    temperature_mat = np.stack(trials["temperature"].values)
    time = trials.iloc[0]["time"]
    mean_h = np.mean(humidity_mat, axis=0)
    mean_t = np.mean(temperature_mat, axis=0)
    std_h = np.std(humidity_mat, axis=0)
    std_t = np.std(temperature_mat, axis=0)

    return time, mean_h, mean_t, std_h, std_t

def plot_measurements(df, region):
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH*3, HEIGHT*2), sharex="col", sharey="row")

    for col, cls in enumerate(CLASSES):
        trials = df[(df["class"] == cls) & (df["region"] == region)]

        for row, measurement in enumerate(["humidity", "temperature"]):
            ax = axes[row, col]

            for record in trials.to_dict("records"):
                t = record["time"]
                h = record[measurement]
                ax.plot(t, h, color=CLASS_COLORS[cls], alpha=0.1)

            axes[0, col].set_title(f"{cls.capitalize()} (n={len(trials)})")
            axes[-1, col].set_xlabel("Time in s")
            axes[row, 0].set_ylabel(f"{measurement.capitalize()} in {"%" if measurement == "humidity" else "°C"}")
            ax.grid()

    plt.suptitle(f"All {region} trials ({region})")
    plt.tight_layout()
    plt.show()

def plot_region_comparison(df, t):
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH*3, HEIGHT*2), sharex=True, sharey="row")
    for col, cls in enumerate(CLASSES):
        for row, measurement in enumerate(["humidity", "temperature"]):
            ax = axes[row, col]
            for region in REGIONS:
                # resample all trials to common time grid
                trials = df[(df["class"] == cls) & (df["region"] == region)]
                trials_resampled = resample_trials(trials, t_common=t, measurement_type=measurement)

                # stats
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    mean = np.nanmean(trials_resampled, axis=0)
                    std = np.nanstd(trials_resampled, axis=0)

                # plot
                ax.plot(t, mean, label=region.capitalize(), color=CLASS_COLORS[cls], linewidth=2.0, linestyle="-" if region == "mouth" else "--")
                ax.fill_between(t, mean-std, mean+std, alpha=0.2, color=CLASS_COLORS[cls])
            axes[0, col].set_title(f"{cls.capitalize()}")
            axes[-1, col].set_xlabel("Time in s")
            axes[row, 0].set_ylabel(f"{measurement.capitalize()} in {"%" if measurement == "humidity" else "°C"}")
            ax.grid()
            ax.legend(loc="upper left")

    plt.suptitle("Comparison of mouth and nose trials")
    plt.tight_layout()
    plt.show()

def plot_interparticipant_variability(df, t, region):
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH*3, HEIGHT*2), sharex=True, sharey="row")
    for row, measurement in enumerate(["humidity", "temperature"]):
        for col, cls in enumerate(CLASSES):
            ax = axes[row, col]
            for participant in PARTICIPANTS:
                # resample all trials to common time grid
                trials = df[(df["class"] == cls) & (df["region"] == region) & (df["participant"] == participant)]
                if len(trials) == 0:
                    continue
                mat  = resample_trials(trials, t_common=t, measurement_type=measurement)

                # stats
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    mean = np.nanmean(mat, axis=0)
                    std  = np.nanstd(mat,  axis=0)

                # plot
                label = f"Participant {participant.upper()}  (n={len(trials)})"
                color = PARTICIPANT_COLORS[participant]
                ax.plot(t, mean, label=label, color=color)
                ax.fill_between(t, mean-std, mean+std, alpha=0.2, color=color)
            axes[0, col].set_title(f"{cls.capitalize()}")
            axes[-1, col].set_xlabel("Time in s")
            axes[row, 0].set_ylabel(f"{measurement.capitalize()} in {"%" if measurement == "humidity" else "°C"}")
            ax.grid()
            # ax.legend(loc="lower right") if row == 0 else ax.legend(loc="upper left")
    plt.suptitle(f"Inter-participant variability: {region}")
    plt.tight_layout()
    plt.show()

def plot_correlation(df, participants=PARTICIPANTS):
    fig, axes = plt.subplots(len(participants), 3, figsize=(WIDTH*3, HEIGHT*len(participants)), sharex=True, sharey=True)

    for p_idx, participant in enumerate(participants):
        for col, cls in enumerate(CLASSES):
            ax = axes[p_idx, col]

            trials = df[(df["class"] == cls) & (df["participant"] == participant)]
            records = trials.to_dict("records")

            if records:
                all_h = np.concatenate([r["humidity"] for r in records])
                all_temp = np.concatenate([r["temperature"] for r in records])
                corr = np.corrcoef(all_h, all_temp)[0, 1]
                ax.scatter(all_h, all_temp, alpha=0.15, s=5, rasterized=True, label=f"r = {corr:.2f}", color=CLASS_COLORS[cls])
                ax.legend(loc="upper left")
            axes[0, col].set_title(f"{cls.capitalize()}")
            axes[-1, col].set_xlabel("Humidity in %")
            axes[p_idx, 0].set_ylabel(f"Participant {participant.upper()}\nTemperature in in °C")
            ax.grid()

    plt.suptitle("Correlation between Humidity and Temperature")
    plt.tight_layout()
    plt.show()

def plot_cir(df):
    t, mouth_h, mouth_t, mouth_h_std, mouth_t_std = cir_stats("mouth", df)
    t, nose_h,  nose_t, nose_h_std, nose_t_std = cir_stats("nose", df)

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH*3, HEIGHT), sharex=True, sharey=False)
    axes[0].plot(t, mouth_h, label="Mouth")
    axes[0].plot(t, nose_h,  label="Nose")
    axes[0].fill_between(t, mouth_h-mouth_h_std, mouth_h+mouth_h_std, alpha=0.2)
    axes[0].fill_between(t, nose_h-nose_h_std, nose_h+nose_h_std, alpha=0.2)
    axes[0].set_xlabel("Time in s")
    axes[0].set_ylabel("Humidity in %")
    axes[0].legend(loc="upper right")
    axes[0].grid()

    axes[1].plot(t, mouth_t, label="Mouth")
    axes[1].plot(t, nose_t,  label="Nose")
    axes[1].fill_between(t, mouth_t-mouth_t_std, mouth_t+mouth_t_std, alpha=0.2)
    axes[1].fill_between(t, nose_t-nose_t_std, nose_t+nose_t_std, alpha=0.2)
    axes[1].set_xlabel("Time in s")
    axes[1].set_ylabel("Temperature in °C")
    axes[1].legend(loc="upper right")
    axes[1].grid()
    plt.suptitle("Channel Impulse Response")
    plt.tight_layout()
    plt.show()

# train-real-test-real ablation
def load_trtr_features():
    trtr_results_dir = "./results/trtr"
    feature_counts = {"mouth": Counter(), "nose": Counter()}

    for region in ["mouth", "nose"]:
        for seed in SEEDS:
            json_path = f"{trtr_results_dir}/{region}_s{seed}_tstr.json"
            with open(json_path, "r") as f:
                data = json.load(f)
                features = data["top_20_features"]
                feature_counts[region].update(features)
    return feature_counts

def plot_trtr_ablation():
    df = pd.read_csv("results/ablation_trtr.csv")
    if "single_split" not in df.columns:
        df["single_split"] = False
    df["single_split"] = df["single_split"].fillna(False).astype(bool)

    df_kfold  = df[~df["single_split"]]
    df_legacy = df[df["single_split"] & (df["pipeline"] == "replication")]

    pipelines = ["original", "replication", "shap_fix", "lgbm_fix", "tsfresh_fix", "smote_fix"]
    pipeline_labels = ["Original\nprotocol",
                       "Replication\n(k-fold)",
                       "+ SHAP\non train",
                       "+ Unified\nLGBM tuning",
                       "+ tsfresh\nper fold",
                       "+ SMOTE\ninside CV"]
    regions = ["mouth", "nose"]
    region_colors = {"mouth": "tab:blue", "nose": "tab:orange"}

    metric_cols = ["accuracy", "f1_weighted", "roc_auc_ovr", "log_loss"]

    # k-fold: average over init_seeds per fold, then mean+/-std across folds
    fold_level = df_kfold.groupby(["region", "pipeline", "fold"])[metric_cols].mean().reset_index()
    agg_kfold = fold_level.groupby(["region", "pipeline"])[metric_cols].agg(["mean", "std"])
    # legacy: mean+/-std across init_seeds (no folds)
    agg_legacy = df_legacy.groupby("region")[metric_cols].agg(["mean", "std"])

    def lookup(region, pipeline, metric_col):
        if pipeline == "original":
            if region in agg_legacy.index:
                return (agg_legacy.loc[region, (metric_col, "mean")],
                        agg_legacy.loc[region, (metric_col, "std")])
        elif (region, pipeline) in agg_kfold.index:
            return (agg_kfold.loc[(region, pipeline), (metric_col, "mean")],
                    agg_kfold.loc[(region, pipeline), (metric_col, "std")])
        return (np.nan, 0.0)

    metrics = [("accuracy", "Accuracy in %", True),
               ("f1_weighted", r"$F_1^w$ in %", True),
               ("roc_auc_ovr", "ROC-AUC (OVR) in %", True),
               ("log_loss", "Log Loss", False)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    width = 0.38
    x = np.arange(len(pipelines))

    for ax, (metric_col, ylabel, scale_pct) in zip(axes.flat, metrics):
        for i, region in enumerate(regions):
            means, stds = [], []
            for p in pipelines:
                m, s = lookup(region, p, metric_col)
                means.append(m)
                stds.append(0.0 if np.isnan(s) else s)
            means = np.array(means, dtype=float)
            stds  = np.array(stds, dtype=float)
            if scale_pct:
                means, stds = means * 100, stds * 100

            offset = (i - 0.5) * width
            ax.bar(x + offset, means, width, yerr=stds, capsize=4,
                   label=region.capitalize(), color=region_colors[region],
                   alpha=0.85, edgecolor="black", linewidth=0.5,
                   error_kw=dict(ecolor="black", lw=0.8))

            fmt = "{:.1f}" if scale_pct else "{:.3f}"
            label_offset = 1.5 if scale_pct else 0.02
            for xi, m, s in zip(x, means, stds):
                if not np.isnan(m):
                    ax.text(xi + offset, m + s + label_offset,
                            fmt.format(m), ha="center", fontsize=8)

        ax.axvline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(pipeline_labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        loc = "upper right" if metric_col == "log_loss" else "lower right"
        ax.legend(loc=loc, fontsize=9)
        ax.set_title(ylabel.replace(" in %", ""))

    plt.tight_layout()
    plt.show()

def plot_trtr_feature_count(feature_counts):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH*4, HEIGHT*3))

    for ax, region in zip(axes, ["mouth", "nose"]):
        top = feature_counts[region].most_common(20)
        features, counts = zip(*top)
        
        # truncate long feature names for readability
        feature_labels = [f if len(f) < 60 else f[:57] + "..." for f in features]
        
        y_pos = np.arange(len(features))
        ax.barh(y_pos, counts)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_labels, fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel("Times in top-20")
        ax.set_title(f"{region.capitalize()}")
        ax.axvline(5, color="gray", linestyle="--", alpha=0.5, label="always selected (5/5)")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3, axis="x")

    fig.suptitle("Most frequently selected features")
    plt.tight_layout()
    plt.show()

def plot_trtr_feature_stability(feature_counts):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH*3, HEIGHT*1.5), sharey=True)

    for ax, region in zip(axes, ["mouth", "nose"]):
        counts_distribution = Counter(feature_counts[region].values())
        
        x = list(range(1, 5 + 1))
        y = [counts_distribution.get(i, 0) for i in x]
        
        ax.bar(x, y)
        ax.set_ylim(0, max(y) + 5)
        ax.set_xlabel("Times feature appears in top-20")
        ax.set_ylabel("Number of features") if ax == axes[0] else None
        ax.set_title(f"{region.capitalize()}")
        ax.set_xticks(x)
        ax.grid(alpha=0.3, axis="y")
        
        for xi, yi in zip(x, y):
            ax.text(xi, yi, str(yi), ha="center", va="bottom")
    fig.suptitle("Feature selection stability")
    plt.tight_layout()
    plt.show()

def plot_trtr_channel_contribution():
    df = pd.read_csv("results/summary.csv")
    df_trtr = df[df["model"]=="trtr"].copy()
    df_temperature_mouth = df_trtr[(df_trtr["channel"]=="temperature") & (df_trtr["region"]=="mouth")]
    df_temperature_nose = df_trtr[(df_trtr["channel"]=="temperature") & (df_trtr["region"]=="nose")]
    df_humidity_mouth = df_trtr[(df_trtr["channel"]=="humidity") & (df_trtr["region"]=="mouth")]
    df_humidity_nose = df_trtr[(df_trtr["channel"]=="humidity") & (df_trtr["region"]=="nose")]
    df_mouth = df_trtr[(df_trtr["region"]=="mouth") & (df_trtr["channel"]!="temperature") & (df_trtr["channel"]!="humidity")]
    df_nose = df_trtr[(df_trtr["region"]=="nose") & (df_trtr["channel"]!="temperature") & (df_trtr["channel"]!="humidity")]

    fig, ax = plt.subplots(1, 2, figsize=(WIDTH*2, HEIGHT*1.5), sharey=True)
    ax[0].bar(["Humidity", "Temperature", "Both"], [df_humidity_mouth["accuracy"].mean() * 100, df_temperature_mouth["accuracy"].mean() * 100, df_mouth["accuracy"].mean() * 100], color=["tab:blue", "tab:orange", "tab:green"])
    ax[0].set_ylim(0, 100)
    ax[0].set_ylabel("Accuracy in %")
    ax[0].set_title("Mouth")
    ax[0].grid(axis="y")

    ax[1].bar(["Humidity", "Temperature", "Both"], [df_humidity_nose["accuracy"].mean() * 100, df_temperature_nose["accuracy"].mean() * 100, df_nose["accuracy"].mean() * 100], color=["tab:blue", "tab:orange", "tab:green"])
    ax[1].set_ylim(0, 100)
    ax[1].set_title("Nose")
    ax[1].grid(axis="y")

    plt.tight_layout()
    plt.show()

# variational autoencoder
def load_vaes(device, latent_dim=32, free_bits=[0.0, 0.1, 2.0]):
    models = {seed: {region: {} for region in REGIONS} for seed in SEEDS}
    datasets = {seed: {} for seed in SEEDS}
    splits = {seed: {} for seed in SEEDS}

    for seed in SEEDS:
        for region in REGIONS:
            df = load_dataset()
            df = df[df["region"] == region].reset_index(drop=True)
            df_train, df_val, df_test = split_dataset(df, random_state=seed)
            train_ds = BreathDataset(df_train)
            datasets[seed][region] = {"train": train_ds,
                                    "val":   BreathDataset(df_val,  stats=train_ds.stats),
                                    "test":  BreathDataset(df_test, stats=train_ds.stats)}
            splits[seed][region] = {"train": df_train, "val": df_val, "test": df_test}

            for free_bit in free_bits:
                ckpt_path = f"results/vae/{region}_s{seed}_ld{latent_dim}_fb{free_bit}_checkpoint.pt"
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model = VAE(latent_dim=ckpt["latent_dim"]).to(device)
                model.load_state_dict(ckpt["model_state"])
                model.eval()
                models[seed][region][free_bit] = model

    return models, datasets

@torch.no_grad()
def encoder_outputs(model, train_ds, device):
    mus, logvars = [], []
    model.eval()
    for i in range(len(train_ds)):
        signal, *_ = train_ds[i]
        mu, logvar = model.encoder(signal.unsqueeze(0).to(device))
        mus.append(mu.squeeze(0).cpu())
        logvars.append(logvar.squeeze(0).cpu())
    return torch.stack(mus), torch.stack(logvars)

def per_dim_kl(mu, logvar):
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(dim=0).numpy()

def plot_vae_training(region, latent_dim=32, free_bits=[0.0, 0.1, 2.0]):
    # training curves
    pairs = [("train_loss", "val_loss", "Total loss"),
            ("train_recon", "val_recon", "Reconstruction loss"),
            ("train_kl", "val_kl", "KL divergence")]

    n_configs = len(free_bits)
    n_rows = len(pairs)
    fig, axes = plt.subplots(n_rows, n_configs, figsize=(WIDTH * n_configs, HEIGHT * n_rows), sharex=True, sharey="row")

    for col, free_bit in enumerate(free_bits):
        hists = [pd.read_csv(f"results/vae/{region}_s{seed}_ld{latent_dim}_fb{free_bit}_train_history.csv") for seed in SEEDS]
        epochs = hists[0]["epoch"].values
        beta_one_median = int(np.median([h.loc[h["beta"] >= 1.0, "epoch"].min() for h in hists]))

        for row, (tr_col, vl_col, title) in enumerate(pairs):
            ax = axes[row, col]
            for data_col, label, color in [(tr_col, "train", "tab:blue"), (vl_col, "val", "tab:orange")]:
                vals = np.stack([h[data_col].values for h in hists])
                mean = vals.mean(axis=0)
                std  = vals.std(axis=0)
                ax.plot(epochs, mean, label=label, color=color)
                ax.fill_between(epochs, mean - std, mean + std, alpha=0.2, color=color)
            ax.axvline(beta_one_median, color="gray", linestyle="--", alpha=0.7, label=r"$\beta=1$")
            if row == 0:
                ax.set_title(f"Free bits: {free_bit}")
            if col == 0:
                ax.set_ylabel(title)
            if row == n_rows - 1:
                ax.set_xlabel("Epoch")
            ax.legend(loc="upper right") if (row, col) == (0, n_configs - 1) else None
            ax.grid()

    plt.tight_layout()
    plt.show()

def plot_vae_active_dims_training(region, latent_dim=32, free_bits=[0.0, 0.1, 2.0]):
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))

    for i, free_bit in enumerate(free_bits):
        curves = []
        for seed in SEEDS:
            path = f"results/vae/{region}_s{seed}_ld{latent_dim}_fb{free_bit}_train_history.csv"
            df = pd.read_csv(path)
            curves.append(df["active_dims"].values)

        min_len = min(len(c) for c in curves)
        arr = np.stack([c[:min_len] for c in curves], axis=0)
        epochs = np.arange(1, min_len + 1)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)

        ax.plot(epochs, mean, color=list(CLASS_COLORS.values())[i], linewidth=1.6, label=rf"$\lambda_\mathrm{{fb}} = {free_bit}$")
        ax.fill_between(epochs, mean - std, mean + std, color=list(CLASS_COLORS.values())[i], alpha=0.18)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Active latent dimensions")
    ax.set_ylim(bottom=0)
    ax.grid()
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

def plot_vae_active_dims_final(models, datasets, region, free_bits=[0.0, 0.1, 2.0], active_threshold=0.1):
    fig, axes = plt.subplots(1, len(free_bits), figsize=(WIDTH * len(free_bits), HEIGHT*1.5), sharey=True)

    for i, (ax, free_bit) in enumerate(zip(axes, free_bits)):
        kl_per_dim_seeds = []
        for seed in SEEDS:
            mu, logvar = encoder_outputs(models[seed][region][free_bit], datasets[seed][region]["train"], device="cpu")
            kl_per_dim_seeds.append(per_dim_kl(mu, logvar))

        kl_mean = np.mean(kl_per_dim_seeds, axis=0)
        order = np.argsort(kl_mean)[::-1]
        x = np.arange(len(kl_mean))

        bars = ax.bar(x, kl_mean[order], width=0.5, color=list(CLASS_COLORS.values())[i], alpha=0.85, edgecolor="none")
        for i, bar in enumerate(bars):
            if kl_mean[order][i] <= active_threshold:
                bar.set_alpha(0.30)

        ax.axhline(active_threshold, color="black", linestyle="--", label=f"active threshold: {active_threshold} nat")
        if free_bit > 0:
            ax.axhline(free_bit, color="tab:red", linestyle=":", label=f"free-bits floor: {free_bit} nat")

        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_xticks([])
        ax.set_xlabel("Sorted latent dimension")
        ax.grid(axis="y")
        ax.legend(loc="upper right")
        ax.set_ylabel("Kullback-Leibler Divergence in nat") if ax == axes[0] else None

    plt.tight_layout()
    plt.show()

def participant_contours(ax, emb, parts, levels=(0.5,), grid_size=120, pad=0.1):
    x_min, x_max = emb[:, 0].min(), emb[:, 0].max()
    y_min, y_max = emb[:, 1].min(), emb[:, 1].max()
    dx = (x_max - x_min) * pad
    dy = (y_max - y_min) * pad
    xx, yy = np.meshgrid(
        np.linspace(x_min - dx, x_max + dx, grid_size),
        np.linspace(y_min - dy, y_max + dy, grid_size),
    )
    grid = np.vstack([xx.ravel(), yy.ravel()])

    for pi, participant in enumerate(PARTICIPANTS):
        pts = emb[parts == pi]
        if len(pts) < 5:  # KDE needs a few points
            continue
        try:
            kde = gaussian_kde(pts.T)
        except np.linalg.LinAlgError:
            continue  # singular covariance (collinear points)
        density = kde(grid).reshape(xx.shape)
        flat = np.sort(density.ravel())[::-1]
        cumulative = np.cumsum(flat) / flat.sum()
        thresholds = [flat[np.searchsorted(cumulative, lvl)] for lvl in levels]
        ax.contour(xx, yy, density, levels=sorted(thresholds),
                colors=[PARTICIPANT_COLORS[participant]], linewidths=1.5, alpha=0.9)

def plot_vae_tSNE(models, datasets, region, free_bits=[0.0, 0.1, 2.0]):
    train_ds = datasets[SEEDS[0]][region]["train"]
    classes = np.array([train_ds[i][2] for i in range(len(train_ds))])
    parts   = np.array([train_ds[i][3] for i in range(len(train_ds))])

    fig, axes = plt.subplots(1, len(free_bits), figsize=(WIDTH * len(free_bits), HEIGHT*1.5), sharey=True)

    for c_idx, free_bit in enumerate(free_bits):
        ax = axes[c_idx]
        mu, _ = encoder_outputs(models[SEEDS[0]][region][free_bit], train_ds, device="cpu")
        mu = mu.numpy()

        eff_perp = min(30, max(5, (len(mu) - 1) // 3))
        emb = TSNE(n_components=2, perplexity=eff_perp,
                random_state=SEEDS[0],
                init="pca", learning_rate="auto").fit_transform(mu)

        # Points colored by class
        for ci, name in enumerate(CLASSES):
            mask = classes == ci
            ax.scatter(emb[mask, 0], emb[mask, 1], c=CLASS_COLORS[name],
                    label=name if c_idx == 0 else None,
                    s=22, alpha=0.8, edgecolors="none")

        # Contours per participant
        participant_contours(ax, emb, parts, levels=(0.8,))

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(rf"$\lambda_\mathrm{{fb}} = {free_bit}$")

    # Build a combined legend: classes (filled) + participants (line)
    class_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASS_COLORS[name],
                            markersize=8, label=name.capitalize())
                     for name in CLASSES]
    part_handles = [Line2D([0], [0], color=PARTICIPANT_COLORS[p], linewidth=2,
                           label=f"Participant {p.capitalize()}")
                    for p in PARTICIPANTS]
    axes[0].legend(handles=class_handles + part_handles, loc="lower left", ncol=1)

    plt.tight_layout()
    plt.show()

# conditional variational autoencoder
def load_cvae_ablation_data(arch_dir="results/ablation_cvae_architecture", dyn_dir="results/ablation_cvae_training_dynamics"):
    def load_histories(directory, key_col):
        pattern = re.compile(r"(?P<name>.+)_(?P<region>mouth|nose)_s(?P<seed>\d+)_history$")
        frames = []
        for path_str in sorted(glob.glob(str(Path(directory) / "*_history.csv"))):
            path = Path(path_str)
            match = pattern.match(path.stem)
            if match is None:
                raise ValueError(f"Unexpected history filename: {path.name}")
            frame = pd.read_csv(path)
            frame[key_col] = match.group("name")
            frame["region"] = match.group("region")
            frame["seed"] = int(match.group("seed"))
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    combined_configs = ["lag_5_250_beta_0.01", "lag_5_250_beta_0.1"]
    dynamics_configs = BETA_CONFIGS + LAG_CONFIGS + combined_configs

    arch_summary = pd.read_csv(f"{arch_dir}/summary.csv")
    arch_summary["variant"] = pd.Categorical(arch_summary["variant"], categories=ARCHITECTURES, ordered=True)
    arch_summary = arch_summary.sort_values(["region", "variant", "seed"]).reset_index(drop=True)

    arch_history = load_histories(arch_dir, "variant")
    arch_history["variant"] = pd.Categorical(arch_history["variant"], categories=ARCHITECTURES, ordered=True)
    arch_history = arch_history.sort_values(["region", "variant", "seed", "epoch"]).reset_index(drop=True)

    dyn_summary = pd.read_csv(f"{dyn_dir}/summary.csv")
    dyn_summary["config"] = pd.Categorical(dyn_summary["config"], categories=dynamics_configs, ordered=True)
    dyn_summary = dyn_summary.sort_values(["region", "config", "seed"]).reset_index(drop=True)

    dyn_history = load_histories(dyn_dir, "config")
    dyn_history["config"] = pd.Categorical(dyn_history["config"], categories=dynamics_configs, ordered=True)
    dyn_history = dyn_history.sort_values(["region", "config", "seed", "epoch"]).reset_index(drop=True)

    return arch_summary, arch_history, dyn_summary, dyn_history

def load_cvaes(device, configs=["beta_cap_0.001", "beta_cap_0.1", "beta_cap_1.0"], seed=0, ckpt_dir="results/ablation_cvae_training_dynamics"):
    models = {region: {} for region in REGIONS}
    datasets = {}

    for region in REGIONS:
        df = load_dataset()
        df = df[df["region"] == region].reset_index(drop=True)
        df_train, _, _ = split_dataset(df, random_state=seed)
        datasets[region] = BreathDataset(df_train)

        for cfg in configs:
            ckpt = torch.load(f"{ckpt_dir}/{cfg}_{region}_s{seed}.pt", map_location=device, weights_only=False)
            model = CVAE(latent_dim=ckpt["latent_dim"],
                         embed_dim=ckpt["embed_dim"],
                         condition_on_participant=True,
                         part_embed_dim=ckpt["part_embed_dim"]).to(device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            models[region][cfg] = model

    return models, datasets

@torch.no_grad()
def cvae_encoder_outputs(model, dataset, device):
    mus, logvars = [], []
    model.eval()
    for i in range(len(dataset)):
        signal, _, label, part = dataset[i]
        y = torch.tensor([label], dtype=torch.long, device=device)
        p = torch.tensor([part], dtype=torch.long, device=device) if model._cond_part else None
        mu, logvar = model.encoder(signal.unsqueeze(0).to(device), y, p)
        mus.append(mu.squeeze(0).cpu())
        logvars.append(logvar.squeeze(0).cpu())
    return torch.stack(mus), torch.stack(logvars)

def plot_cvae_architecture_ablation(arch_summary):
    metrics = ["accuracy", "f1_weighted", "roc_auc"]
    titles = {"accuracy": "Accuracy", "f1_weighted": "F1 (weighted)", "roc_auc": "ROC-AUC"}
    stats = arch_summary.groupby(["region", "variant"], observed=True)[metrics].agg(["mean", "std"])

    fig, axes = plt.subplots(2, 3, figsize=(WIDTH * 3, HEIGHT * 2), sharex="col", sharey="row")
    x = np.arange(len(ARCHITECTURES))
    for row, region in enumerate(REGIONS):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            means = stats.loc[(region,), (metric, "mean")].reindex(ARCHITECTURES).to_numpy()
            stds  = stats.loc[(region,), (metric, "std")].reindex(ARCHITECTURES).to_numpy()
            ax.bar(x, means, yerr=stds, capsize=4, color=[ARCHITECTURES_COLORS[v] for v in ARCHITECTURES], edgecolor="black", linewidth=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(ARCHITECTURES, rotation=45)
            ax.set_ylim(0, 1.05)
            axes[0, col].set_title(titles[metric])
            axes[row, 0].set_ylabel(f"{region.capitalize()}\nScore")
            ax.grid(axis="y")

    plt.suptitle("Architecture ablation")
    plt.tight_layout()
    plt.show()

def plot_cvae_architecture_dynamics(arch_history, warmup_end=250, latent_dim=16):
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH * 3, HEIGHT * 2), sharex=True, sharey="row")
    for col, region in enumerate(REGIONS):
        ax_kl = axes[0, col]
        ax_ad = axes[1, col]
        for variant in ARCHITECTURES:
            curve = (arch_history[(arch_history["region"] == region) & (arch_history["variant"] == variant)]
                     .groupby("epoch", observed=True)[["train_kl", "active_dims"]]
                     .agg(["mean", "std"]))
            epochs = curve.index.to_numpy()
            kl_mean = curve[("train_kl", "mean")].to_numpy()
            kl_std  = curve[("train_kl", "std")].fillna(0).to_numpy()
            ad_mean = curve[("active_dims", "mean")].to_numpy()
            ad_std  = curve[("active_dims", "std")].fillna(0).to_numpy()
            color = ARCHITECTURES_COLORS[variant]

            ax_kl.plot(epochs, kl_mean, color=color, linewidth=2, label=variant)
            ax_kl.fill_between(epochs, kl_mean - kl_std, kl_mean + kl_std, color=color, alpha=0.15)
            ax_ad.plot(epochs, ad_mean, color=color, linewidth=2)
            ax_ad.fill_between(epochs, ad_mean - ad_std, ad_mean + ad_std, color=color, alpha=0.15)

        ax_kl.axvline(warmup_end, color="black", linestyle="--")
        ax_ad.axvline(warmup_end, color="black", linestyle="--")
        ax_ad.axhline(latent_dim, color="black", linestyle=":")
        ax_kl.set_title(region.capitalize())
        axes[0, 0].set_ylabel("KL divergence")
        axes[1, 0].set_ylabel("Active dimensions")
        ax_ad.set_xlabel("Epoch")
        ax_ad.set_ylim(0, latent_dim + 1)
        ax_kl.grid()
        ax_ad.grid()

    handles = [Line2D([0], [0], color=c, lw=2, label=name) for name, c in ARCHITECTURES_COLORS.items()]
    handles += [Line2D([0], [0], color="black", lw=1, linestyle="--", label=rf"$\beta$ warmup end (epoch {warmup_end})"),
                Line2D([0], [0], color="black", lw=1, linestyle=":", label=f"Latent dim = {latent_dim}")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=len(handles), frameon=False)

    plt.suptitle("Architecture ablations: training dynamics")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

def plot_cvae_beta_sweep(dyn_summary):
    region_colors = {"mouth": "tab:blue", "nose": "tab:orange"}
    metrics = ["accuracy", "active_dims", "kl_final"]
    ylabels = {"accuracy": "TSTR accuracy", "active_dims": "Active dimensions", "kl_final": "Final KL"}

    beta_only = dyn_summary[dyn_summary["config"].isin(BETA_CONFIGS)].copy()
    beta_only["beta_max"] = beta_only["config"].map(BETA_VALUES).astype(float)

    beta_best = (beta_only.groupby(["region", "config"], observed=True)["accuracy"].mean()
                 .groupby("region", observed=True)
                 .idxmax())
    beta_best = {region: cfg for region, (_, cfg) in beta_best.items()}

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH * 3, HEIGHT * 1.5))
    for ax, metric in zip(axes, metrics):
        for region, linestyle in zip(REGIONS, ["-", "--"]):
            stats = (beta_only[beta_only["region"] == region]
                     .groupby("beta_max", observed=True)[metric]
                     .agg(["mean", "std"])
                     .reindex([BETA_VALUES[cfg] for cfg in BETA_CONFIGS]))
            x = stats.index.to_numpy(dtype=float)
            y = stats["mean"].to_numpy(dtype=float)
            std = stats["std"].fillna(0).to_numpy(dtype=float)

            ax.plot(x, y, color=region_colors[region], linewidth=2.0, linestyle=linestyle, label=region.capitalize())
            ax.fill_between(x, y - std, y + std, color=region_colors[region], alpha=0.12)
            ax.scatter(x, y, s=40, c=region_colors[region], zorder=3)

            best_x = BETA_VALUES[beta_best[region]]
            best_y = stats.loc[best_x, "mean"]
            ax.scatter([best_x], [best_y], marker="*", s=160, color=region_colors[region],
                       edgecolor="black", linewidth=0.8, zorder=4)

        ax.set_xscale("log")
        ax.set_xticks([BETA_VALUES[cfg] for cfg in BETA_CONFIGS], [str(BETA_VALUES[cfg]) for cfg in BETA_CONFIGS])
        ax.set_xlabel(r"$\beta_{\max}$")
        ax.set_ylabel(ylabels[metric])
        ax.grid()

    handles = [Line2D([0], [0], color=region_colors[r], linestyle=style, lw=2.0, label=r.capitalize())
               for r, style in zip(REGIONS, ["-", "--"])]
    handles.append(Line2D([0], [0], marker="*", color="black", lw=0, markersize=10, label=r"Best $\beta$-cap per region"))
    axes[-1].legend(handles=handles, loc="upper right")

    plt.suptitle(r"$\beta$-cap sweep")
    plt.tight_layout()
    plt.show()

def plot_cvae_beta_active_dims(dyn_history, warmup_end=250, latent_dim=16):
    beta_colors = dict(zip(BETA_CONFIGS, sns.color_palette("Blues", n_colors=len(BETA_CONFIGS) + 2)[2:]))

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH * 3, HEIGHT * 1.5), sharey=True)
    for ax, region in zip(axes, REGIONS):
        for cfg in BETA_CONFIGS:
            curve = (dyn_history[(dyn_history["region"] == region) & (dyn_history["config"] == cfg)]
                     .groupby("epoch", observed=True)["active_dims"]
                     .agg(["mean", "std"]))
            epochs = curve.index.to_numpy()
            mean = curve["mean"].to_numpy(dtype=float)
            std = curve["std"].fillna(0).to_numpy(dtype=float)
            color = beta_colors[cfg]
            ax.plot(epochs, mean, color=color, linewidth=2.0, label=CONFIG_LABELS[cfg])
            ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.14)

        ax.axvline(warmup_end, color="black", linestyle="--", linewidth=1)
        ax.axhline(latent_dim, color="black", linestyle=":", linewidth=1)
        ax.set_title(region.capitalize())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Active dimensions") if ax == axes[0] else None
        ax.set_ylim(0, latent_dim + 1)
        ax.grid()

    handles = [Line2D([0], [0], color=beta_colors[cfg], lw=2.0, label=CONFIG_LABELS[cfg]) for cfg in BETA_CONFIGS]
    handles += [Line2D([0], [0], color="black", lw=1, linestyle="--", label=rf"$\beta$ warmup end (epoch {warmup_end})"),
                Line2D([0], [0], color="black", lw=1, linestyle=":", label=f"Latent dim = {latent_dim}")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=len(handles), frameon=False)


    plt.suptitle(r"$\beta$-cap sweep: training dynamics")
    plt.tight_layout()
    plt.show()

def plot_cvae_tSNE(models, datasets, region, configs=["beta_cap_0.001", "beta_cap_0.1", "beta_cap_1.0"], device="cpu", random_state=0, perplexity=30.0):
    dataset = datasets[region]
    region_models = models[region]
    classes = np.array([dataset[i][2] for i in range(len(dataset))])
    parts   = np.array([dataset[i][3] for i in range(len(dataset))])

    fig, axes = plt.subplots(1, len(configs), figsize=(WIDTH * len(configs), HEIGHT * 1.5), sharey=True)

    for c_idx, cfg in enumerate(configs):
        ax = axes[c_idx]
        mu, _ = cvae_encoder_outputs(region_models[cfg], dataset, device=device)
        mu = mu.numpy()

        eff_perp = min(perplexity, max(5, (len(mu) - 1) // 3))
        emb = TSNE(n_components=2, perplexity=eff_perp, random_state=random_state,
                   init="pca", learning_rate="auto").fit_transform(mu)

        for ci, name in enumerate(CLASSES):
            mask = classes == ci
            ax.scatter(emb[mask, 0], emb[mask, 1], c=CLASS_COLORS[name],
                       s=22, alpha=0.8, edgecolors="none")

        participant_contours(ax, emb, parts, levels=(0.8,))

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(rf"$\beta_{{\max}} = {BETA_VALUES[cfg]}$")

    class_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASS_COLORS[name],
                            markersize=8, label=name.capitalize()) for name in CLASSES]
    part_handles = [Line2D([0], [0], color=PARTICIPANT_COLORS[p], linewidth=2,
                           label=f"Participant {p.capitalize()}") for p in PARTICIPANTS]
    axes[0].legend(handles=class_handles + part_handles, loc="lower left", ncol=1)

    plt.tight_layout()
    plt.show()

def plot_cvae_lag(dyn_summary):
    lag_colors = {"beta_cap_1.0": "#4d4d4d", **dict(zip(LAG_CONFIGS, sns.color_palette("viridis", n_colors=len(LAG_CONFIGS) + 3)[1:-2]))}
    plot_configs = ["beta_cap_1.0", *LAG_CONFIGS]
    labels = {"beta_cap_1.0": r"$\beta$ 1.0 baseline", **{cfg: CONFIG_LABELS[cfg] for cfg in LAG_CONFIGS}}

    stats = (dyn_summary[dyn_summary["config"].isin(plot_configs)]
             .groupby(["region", "config"], observed=True)[["accuracy"]]
             .agg(["mean", "std"]))

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH * 3, HEIGHT * 1.5), sharey=True)
    x = np.arange(len(plot_configs))
    colors = [lag_colors[cfg] for cfg in plot_configs]

    for ax, region in zip(axes, REGIONS):
        means = stats.loc[(region,), ("accuracy", "mean")].reindex(plot_configs).to_numpy()
        stds  = stats.loc[(region,), ("accuracy", "std")].reindex(plot_configs).to_numpy()
        baseline_acc = means[0]

        ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.7)
        ax.axhline(baseline_acc, color=lag_colors["beta_cap_1.0"], linestyle="--", linewidth=1.2)
        ax.set_xticks(x, [labels[cfg] for cfg in plot_configs], rotation=20, ha="right")
        ax.set_title(region.capitalize())
        ax.set_xlabel("Configuration")
        ax.set_ylabel("TSTR accuracy") if ax == axes[0] else None
        ax.set_ylim(0, max(1.0, means.max() + stds.max() + 0.05))
        ax.grid(axis="y")

    plt.suptitle("Lagging inference")
    plt.tight_layout()
    plt.show()

def plot_cvae_summary(dyn_summary, arch_summary):
    rank_colors = {"beta": "tab:green", "lag": "tab:purple", "combined": "tab:red"}

    def family_of(config):
        if config in BETA_CONFIGS:
            return "beta"
        if config in LAG_CONFIGS:
            return "lag"
        return "combined"

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH * 3, HEIGHT * 2), sharey="row")
    for ax, region in zip(axes, REGIONS):
        stats = (dyn_summary[dyn_summary["region"] == region]
                 .groupby("config", observed=True)["accuracy"]
                 .agg(["mean", "std"])
                 .sort_values("mean", ascending=True))
        labels = [CONFIG_LABELS[cfg] for cfg in stats.index]
        colors = [rank_colors[family_of(cfg)] for cfg in stats.index]
        ax.barh(labels, stats["mean"].values,
                xerr=stats["std"].fillna(0).values,
                color=colors, edgecolor="black", linewidth=0.6, capsize=3,
                error_kw={"elinewidth": 1.0, "ecolor": "black"})
        baseline = arch_summary[(arch_summary["region"] == region) & (arch_summary["variant"] == "conv_baseline")]["accuracy"].mean()
        ax.axvline(baseline, color="black", linestyle="--", linewidth=1.2)
        ax.set_title(region.capitalize())
        ax.set_xlabel("TSTR accuracy")
        ax.set_ylabel("Configuration") if ax == axes[0] else None
        ax.set_xlim(0, 1.0)
        ax.grid(axis="x")

    handles = [Patch(facecolor=rank_colors["beta"], edgecolor="black", label=r"$\beta$-cap"),
               Patch(facecolor=rank_colors["lag"], edgecolor="black", label="Lagging"),
               Patch(facecolor=rank_colors["combined"], edgecolor="black", label="Combined"),
               Line2D([0], [0], color="black", linestyle="--", lw=1.2, label="Architecture baseline")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=len(handles), frameon=False)

    plt.suptitle("Summary of training dynamics configs")
    plt.tight_layout()
    plt.show()

def plot_cvae_jittering():
    df = pd.read_csv("results/ablation_cvae_jittering/summary.csv")

    metrics = ["accuracy", "f1_weighted", "roc_auc"]
    titles = {"accuracy": "Accuracy", "f1_weighted": "F1 (weighted)", "roc_auc": "ROC-AUC"}

    configs = df.drop_duplicates("config").sort_values(["alpha", "n_copies"])["config"].tolist()
    alphas = sorted({a for a in df["alpha"].unique() if a > 0})
    alpha_colors = dict(zip(alphas, sns.color_palette("viridis", n_colors=len(alphas))))
    bar_colors = ["lightgray" if cfg == "baseline" else alpha_colors[df[df["config"] == cfg]["alpha"].iloc[0]] for cfg in configs]

    def label(cfg):
        if cfg == "baseline":
            return "baseline"
        a = df[df["config"] == cfg]["alpha"].iloc[0]
        n = df[df["config"] == cfg]["n_copies"].iloc[0]
        return rf"$\alpha$={a}, n={n}"

    stats = df.groupby(["region", "config"], observed=True)[metrics].agg(["mean", "std"])

    fig, axes = plt.subplots(2, 3, figsize=(WIDTH * 3, HEIGHT * 2.2), sharex="col", sharey="row")
    x = np.arange(len(configs))
    for row, region in enumerate(REGIONS):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            means = stats.loc[(region,), (metric, "mean")].reindex(configs).to_numpy()
            stds = stats.loc[(region,), (metric, "std")].reindex(configs).to_numpy()
            best_idx = int(np.nanargmax(means))

            ax.bar(x, means, yerr=stds, capsize=3, color=bar_colors, edgecolor="black", linewidth=0.6)
            ax.bar(x[best_idx], means[best_idx], yerr=stds[best_idx], capsize=3,
                   color=bar_colors[best_idx], edgecolor="tab:red", linewidth=1.6)

            baseline_mean = stats.loc[(region, "baseline"), (metric, "mean")]
            ax.axhline(baseline_mean, color="black", linestyle="--", linewidth=1.0, alpha=0.6)

            ax.set_xticks(x)
            ax.set_xticklabels([label(c) for c in configs], rotation=45)
            ax.set_ylim(0, 1.05)
            axes[0, col].set_title(titles[metric])
            axes[row, 0].set_ylabel(f"{region.capitalize()}\nScore")
            ax.grid(axis="y")

    handles = [Patch(facecolor="lightgray", edgecolor="black", label="Baseline")]
    handles += [Patch(facecolor=alpha_colors[a], edgecolor="black", label=rf"$\alpha$={a}") for a in alphas]
    handles += [Line2D([0], [0], color="black", linestyle="--", lw=1.0, label="Baseline mean"),
                Patch(facecolor="none", edgecolor="tab:red", linewidth=1.6, label="Best per panel")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=len(handles), frameon=False)

    plt.suptitle("Jittering augmentation ablation")
    plt.tight_layout()
    plt.show()
