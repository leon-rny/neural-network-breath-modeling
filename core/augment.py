"""Synthetic training trials for the classifiers of core.nested.

The DHT22 does not report baseline-corrected physical signals directly (0.1-unit reporting grid, 99.9 %RH ceiling,
ambient baseline); `observe_with` applies that measurement chain so synthetic trials live in the space of the real
recordings.
  rerender           re-renderings of real trials under a random sensor gain x/÷1.8 per channel + the DHT22 observation
                     (physics gain, policy physgain); also the gain step of the generator posterior samples (postgain)
  posterior_samples  one posterior sample per real parent trial: encode (trial, class, participant), draw z ~ q(z|x),
                     decode with the parent's class and participant, add the parent's baseline, DHT22 observation
  prior_samples      class-balanced prior samples decoded with the null participant token (fidelity, TSTR)
  jitter / mixup     non-generative controls: Gaussian jitter of real trials / mixup of two trials of the same class
Every draw comes from a named random stream (crc32 of seed / task / stream name), so results do not depend on what
else runs in the same process, and matched comparisons (e.g. CVAE vs. PI-CVAE posterior samples) share their noise.
"""

import zlib

import numpy as np
import pandas as pd
import torch

from core.data import CLASS_TO_IDX

H_CEILING = 99.9
RESOLUTION = 0.1
DT = 2.004  # sampling interval (s)
GAIN_MAX = 1.8  # physics gain: log-uniform in [1 / GAIN_MAX, GAIN_MAX] per trial and channel
JITTER_ALPHA = 0.05  # jitter SD in units of the per-channel training SD (the CVAE's training jitter)
MIXUP_BETA = 0.4  # mixup weight lambda ~ Beta(0.4, 0.4)


def stream_rng(seed: int, key, name: str) -> np.random.RandomState:
    """Named random stream for one (seed, task key) partition."""
    return np.random.RandomState(zlib.crc32(f'{seed}/{key}/{name}'.encode()))


def torch_stream(rng: np.random.RandomState) -> torch.Generator:
    """torch generator seeded from a named stream."""
    return torch.Generator().manual_seed(int(rng.randint(2 ** 31)))


# sensor observation model
def parent_baselines(df_parents: pd.DataFrame, nb: int = 5) -> np.ndarray:
    """(n, 2) absolute pre-onset baselines of the given trials."""
    return np.array([[np.mean(h[:nb]), np.mean(t[:nb])] for h, t in zip(df_parents['humidity'], df_parents['temperature'])])


def observe_with(signals: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Baseline-corrected signals (n, 2, T) + per-sample absolute baselines (n, 2) -> DHT22 readings."""
    out = np.asarray(signals, dtype=float) + base[:, :, None]
    out = np.round(out / RESOLUTION) * RESOLUTION
    out[:, 0] = np.clip(out[:, 0], 0.0, H_CEILING)
    return out


# physics re-rendering of real trials
def sensor_response(x: np.ndarray, dtau: float) -> np.ndarray:
    """First-order sensor response change of (n, T) baseline-corrected signals: dtau > 0 adds a lag of time constant
    dtau (s), dtau < 0 inverts one (sharper), 0 leaves the signal unchanged."""
    if dtau == 0:
        return x
    a = 1.0 - np.exp(-DT / abs(dtau))
    y = np.empty_like(x)
    if dtau > 0:
        y[:, 0] = x[:, 0]
        for n in range(1, x.shape[1]):
            y[:, n] = y[:, n - 1] + a * (x[:, n] - y[:, n - 1])
    else:
        y[:, 0] = x[:, 0]
        y[:, 1:] = (x[:, 1:] - (1 - a) * x[:, :-1]) / a
    return y


def rerender(signals: np.ndarray, copies: int, gain_max: float, tau_max: float, rng: np.random.RandomState) -> np.ndarray:
    """`copies` re-renderings of absolute (n, 2, T) trials under random sensor physics: per trial and channel a gain g
    (log-uniform in [1/gain_max, gain_max]: breath strength x sensor coupling) and a response change dtau (uniform in
    [-tau_max / 2, tau_max] s) act on the baseline-corrected signal, then the DHT22 observation (0.1 grid, 99.9 %RH
    ceiling). tau_max = 0 randomizes the gain only (physics gain)."""
    S = np.asarray(signals, dtype=float)
    out = []
    for _ in range(copies):
        for i in range(len(S)):
            theta = [(float(np.exp(rng.uniform(-np.log(gain_max), np.log(gain_max)))), float(rng.uniform(-tau_max / 2, tau_max))) for _c in range(2)]
            sig = S[i:i + 1]
            base = sig[:, :, :5].mean(2, keepdims=True)
            r = np.empty_like(sig)
            for c, (g, dt) in enumerate(theta):
                r[:, c] = g * sensor_response(sig[:, c] - base[:, c], dt) + base[:, c]
            r = np.round(r / RESOLUTION) * RESOLUTION
            r[:, 0] = np.clip(r[:, 0], 0.0, H_CEILING)
            out.append(r)
    return np.concatenate(out)


# generator samples
def sample_parents(df: pd.DataFrame, n_add: int, rng: np.random.RandomState) -> tuple[pd.DataFrame, np.ndarray]:
    """Class-balanced parent trials (without replacement while the class pool lasts) and their labels."""
    lab = np.array([CLASS_TO_IDX[c] for c in df['class']])
    counts = [n_add // 3 + (1 if i < n_add % 3 else 0) for i in range(3)]
    idx = np.concatenate([rng.choice(np.where(lab == c)[0], k, replace=k > (lab == c).sum()) for c, k in zip(range(3), counts)])
    return df.iloc[idx].reset_index(drop=True), lab[idx]


def posterior_samples(gen, df_parents: pd.DataFrame, rng: np.random.RandomState) -> np.ndarray:
    """One posterior sample per real parent trial (core.train.Generator), decoded with the parent's class and
    participant, + the parent's baseline and the DHT22 observation. Returns observed absolute signals (n, 2, T)."""
    x, y, p = gen.inputs(df_parents)
    g = torch_stream(rng)
    with torch.no_grad():
        mu, logvar = gen.model.encoder(x, y, p)
        z = mu + (0.5 * logvar).exp() * torch.randn(mu.shape, generator=g)
        xh = gen.model.decode(z, y, p)
    return observe_with(gen.signals(xh), parent_baselines(df_parents))


def prior_samples(gen, n: int, rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    """n class-balanced prior samples z ~ N(0, I) decoded with the null participant token (an unseen participant).
    Returns baseline-corrected signals (n, 2, T) in physical units and their labels."""
    y = np.concatenate([np.full(n // 3 + (1 if c < n % 3 else 0), c) for c in range(3)])
    z = torch.randn((n, gen.model.latent_dim), generator=torch_stream(rng))
    with torch.no_grad():
        xh = gen.model.decode(z, torch.as_tensor(y), torch.full((n,), gen.model.null_part_idx))
    return gen.signals(xh), y


# non-generative controls
def jitter(signals: np.ndarray, copies: int, std: np.ndarray, rng: np.random.RandomState, alpha: float = JITTER_ALPHA) -> np.ndarray:
    """`copies` jittered copies of absolute (n, 2, T) trials: + alpha x std[channel] x N(0, 1) per sample."""
    S = np.tile(np.asarray(signals, dtype=float), (copies, 1, 1))
    return S + alpha * np.asarray(std, dtype=float).reshape(1, 2, 1) * rng.standard_normal(S.shape)


def mixup(signals: np.ndarray, labels: np.ndarray, copies: int, rng: np.random.RandomState, beta: float = MIXUP_BETA) -> np.ndarray:
    """`copies` mixups of every trial with a random trial of the same class: lambda x + (1 - lambda) x', lambda ~ Beta."""
    S, labels = np.asarray(signals, dtype=float), np.asarray(labels)
    out = []
    for _ in range(copies):
        for i in range(len(S)):
            j = rng.choice(np.where(labels == labels[i])[0])
            lam = rng.beta(beta, beta)
            out.append(lam * S[i] + (1 - lam) * S[j])
    return np.stack(out)
