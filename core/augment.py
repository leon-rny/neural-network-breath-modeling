"""Physics-generator augmentation policies for TSTR+ (core.tstr --mode tstr_plus --aug_policy ...) and
signal-space MMD (fidelity).

The physics generators emit baseline-corrected signals in physical units; the DHT22 does not report those
directly (0.1-unit reporting grid, 99.9 %RH ceiling, ambient baseline). `observe_with` / `sensor_observe`
apply that measurement chain so synthetic trials live in the same space as real recordings.

Policies (synthetic : real ratio via --augmentation_ratio):
  prior         samples from the generator prior (core.tstr.generate_synthetic_signals; legacy TSTR+)
  steer_margin  boundary steering: posterior samples of class-balanced real parents, searched (cross-entropy
                method, 12 candidates x 3 iterations inside the posterior SD) for observed samples whose
                real-only top-20 features lie near the class boundary but on the correct side
                (k-NN class margin ~0.5)
  steer_knn     the same search towards class-typical features (largest k-NN class margin)
  obsdr         prior samples (3x pool) through the sensor model, kept by an out-of-fold real-vs-synthetic
                random forest (discriminator rejection), class-balanced
Every policy draws from a named random stream (crc32 of seed / fold / stream name) so results do not depend
on what else runs in the same process.
`rerender` re-renders real trials under random sensor gain / response + the DHT22 observation (core.cnn).
"""

import zlib

import numpy as np
import pandas as pd

from core.data import CLASS_TO_IDX, PARTICIPANT_TO_IDX

H_CEILING = 99.9
RESOLUTION = 0.1
POLICY_KINDS = {'steer_margin': 'steer@margin', 'steer_knn': 'steer@knn', 'obsdr': 'obsdr'}


def stream_rng(seed: int, fold: int, name: str) -> np.random.RandomState:
    """Named random stream for one (seed, fold) partition."""
    return np.random.RandomState(zlib.crc32(f'{seed}/{fold}/{name}'.encode()))


def top_features(signals: np.ndarray, cache: dict, preprocessing: str = 'baseline') -> np.ndarray:
    """Preprocess (n, 2, T) signals like the real data and extract the cached top-20 features."""
    from core.tstr import _feat_top, preprocess_synth_signals
    sig = preprocess_synth_signals(signals, preprocessing)
    n, _, L = sig.shape
    dl = pd.DataFrame({'id': np.repeat(np.arange(n), L), 'time': np.tile(np.arange(L), n),
                       'Humidity': sig[:, 0].ravel(), 'Temperature': sig[:, 1].ravel()})
    return _feat_top(dl, cache, 0)


# sensor observation model
def training_baselines(df_train: pd.DataFrame, nb: int = 5) -> dict[int, np.ndarray]:
    """Per-class arrays of absolute pre-onset baselines [humidity, temperature] of real training trials."""
    out: dict[int, list] = {}
    for cls, h, t in zip(df_train['class'], df_train['humidity'], df_train['temperature']):
        out.setdefault(CLASS_TO_IDX[cls], []).append([np.mean(h[:nb]), np.mean(t[:nb])])
    return {k: np.asarray(v) for k, v in out.items()}


def parent_baselines(df_parents: pd.DataFrame, nb: int = 5) -> np.ndarray:
    """(n, 2) absolute pre-onset baselines of the given trials."""
    return np.array([[np.mean(h[:nb]), np.mean(t[:nb])] for h, t in zip(df_parents['humidity'], df_parents['temperature'])])


def sensor_observe(signals: np.ndarray, labels: np.ndarray, baselines: dict[int, np.ndarray], rng: np.random.RandomState) -> np.ndarray:
    """Baseline-corrected signals (n, 2, T) -> DHT22 readings: same-class real baseline, 0.1 grid, 99.9 %RH ceiling."""
    out = np.asarray(signals, dtype=float).copy()
    for i, y in enumerate(labels):
        pool = baselines[int(y)]
        out[i] += pool[rng.randint(len(pool))][:, None]
    out = np.round(out / RESOLUTION) * RESOLUTION
    out[:, 0] = np.clip(out[:, 0], 0.0, H_CEILING)
    return out


def observe_with(signals: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Observation model with given per-sample absolute baselines (n, 2)."""
    out = np.asarray(signals, dtype=float) + base[:, :, None]
    out = np.round(out / RESOLUTION) * RESOLUTION
    out[:, 0] = np.clip(out[:, 0], 0.0, H_CEILING)
    return out


# physics re-rendering of real trials (sensor gain / response randomization, core.cnn)
DT = 2.004  # sampling interval (s)


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
    """`copies` re-renderings of absolute (n, 2, T) real trials under random sensor physics: per trial and channel a
    gain g (log-uniform in [1/gain_max, gain_max]: breath strength x sensor coupling) and a response change dtau
    (uniform in [-tau_max / 2, tau_max] s) act on the baseline-corrected signal, then the DHT22 observation (0.1 grid,
    99.9 %RH ceiling). tau_max = 0 randomizes the gain only."""
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


def denormalize(x, stats: dict) -> np.ndarray:
    """Physics-generator output (n, 2, T) in training space -> physical units (as generate_synthetic_signals)."""
    mode = stats.get('phys_prep', 'peakscale')
    if mode == 'shared':
        scale = (stats['shared_scale'], stats['shared_scale'])
    elif mode == 'stdscale':
        scale = (stats['h_bcstd'], stats['t_bcstd'])
    else:
        scale = (stats['h_scale'], stats['t_scale'])
    return np.asarray(x, dtype=float) * np.asarray(scale, dtype=float).reshape(1, 2, 1)


# selection helpers
def select_balanced(labels: np.ndarray, n_total: int, score: np.ndarray | None = None, rng: np.random.RandomState | None = None) -> np.ndarray:
    """Indices of a class-balanced subset of size n_total, by descending score (or random order when score is None)."""
    classes = np.unique(labels)
    counts = [n_total // len(classes) + (1 if i < n_total % len(classes) else 0) for i in range(len(classes))]
    keep = []
    for c, k in zip(classes, counts):
        idx = np.where(labels == c)[0]
        if score is None:
            idx = idx if rng is None else rng.permutation(idx)
        else:
            idx = idx[np.argsort(-score[idx], kind='stable')]
        keep.extend(idx[:k].tolist())
    return np.asarray(sorted(keep))


def sample_parents(df: pd.DataFrame, n_add: int, rng: np.random.RandomState) -> tuple[pd.DataFrame, np.ndarray]:
    """Class-balanced parent trials (without replacement while the class pool lasts) and their labels."""
    lab = np.array([CLASS_TO_IDX[c] for c in df['class']])
    counts = [n_add // 3 + (1 if i < n_add % 3 else 0) for i in range(3)]
    idx = np.concatenate([rng.choice(np.where(lab == c)[0], k, replace=k > (lab == c).sum()) for c, k in zip(range(3), counts)])
    return df.iloc[idx].reset_index(drop=True), lab[idx]


def realness(X_real: np.ndarray, X_syn: np.ndarray, seed: int) -> np.ndarray:
    """Out-of-fold probability that each synthetic row is real (real-vs-synthetic random forest, feature space)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict
    X = np.vstack([X_real, X_syn])
    d = np.r_[np.zeros(len(X_real)), np.ones(len(X_syn))]
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=1)
    p = cross_val_predict(rf, X, d, cv=5, method='predict_proba')[:, 1]
    return 1.0 - p[len(X_real):]


# feature-guided posterior sampling
def steer_samples(model, stats: dict, df_parents: pd.DataFrame, feat_fn, X_real: np.ndarray, y_real: np.ndarray,
                  rng: np.random.RandomState, score: str = 'margin', K: int = 12, iters: int = 3, n_elite: int = 3,
                  k_nn: int = 5, margin_target: float = 0.5) -> np.ndarray:
    """Posterior samples of real parents steered in the real-only classifier's feature space.

    For every parent: start at its posterior mean, sample K latents around the current centre (scale bounded by
    the posterior SD), decode, apply the sensor model with the parent's baseline, compute the top-20 features
    (feat_fn) and score them against the real training trials (standardized features):
      margin = mean distance to the k_nn nearest trials of the other classes - same for the own class
      'margin': |margin - margin_target| + 2 max(0, -margin)   (near the boundary, on the correct side)
      'knn'   : -margin                                         (class-typical)
    The centre moves to the mean of the n_elite best candidates (cross-entropy method); the best candidate over
    all iterations is returned. Returns observed absolute signals (n, 2, T).
    """
    import torch
    from sklearn.neighbors import NearestNeighbors

    from core.data import PhysicsInformedDataset
    ds = PhysicsInformedDataset(df_parents, stats=stats)
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    y = torch.tensor([CLASS_TO_IDX[c] for c in df_parents['class']], dtype=torch.long)
    p = torch.tensor([PARTICIPANT_TO_IDX[q] for q in df_parents['participant']], dtype=torch.long)
    base = parent_baselines(df_parents)
    mu_f, sd_f = X_real.mean(0), X_real.std(0) + 1e-9
    Z = (X_real - mu_f) / sd_f
    nn_own = {c: NearestNeighbors(n_neighbors=k_nn).fit(Z[y_real == c]) for c in np.unique(y_real)}
    nn_oth = {c: NearestNeighbors(n_neighbors=k_nn).fit(Z[y_real != c]) for c in np.unique(y_real)}
    gen = torch.Generator().manual_seed(int(rng.randint(2 ** 31)))
    model.eval()
    with torch.no_grad():
        mu, logvar = model.encoder(x, y, p)
    post_sd = (0.5 * logvar).exp()
    n, D = mu.shape
    centre, scale = mu.clone(), post_sd.clone()
    best_score = np.full(n, np.inf)
    best_sig = None
    yv = y.numpy()
    for _ in range(iters):
        eps = torch.randn((n, K, D), generator=gen)
        zc = (centre[:, None] + scale[:, None] * eps).reshape(n * K, D)
        with torch.no_grad():
            xh = model._physics(zc, y.repeat_interleave(K), p.repeat_interleave(K))[0]
        sig = observe_with(denormalize(xh.numpy(), stats), np.repeat(base, K, axis=0))
        F = feat_fn(sig)
        lab = np.repeat(yv, K)
        Fz = (F - mu_f) / sd_f
        s = np.empty(len(lab))
        for c in np.unique(lab):
            m = lab == c
            d_own, d_oth = nn_own[c].kneighbors(Fz[m])[0], nn_oth[c].kneighbors(Fz[m])[0]
            margin = d_oth.mean(1) - d_own.mean(1)  # > 0: on the own-class side
            s[m] = np.abs(margin - margin_target) + 2.0 * np.maximum(0.0, -margin) if score == 'margin' else -margin
        s = s.reshape(n, K)
        sig = sig.reshape(n, K, *sig.shape[1:])
        arg = s.argmin(1)
        cand = s[np.arange(n), arg]
        if best_sig is None:
            best_sig = sig[np.arange(n), arg].copy()
        better = cand < best_score
        best_score[better] = cand[better]
        best_sig[better] = sig[better, arg[better]]
        elite = np.argsort(s, axis=1)[:, :n_elite]
        ze = zc.reshape(n, K, D)[torch.arange(n)[:, None], torch.as_tensor(elite)]
        centre = ze.mean(1)
        scale = torch.minimum(torch.maximum(ze.std(1), 0.1 * post_sd), post_sd)
    return best_sig


def augment(policy: str, model, model_name: str, stats: dict, df_train: pd.DataFrame, cache: dict, n_add: int, seed: int,
            fold: int, stream: str, feat_fn, cv_mode: str, pool_mult: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Observed synthetic signals (n_add, 2, T) and labels for a physics-generator TSTR+ policy (see module doc)."""
    if policy in ('steer_margin', 'steer_knn'):
        dfp, labels = sample_parents(df_train, n_add, stream_rng(seed, fold, stream))
        sig = steer_samples(model, stats, dfp, feat_fn, cache['X_train_top'], cache['y_train'], stream_rng(seed, fold, 's' + stream),
                            score='margin' if policy == 'steer_margin' else 'knn')
        return sig, labels
    if policy == 'obsdr':
        import torch

        from core.tstr import generate_synthetic_signals
        null = model.null_part_idx if (cv_mode == 'loso' and getattr(model, '_cond_part', False)) else None
        sig, lab = generate_synthetic_signals(model, model_name, pool_mult * len(cache['y_train']), stats, torch.device('cpu'), seed + 1000, participant_idx=null)
        obs = sensor_observe(sig, lab, training_baselines(df_train), np.random.RandomState(seed + 17))
        keep = select_balanced(lab, n_add, score=realness(cache['X_train_top'], feat_fn(obs), seed), rng=stream_rng(seed, fold, stream))
        return obs[keep], lab[keep]
    raise ValueError(f'unknown aug_policy: {policy!r}')


# fidelity
def signal_mmd(X: np.ndarray, Y: np.ndarray) -> float:
    """Squared MMD between two sets of (n, 2, T) signals, as defined in the thesis (Sec. Distributional Fidelity):
    Gaussian kernel on the flattened two-channel trials (72-dim signal space), bandwidth sigma = median pairwise
    distance of the pooled sets (median heuristic), expectations replaced by empirical averages over all real,
    all synthetic and all cross pairs (biased V-statistic, >= 0)."""
    from scipy.spatial.distance import cdist
    A, B = np.asarray(X, dtype=float).reshape(len(X), -1), np.asarray(Y, dtype=float).reshape(len(Y), -1)
    d_aa, d_bb, d_ab = cdist(A, A, 'sqeuclidean'), cdist(B, B, 'sqeuclidean'), cdist(A, B, 'sqeuclidean')
    sigma2 = np.median(np.concatenate([d_aa[np.triu_indices(len(A), 1)], d_bb[np.triu_indices(len(B), 1)], d_ab.ravel()]))
    k = lambda d: np.exp(-d / (2 * sigma2))  # noqa: E731
    return float(k(d_aa).mean() + k(d_bb).mean() - 2 * k(d_ab).mean())


def mmd_report(real: np.ndarray, y_real: np.ndarray, synth: np.ndarray, y_synth: np.ndarray, real_ref: np.ndarray, with_test: bool = False) -> dict:
    """MMD of the synthetic set vs the real training trials (all classes: 'mmd'; mean over classes: 'mmd_class') and,
    as a reference level, of the held-out real test trials vs the same training trials ('mmd_real_ref').
    with_test (k-fold / TSTR+ only): also the synthetic set vs the held-out real test trials of the same participants
    ('mmd_test'; held-out fidelity that does not reward reproducing training trials, its floor is mmd_real_ref).
    All signals in the classifier's input space (after the run's preprocessing, e.g. baseline-corrected)."""
    per_class = [signal_mmd(real[y_real == c], synth[y_synth == c]) for c in np.unique(y_real) if (y_synth == c).sum() > 1]
    out = {'mmd': signal_mmd(real, synth), 'mmd_class': float(np.mean(per_class)), 'mmd_real_ref': signal_mmd(real, real_ref)}
    if with_test:
        out['mmd_test'] = signal_mmd(real_ref, synth)
    return out
