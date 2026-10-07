"""Per-trial "combo" features for the LightGBM classifier (core.lgbm): hand-crafted breathing features + spectral /
wavelet features + catch22, computed on the baseline-corrected trial (first 5 samples subtracted per channel).

  breath    (86) amplitude statistics, rise / fall slopes and their timing, rise times, duty cycle, area, peaks and
                 inter-peak intervals, high-pass oscillation, dominant frequency, spectral centroid / entropy,
                 autocorrelation per channel + humidity–temperature coupling (correlation, lag, range ratio, peak delay)
  spectral  (88) log power and relative FFT power spectrum of the detrended trial, relative Welch band powers
                 (nperseg 16), db2 wavelet level energies / statistics per channel
  catch22   (48) catch22 + mean / SD (catch24) per channel (pycatch22)
Per-trial extraction has no fitted state; imputation, scaling and selection happen inside the training folds (core.lgbm).
"""
import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy import stats

DT = 2.004  # s per sample
CH = ('H', 'T')


def _breath_channel(x: np.ndarray) -> dict:
    t = np.arange(len(x)) * DT
    rng = x.max() - x.min() + 1e-9
    d = np.diff(x)
    f = {'max': x.max(), 'min': x.min(), 'range': rng, 'mean': x.mean(), 'std': x.std(), 'median': np.median(x),
         'iqr': stats.iqr(x), 'skew': stats.skew(x), 'kurt': stats.kurtosis(x), 'energy': np.mean(x ** 2),
         'mad_diff': np.mean(np.abs(d)), 'std_diff': d.std(), 'max_rise': d.max(), 'max_fall': d.min(),
         't_peak': t[np.argmax(x)], 't_trough': t[np.argmin(x)], 't_max_rise': t[np.argmax(d)], 't_max_fall': t[np.argmin(d)]}
    for q in (0.1, 0.5, 0.9):  # rise time to q x peak (relative to the start level 0)
        above = np.nonzero(x >= q * x.max())[0] if x.max() > 0 else np.array([])
        f[f't_rise{int(q * 100)}'] = t[above[0]] if len(above) else t[-1]
    f['duty50'] = np.mean(x >= 0.5 * x.max()) if x.max() > 0 else 0.0
    f['end_rel'] = x[-1] / rng
    f['auc'] = x.sum() * DT
    f['centroid_t'] = (t * np.abs(x)).sum() / (np.abs(x).sum() + 1e-9)
    hp = x - np.convolve(x, np.ones(5) / 5, mode='same')  # moving-average high-pass
    peaks, props = sps.find_peaks(x, prominence=0.05 * rng)
    f['n_peaks'] = len(peaks)
    f['peak_prom_mean'] = props['prominences'].mean() if len(peaks) else 0.0
    ipi = np.diff(peaks) * DT
    f['ipi_mean'] = ipi.mean() if len(ipi) else 0.0
    f['ipi_std'] = ipi.std() if len(ipi) > 1 else 0.0
    f['hp_std'] = hp.std()
    f['hp_zero_cross'] = np.sum(np.diff(np.sign(hp - hp.mean())) != 0)
    spec = np.abs(np.fft.rfft(x - np.linspace(x[0], x[-1], len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), DT)
    p = spec[1:] / (spec[1:].sum() + 1e-12)
    f['dom_freq'] = freqs[1:][np.argmax(spec[1:])]
    f['dom_power'] = p.max()
    f['spec_centroid'] = (freqs[1:] * p).sum()
    f['spec_entropy'] = -(p * np.log(p + 1e-12)).sum()
    xc = x - x.mean()
    ac = np.correlate(xc, xc, mode='full')[len(x) - 1:] / (np.dot(xc, xc) + 1e-12)
    for lag in (1, 2, 3, 5, 8):
        f[f'ac{lag}'] = ac[lag]
    zc = np.nonzero(ac < 0)[0]
    f['ac_first_zero'] = zc[0] if len(zc) else len(x)
    return f


def breath_features(S: np.ndarray) -> pd.DataFrame:
    rows = []
    for s in S:
        f = {}
        for c, x in zip(CH, s):
            f.update({f'{c}_{k}': v for k, v in _breath_channel(x).items()})
        h, tt = s
        f['HT_corr'] = np.corrcoef(h, tt)[0, 1] if h.std() > 0 and tt.std() > 0 else 0.0
        cc = np.correlate(h - h.mean(), tt - tt.mean(), mode='full')
        f['HT_lag'] = (np.argmax(cc) - (len(h) - 1)) * DT
        f['HT_range_ratio'] = (h.max() - h.min()) / (tt.max() - tt.min() + 1e-9)
        f['HT_peak_dt'] = (np.argmax(h) - np.argmax(tt)) * DT
        rows.append(f)
    return pd.DataFrame(rows)


def spectral_features(S: np.ndarray) -> pd.DataFrame:
    import pywt
    rows = []
    level = pywt.dwt_max_level(S.shape[2], 'db2')
    for s in S:
        f = {}
        for c, x in zip(CH, s):
            xd = x - np.linspace(x[0], x[-1], len(x))
            mag = np.abs(np.fft.rfft(xd))
            tot = (mag ** 2).sum() + 1e-12
            f[f'{c}_logpower'] = np.log(tot)
            f.update({f'{c}_fft{i}': m ** 2 / tot for i, m in enumerate(mag[1:], start=1)})
            _, pw = sps.welch(x - x.mean(), fs=1 / DT, nperseg=16)
            pw = pw / (pw.sum() + 1e-12)
            f.update({f'{c}_welch{i}': v for i, v in enumerate(pw)})
            for li, coef in enumerate(pywt.wavedec(x, 'db2', level=level)):
                f[f'{c}_wl{li}_energy'] = np.sum(coef ** 2)
                f[f'{c}_wl{li}_absmean'] = np.mean(np.abs(coef))
                f[f'{c}_wl{li}_std'] = coef.std()
                f[f'{c}_wl{li}_absmax'] = np.max(np.abs(coef))
        rows.append(f)
    return pd.DataFrame(rows)


def catch22_features(S: np.ndarray) -> pd.DataFrame:
    import pycatch22
    rows = []
    for s in S:
        f = {}
        for c, x in zip(CH, s):
            r = pycatch22.catch22_all(list(map(float, x)), catch24=True)
            f.update({f'{c}_{n}': v for n, v in zip(r['names'], r['values'])})
        rows.append(f)
    return pd.DataFrame(rows)


def combo_features(signals: np.ndarray) -> pd.DataFrame:
    """(n, 222) combo features of (n, 2, T) absolute trials (baseline-corrected here); +/-inf -> NaN (imputed in fold)."""
    S = np.asarray(signals, dtype=float)
    S = S - S[:, :, :5].mean(2, keepdims=True)
    X = pd.concat([breath_features(S).add_prefix('br_'), spectral_features(S).add_prefix('sp_'), catch22_features(S).add_prefix('ca_')], axis=1)
    return X.replace([np.inf, -np.inf], np.nan)
