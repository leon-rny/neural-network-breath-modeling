"""Calibration of the PI-CVAE transport per region: advection-diffusion channel + humidity sensor (tau_s = 15 s) fitted
to the mean single-exhalation recording (dataset/sir/), temperature with the same transport and a bi-exponential sensor.

Usage: python -m core.calibrate   -> results/pinn/params_<region>.npy (A, D, v, t_shift of the humidity fit) + manifest
"""

import argparse
import json
import os
import re
from collections.abc import Callable
from functools import partial

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from core.utils import file_digest

SEED = 0
REGIONS = ['mouth', 'nose']
TAU_S_HUMIDITY = 15.0
D0 = 0.03  # source-detector separation (m)

# data loading
def load_sir(dataset_dir: str = 'dataset') -> pd.DataFrame:
    """Calibration recordings dataset/sir/sir_<region>_trial_<n>.dat, one row per trial (time in s)."""
    dataset = []
    folder = os.path.join(dataset_dir, 'sir')
    for fname in sorted(os.listdir(folder)):
        m = re.compile(r'^(sir)_(mouth|nose)_trial_(\d+)\.dat$').match(fname)
        if m is None:
            continue
        df = pd.read_csv(os.path.join(folder, fname))
        dataset.append({'filename': fname,
                        'class': m.group(1),
                        'region': m.group(2),
                        'trial_num': int(m.group(3)),
                        'time': df['Time'].values / 1000.0,
                        'humidity': df['Humidity'].values.astype(float),
                        'temperature': df['Temperature'].values.astype(float)})
    return pd.DataFrame(dataset)

def preprocess_cir(humidity: np.ndarray, temperature: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # baseline = mean of the first 6 samples
    return humidity - np.mean(humidity[:6]), temperature - np.mean(temperature[:6])

def common_time_grid(df: pd.DataFrame) -> np.ndarray:
    # uniform grid (median sampling interval) over the time span shared by all trials
    times = df['time'].values
    dts = [float(np.median(np.diff(t))) for t in times]
    dt = float(np.median(dts))
    t_start = max(float(t[0]) for t in times)
    t_end = min(float(t[-1]) for t in times)
    n = int(np.floor((t_end - t_start) / dt)) + 1
    return t_start + np.arange(n) * dt

def cir_stats(df: pd.DataFrame, t_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # mean and SD over the trials resampled onto t_grid: (humidity mean, temperature mean, humidity SD, temperature SD)
    humidity_all = np.stack([np.interp(t_grid, t, h) for t, h in zip(df['time'].values, df['humidity'].values)])
    temperature_all = np.stack([np.interp(t_grid, t, tp) for t, tp in zip(df['time'].values, df['temperature'].values)])
    return np.mean(humidity_all, axis=0), np.mean(temperature_all, axis=0), np.std(humidity_all, axis=0), np.std(temperature_all, axis=0)

# physical model
def advection_diffusion(time: np.ndarray, A: float, D: float, v: float, t_shift: float, d0: float = D0) -> np.ndarray:
    """Advection-diffusion channel response A / sqrt(4 pi D t) exp(-(d0 - v t)^2 / (4 D t)), t = time - t_shift > 0.

    :param time: time samples (s)
    :param D: diffusion coefficient (m^2/s)
    :param v: advection velocity (m/s)
    :param t_shift: onset delay (s)
    :return: channel response, zero for time <= t_shift
    """
    t = np.asarray(time, dtype=float)
    out = np.zeros_like(t)
    t_model = t - t_shift
    mask = t_model > 1e-9
    tt = t_model[mask]
    prefactor = A / np.sqrt(4.0 * np.pi * D * tt)
    exponent = -((d0 - v * tt) ** 2) / (4.0 * D * tt)
    out[mask] = prefactor * np.exp(exponent)
    return out

def sensor_kernel(t: np.ndarray, tau_s: float = 15.0) -> np.ndarray:
    return np.where(t > 0, (1.0 / tau_s) * np.exp(-t / tau_s), 0.0)

def sensor_kernel_biexp(t: np.ndarray, tau_fast: float, alpha: float, tau_slow: float) -> np.ndarray:
    fast = (alpha / tau_fast) * np.exp(-t / tau_fast)
    slow = ((1.0 - alpha) / tau_slow) * np.exp(-t / tau_slow)
    return np.where(t > 0, fast + slow, 0.0)

def system_impulse_response(time: np.ndarray, A: float, D: float, v: float, t_shift: float, sensor_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Channel response convolved with a sensor kernel.

    :param time: uniformly sampled time (s)
    :param sensor_fn: time grid -> sensor kernel values
    :return: system impulse response on time
    """
    t = np.asarray(time, dtype=float)
    dt = t[1] - t[0]  # assumes uniform sampling
    t_kernel = np.arange(0, t[-1] + dt, dt)
    cir = advection_diffusion(t, A, D, v, t_shift)
    s = sensor_fn(t_kernel)
    return np.convolve(cir, s, mode='full')[:len(t)] * dt

# fitting
def fit_sir(signal: dict, tau_s_humidity: float = TAU_S_HUMIDITY) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fit humidity (A, D, v, t_shift) and then temperature (A, tau_fast, alpha, tau_slow) with the humidity transport.

    :param signal: {'time', 'humidity', 'temperature'} of the mean calibration recording
    :return: humidity parameters, temperature parameters, their MSEs
    """
    time = signal['time']
    h, t_data = preprocess_cir(signal['humidity'], signal['temperature'])
    # humidity: advection-diffusion + single-exponential sensor
    sensor_h = partial(sensor_kernel, tau_s=tau_s_humidity)

    def obj_h(params: np.ndarray, t: np.ndarray, h: np.ndarray) -> float:
        A, D, v, ts = params
        pred = system_impulse_response(t, A, D, v, ts, sensor_h)
        return np.mean((pred - h) ** 2)

    bounds_h = [(1e-3, 1e3), (1e-7, 1e-2), (0.0, 1e-1), (0.0, 15.0)]  # A, D, v, t_shift
    res_h = differential_evolution(obj_h, bounds=bounds_h, args=(time, h), seed=SEED, tol=1e-8, maxiter=1000, polish=True)
    _A_h, D, v, t_shift = res_h.x
    # temperature: bi-exponential sensor, humidity transport
    def obj_t(params: np.ndarray, t: np.ndarray, td: np.ndarray) -> float:
        A, tau_fast, alpha, tau_slow = params
        sensor_t = partial(sensor_kernel_biexp, tau_fast=tau_fast, alpha=alpha, tau_slow=tau_slow)
        pred = system_impulse_response(t, A, D, v, t_shift, sensor_t)
        return np.mean((pred - td) ** 2)

    bounds_t = [(1e-3, 1e2), (0.1, 5.0), (0.2, 1.5), (30.0, 350.0)]  # A, tau_fast, alpha, tau_slow
    res_t = differential_evolution(obj_t, bounds=bounds_t, args=(time, t_data), seed=SEED, tol=1e-8, maxiter=1000, polish=True)
    return res_h.x, res_t.x, res_h.fun, res_t.fun

def _r2(signal: dict, params_h: np.ndarray, params_t: np.ndarray, tau_s_humidity: float = TAU_S_HUMIDITY) -> tuple[float, float]:
    # R^2 of the humidity and temperature fits (diagnostic only)
    time = signal['time']
    h_bc, t_bc = preprocess_cir(signal['humidity'], signal['temperature'])
    A_h, D, v, t_shift = params_h
    A_t, tau_fast, alpha, tau_slow = params_t
    sensor_h = partial(sensor_kernel, tau_s=tau_s_humidity)
    sensor_t = partial(sensor_kernel_biexp, tau_fast=tau_fast, alpha=alpha, tau_slow=tau_slow)
    sol_h = system_impulse_response(time, A_h, D, v, t_shift, sensor_h)
    sol_t = system_impulse_response(time, A_t, D, v, t_shift, sensor_t)

    def r2(y: np.ndarray, yhat: np.ndarray) -> float:
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot

    return r2(h_bc, sol_h), r2(t_bc, sol_t)

# driver
def fit_group(df_group: pd.DataFrame, label: str) -> np.ndarray:
    """Fit the trial mean of df_group, print a report and return the humidity parameters (A, D, v, t_shift)."""
    t_grid = common_time_grid(df_group)
    hum_mean, temp_mean, _, _ = cir_stats(df_group, t_grid)
    signal = {'time': t_grid, 'humidity': hum_mean, 'temperature': temp_mean}
    params_h, params_t, loss_h, loss_t = fit_sir(signal)
    r2_h, r2_t = _r2(signal, params_h, params_t)
    A_h, D, v, t_shift = params_h
    A_t, tau_fast, alpha, tau_slow = params_t
    print(f'\n=== {label} ({len(df_group)} trials, grid {t_grid[0]:.1f}-{t_grid[-1]:.1f}s '
          f'x{len(t_grid)} @ dt={t_grid[1] - t_grid[0]:.3f}s) ===')
    print('  Humidity    (A, D, v, t_shift) = '
          f'({A_h:.6f}, {D:.6e}, {v:.6f}, {t_shift:.6f})')
    print('  Temperature (A, tau_fast, alpha, tau_slow) = '
          f'({A_t:.6f}, {tau_fast:.6f}, {alpha:.6f}, {tau_slow:.6f})')
    print(f'  loss(humidity)={loss_h:.6e}  loss(temperature)={loss_t:.6e}')
    print(f'  R^2(humidity)={r2_h:.4f}  R^2(temperature)={r2_t:.4f}')
    return params_h

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # data
    p.add_argument('--dataset_dir', default='dataset')
    p.add_argument('--region', choices=REGIONS, default=None)
    return p.parse_args()

def main() -> None:
    args = parse_args()
    out_dir = os.path.join('results', 'pinn')
    os.makedirs(out_dir, exist_ok=True)
    df_sir = load_sir(args.dataset_dir)
    if df_sir.empty:
        raise SystemExit(f"No SIR files matched in {os.path.join(args.dataset_dir, 'sir')}")
    for region in [args.region] if args.region else REGIONS:
        # fit
        df_r = df_sir[df_sir['region'] == region]
        if df_r.empty:
            raise ValueError(f'No calibration trials for region={region}')
        params_h = fit_group(df_r, f'region={region}')
        # save parameters + manifest of the input files
        out_path = os.path.join(out_dir, f'params_{region}.npy')
        np.save(out_path, params_h)
        inputs = {filename: file_digest(os.path.join(args.dataset_dir, 'sir', filename)) for filename in df_r['filename']}
        with open(out_path + '.json', 'w') as manifest:
            json.dump({'calibration': 'sir', 'files': inputs, 'seed': SEED}, manifest, indent=2)
        print(f'  saved -> {out_path}')

if __name__ == '__main__':
    main()
