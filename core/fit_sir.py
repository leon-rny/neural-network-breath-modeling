import os
import re
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution


# constants
SEED = 0
REGIONS = ["mouth", "nose"]
CLASSES = ["bradypnea", "eupnea", "tachypnea"]
TAU_S_HUMIDITY = 15.0
D0 = 0.03  # source-detector separation (m), matches advection_diffusion default

# data loading
def load_sir(dataset_dir="dataset"):
    """Load all matched SIR .dat files into a DataFrame.

    Columns: filename, class, region, trial_num, time (s), humidity, temperature.
    Time is stored in milliseconds in the files and converted to seconds here.
    """
    dataset = []
    folder = os.path.join(dataset_dir, "sir")

    for fname in sorted(os.listdir(folder)):
        m = re.compile(r"^(sir_long|bradypnea|eupnea|tachypnea)_(mouth|nose)_trial_(\d+)\.dat$").match(fname)
        if m is None:
            continue

        df = pd.read_csv(os.path.join(folder, fname))
        dataset.append({"filename": fname,
                        "class": m.group(1),
                        "region": m.group(2),
                        "trial_num": int(m.group(3)),
                        "time": df["Time"].values / 1000.0,
                        "humidity": df["Humidity"].values.astype(float),
                        "temperature": df["Temperature"].values.astype(float)})

    return pd.DataFrame(dataset)

def preprocess_cir(humidity, temperature):
    """Baseline-correct by subtracting the mean of the first 6 samples."""
    baseline_h = np.mean(humidity[:6])
    baseline_t = np.mean(temperature[:6])
    return humidity - baseline_h, temperature - baseline_t

def common_time_grid(df):
    """Build a uniform common time grid shared across all trials in `df`.

    Trial lengths/durations differ slightly, so we use the median per-trial dt
    and the minimum common end time (so every trial fully covers the grid),
    starting at the maximum start time (typically 0).
    """
    times = df["time"].values
    dts = [float(np.median(np.diff(t))) for t in times]
    dt = float(np.median(dts))
    t_start = max(float(t[0]) for t in times)
    t_end = min(float(t[-1]) for t in times)
    n = int(np.floor((t_end - t_start) / dt)) + 1
    return t_start + np.arange(n) * dt

def cir_stats(df, t_grid):
    """Resample each trial onto `t_grid` (via np.interp) then average over trials.

    Returns (mean_humidity, mean_temperature, std_humidity, std_temperature),
    each aligned to `t_grid`.
    """
    humidity_all = np.stack([np.interp(t_grid, t, h)
                             for t, h in zip(df["time"].values, df["humidity"].values)])
    temperature_all = np.stack([np.interp(t_grid, t, tp)
                                for t, tp in zip(df["time"].values, df["temperature"].values)])

    return (np.mean(humidity_all, axis=0), np.mean(temperature_all, axis=0),
            np.std(humidity_all, axis=0), np.std(temperature_all, axis=0))

# physical model
def advection_diffusion(time, A, D, v, t_shift, d0=D0):
    """Advection-diffusion channel response at a fixed source-detector distance.

    Evaluates A / sqrt(4 pi D (t - t_shift)) * exp(-(d0 - v (t - t_shift))^2
    / (4 D (t - t_shift))), zeroed for t <= t_shift.

    :param time: time samples (s), shape (N,).
    :param A: amplitude scale.
    :param D: diffusion coefficient (m^2/s).
    :param v: advection velocity (m/s).
    :param t_shift: onset delay (s).
    :param d0: source-detector separation (m).
    :return: channel response, shape (N,).
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

def sensor_kernel(t, tau_s=15.0):
    """Single-exponential sensor-lag kernel (1/tau_s) exp(-t/tau_s), zero for t <= 0.

    :param t: time samples (s), shape (N,).
    :param tau_s: sensor time constant (s).
    :return: kernel values, shape (N,).
    """
    return np.where(t > 0, (1.0 / tau_s) * np.exp(-t / tau_s), 0.0)

def sensor_kernel_biexp(t, tau_fast, alpha, tau_slow):
    """Bi-exponential sensor-lag kernel: alpha fast plus (1 - alpha) slow, zero for t <= 0.

    :param t: time samples (s), shape (N,).
    :param tau_fast: fast time constant (s).
    :param alpha: weight of the fast component in [0, 1].
    :param tau_slow: slow time constant (s).
    :return: kernel values, shape (N,).
    """
    fast = (alpha / tau_fast) * np.exp(-t / tau_fast)
    slow = ((1.0 - alpha) / tau_slow) * np.exp(-t / tau_slow)
    return np.where(t > 0, fast + slow, 0.0)

def system_impulse_response(time, A, D, v, t_shift, sensor_fn):
    """Channel response convolved with a sensor kernel, sampled on `time`.

    Convolves advection_diffusion(time, A, D, v, t_shift) with sensor_fn
    evaluated on a uniform kernel grid, then truncates to the input length.

    :param time: uniformly sampled time (s), shape (N,).
    :param A: channel amplitude scale.
    :param D: diffusion coefficient (m^2/s).
    :param v: advection velocity (m/s).
    :param t_shift: onset delay (s).
    :param sensor_fn: callable mapping a time grid to sensor-kernel values.
    :return: system impulse response, shape (N,).
    """
    t = np.asarray(time, dtype=float)
    dt = t[1] - t[0]  # assume uniform sampling
    t_kernel = np.arange(0, t[-1] + dt, dt)

    cir = advection_diffusion(t, A, D, v, t_shift)
    s = sensor_fn(t_kernel)

    return np.convolve(cir, s, mode="full")[:len(t)] * dt

# fitting
def fit_sir(signal, tau_s_humidity=TAU_S_HUMIDITY):
    """Fit humidity (A, D, v, t_shift) and temperature (A, tau_fast, alpha, tau_slow).

    Returns (params_humidity, params_temperature, loss_humidity, loss_temperature),
    where the losses are the final differential_evolution objective (MSE) values.
    """
    time = signal["time"]
    h, t_data = preprocess_cir(signal["humidity"], signal["temperature"])

    # humidity: advection-diffusion + single-exponential sensor
    sensor_h = lambda tk: sensor_kernel(tk, tau_s_humidity)

    def obj_h(params, t, h):
        """MSE between the humidity SIR for (A, D, v, t_shift) and measured h."""
        A, D, v, ts = params
        pred = system_impulse_response(t, A, D, v, ts, sensor_h)
        return np.mean((pred - h) ** 2)

    bounds_h = [(1e-3, 1e3), (1e-7, 1e-2), (0.0, 1e-1), (0.0, 15.0)]  # A, D, v, t_shift
    res_h = differential_evolution(obj_h, bounds=bounds_h, args=(time, h),
                                   seed=SEED, tol=1e-8, maxiter=1000, polish=True)
    A_h, D, v, t_shift = res_h.x

    # temperature: bi-exponential sensor, reusing humidity transport
    def obj_t(params, t, td):
        """MSE between the temperature SIR (bi-exp sensor, fixed transport) and measured td."""
        A, tau_fast, alpha, tau_slow = params
        sensor_t = lambda tk: sensor_kernel_biexp(tk, tau_fast, alpha, tau_slow)
        pred = system_impulse_response(t, A, D, v, t_shift, sensor_t)
        return np.mean((pred - td) ** 2)

    bounds_t = [(1e-3, 1e2), (0.1, 5.0), (0.2, 1.5), (30.0, 350.0)]  # A, tau_fast, alpha, tau_slow
    res_t = differential_evolution(obj_t, bounds=bounds_t, args=(time, t_data), seed=SEED, tol=1e-8, maxiter=1000, polish=True)

    return res_h.x, res_t.x, res_h.fun, res_t.fun

def _r2(signal, params_h, params_t, tau_s_humidity=TAU_S_HUMIDITY):
    """Compute R^2 of the humidity and temperature fits (diagnostic only)."""
    time = signal["time"]
    h_bc, t_bc = preprocess_cir(signal["humidity"], signal["temperature"])
    A_h, D, v, t_shift = params_h
    A_t, tau_fast, alpha, tau_slow = params_t

    sensor_h = lambda tk: sensor_kernel(tk, tau_s=tau_s_humidity)
    sensor_t = lambda tk: sensor_kernel_biexp(tk, tau_fast, alpha, tau_slow)

    sol_h = system_impulse_response(time, A_h, D, v, t_shift, sensor_h)
    sol_t = system_impulse_response(time, A_t, D, v, t_shift, sensor_t)

    def r2(y, yhat):
        """Coefficient of determination of yhat against y."""
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot

    return r2(h_bc, sol_h), r2(t_bc, sol_t)

# driver
def fit_group(df_group, label):
    """Average over trials and fit; print a report; return the humidity 4-vector."""
    t_grid = common_time_grid(df_group)
    hum_mean, temp_mean, _, _ = cir_stats(df_group, t_grid)
    signal = dict(time=t_grid, humidity=hum_mean, temperature=temp_mean)

    params_h, params_t, loss_h, loss_t = fit_sir(signal)
    r2_h, r2_t = _r2(signal, params_h, params_t)

    A_h, D, v, t_shift = params_h
    A_t, tau_fast, alpha, tau_slow = params_t

    print(f"\n=== {label} ({len(df_group)} trials, grid {t_grid[0]:.1f}-{t_grid[-1]:.1f}s "
          f"x{len(t_grid)} @ dt={t_grid[1] - t_grid[0]:.3f}s) ===")
    print("  Humidity    (A, D, v, t_shift) = "
          f"({A_h:.6f}, {D:.6e}, {v:.6f}, {t_shift:.6f})")
    print("  Temperature (A, tau_fast, alpha, tau_slow) = "
          f"({A_t:.6f}, {tau_fast:.6f}, {alpha:.6f}, {tau_slow:.6f})")
    print(f"  loss(humidity)={loss_h:.6e}  loss(temperature)={loss_t:.6e}")
    print(f"  R^2(humidity)={r2_h:.4f}  R^2(temperature)={r2_t:.4f}")

    return params_h

def main():
    """Fit the SIR model per region (and optionally per class) and save humidity params."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_dir", default="dataset")
    parser.add_argument("--per_class", action="store_true")
    parser.add_argument("--region", choices=REGIONS, default=None)
    args = parser.parse_args()

    out_dir = os.path.join("results", "pinn")
    os.makedirs(out_dir, exist_ok=True)

    df_sir = load_sir(args.dataset_dir)
    if df_sir.empty:
        raise SystemExit(f"No SIR files matched in {os.path.join(args.dataset_dir, 'sir')}")

    regions = [args.region] if args.region else REGIONS

    # generic fit: sir_long step-input, per region
    df_long = df_sir[df_sir["class"] == "sir_long"]
    for region in regions:
        df_r = df_long[df_long["region"] == region]
        if df_r.empty:
            print(f"[warn] no sir_long trials for region={region}; skipping.")
            continue
        params_h = fit_group(df_r, f"GENERIC region={region}")
        out_path = os.path.join(out_dir, f"params_{region}.npy")
        np.save(out_path, params_h)
        print(f"  saved -> {out_path}")

    # optional per-class fits
    if args.per_class:
        for region in regions:
            for cls in CLASSES:
                df_rc = df_sir[(df_sir["region"] == region) & (df_sir["class"] == cls)]
                if df_rc.empty:
                    print(f"[warn] no trials for region={region}, class={cls}; skipping.")
                    continue
                params_h = fit_group(df_rc, f"PER-CLASS region={region} class={cls}")
                out_path = os.path.join(out_dir, f"params_{region}_{cls}.npy")
                np.save(out_path, params_h)
                print(f"  saved -> {out_path}")

if __name__ == "__main__":
    # main()

    import matplotlib.pyplot as plt
    TAU_S = 15.0
    REGION = ["mouth", "nose"]
    T_MAX = 300.0  # truncate sir_long to the ~70 s short-sir window before fitting (None = full length)

    df_sir = load_sir(dataset_dir='../dataset/')
    df_long = df_sir[df_sir["class"] == "sir_long"]  # generic step-input trials

    # fitting (mirrors fit_sir.fit_group, with plotting); separate dict so mean_then_fit is untouched
    module_fit = {}
    fig, ax = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    ax_twin = [ax[0].twinx(), ax[1].twinx()]
    ax_twin[1].sharey(ax_twin[0])
    ax_twin[0].tick_params(right=False, labelright=False)
    for col, region in enumerate(REGION):
        df_region = df_long[df_long["region"] == region]

        # common time grid + interpolated mean over trials
        t_grid = common_time_grid(df_region)
        humidity_mean, temperature_mean, _, _ = cir_stats(df_region, t_grid)

        # shorten the recording before fitting
        if T_MAX is not None:
            keep = t_grid <= T_MAX
            t_grid = t_grid[keep]
            humidity_mean = humidity_mean[keep]
            temperature_mean = temperature_mean[keep]

        signal = dict(time=t_grid, humidity=humidity_mean, temperature=temperature_mean)

        # fit model to mean signal
        params_humidity, params_temperature, loss_h, loss_t = fit_sir(signal)
        A_h, D, v, t_shift = params_humidity
        A_t, tau_fast, alpha, tau_slow = params_temperature
        r2_humidity, r2_temperature = _r2(signal, params_humidity, params_temperature)

        # reconstruct fitted curves
        sensor_h = lambda tk: sensor_kernel(tk, tau_s=TAU_S)
        sensor_t = lambda tk: sensor_kernel_biexp(tk, tau_fast, alpha, tau_slow)
        solution_humidity = system_impulse_response(t_grid, A_h, D, v, t_shift, sensor_h)
        solution_temperature = system_impulse_response(t_grid, A_t, D, v, t_shift, sensor_t)
        humidity_baseline_corrected, temperature_baseline_corrected = preprocess_cir(humidity_mean, temperature_mean)

        module_fit[region] = {"humidity_params": params_humidity,
                            "temperature_params": params_temperature,
                            "r2_humidity": r2_humidity,
                            "r2_temperature": r2_temperature}

        # humidity
        ax[col].scatter(t_grid, humidity_baseline_corrected, marker="o", color="tab:blue", alpha=0.5)
        ax[col].plot(t_grid, solution_humidity, label=rf"$R^2_{{humidity}}=${r2_humidity:.4f}", color="tab:blue")
        ax[col].set_xlabel("Time in s")
        ax[col].tick_params(axis="y", labelcolor="tab:blue")
        ax[col].grid()

        # temperature
        ax_twin[col].scatter(t_grid, temperature_baseline_corrected, marker="o", color="tab:orange", alpha=0.5)
        ax_twin[col].plot(t_grid, solution_temperature, label=rf"$R^2_{{temperature}}=${r2_temperature:.4f}", color="tab:orange")
        ax_twin[col].tick_params(axis="y", labelcolor="tab:orange")

        ax[0].set_ylabel("Humidity in %", color="tab:blue")
        ax_twin[1].set_ylabel("Temperature in °C", color="tab:orange")
        h1, l1 = ax[col].get_legend_handles_labels()
        h2, l2 = ax_twin[col].get_legend_handles_labels()
        ax[col].legend(h1 + h2, l1 + l2, loc="upper right")
        window = f"<= {T_MAX:.0f}s" if T_MAX is not None else "full"
        ax[col].set_title(f"{region.capitalize()} (sir_long {window}, from fit_sir.py)")

    plt.tight_layout()
    plt.show()

    # print
    header = f"{'Parameter':<16}{'Mouth':>14}{'Nose':>14}"
    sep    = "─" * len(header)
    print(sep)
    print(header)
    print(sep)
    print("Humidity")
    rows_h = [("  A",         "{:.4f}",   "humidity_params", 0),
            ("  D (m²/s)",  "{:.2e}",   "humidity_params", 1),
            ("  v (m/s)",   "{:.4f}",   "humidity_params", 2),
            ("  t_shift (s)","{:.4f}",  "humidity_params", 3)]
    for label, fmt, key, idx in rows_h:
        m = fmt.format(module_fit["mouth"][key][idx])
        n = fmt.format(module_fit["nose"][key][idx])
        print(f"{label:<16}{m:>14}{n:>14}")

    print("Temperature")
    rows_t = [("  A",          "{:.4f}", "temperature_params", 0),
            ("  tau_fast (s)","{:.4f}","temperature_params", 1),
            ("  alpha",      "{:.4f}", "temperature_params", 2),
            ("  tau_slow (s)","{:.4f}","temperature_params", 3)]
    for label, fmt, key, idx in rows_t:
        m = fmt.format(module_fit["mouth"][key][idx])
        n = fmt.format(module_fit["nose"][key][idx])
        print(f"{label:<16}{m:>14}{n:>14}")
    print(sep)