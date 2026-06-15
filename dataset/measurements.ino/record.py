import os
import re
import time
import threading
from collections import deque

import numpy as np
import serial
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

PORT = '/dev/cu.usbmodem21101'
BAUD = 9600
DATASET_ROOT = './dataset/new_measurements'

MONITOR_WINDOW = 70.0
DEFAULT_DURATION = 70
# impulse-response settle: the existing 70-76s SIR trials truncate the decay (temperature,
# esp. nose, is still ~25-43% of peak at 76s), so capture the full return-to-baseline (~4 min).
IMPULSE_DURATION = 240
BASELINE_SECONDS = 10
IMPULSE_SECONDS = 2
# the class-dependent SIR is elicited by ONE paced breath at the class rate (window = 60/bpm,
# from BREATH_RATES below): a slow deep cycle for bradypnea ... a short sharp one for tachypnea,
# so the single SIR breath carries the class-characteristic emission shape.
IMPULSE_CUE = {'bradypnea': 'ONE SLOW, DEEP BREATH',
               'eupnea': 'ONE NORMAL BREATH',
               'tachypnea': 'ONE SHORT, SHARP BREATH'}
IMPULSE_SETTLE_TAU = 50.0  # realistic decay constant for the settle cue (matches the long tail)
CAPTURE_MARGIN_S = 4
MEAS_TYPES = ['breathing', 'impulse']

# baseline-stability
BASE_WINDOW = 20     # number of recent samples to judge stability
TEMP_TOL = 0.1       # max std of consecutive-sample changes in °C
HUM_TOL = 0.8        # max std of consecutive-sample changes in %RH
TEMP_DRIFT_TOL = 0.2 # max |last-third mean − first-third mean| in °C
HUM_DRIFT_TOL = 1.0  # max |last-third mean − first-third mean| in %RH

# SIR live quality-control (run on save): humidity is the reliable channel, so QC keys off it.
QC_BASELINE_HUM_STD = 0.3   # %RH: pre-breath baseline (first ~10s) must be flat
QC_MIN_HUM_PEAK = 5.0       # %RH: the single breath must clearly register
QC_RETURN_FRAC = 0.10       # humidity must decay back below 10% of peak by the end of the window
QC_OUTLIER_Z = 3.5          # robust (MAD) z-score of this trial's peak vs the cell's other trials

CLASSES = ['bradypnea', 'eupnea', 'tachypnea']
REGIONS = ['mouth', 'nose']
# pacing guide: typical breathing rate per class, in breaths per minute
BREATH_RATES = {'bradypnea': 10, 'eupnea': 15, 'tachypnea': 20}
PARTICIPANTS = ['b', 'c', 'd']
PARTICIPANT_PREFIX = { 'b': 'b_', 'c': 'c_', 'd': 'd_'}

# serial reader
class SerialReader(threading.Thread):
    def __init__(self, port, baud):
        super().__init__(daemon=True)
        self.ser = serial.Serial(port, baud, timeout=1)
        self.lock = threading.Lock()
        self.monitor = deque()
        self.recording = False
        self.record_buffer = []
        self.stop_event = threading.Event()

    def run(self):
        self.ser.reset_input_buffer()
        while not self.stop_event.is_set():
            try:
                raw = self.ser.readline().decode('utf-8', errors='ignore').strip()
            except Exception:
                continue
            if not raw:
                continue
            parts = raw.split(',')
            if len(parts) != 3:
                continue
            try:
                ardu_t = float(parts[0])
                hum = float(parts[1])
                temp = float(parts[2])
            except ValueError:
                continue

            host_t = time.time()
            with self.lock:
                self.monitor.append((host_t, hum, temp))
                cutoff = host_t - MONITOR_WINDOW
                while self.monitor and self.monitor[0][0] < cutoff:
                    self.monitor.popleft()
                if self.recording:
                    self.record_buffer.append((ardu_t, hum, temp))

    # recording control
    def start_recording(self):
        with self.lock:
            self.record_buffer = []
            self.recording = True

    def stop_recording(self):
        with self.lock:
            self.recording = False
            return list(self.record_buffer)

    def get_monitor(self):
        with self.lock:
            return list(self.monitor)

    def close(self):
        self.stop_event.set()
        time.sleep(0.2)
        try:
            self.ser.close()
        except Exception:
            pass

# saving
def _next_index(folder, pat):
    """Next trial number = max existing match + 1, so discards never collide."""
    existing = [int(m.group(1)) for f in os.listdir(folder) if (m := pat.match(f))]
    return max(existing) + 1 if existing else 1

def _trim_to_duration(rows, duration_s):
    """Keep samples up to and including the first that reaches duration_s, so the
    saved span is always >= duration_s (Arduino time is in ms)."""
    if not rows:
        return rows
    t0 = rows[0][0]
    cutoff = duration_s * 1000.0
    for i, r in enumerate(rows):
        if r[0] - t0 >= cutoff:
            return rows[:i + 1]
    return rows  # never reached the target (e.g. dropped samples) -> keep all

def _write_dat(path, rows):
    # Arduino sends raw millis(); rebase so each trial starts at t=0 (still in ms).
    t0 = rows[0][0] if rows else 0.0
    with open(path, 'w') as f:
        f.write('Time,Humidity,Temperature\n')
        for t, h, temp in rows:
            f.write(f'{t - t0},{h},{temp}\n')

def save_measurement(rows, breath_class, region, participant):
    folder = os.path.join(DATASET_ROOT, breath_class)
    os.makedirs(folder, exist_ok=True)
    prefix = PARTICIPANT_PREFIX[participant]
    pat = re.compile(rf'^{re.escape(prefix)}{region}_trial_(\d+)\.dat$')
    fname = f'{prefix}{region}_trial_{_next_index(folder, pat)}.dat'
    path = os.path.join(folder, fname)
    _write_dat(path, rows)
    return path

def save_sir(rows, region, breath_class, participant):
    """Class-dependent impulse-response trial ->
    dataset/sir/{participant}_{class}_{region}_trial_N.dat. Class is encoded so the per-class
    emission profile is recoverable; region keeps mouth/nose separate (do not pool). Mirrors the
    breathing-trial naming (participant prefix + region + trial), with the class added."""
    folder = os.path.join(DATASET_ROOT, 'sir')
    os.makedirs(folder, exist_ok=True)
    prefix = PARTICIPANT_PREFIX[participant]
    pat = re.compile(rf'^{re.escape(prefix)}{re.escape(breath_class)}_{re.escape(region)}_trial_(\d+)\.dat$')
    fname = f'{prefix}{breath_class}_{region}_trial_{_next_index(folder, pat)}.dat'
    path = os.path.join(folder, fname)
    _write_dat(path, rows)
    return path

# SIR live quality control
def _sir_metrics(rows, nb=5):
    """Baseline-corrected humidity peak, pre-breath baseline std, and end-fraction for a SIR trial.
    Humidity is the reliable channel (temperature is too weak/quantized to gate on)."""
    H = np.array([r[1] for r in rows], dtype=float)
    base = H[:nb].mean()
    corr = H - base
    pk = float(corr.max())
    base_std = float(H[:nb].std())
    end_frac = float(corr[-3:].mean() / pk) if pk > 1e-6 else float('nan')
    return pk, base_std, end_frac

def _cell_peaks(folder, prefix, breath_class, region, exclude_path):
    """Humidity peaks of the already-saved SIR trials in this (participant, class, region) cell."""
    pat = re.compile(rf'^{re.escape(prefix)}{re.escape(breath_class)}_{re.escape(region)}_trial_\d+\.dat$')
    peaks = []
    for f in sorted(os.listdir(folder)):
        full = os.path.join(folder, f)
        if not pat.match(f) or full == exclude_path:
            continue
        try:
            d = np.genfromtxt(full, delimiter=',', skip_header=1)
            peaks.append(float((d[:, 1] - d[:5, 1].mean()).max()))
        except Exception:
            continue
    return peaks

def qc_sir(rows, folder, prefix, breath_class, region, saved_path):
    """Flag a just-saved SIR trial: unsettled baseline, weak/failed breath, no return to baseline,
    or an amplitude outlier vs the rest of the cell (warm-up). Returns a list of warning strings."""
    warns = []
    pk, base_std, end_frac = _sir_metrics(rows)
    if base_std > QC_BASELINE_HUM_STD:
        warns.append(f'unsettled baseline (Hstd {base_std:.2f}%RH) — breath may have started early')
    if pk < QC_MIN_HUM_PEAK:
        warns.append(f'weak breath (peak {pk:.1f}%RH) — did the exhale reach the sensor?')
    elif np.isfinite(end_frac) and end_frac > QC_RETURN_FRAC:
        warns.append(f'no return to baseline (end {100 * end_frac:.0f}% of peak) — extend duration / 2nd breath?')
    peers = _cell_peaks(folder, prefix, breath_class, region, saved_path)
    if len(peers) >= 3 and pk >= QC_MIN_HUM_PEAK:
        med = float(np.median(peers))
        mad = float(np.median(np.abs(np.array(peers) - med))) + 1e-9
        z = 0.6745 * (pk - med) / mad
        if abs(z) > QC_OUTLIER_Z:
            warns.append(f'amplitude outlier (peak {pk:.1f} vs cell median {med:.1f}%RH) — likely warm-up')
    return warns

# gui
class CampaignGUI:
    def __init__(self, root, reader):
        self.root = root
        self.reader = reader
        self.last_saved_path = None
        self.measuring = False
        self._countdown_id = None
        self._measure_start = None
        self._measure_total = None
        self._impulse_win = None
        self._sir_warnings = []

        root.title('Breath Measurement Campaign')

        # live monitor
        self.mon_fig = Figure(figsize=(8, 3), dpi=100)
        self.ax_hum = self.mon_fig.add_subplot(2, 1, 1)
        self.ax_temp = self.mon_fig.add_subplot(2, 1, 2, sharex=self.ax_hum)
        self.mon_fig.tight_layout()
        self.mon_canvas = FigureCanvasTkAgg(self.mon_fig, master=root)
        self.mon_canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew',padx=6, pady=(6, 0))

        # control bar
        ctrl = ttk.Frame(root, padding=8)
        ctrl.grid(row=1, column=0, sticky='ew')

        ttk.Label(ctrl, text='Type:').grid(row=0, column=0, sticky='w')
        self.type_var = tk.StringVar(value=MEAS_TYPES[0])
        ttk.Combobox(ctrl, textvariable=self.type_var, values=MEAS_TYPES,
                     state='readonly', width=10).grid(row=0, column=1, padx=(2, 14))
        self.type_var.trace_add('write', self._on_type_change)

        ttk.Label(ctrl, text='Participant:').grid(row=0, column=2, sticky='w')
        self.part_var = tk.StringVar(value=PARTICIPANTS[0])
        ttk.Combobox(ctrl, textvariable=self.part_var, values=PARTICIPANTS,
                     state='readonly', width=4).grid(row=0, column=3, padx=(2, 14))

        ttk.Label(ctrl, text='Class:').grid(row=0, column=4, sticky='w')
        self.class_var = tk.StringVar(value=CLASSES[1])
        self.class_menu = ttk.Combobox(ctrl, textvariable=self.class_var, values=CLASSES,
                                       state='readonly', width=12)
        self.class_menu.grid(row=0, column=5, padx=(2, 14))

        ttk.Label(ctrl, text='Region:').grid(row=0, column=6, sticky='w')
        self.region_var = tk.StringVar(value=REGIONS[0])
        ttk.Combobox(ctrl, textvariable=self.region_var, values=REGIONS,
                     state='readonly', width=6).grid(row=0, column=7, padx=(2, 14))

        ttk.Label(ctrl, text='Duration in s:').grid(row=0, column=8, sticky='w')
        self.dur_var = tk.StringVar(value=str(DEFAULT_DURATION))
        ttk.Entry(ctrl, textvariable=self.dur_var, width=5).grid(row=0, column=9, padx=(2, 14))

        self.start_btn = ttk.Button(ctrl, text='Start measurement', command=self.start_measurement, state='disabled')
        self.start_btn.grid(row=0, column=10, padx=(0, 8))

        self.abort_btn = ttk.Button(ctrl, text='Abort', command=self.abort_measurement, state='disabled')
        self.abort_btn.grid(row=0, column=11, padx=(0, 8))

        self.discard_btn = ttk.Button(ctrl, text='Discard last',command=self.discard_last, state='disabled')
        self.discard_btn.grid(row=0, column=12, padx=(0, 14))

        self.baseline_lbl = ttk.Label(ctrl, text='Baseline: --', width=22)
        self.baseline_lbl.grid(row=0, column=13, sticky='w')

        self.met_canvas = tk.Canvas(root, height=160, highlightthickness=0, bg='white')
        self.met_canvas.grid(row=2, column=0, sticky='ew', padx=6, pady=(6, 0))
        self._met_phase = 0.0
        self._met_last_t = time.time()

        self.status = ttk.Label(root, text='Monitoring...', padding=(8, 0, 8, 6))
        self.status.grid(row=3, column=0, sticky='w')

        # result plot
        self.res_fig = Figure(figsize=(8, 3), dpi=100)
        self.res_hum = self.res_fig.add_subplot(1, 2, 1)
        self.res_temp = self.res_fig.add_subplot(1, 2, 2)
        self.res_fig.tight_layout()
        self.res_canvas = FigureCanvasTkAgg(self.res_fig, master=root)
        self.res_canvas.get_tk_widget().grid(row=4, column=0, sticky='nsew', padx=6, pady=(0, 6))

        root.rowconfigure(0, weight=1)
        root.rowconfigure(4, weight=1)
        root.columnconfigure(0, weight=1)

        self.update_monitor()
        self.update_metronome()

    # class applies to BOTH types: impulse uses it to pace the per-class emission breath
    def _on_type_change(self, *_):
        impulse = self.type_var.get() == 'impulse'
        self.dur_var.set(str(IMPULSE_DURATION if impulse else DEFAULT_DURATION))

    def _impulse_window(self):
        """Class-dependent SIR emission window = one paced breath at the class rate (60/bpm)."""
        return 60.0 / BREATH_RATES.get(self.class_var.get(), 15)

    # baseline stability
    def baseline_state(self, data):
        if len(data) < BASE_WINDOW:
            return None
        recent = data[-BASE_WINDOW:]
        hum = np.array([d[1] for d in recent])
        temp = np.array([d[2] for d in recent])
        noise_ok = (np.std(np.diff(temp)) < TEMP_TOL) and (np.std(np.diff(hum)) < HUM_TOL)
        k = max(1, BASE_WINDOW // 3)
        hum_drift = abs(hum[-k:].mean() - hum[:k].mean())
        temp_drift = abs(temp[-k:].mean() - temp[:k].mean())
        drift_ok = (hum_drift < HUM_DRIFT_TOL) and (temp_drift < TEMP_DRIFT_TOL)
        return bool(noise_ok and drift_ok)

    # live monitor
    def update_monitor(self):
        data = self.reader.get_monitor()
        if data:
            now = data[-1][0]
            xs = [d[0] - now for d in data]
            hum = [d[1] for d in data]
            temp = [d[2] for d in data]

            self.ax_hum.cla()
            self.ax_hum.plot(xs, hum, color='steelblue', marker='.')
            self.ax_hum.set_ylabel('Humidity in %')
            self.ax_hum.set_xlim(-MONITOR_WINDOW, 0)
            self.ax_hum.set_title('Live monitor')
            self.ax_hum.grid()
            self.ax_temp.cla()
            self.ax_temp.plot(xs, temp, color='orange', marker='.')
            self.ax_temp.set_ylabel('Temperature in °C')
            self.ax_temp.set_xlabel('seconds ago')
            self.ax_temp.set_xlim(-MONITOR_WINDOW, 0)
            self.ax_temp.grid()

            self.mon_canvas.draw_idle()

            stable = self.baseline_state(data)
            if stable is True:
                self.baseline_lbl.config(text='Baseline: Stable', foreground='green')
            elif stable is False:
                self.baseline_lbl.config(text='Baseline: Unstable', foreground='red')
            else:
                self.baseline_lbl.config(text='Baseline: --', foreground='black')

            # gate Start on a stable baseline
            if not self.measuring:
                self.start_btn.config(state='normal' if stable is True else 'disabled')

        self.root.after(100, self.update_monitor) # refresh 100ms

    def update_metronome(self):
        now = time.time()
        dt = now - self._met_last_t
        self._met_last_t = now

        impulse = self.type_var.get() == 'impulse'
        cls = self.class_var.get()
        bpm = BREATH_RATES.get(cls, 15)
        f = bpm / 60.0

        # elapsed within the active measurement (None when idle / not measuring)
        elapsed = (now - self._measure_start) if (self.measuring and self._measure_start is not None) else None

        c = self.met_canvas
        c.delete('all')
        w, h = c.winfo_width(), c.winfo_height()
        if w < 10 or h < 10:                       # not laid out yet
            self.root.after(50, self.update_metronome)
            return

        # data coords: x in [-2, 10] s, y in [-1.25, 1.25]; map to canvas pixels
        x_min, x_max, y_min, y_max = -2.0, 10.0, -1.25, 1.25
        pad_l, pad_r, pad_top, pad_bot = 10, 10, 26, 20
        plot_w, plot_h = w - pad_l - pad_r, h - pad_top - pad_bot
        px = lambda x: pad_l + (x - x_min) / (x_max - x_min) * plot_w
        py = lambda y: pad_top + (y_max - y) / (y_max - y_min) * plot_h
        def curve(xs, ys, color, width=2):
            pts = []
            for xv, yv in zip(xs, ys):
                pts += [px(xv), py(yv)]
            c.create_line(*pts, fill=color, width=width, smooth=True)
        def dot(x, y, color, r=6):
            c.create_oval(px(x) - r, py(y) - r, px(x) + r, py(y) + r, fill=color, outline='')

        title = ''
        if elapsed is not None and elapsed < BASELINE_SECONDS:
            # shared baseline hold for both measurement types
            self._met_phase = 0.0
            left = int(np.ceil(BASELINE_SECONDS - elapsed))
            cue = "follow the paced breath, once" if impulse else "exhale when it turns green"
            nxt = f"one paced {cls} breath" if impulse else f"breathe {cls}"
            curve([-2, 10], [-1, -1], "#e8a33d")
            dot(0, -1, "#e8a33d")
            c.create_text(w / 2, h / 2, justify='center', fill="#b9791f",
                          font=('TkDefaultFont', 13, 'bold'),
                          text=f"Baseline measurement: hold still · {left}s\n({cue})")
            title = f"Step 1/2 · Baseline (no breathing) → then {nxt}"
        elif impulse:
            # class-dependent SIR: reuse the SAME paced-breath guide as the breathing recordings,
            # but for a SINGLE cycle at the class rate -> then settle/decay back to baseline.
            win = self._impulse_win if (self.measuring and self._impulse_win) else self._impulse_window()
            x = np.linspace(-2.0, 10.0, 400)
            in_breath = elapsed is not None and elapsed < BASELINE_SECONDS + win
            if elapsed is None or in_breath:
                self._met_phase = (self._met_phase + 2 * np.pi * f * dt) % (2 * np.pi)
                y = -np.cos(self._met_phase + 2 * np.pi * f * x)
                exhaling = np.sin(self._met_phase) > 0
                curve(x, y, "#33aa77")
                dot(0, -np.cos(self._met_phase), "#cc3333")
                phase_txt = "EXHALE ▲" if exhaling else "INHALE ▼"
                if in_breath:
                    title = f"Step 2/2 · Impulse ({cls}) · ONE breath · {phase_txt}"
                else:
                    title = f"Impulse ({cls}) · baseline → one paced breath → settle"
            else:
                # settle: hold still, let the response decay back to baseline
                self._met_phase = 0.0
                t_set = elapsed - (BASELINE_SECONDS + win)
                y = -1 + 2 * np.exp(-np.clip(t_set + x, 0, None) / IMPULSE_SETTLE_TAU)
                curve(x, y, "#33aa77")
                dot(0, -1 + 2 * np.exp(-t_set / IMPULSE_SETTLE_TAU), "#33aa77")
                left = max(0, int(np.ceil((self._measure_total or 0) - elapsed)))
                c.create_text(w / 2, py(1.0), fill="#22aa77", font=('TkDefaultFont', 13, 'bold'),
                              text=f"Settling — hold still, no breathing · {left}s left")
                title = "Step 2/2 · Settling (no breathing)"
        else:
            # paced breathing: free-running before Start, the breathing portion after
            self._met_phase = (self._met_phase + 2 * np.pi * f * dt) % (2 * np.pi)
            x = np.linspace(-2.0, 10.0, 400)
            y = -np.cos(self._met_phase + 2 * np.pi * f * x)
            exhaling = np.sin(self._met_phase) > 0
            curve(x, y, "#33aa77")
            dot(0, -np.cos(self._met_phase), "#cc3333")
            phase_txt = "EXHALE ▲" if exhaling else "INHALE ▼"
            step = "Step 2/2 · " if self.measuring else ""
            title = f"{step}{cls.capitalize()} · {bpm}/min · {phase_txt}"

        c.create_line(px(0), pad_top, px(0), h - pad_bot, fill="#666666", width=1)  # x=0
        c.create_text(w / 2, 13, text=title, font=('TkDefaultFont', 12, 'bold'))
        c.create_text(w - pad_r, h - 9, text="seconds", anchor='e', font=('TkDefaultFont', 9))

        self.root.after(50, self.update_metronome)

    # measurement control
    def start_measurement(self):
        try:
            duration = float(self.dur_var.get())
            assert duration > 0
        except (ValueError, AssertionError):
            messagebox.showerror('Invalid duration', 'Enter a positive number of seconds.')
            return
        impulse = self.type_var.get() == 'impulse'
        self._impulse_win = self._impulse_window()
        min_dur = BASELINE_SECONDS + (self._impulse_win if impulse else 0)
        if duration <= min_dur:
            tail = 'the impulse + some settle time' if impulse else 'time to breathe the pattern'
            messagebox.showerror('Duration too short',
                                 f'Duration must exceed the {min_dur}s baseline/impulse phase '
                                 f'so there is room left for {tail}.')
            return

        self.measuring = True
        self._measure_start = time.time()
        self._measure_total = duration
        self.start_btn.config(state='disabled')
        self.abort_btn.config(state='normal')
        self.discard_btn.config(state='disabled')
        self.reader.start_recording()
        self.status.config(foreground='black')  # clear any prior QC colour
        # breathing trials record a touch past the target so the trailing sample
        # always crosses the full span; trimmed back to `duration` on save.
        rec_window = duration + (0 if impulse else CAPTURE_MARGIN_S)
        self._countdown(int(round(rec_window)), rec_window)

    def _countdown(self, remaining, total):
        if remaining > 0:
            cls = self.class_var.get()
            reg = self.region_var.get()
            impulse = self.type_var.get() == 'impulse'
            elapsed = total - remaining
            win = self._impulse_win if (self.measuring and self._impulse_win) else self._impulse_window()
            if elapsed < BASELINE_SECONDS:
                nxt = f'one paced {cls} breath' if impulse else f'breathe {cls}'
                self.status.config(text=f'BASELINE (hold still) … {int(BASELINE_SECONDS - elapsed)}s — then {nxt}')
            elif impulse and elapsed < BASELINE_SECONDS + win:
                self.status.config(text=f'IMPULSE — {IMPULSE_CUE.get(cls, "one breath").lower()} NOW! ({cls}/{reg})')
            elif impulse:
                self.status.config(text=f'Settling (hold still, no breathing) … {remaining}s left')
            else:
                self.status.config(text=f'Breathe {cls}/{reg} … {remaining}s left')
            self._countdown_id = self.root.after(1000, lambda: self._countdown(remaining - 1, total))
        else:
            self._countdown_id = None
            self.finish_measurement()

    def abort_measurement(self):
        if not self.measuring:
            return
        if self._countdown_id is not None:
            self.root.after_cancel(self._countdown_id)
            self._countdown_id = None
        self.reader.stop_recording()
        self.measuring = False
        self.abort_btn.config(state='disabled')
        self.status.config(text='Measurement aborted — nothing saved.')

    def finish_measurement(self):
        rows = self.reader.stop_recording()
        self.measuring = False
        self.abort_btn.config(state='disabled')

        if not rows:
            self.status.config(text='No data captured, check the sensor/connection.')
            return

        reg = self.region_var.get()
        if self.type_var.get() == 'impulse':
            cls = self.class_var.get()
            part = self.part_var.get()
            path = save_sir(rows, reg, cls, part)
            label = f'SIR/{part}/{reg.capitalize()}/{cls.capitalize()}'
            self._sir_warnings = qc_sir(rows, os.path.join(DATASET_ROOT, 'sir'),
                                        PARTICIPANT_PREFIX[part], cls, reg, path)
        else:
            rows = _trim_to_duration(rows, self._measure_total)
            cls = self.class_var.get()
            part = self.part_var.get()
            path = save_measurement(rows, cls, reg, part)
            label = f'{part}/{reg.capitalize()}/{cls.capitalize()}'
            self._sir_warnings = []
        self.last_saved_path = path
        self.discard_btn.config(state='normal')

        # plot the captured run
        t0 = rows[0][0]
        ts = [(r[0] - t0) / 1000.0 for r in rows]
        hum = [r[1] for r in rows]
        temp = [r[2] for r in rows]

        self.res_hum.cla()
        self.res_hum.plot(ts, hum, color='steelblue')
        self.res_hum.set_xlabel('Time in s')
        self.res_hum.set_ylabel('Humidity in %')
        self.res_hum.set_title(f'{label}: Humidity')
        self.res_hum.grid()
        self.res_temp.cla()
        self.res_temp.plot(ts, temp, color='orange')
        self.res_temp.set_xlabel('Time in s')
        self.res_temp.set_ylabel('Temperature in °C')
        self.res_temp.set_title(f'{label}: Temperature')
        self.res_temp.grid()
        self.res_fig.suptitle(('⚠ QC: ' + ' | '.join(self._sir_warnings)) if self._sir_warnings else '',
                              color='#cc3333', fontsize=9)
        self.res_fig.tight_layout()
        self.res_canvas.draw_idle()

        if self._sir_warnings:
            self.status.config(
                text='⚠ Saved ' + os.path.basename(path) + ' — QC: ' + ' | '.join(self._sir_warnings)
                     + '  →  consider "Discard last" & re-record',
                foreground='red')
        else:
            tag = 'QC OK · ' if self.type_var.get() == 'impulse' else ''
            self.status.config(text=f'✓ {tag}Saved {len(rows)} samples to {os.path.relpath(path)}',
                               foreground='green')

    def discard_last(self):
        if self.last_saved_path and os.path.exists(self.last_saved_path):
            os.remove(self.last_saved_path)
            self.status.config(text=f'Discarded {os.path.basename(self.last_saved_path)}')
            self.last_saved_path = None
            self.discard_btn.config(state='disabled')

# main
def main():
    root = tk.Tk()
    root.update_idletasks()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f'{sw}x{sh}+0+0')
    root.minsize(900, 600)
    try:
        reader = SerialReader(PORT, BAUD)
    except serial.SerialException as e:
        messagebox.showerror('Serial error',
                             f'Could not open {PORT}:\n{e}\n\nCheck the port name and '
                             f'that the Arduino IDE Serial Monitor is closed.')
        root.destroy()
        return

    reader.start()
    gui = CampaignGUI(root, reader) #noqa

    def on_close():
        reader.close()
        root.destroy()

    root.protocol('WM_DELETE_WINDOW', on_close)
    root.mainloop()

if __name__ == '__main__':
    main()
