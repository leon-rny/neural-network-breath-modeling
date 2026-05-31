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

# baseline-stability
BASE_WINDOW = 10     # number of recent samples to judge stability
TEMP_TOL = 0.2       # max std of consecutive-sample changes in °C
HUM_TOL = 0.9        # max std of consecutive-sample changes in %RH
TEMP_DRIFT_TOL = 0.3 # max |last-third mean − first-third mean| in °C
HUM_DRIFT_TOL = 1.0  # max |last-third mean − first-third mean| in %RH

CLASSES = ['bradypnea', 'eupnea', 'tachypnea']
REGIONS = ['mouth', 'nose']
PARTICIPANTS = ['b', 'c', 'd']
PARTICIPANT_PREFIX = {'b': '', 'c': 'c_', 'd': 'd_'}

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
def save_measurement(rows, breath_class, region, participant):
    folder = os.path.join(DATASET_ROOT, breath_class)
    os.makedirs(folder, exist_ok=True)
    prefix = PARTICIPANT_PREFIX[participant]
    pat = re.compile(rf'^{re.escape(prefix)}{region}_trial_(\d+)\.dat$')
    existing = [int(m.group(1)) for f in os.listdir(folder) if (m := pat.match(f))]
    idx = max(existing) + 1 if existing else 1
    fname = f'{prefix}{region}_trial_{idx}.dat'
    path = os.path.join(folder, fname)
    with open(path, 'w') as f:
        f.write('Time,Humidity,Temperature\n')
        for t, h, temp in rows:
            f.write(f'{t},{h},{temp}\n')
    return path

# gui
class CampaignGUI:
    def __init__(self, root, reader):
        self.root = root
        self.reader = reader
        self.last_saved_path = None
        self.measuring = False
        self._countdown_id = None

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

        ttk.Label(ctrl, text='Participant:').grid(row=0, column=0, sticky='w')
        self.part_var = tk.StringVar(value=PARTICIPANTS[0])
        ttk.OptionMenu(ctrl, self.part_var, PARTICIPANTS[0], *PARTICIPANTS).grid(row=0, column=1, padx=(2, 14))

        ttk.Label(ctrl, text='Class:').grid(row=0, column=2, sticky='w')
        self.class_var = tk.StringVar(value=CLASSES[1])
        ttk.OptionMenu(ctrl, self.class_var, CLASSES[1], *CLASSES).grid(row=0, column=3, padx=(2, 14))

        ttk.Label(ctrl, text='Region:').grid(row=0, column=4, sticky='w')
        self.region_var = tk.StringVar(value=REGIONS[0])
        ttk.OptionMenu(ctrl, self.region_var, REGIONS[0], *REGIONS).grid(row=0, column=5, padx=(2, 14))

        ttk.Label(ctrl, text='Duration in s:').grid(row=0, column=6, sticky='w')
        self.dur_var = tk.StringVar(value=str(DEFAULT_DURATION))
        ttk.Entry(ctrl, textvariable=self.dur_var, width=5).grid(row=0, column=7, padx=(2, 14))

        self.start_btn = ttk.Button(ctrl, text='Start measurement', command=self.start_measurement, state='disabled')
        self.start_btn.grid(row=0, column=8, padx=(0, 8))

        self.abort_btn = ttk.Button(ctrl, text='Abort', command=self.abort_measurement, state='disabled')
        self.abort_btn.grid(row=0, column=9, padx=(0, 8))

        self.discard_btn = ttk.Button(ctrl, text='Discard last',command=self.discard_last, state='disabled')
        self.discard_btn.grid(row=0, column=10, padx=(0, 14))

        self.baseline_lbl = ttk.Label(ctrl, text='Baseline: --', width=22)
        self.baseline_lbl.grid(row=0, column=11, sticky='w')

        self.status = ttk.Label(root, text='Monitoring...', padding=(8, 0, 8, 6))
        self.status.grid(row=2, column=0, sticky='w')

        # result plot
        self.res_fig = Figure(figsize=(8, 3), dpi=100)
        self.res_hum = self.res_fig.add_subplot(1, 2, 1)
        self.res_temp = self.res_fig.add_subplot(1, 2, 2)
        self.res_fig.tight_layout()
        self.res_canvas = FigureCanvasTkAgg(self.res_fig, master=root)
        self.res_canvas.get_tk_widget().grid(row=3, column=0, sticky='nsew', padx=6, pady=(0, 6))

        root.rowconfigure(0, weight=1)
        root.rowconfigure(3, weight=1)
        root.columnconfigure(0, weight=1)

        self.update_monitor()

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

            self.mon_fig.tight_layout()
            self.mon_canvas.draw_idle()

            stable = self.baseline_state(data)
            if stable is True:
                self.baseline_lbl.config(text='baseline: STABLE', foreground='green')
            elif stable is False:
                self.baseline_lbl.config(text='Baseline: Settling…', foreground='red')
            else:
                self.baseline_lbl.config(text='Baseline: --', foreground='black')

            # gate Start on a stable baseline
            if not self.measuring:
                self.start_btn.config(state='normal' if stable is True else 'disabled')

        self.root.after(100, self.update_monitor) # refresh 100ms

    # measurement control
    def start_measurement(self):
        try:
            duration = float(self.dur_var.get())
            assert duration > 0
        except (ValueError, AssertionError):
            messagebox.showerror('Invalid duration', 'Enter a positive number of seconds.')
            return

        self.measuring = True
        self.start_btn.config(state='disabled')
        self.abort_btn.config(state='normal')
        self.discard_btn.config(state='disabled')
        self.reader.start_recording()
        self._countdown(int(round(duration)), duration)

    def _countdown(self, remaining, total):
        if remaining > 0:
            cls = self.class_var.get()
            reg = self.region_var.get()
            self.status.config(text=f'Recording {cls}/{reg} … {remaining}s left')
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

        cls = self.class_var.get()
        reg = self.region_var.get()
        part = self.part_var.get()
        path = save_measurement(rows, cls, reg, part)
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
        self.res_hum.set_title(f'{part}/{reg.capitalize()}/{cls.capitalize()}: Humidity')
        self.res_hum.grid()
        self.res_temp.cla()
        self.res_temp.plot(ts, temp, color='orange')
        self.res_temp.set_xlabel('Time in s')
        self.res_temp.set_ylabel('Temperature in °C')
        self.res_temp.set_title(f'{part}/{reg.capitalize()}/{cls.capitalize()}: Temperature')
        self.res_temp.grid()
        self.res_fig.tight_layout()
        self.res_canvas.draw_idle()

        self.status.config(text=f'Saved {len(rows)} samples to {os.path.relpath(path)}')

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