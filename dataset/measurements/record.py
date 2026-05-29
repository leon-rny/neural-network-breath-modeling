import argparse
import glob
import os
import re
import sys
import time

import serial

CLASSES = ['bradypnea', 'eupnea', 'tachypnea']
REGIONS = ['mouth', 'nose']
EXPECTED_HEADER = 'Time,Humidity,Temperature'


def participant_id(v: str) -> str:
    if not re.fullmatch(r'[a-z]', v):
        raise argparse.ArgumentTypeError(f'participant must be a single lowercase letter, got {v!r}')
    return v
N_SAMPLES = 36
DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def autodetect_port() -> str:
    candidates = sorted(
        glob.glob('/dev/cu.usbmodem*')
        + glob.glob('/dev/cu.usbserial*')
        + glob.glob('/dev/cu.wchusbserial*')
        + glob.glob('/dev/ttyACM*')
        + glob.glob('/dev/ttyUSB*')
    )
    if not candidates:
        sys.exit('No Arduino serial port found. Plug in the board or pass --port explicitly.')
    if len(candidates) > 1:
        print(f'Multiple ports found ({candidates}); using {candidates[0]}. Pass --port to override.')
    return candidates[0]


def next_trial_num(folder: str, participant: str, region: str) -> int:
    prefix = f'{participant}_' if participant != 'a' else ''
    pattern = re.compile(rf'^{prefix}{region}_trial_(\d+)\.dat$')
    nums = [int(m.group(1)) for f in os.listdir(folder) if (m := pattern.match(f))]
    return max(nums, default=0) + 1


def target_path(cls: str, participant: str, region: str, trial: int) -> str:
    folder = os.path.join(DATASET_DIR, cls)
    prefix = f'{participant}_' if participant != 'a' else ''
    return os.path.join(folder, f'{prefix}{region}_trial_{trial}.dat')


def read_until_header(ser: serial.Serial, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line == EXPECTED_HEADER:
            return
    sys.exit(f'Did not see header "{EXPECTED_HEADER}" within {timeout_s:.0f}s. '
             'Is the sketch flashed and Serial Monitor closed?')


def capture(ser: serial.Serial, n: int) -> list[str]:
    rows: list[str] = []
    while len(rows) < n:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) != 3:
            continue
        try:
            int(parts[0])
            float(parts[1])
            float(parts[2])
        except ValueError:
            continue
        rows.append(line)
        print(f'  {len(rows):2d}/{n}  t={parts[0]:>6}ms  H={parts[1]}%  T={parts[2]}°C')
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--class', dest='cls', required=True, choices=CLASSES)
    p.add_argument('--region', required=True, choices=REGIONS)
    p.add_argument('--participant', default='a', type=participant_id,
                   help='Single lowercase letter identifying the participant (default: a)')
    p.add_argument('--trial', type=int, default=None, help='Override auto-detected trial number')
    p.add_argument('--port', default=None, help='Serial port (auto-detected if omitted)')
    p.add_argument('--baud', type=int, default=9600)
    p.add_argument('--samples', type=int, default=N_SAMPLES)
    p.add_argument('--force', action='store_true', help='Overwrite if target file exists')
    args = p.parse_args()

    folder = os.path.join(DATASET_DIR, args.cls)
    os.makedirs(folder, exist_ok=True)
    trial = args.trial if args.trial is not None else next_trial_num(folder, args.participant, args.region)
    out_path = target_path(args.cls, args.participant, args.region, trial)
    if os.path.exists(out_path) and not args.force:
        sys.exit(f'Refusing to overwrite {out_path} (pass --force).')

    port = args.port or autodetect_port()
    print(f'Port:        {port} @ {args.baud} baud')
    print(f'Target:      {os.path.relpath(out_path, DATASET_DIR)}  ({args.samples} samples)')
    print('Connecting (Arduino will reset on DTR)...')

    with serial.Serial(port, args.baud, timeout=5) as ser:
        time.sleep(2.0)  # wait out the post-reset boot
        ser.reset_input_buffer()
        read_until_header(ser)
        print(f'Header seen. Recording {args.samples} samples (~{args.samples * 2}s). Ctrl+C to abort.\n')
        try:
            rows = capture(ser, args.samples)
        except KeyboardInterrupt:
            sys.exit('\nAborted — nothing written.')

    content = EXPECTED_HEADER + '\n' + '\n'.join(rows)
    with open(out_path, 'w') as f:
        f.write(content)
    print(f'\nSaved {len(rows)} samples to {out_path}')


if __name__ == '__main__':
    main()
