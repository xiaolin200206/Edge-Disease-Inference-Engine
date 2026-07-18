#!/usr/bin/env python3
"""
reproduce_table3.py - Reproduce Table 3 (duty-cycle thermal characterisation)
of the manuscript directly from the released telemetry logs.

For each of the five configurations (A continuous ... E 60-60), this
recomputes every column of Table 3:

  - Mean and maximum CPU temperature
  - Percentage of active-window samples carrying the hardware throttle flag
  - Software thermal cut-off events (from the event log)
  - Cut-off downtime (cut-offs x 5 s, in minutes)
  - Median inference latency (Tukey-filtered)
  - Nominal coverage (the scheduled duty cycle)
  - Effective coverage (nominal minus cut-off downtime)

The cut-off count, downtime, and effective-coverage columns are the
paper's primary thermal contribution.  All values are computed from
the raw CSV logs with no intermediate files.

Usage:
    python reproduce_table3.py --data-dir data/thermal_telemetry
"""
import argparse, csv, glob, os
from datetime import datetime

GROUPS = [   # (label, sleep_s, folder-substring)
    ('A (cont.)', 0,  'groupA_continuous'),
    ('B',        15,  'groupB_60-15'),
    ('C',        30,  'groupC_60-30'),
    ('D',        45,  'groupD_60-45'),
    ('E',        60,  'groupE_60-60'),
]
ACTIVE_S = 60.0
CUTOFF_SUSPEND_S = 5.0   # each THERMAL_CUTOFF event suspends inference for this long


def psec(t):
    """Parse HH:MM:SS[.fff] to seconds-of-day."""
    for fmt in ("%H:%M:%S.%f", "%H:%M:%S"):
        try:
            d = datetime.strptime(t.strip(), fmt)
            return d.hour * 3600 + d.minute * 60 + d.second + d.microsecond / 1e6
        except ValueError:
            continue
    return None


def load_csv(path):
    rows = list(csv.DictReader(open(path, newline='', encoding='utf-8-sig')))
    # drop trailing rows with un-parseable timestamps (e.g. summary lines)
    while rows and psec(rows[-1].get('Timestamp') or '') is None:
        rows.pop()
    return rows


def tukey_median(values):
    """Median of Tukey-fence-filtered values (1.5 x IQR)."""
    s = sorted(values)
    n = len(s)
    q1 = s[n // 4]
    q3 = s[3 * n // 4]
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    clean = [v for v in s if lo <= v <= hi]
    return clean[len(clean) // 2] if clean else s[n // 2]


def find_file(data_dir, sub, suffix):
    """Locate a CSV inside the telemetry folder (handles slight naming variations)."""
    patterns = [
        os.path.join(data_dir, f'{sub}_*', f'{sub}_{suffix}.csv'),
        os.path.join(data_dir, sub, f'{sub}_{suffix}.csv'),
        os.path.join(data_dir, f'{sub}*{suffix}.csv'),
    ]
    for p in patterns:
        hits = glob.glob(p)
        if hits:
            return hits[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data/thermal_telemetry')
    a = ap.parse_args()

    hdr = ['Config', 'MeanT', 'MaxT', 'Thr%', 'Cut-offs',
           'Downtime(min)', 'MedLat(ms)', 'NomCov%', 'EffCov%']
    print(''.join(f'{h:>14}' for h in hdr))
    print('-' * len(hdr) * 14)

    for label, sleep, sub in GROUPS:
        # --- data log ---
        data_path = find_file(a.data_dir, sub, 'data')
        if not data_path:
            print(f'  {label}: data CSV not found'); continue
        rows = load_csv(data_path)
        N = len(rows)

        temps = [float(r['CPU_Temp_C']) for r in rows if r.get('CPU_Temp_C')]
        latencies = [float(r['Latency_ms']) for r in rows if r.get('Latency_ms')]
        secs = [psec(r['Timestamp']) for r in rows]
        secs = [s for s in secs if s is not None]

        # handle midnight wrap
        dur = secs[-1] - secs[0]
        if dur < 0:
            dur += 86400

        mean_t = sum(temps) / len(temps)
        max_t = max(temps)

        # throttle percentage
        thr_flags = [r['Throttled'].strip().lower() in ('yes', 'true', '1')
                     for r in rows if r.get('Throttled')]
        thr_pct = 100.0 * sum(thr_flags) / len(thr_flags)

        # median latency (Tukey-filtered)
        med_lat = tukey_median(latencies)

        # --- event log: count THERMAL_CUTOFF events ---
        evt_path = find_file(a.data_dir, sub, 'events')
        cutoffs = 0
        if evt_path:
            evt_rows = load_csv(evt_path)
            cutoffs = sum(1 for r in evt_rows if r.get('Event', '').strip() == 'THERMAL_CUTOFF')

        # --- derived: downtime and effective coverage ---
        cutoff_downtime_min = (cutoffs * CUTOFF_SUSPEND_S) / 60.0
        duty = ACTIVE_S / (ACTIVE_S + sleep) if sleep > 0 else 1.0
        nominal_cov = duty * 100.0
        # Cut-off downtime is lost from the active portion of the run.
        # Effective coverage = (nominal_active_minutes - downtime) / wall_minutes * 100
        wall_min = dur / 60.0
        active_min = wall_min * duty
        effective_cov = (active_min - cutoff_downtime_min) / wall_min * 100.0 if wall_min > 0 else nominal_cov

        vals = [
            label,
            f'{mean_t:.1f}',
            f'{max_t:.1f}',
            f'{thr_pct:.1f}',
            f'{cutoffs}',
            f'{cutoff_downtime_min:.1f}',
            f'{med_lat:.1f}',
            f'{nominal_cov:.1f}',
            f'{effective_cov:.1f}',
        ]
        print(''.join(f'{v:>14}' for v in vals))

    print()
    print('Notes:')
    print(f'  - Each THERMAL_CUTOFF suspends inference for {CUTOFF_SUSPEND_S:.0f} s (bare retry, no hysteresis)')
    print(f'  - Active window = {ACTIVE_S:.0f} s for all configurations')
    print('  - Median latency is Tukey-filtered (1.5 x IQR fence)')
    print('  - Effective coverage = nominal coverage minus cut-off downtime as a fraction of wall time')


if __name__ == '__main__':
    main()
