#!/usr/bin/env python3
"""
reproduce_table3.py - Reproduce Table 3 (duty-cycle thermal characterisation)
from the released telemetry logs.

This version supersedes the single-run version. Three changes:

1. REPLICATES. Configurations B, C and D were each benchmarked twice under the
   identical script (the four duty*.py differ only in RUN_LABEL and the sleep
   constant; verified by md5 and a digit-normalised diff). Both runs are now
   reported. The second run of each was previously set aside during folder
   housekeeping - one directory per configuration - not on any methodological
   ground, and the shell history shows the 5 July runs were already in
   logs_completed/ before the 6 July runs displaced them. Reporting only one run
   per configuration would therefore be a selection made after seeing the
   results.

2. IDLE BASELINE. The SYSTEM_START temperature of each run is reported. It is
   the best available proxy for the thermal environment at the start of a run,
   and it varies by 5 C across the released set. Cut-off count is an exceedance
   count against a fixed 82 C threshold, so it is sensitive to baseline in a way
   that mean temperature is not: the two runs of configuration B differ by 0.4 C
   in mean temperature and by a factor of eight in cut-offs.

3. THROTTLE FLAG. detection.py records vcgencmd get_throttled as Yes whenever the
   raw word is non-zero, which includes the sticky bits 16-19 ("has occurred
   since boot"). The flag therefore never clears within a run - verified: zero
   Yes->No transitions across all released runs - and the percentage column is
   not an independent measurement but a restatement of the time to first
   throttle. Both are printed so the relationship is visible. New runs should
   mask with 0b1111 and log the raw word separately.

Usage:
    python reproduce_table3.py --data-dir data/thermal_telemetry
    python reproduce_table3.py --data-dir data/thermal_telemetry --published-only
"""
import argparse
import csv
import glob
import os
import re
from datetime import datetime

# (label, sleep_s, folder-substring)
GROUPS = [
    ('A (cont.)', 0,  'groupA_continuous'),
    ('B',        15,  'groupB_60-15'),
    ('C',        30,  'groupC_60-30'),
    ('D',        45,  'groupD_60-45'),
    ('E',        60,  'groupE_60-60'),
]
ACTIVE_S = 60.0
CUTOFF_SUSPEND_S = 5.0
MIN_ROWS = 5000          # runs shorter than this were aborted; see --min-rows

# runs published in the original submission, one per configuration
PUBLISHED = {
    'groupA_continuous_20260705_154109',
    'groupB_60-15_20260706_132828',
    'groupC_60-30_20260706_171747',
    'groupD_60-45_20260706_204557',
    'groupD_60-45_20260706_204556',
    'groupE_60-60_20260706_095802',
}


def set_min_rows(n):
    global MIN_ROWS
    MIN_ROWS = n


def psec(t):
    for fmt in ("%H:%M:%S.%f", "%H:%M:%S"):
        try:
            d = datetime.strptime(t.strip(), fmt)
            return d.hour * 3600 + d.minute * 60 + d.second + d.microsecond / 1e6
        except ValueError:
            continue
    return None


def load_csv(path):
    rows = list(csv.DictReader(open(path, newline='', encoding='utf-8-sig')))
    while rows and psec(rows[-1].get('Timestamp') or '') is None:
        rows.pop()
    return rows


def tukey_median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return float('nan')
    q1, q3 = s[n // 4], s[3 * n // 4]
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    clean = [v for v in s if lo <= v <= hi]
    return clean[len(clean) // 2] if clean else s[n // 2]


def find_runs(data_dir, sub):
    """Every run directory for a configuration, published or not, sorted by date."""
    hits = []
    for d in glob.glob(os.path.join(data_dir, '*')):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        if sub not in name:
            continue
        data = glob.glob(os.path.join(d, f'*{sub}*_data.csv')) or \
               glob.glob(os.path.join(d, '*_data.csv'))
        events = glob.glob(os.path.join(d, f'*{sub}*_events.csv')) or \
                 glob.glob(os.path.join(d, '*_events.csv'))
        if data and events:
            hits.append((name, data[0], events[0]))
    return sorted(hits, key=lambda x: re.sub(r'^DISCARDED_', '', x[0]))


def analyse(name, data_path, evt_path, sleep):
    rows = load_csv(data_path)
    if len(rows) < MIN_ROWS:
        return None

    temps = [float(r['CPU_Temp_C']) for r in rows if r.get('CPU_Temp_C')]
    lats = [float(r['Latency_ms']) for r in rows if r.get('Latency_ms')]
    secs = [s for s in (psec(r['Timestamp']) for r in rows) if s is not None]
    dur = secs[-1] - secs[0]
    if dur < 0:
        dur += 86400

    flags = [r['Throttled'].strip().lower() in ('yes', 'true', '1')
             for r in rows if r.get('Throttled')]
    thr_pct = 100.0 * sum(flags) / len(flags) if flags else 0.0

    # time to first throttle, and whether the flag ever clears again
    first_idx = next((i for i, f in enumerate(flags) if f), None)
    t_first = (secs[first_idx] - secs[0]) / 60.0 if first_idx is not None else None
    reverts = sum(1 for a, b in zip(flags, flags[1:]) if a and not b)

    evts = load_csv(evt_path)
    cutoffs = sum(1 for r in evts if r.get('Event', '').strip() == 'THERMAL_CUTOFF')
    idle = next((float(r['CPU_Temp_C']) for r in evts
                 if r.get('Event', '').strip() == 'SYSTEM_START'), float('nan'))

    downtime = cutoffs * CUTOFF_SUSPEND_S / 60.0
    duty = ACTIVE_S / (ACTIVE_S + sleep) if sleep > 0 else 1.0
    wall = dur / 60.0
    eff = ((wall * duty - downtime) / wall * 100.0) if wall > 0 else duty * 100

    return dict(
        run=re.sub(r'^DISCARDED_', '', name),
        published='Y' if name in PUBLISHED else '-',
        start=rows[0]['Timestamp'][:5],
        n=len(rows),
        idle=idle,
        mean_t=sum(temps) / len(temps),
        max_t=max(temps),
        thr_pct=thr_pct,
        t_first=t_first,
        reverts=reverts,
        cutoffs=cutoffs,
        downtime=downtime,
        med_lat=tukey_median(lats),
        nom_cov=duty * 100.0,
        eff_cov=eff,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data/thermal_telemetry')
    ap.add_argument('--published-only', action='store_true',
                    help='reproduce the original submission exactly (one run per config)')
    ap.add_argument('--min-rows', type=int, default=MIN_ROWS)
    a = ap.parse_args()
    set_min_rows(a.min_rows)

    cols = [('Config', 10), ('Run start', 10), ('Pub', 4), ('Idle C', 7),
            ('MeanT', 7), ('MaxT', 7), ('Thr%', 7), ('t_thr(m)', 9),
            ('Cut-offs', 9), ('Down(m)', 8), ('MedLat', 8),
            ('NomCov%', 8), ('EffCov%', 8)]
    print(''.join(f'{h:>{w}}' for h, w in cols))
    print('-' * sum(w for _, w in cols))

    summary = []
    for label, sleep, sub in GROUPS:
        runs = find_runs(a.data_dir, sub)
        results = []
        for name, dp, ep in runs:
            if a.published_only and name not in PUBLISHED:
                continue
            r = analyse(name, dp, ep, sleep)
            if r:
                results.append(r)

        if not results:
            print(f'{label:>10}   no complete run found')
            continue

        for i, r in enumerate(results):
            tf = f'{r["t_first"]:.1f}' if r['t_first'] is not None else '-'
            vals = [label if i == 0 else '', r['start'], r['published'],
                    f'{r["idle"]:.1f}', f'{r["mean_t"]:.1f}', f'{r["max_t"]:.1f}',
                    f'{r["thr_pct"]:.1f}', tf, f'{r["cutoffs"]}',
                    f'{r["downtime"]:.1f}', f'{r["med_lat"]:.1f}',
                    f'{r["nom_cov"]:.1f}', f'{r["eff_cov"]:.1f}']
            print(''.join(f'{v:>{w}}' for v, (_, w) in zip(vals, cols)))

        summary.append((label, results))

    if not a.published_only:
        print()
        print('Per-configuration summary across replicates')
        print(f'{"Config":>10}{"n runs":>8}{"idle C":>16}{"max T":>16}{"cut-offs":>16}')
        print('-' * 66)
        for label, rs in summary:
            idles = [r['idle'] for r in rs]
            maxts = [r['max_t'] for r in rs]
            cuts = [r['cutoffs'] for r in rs]
            def rng(v, f='{:.1f}'):
                return f.format(v[0]) if len(v) == 1 else \
                    f'{f.format(min(v))}-{f.format(max(v))}'
            print(f'{label:>10}{len(rs):>8}{rng(idles):>16}{rng(maxts):>16}'
                  f'{rng(cuts, "{:.0f}"):>16}')

    print()
    print('Notes:')
    print(f'  - Each THERMAL_CUTOFF suspends inference for {CUTOFF_SUSPEND_S:.0f} s '
          f'(bare retry, no hysteresis); active window {ACTIVE_S:.0f} s throughout')
    print('  - Idle C is the SYSTEM_START temperature: the thermal environment at run start')
    print('  - Median latency is Tukey-filtered (1.5 x IQR fence)')
    print('  - Thr% includes the sticky bits of vcgencmd get_throttled and therefore')
    print('    never clears within a run; it equals 100 x (1 - t_thr / duration) and is')
    print('    not independent of the t_thr column. See module docstring.')
    print(f'  - Runs shorter than {MIN_ROWS} rows are aborted starts and are excluded;')
    print('    this criterion is on run length only, not on any outcome.')


if __name__ == '__main__':
    main()
