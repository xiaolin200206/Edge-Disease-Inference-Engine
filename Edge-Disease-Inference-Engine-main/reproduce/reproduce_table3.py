#!/usr/bin/env python3
"""
reproduce_table3.py - Reproduce Table 3 (duty-cycle thermal characterisation)
of the manuscript directly from the released telemetry logs.

For each of the five configurations (A continuous ... E 60-60), this
recomputes: sample count N, run duration, mean and maximum CPU
temperature, time to first hardware-throttle flag, the percentage of
samples recorded under the throttle flag, and the resulting monitoring
coverage (duty cycle).

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

def psec(t):
    for fmt in ("%H:%M:%S.%f", "%H:%M:%S"):
        try:
            d = datetime.strptime(t.strip(), fmt)
            return d.hour*3600 + d.minute*60 + d.second + d.microsecond/1e6
        except ValueError:
            continue
    return None

def load(path):
    rows = list(csv.DictReader(open(path)))
    while rows and psec(rows[-1].get('Timestamp') or '') is None:
        rows.pop()
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data/thermal_telemetry')
    a = ap.parse_args()

    hdr = ['Config','Sleep(s)','Duty','N','Dur(h)','MeanT','MaxT','1stThrottle','Thr%','Coverage']
    print(''.join(f'{h:>11}' for h in hdr))
    print('-'*len(hdr)*11)
    for label, sleep, sub in GROUPS:
        d = glob.glob(os.path.join(a.data_dir, f'{sub}_*', f'{sub}_data.csv'))
        if not d:
            d = glob.glob(os.path.join(a.data_dir, sub, f'{sub}_data.csv')) or \
                glob.glob(os.path.join(a.data_dir, f'{sub}*data.csv'))
        rows = load(d[0])
        N = len(rows)
        temps = [float(r['CPU_Temp_C']) for r in rows if r.get('CPU_Temp_C')]
        secs  = [psec(r['Timestamp']) for r in rows if psec(r.get('Timestamp') or '')]
        dur = secs[-1]-secs[0]; dur += 86400 if dur < 0 else 0
        duty = ACTIVE_S/(ACTIVE_S+sleep)
        thr_flags = [r['Throttled'].strip().lower() in ('yes','true','1') for r in rows if r.get('Throttled')]
        thr_pct = 100.0*sum(thr_flags)/len(thr_flags)
        # time to first throttle
        t0 = secs[0]; first = 'never'
        for r in rows:
            if r.get('Throttled','').strip().lower() in ('yes','true','1'):
                s = psec(r.get('Timestamp') or '')
                if s is not None:
                    m = (s-t0); m += 86400 if m < 0 else 0
                    first = f'{m/60:.1f} min'; break
        vals = [label, str(sleep), f'{duty*100:.0f}%', f'{N:,}', f'{dur/3600:.2f}',
                f'{sum(temps)/len(temps):.1f}', f'{max(temps):.1f}', first,
                f'{thr_pct:.1f}', f'{duty*100:.1f}%']
        print(''.join(f'{v:>11}' for v in vals))

if __name__ == '__main__':
    main()
