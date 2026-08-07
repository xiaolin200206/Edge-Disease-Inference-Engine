# Power measurement, third round

Section 4.4 and Table 4 of the manuscript. Five duty-cycle configurations, three
hours each, executed 6-7 August 2026.

## What is measured

`vcgencmd pmic_read_adc` exposes fourteen rails of the system-on-chip's
power-management controller. Twelve report both voltage and current; their
product is summed to give SoC-domain power. Of the remaining two, which report
voltage only, `EXT5V` is logged in the `EXT5V_V` column as a check on supply
stability and `BATT` is not logged. Neither is summed. Sampling interval 2 s,
5,445 samples per configuration.

**This is SoC power, not system power.** It excludes the USB camera and the
Active Cooler. Neither is modulated by the duty cycle - the camera stays
enumerated and the fan responds to temperature - so the comparison between
configurations is unaffected, but the absolute figures must be reported as SoC
power. Instrumentation at the battery rail is future work.

## Files

```
group{A,B,C,D,E}_*_power.csv        power sampler output, 2 s interval
group{A,B,C,D,E}_*_2026*/           the inference engine's own telemetry
  *_data.csv                          per-inference latency, temperature, throttle word
  *_events.csv                        THERMAL_CUTOFF and SYSTEM_START records
```

Columns in the power CSVs: `Timestamp`, `Elapsed_s`, `Label`, `Total_W`,
`CPU_Temp_C`, `Throttled_raw`, `EXT5V_V`, then one `<rail>_W` column per summed
rail. `Throttled_raw` is the raw word from `vcgencmd get_throttled` and is
**not** decoded here: its upper bits latch until reboot, which is the error
Section 4.3 discloses having made in the first submission.

## Reproducing Table 4

Run from this directory:

    python3 ../../reproduce/analyse_power.py

Or inline, for the mean/idle/active columns:

    python3 -c "
    import csv, glob, statistics as st
    DUTY = {'A':1.0, 'B':60/75, 'C':60/90, 'D':60/105, 'E':60/120}
    for f in sorted(glob.glob('group*_power.csv')):
        k = f[5]
        d = list(csv.DictReader(open(f)))
        W = [float(r['Total_W']) for r in d]
        idle = st.mean(W[:25])
        mean = st.mean(W)
        act  = (mean - (1 - DUTY[k]) * idle) / DUTY[k]
        print(f'{k}  mean={mean:.3f}  idle={idle:.3f}  active={act:.3f}')
    "

`idle` is the mean of the first 25 samples (50 s), logged before the inference
process starts. Energy per inference is
`mean_W * 3600 / (duty * 3600 / median_latency_s)`, with the Tukey-filtered
median latency taken from the same round's `*_data.csv`.

## Why this round's thermal record is not used

All five configurations reached 82.9-84.0 degrees C here, and cut-offs were
recorded at every sleep interval including the three that produced none in
either of the two rounds in `../thermal_telemetry*/`. Three conditions differed
and cannot be separated:

1. the battery pack was charging throughout, adding heat from the power board;
2. the power sampler added roughly 2% to median latency and a corresponding
   small load (three `vcgencmd` invocations every 2 s);
3. the five runs spanned an overnight session with uncontrolled ambient
   temperature.

The power figures are unaffected - all five configurations were measured under
the same conditions and the comparison is between them. The same argument applies
to the temperatures *within* this round: their ordering relative to one another
is interpretable, and Section 4.4 uses it (configuration A peaked at 84.0 C
against 82.9-83.4 C for B to E, with 377 cut-offs against 25-70). What does not
carry across rounds is the absolute level, so these temperatures are excluded
from Table 3 and from every cross-round thermal claim.

What the round does establish is recorded in Section 6: the margin at a 30 s
sleep interval is thinner than Table 3 suggests. Configuration C peaked at
81.0 degrees C in the first round, 1.0 degrees C below the cut-off threshold,
and a modest increase in total platform thermal load was enough to cross it.
The round is released in full rather than withheld, so that a reader can see
this for themselves.
