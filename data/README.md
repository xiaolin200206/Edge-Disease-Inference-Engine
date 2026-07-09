# Duty-cycle thermal telemetry (Table 4)

Five three-hour benchmarks on a passively-cooled Raspberry Pi 5, indoor
ambient ~28 C, active window fixed at 60 s, sleep interval swept:

| Folder | Config | Sleep (s) | Duty |
|--------|--------|-----------|------|
| groupA_continuous_* | A | 0  | 100% |
| groupB_60-15_*      | B | 15 | 80%  |
| groupC_60-30_*      | C | 30 | 67%  |
| groupD_60-45_*      | D | 45 | 57%  |
| groupE_60-60_*      | E | 60 | 50%  |

Each folder holds `<group>_data.csv` (0.5 s telemetry: Timestamp, Mode,
Latency_ms, FPS, CPU_Usage_%, RAM_Usage_%, CPU_Temp_C, Throttled,
Confidence_Max, Detection_Result) and `<group>_events.csv` (state
transitions and THERMAL_CUTOFF events).

Every value in Table 4 of the paper is recomputed from these logs by
`reproduce/reproduce_table4.py`. The `Throttled` column reports the
Raspberry Pi hardware throttle flag; note that this flag responds to the
~80 C soft temperature limit, so configurations that reach 80-83 C show
a non-zero throttle fraction while configurations D and E (max < 80 C)
show 0.0%, consistent with the events logs containing no THERMAL_CUTOFF.
