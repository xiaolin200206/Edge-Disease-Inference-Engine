# Duty-cycle thermal telemetry (Table 3)

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

Every value in Table 3 of the paper is recomputed from these logs by
`reproduce/reproduce_table3.py`.

# Field-deployment telemetry (§5.1)

`field_test.csv` — 46,576 samples from four field sessions totalling
3.08 h of active logging (February 2026). Columns match the duty-cycle
logs above. This log predates the duty-cycle build and was collected
with an earlier, lighter inference configuration (Section 3.3.2 of the
paper); its absolute latency is not comparable with the duty-cycle runs
and no such comparison is made in the paper. The quantities drawn from
this log are the median inference latency (177.8 ms), the coefficient
of variation (5.2%), and the thermal conditions that motivated the
duty-cycle study (91% of samples throttled, peak 82 °C).

# Note on the `Throttled` column

The `Throttled` column in both the duty-cycle and field-deployment logs
reports the Raspberry Pi hardware throttle flag. This flag responds to
the ~80 °C soft temperature limit, so configurations that reach 80–83 °C
show a non-zero throttle fraction while configurations D and E
(max < 80 °C) show 0.0%, consistent with the event logs containing no
THERMAL_CUTOFF.
