# Duty-cycle telemetry — round 2, August 2026

Replication of the five duty-cycle configurations, four weeks after round 1
(`../thermal_telemetry/`). Same Raspberry Pi 5, same ONNX Runtime 1.23.2, same
model checksum, same USB camera (Sunplus HK 5M CAM, 1bcf:28c4), mains powered,
governor `ondemand` at a 2.4 GHz ceiling.

Difference in protocol from round 1: each run waits automatically for the SoC to
fall below 55 °C before starting, rather than relying on a fixed interval between
runs. Idle baselines consequently span 52.4–56.8 °C against 49.0–54.0 °C in
round 1, so round 2 is the warmer of the two.

Populate this directory by copying the five run folders from the device:

    scp -r pi:~/logs/group?_*_20260804_* data/thermal_telemetry_aug2026/

Then recompute the table over both rounds:

    python reproduce/reproduce_table3.py --data-dir data/thermal_telemetry
