# Field deployment engine (`detection.py`)

The always-on, fully-offline edge inference engine used for the field
deployment described in Section 5 of the paper. It runs an ONNX YOLO
model on a Raspberry Pi 5 with:

- **Cyclic duty-cycling for thermal protection** — `CYCLE_ACTIVE_SEC = 30`,
  `CYCLE_SLEEP_SEC = 30` (30 s on / 30 s off, ~50% duty). This is the
  deployment default; Table 4 of the paper characterizes the wider
  duty-cycle design space (60 s active, sleep swept 0-60 s) that motivates
  operating below the thermal limit.
- **Hard thermal cutoff** — inference suspends if CPU core temperature
  exceeds `MAX_TEMP_LIMIT = 82 °C`.
- **Post-processing class merge** (`CLASS_MERGE_MAP`) — the eight raw
  label classes are merged into deployment categories (e.g. Algal_leave +
  Anthracnose -> "Fungal"), matching the post-processing merge discussed
  in Section 4.2.
- **Class-specific confidence thresholds** (`THRESHOLDS`) — Fungal 0.55,
  Pink Disease 0.55, Root Disease 0.40, Phomopsis 0.40, Leaf Rot 0.30,
  Early Blight 0.35, default 0.40.
- **Temporal confirmation** — a detection is confirmed only after it
  recurs in `CONFIRMATION_FRAMES = 3` frames, suppressing transient
  wind/glare false positives.
- **Telemetry logging** — per-frame latency, CPU temperature and throttle
  flag, CPU/RAM usage, and filtered detections are written to CSV.

## Usage
Provide your own trained model as `Leave_disease.onnx` (weights are not
released) and run on the Raspberry Pi with a connected camera:
```bash
python detection.py
```
