# Edge Disease Inference Engine

A fully offline edge inference system designed for long-term agricultural
disease monitoring on resource-constrained devices such as Raspberry Pi.

This project prioritizes **system stability, thermal safety, and low false
positive rates** over benchmark accuracy.

---

## Key Characteristics

- Fully offline inference (no cloud dependency)
- Thermal-aware cyclic scheduling
- Class-specific confidence thresholds
- Temporal confirmation to suppress noise
- Designed for continuous, unattended operation

---

## Architecture Overview

1. Camera captures fixed-view frames
2. YOLO-based object detector runs locally
3. Post-processing applies:
   - class merging
   - class-specific thresholds
   - temporal confirmation
4. Confirmed detections are logged locally
5. System telemetry is continuously monitored

---

## Thermal Management

Inference runs in a fixed duty cycle (e.g. 30s ON / 30s OFF) to prevent
thermal throttling under high ambient temperatures.

If CPU temperature exceeds the safety limit, inference is automatically paused.

---

## Post-processing Logic

### Class Merging
Visually similar disease classes are merged at inference time to reduce
semantic ambiguity on lightweight models.

### Class-Specific Thresholds
Each disease category uses a tailored confidence threshold to minimize
false positives and prevent unnecessary chemical intervention.

### Temporal Confirmation
A disease is only confirmed after appearing consistently across multiple
consecutive frames, suppressing transient noise caused by wind or motion blur.

---

## Logging

The system records:
- Inference latency
- Frame rate
- CPU usage and temperature
- Thermal throttling events
- Detection results

All logs are stored locally in CSV format.

---

## Disclaimer

Trained models and proprietary datasets are not included in this repository.

This project is intended for research, engineering evaluation, and controlled
field deployment only.
