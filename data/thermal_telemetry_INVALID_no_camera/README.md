# Invalid runs — camera absent

Runs executed on 3–4 August 2026 with no USB camera attached. The deployment
engine of the time fell back to `cv2.VideoCapture(0)` without checking that the
device had opened; the inference framework then substituted its own bundled
sample images, and the runs completed normally while inferring on photographs
unrelated to the crop.

The signature is a doubling of median inference latency — 819–821 ms against
407–419 ms with the camera attached — accompanied by a halving of the sample
count over the same wall-clock duration and roughly two cores of CPU occupancy
instead of four.

These runs are retained rather than deleted so that the failure mode can be
recognised. **They are not reported in the manuscript and must not be pooled
with either round.** `deployment/detection.py` now aborts in this condition.
