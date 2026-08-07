#!/usr/bin/env python3
import time
import os
import sys
import csv
import cv2
import gc
import psutil
import subprocess
import numpy as np
import traceback
from datetime import datetime
from ultralytics import YOLO


# ==================================================
# Runtime Configuration
# ==================================================

def get_absolute_path(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, filename)


MODEL_PATH = get_absolute_path("Leave_disease.onnx")

# Cyclic workload scheduling (thermal protection)
# Overridable from the environment so that a sweep can vary the duty cycle
# without editing this file; defaults reproduce the deployed configuration.
CYCLE_ACTIVE_SEC = int(os.environ.get("EDIE_CYCLE_ACTIVE_SEC", 30))
CYCLE_SLEEP_SEC = int(os.environ.get("EDIE_CYCLE_SLEEP_SEC", 30))

# Logging and storage
SAVE_DATA_LOG = True
SAVE_IMAGES = os.environ.get("EDIE_SAVE_IMAGES", "1") != "0"
SAVE_IMG_INTERVAL = 2.0   # seconds
INFERENCE_SIZE = 640

# Safety constraints
CONFIRMATION_FRAMES = 3   # temporal confirmation window
MAX_TEMP_LIMIT = 82.0     # CPU thermal cutoff (°C)


# ==================================================
# Class Definitions
# ==================================================

RAW_CLASS_NAMES = [
    "Algal_leave", "Early_Blight", "Leaf_rot", "Phomopsis",
    "Pink_Disease", "early_blight", "root_disease", "Anthracnose"
]

CLASS_MERGE_MAP = {
    "Algal_leave": "Fungal",
    "Anthracnose": "Fungal",
    "Early_Blight": "Early Blight",
    "early_blight": "Early Blight",
    "Leaf_rot": "Leaf Rot",
    "Pink_Disease": "Pink Disease",
    "root_disease": "Root Disease",
    "Phomopsis": "Phomopsis"
}

# Class-specific confidence thresholds
THRESHOLDS = {
    "Fungal": 0.55,
    "Pink Disease": 0.55,
    "Root Disease": 0.40,
    "Phomopsis": 0.40,
    "Leaf Rot": 0.30,
    "Early Blight": 0.35,
    "default": 0.40
}

# Classes prioritized during suppression
VIP_CLASSES = ["Fungal", "Root Disease", "Pink Disease"]


# ==================================================
# System Telemetry
# ==================================================

class SystemMonitor:
    """
    Lightweight system telemetry for edge deployment.
    """

    def get_cpu_temp(self):
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                return float(f.read()) / 1000.0
        except Exception:
            return 0.0

    def get_throttled_state(self):
        """
        Current throttle state only.

        vcgencmd get_throttled returns a bit field in which bits 0-3 describe
        the present state and bits 16-19 latch whether each condition has
        occurred at any point since boot. The sticky bits do not clear until
        the device is rebooted, so testing the raw word against zero - as
        earlier versions of this file did - reports "throttled" for the whole
        remainder of a session after the first transient event, and makes the
        resulting percentage a restatement of the time to first throttle
        rather than an independent measurement. Only bits 0-3 are tested here.
        The raw word is returned alongside so that the sticky history remains
        available for auditing.

        bit 0  under-voltage now        bit 16  under-voltage has occurred
        bit 1  ARM frequency capped     bit 17  frequency capping has occurred
        bit 2  currently throttled      bit 18  throttling has occurred
        bit 3  soft temperature limit   bit 19  soft limit has occurred
        """
        try:
            output = subprocess.check_output(
                ["vcgencmd", "get_throttled"]
            ).decode()
            raw = int(output.split("=")[1].strip(), 16)
            return ("Yes" if raw & 0b1111 else "No"), hex(raw)
        except Exception:
            return "Unknown", ""

    def get_ram_usage(self):
        return psutil.virtual_memory().percent

    def get_cpu_usage(self):
        return psutil.cpu_percent(interval=None)


# ==================================================
# Edge Inference Engine
# ==================================================

class EdgeDiseaseMonitor:
    """
    Fully offline edge inference engine for plant disease monitoring.
    """

    def __init__(self):
        print("Starting edge inference engine...")
        self.monitor = SystemMonitor()
        self.detection_counters = {}
        self.session_start_time = datetime.now()
        self.total_detections = 0

        if not os.path.exists(MODEL_PATH):
            print(f"Model file not found: {MODEL_PATH}")
            sys.exit(1)

        try:
            self.model = YOLO(MODEL_PATH, task="detect")
            self.model(
                np.zeros((INFERENCE_SIZE, INFERENCE_SIZE, 3), dtype=np.uint8),
                verbose=False
            )
        except Exception as e:
            print(f"Model initialization failed: {e}")
            sys.exit(1)

        self._setup_logging()
        self._init_camera()

    # --------------------------------------------------

    def _setup_logging(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.dirname(os.path.abspath(__file__))

        label = os.environ.get("EDIE_RUN_LABEL")
        self.log_dir = os.environ.get(
            "EDIE_LOG_DIR", os.path.join(base_dir, "logs", f"session_{ts}"))
        self.img_dir = os.path.join(self.log_dir, "images")

        os.makedirs(self.img_dir, exist_ok=True)

        stem = label or "data_for_thesis"
        self.data_log = os.path.join(self.log_dir, f"{stem}_data.csv")
        self.event_log = os.path.join(
            self.log_dir, f"{stem}_events.csv" if label else "cycle_events.csv")

        if SAVE_DATA_LOG:
            with open(self.data_log, "w", newline="") as f:
                csv.writer(f).writerow([
                    "Timestamp", "Mode", "Latency_ms", "FPS",
                    "CPU_Usage_%", "RAM_Usage_%", "CPU_Temp_C",
                    "Throttled", "Throttled_Raw",
                    "Confidence_Max", "Detection_Result"
                ])

            with open(self.event_log, "w", newline="") as f:
                csv.writer(f).writerow([
                    "Timestamp", "Event", "CPU_Temp_C", "Note"
                ])

        print(f"Logging directory: {self.log_dir}")

    # --------------------------------------------------

    def _log_event(self, event, note=""):
        if not SAVE_DATA_LOG:
            return
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        temp = self.monitor.get_cpu_temp()
        with open(self.event_log, "a", newline="") as f:
            csv.writer(f).writerow([ts, event, f"{temp:.1f}", note])

    # --------------------------------------------------

    def _init_camera(self):
        """
        Acquire a camera, or abort.

        An earlier version fell back to cv2.VideoCapture(0) without checking
        that the device had opened. With no camera attached the capture object
        is created but yields no frames, the inference call receives no source,
        and the framework silently substitutes its own bundled sample images.
        A benchmark run in that state completes normally and produces telemetry
        that looks valid: in one three-hour run the median latency was 819 ms
        against 410 ms with the camera present, and the recorded detections
        described photographs that had nothing to do with the crop. Aborting is
        the only safe behaviour, because the failure is otherwise invisible
        until the numbers are compared against an earlier round.

        Set EDIE_ALLOW_NO_CAMERA=1 to bypass this check. Do not set it for any
        run whose telemetry will be reported.
        """
        try:
            from picamera2 import Picamera2
            self.camera = Picamera2()
            cfg = self.camera.create_preview_configuration(
                main={"size": (640, 640), "format": "RGB888"}
            )
            self.camera.configure(cfg)
            self.camera.start()
            self.camera_type = "picamera"
            return
        except Exception as e:
            picam_error = e

        self.camera = cv2.VideoCapture(0)
        self.camera_type = "usb"
        if not self.camera.isOpened():
            if os.environ.get("EDIE_ALLOW_NO_CAMERA") == "1":
                print("WARNING: no camera; EDIE_ALLOW_NO_CAMERA is set. "
                      "Telemetry from this run must not be reported.")
                return
            raise SystemExit(
                "FATAL: no camera available.\n"
                f"  picamera2: {type(picam_error).__name__}: {picam_error}\n"
                "  cv2.VideoCapture(0): device did not open\n"
                "Attach the camera and check `v4l2-ctl --list-devices`. "
                "Aborting rather than benchmarking against sample images.")

        ok, _ = self.camera.read()
        if not ok:
            raise SystemExit(
                "FATAL: camera opened but returned no frame. Aborting.")

    # --------------------------------------------------

    def _normalize_class(self, name):
        return CLASS_MERGE_MAP.get(name, name)

    # --------------------------------------------------

    def _iou(self, a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter) if area_a else 0

    # --------------------------------------------------

    def infer(self, frame):
        """
        Single-frame inference with post-processing:
        - class merging
        - class-specific thresholds
        - temporal confirmation
        """
        t0 = time.time()
        results = self.model(frame, imgsz=INFERENCE_SIZE, conf=0.20, iou=0.45, verbose=False)
        latency = (time.time() - t0) * 1000

        detections = []
        if results[0].boxes:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                if cls >= len(RAW_CLASS_NAMES):
                    continue
                disease = self._normalize_class(RAW_CLASS_NAMES[cls])
                conf = float(box.conf[0])
                if conf < THRESHOLDS.get(disease, THRESHOLDS["default"]):
                    continue
                detections.append({
                    "disease": disease,
                    "conf": conf,
                    "bbox": box.xyxy[0].cpu().numpy().astype(int).tolist(),
                    "vip": disease in VIP_CLASSES
                })

        detections.sort(key=lambda x: (x["vip"], x["conf"]), reverse=True)

        confirmed = []
        for d in detections:
            count = self.detection_counters.get(d["disease"], 0) + 1
            self.detection_counters[d["disease"]] = count
            if count >= CONFIRMATION_FRAMES:
                confirmed.append(d)

        return confirmed, latency

    # --------------------------------------------------

    def run(self):
        self._log_event("SYSTEM_START", "Inference loop initialized")

        fps_cnt, fps_start, fps = 0, time.time(), 0.0
        active = True
        cycle_start = time.time()
        frame_id = 0

        while True:
            now = time.time()
            elapsed = now - cycle_start

            if active and elapsed > CYCLE_ACTIVE_SEC:
                self._log_event("CYCLE_SLEEP")
                active = False
                cycle_start = now
                continue

            if not active and elapsed > CYCLE_SLEEP_SEC:
                self._log_event("CYCLE_ACTIVE")
                active = True
                cycle_start = now
                fps_start = time.time()

            if not active:
                time.sleep(0.5)
                continue

            if self.monitor.get_cpu_temp() > MAX_TEMP_LIMIT:
                self._log_event("THERMAL_CUTOFF")
                time.sleep(5)
                continue

            frame = (
                self.camera.capture_array()
                if self.camera_type == "picamera"
                else self.camera.read()[1]
            )

            detections, latency = self.infer(frame)

            fps_cnt += 1
            if fps_cnt >= 10:
                fps = 10 / (time.time() - fps_start)
                fps_start, fps_cnt = time.time(), 0

            if SAVE_DATA_LOG:
                with open(self.data_log, "a", newline="") as f:
                    csv.writer(f).writerow([
                        datetime.now().strftime("%H:%M:%S.%f")[:-3],
                        "Active",
                        f"{latency:.1f}",
                        f"{fps:.1f}",
                        f"{self.monitor.get_cpu_usage():.1f}",
                        f"{self.monitor.get_ram_usage():.1f}",
                        f"{self.monitor.get_cpu_temp():.1f}",
                        *self.monitor.get_throttled_state(),
                        max([d["conf"] for d in detections], default=0.0),
                        "|".join([d["disease"] for d in detections]) or "None"
                    ])

            frame_id += 1


# ==================================================
# Entry Point
# ==================================================

if __name__ == "__main__":
    EdgeDiseaseMonitor().run()
