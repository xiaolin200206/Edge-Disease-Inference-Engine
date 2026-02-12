Edge-AG: Robust Offline Disease Inference Engine

    A fault-tolerant, fully offline edge inference system designed for long-term agricultural monitoring in unstructured field environments.

📖 Project Abstract

This project implements a specialized inference engine optimized for resource-constrained edge devices (e.g., Raspberry Pi). Unlike standard computer vision demos, this system addresses the specific challenges of real-world deployment: thermal throttling under direct sunlight, transient environmental noise (wind/glare), and the need for autonomous, unattended operation.

It prioritizes System Stability and False Positive Suppression over raw benchmark metrics, making it suitable for actionable agricultural intervention.

🚀 Key Engineering Features
1. Air-Gapped Autonomy

    Fully Offline: No cloud dependencies or API calls. All inference and logging happen locally on the edge, ensuring functionality in remote farms with zero connectivity.

2. Thermal-Aware Scheduling

    Active Duty-Cycling: Implements a strict 30s ON / 30s OFF logic to manage the thermal envelope of the Raspberry Pi 5 without heavy active cooling.

    Safety Cutoff: Automatic inference suspension if CPU core temperature exceeds 82°C, preventing hardware damage.

3. Noise-Resistant Inference Logic

    Temporal Confirmation: A disease is only "confirmed" if detected consistently across N consecutive frames. This filters out transient false positives caused by motion blur or camera shake.

    Class-Specific Thresholding: abandons a "one-size-fits-all" confidence threshold in favor of dynamic thresholds (e.g., stricter for confusion-prone classes like Algal Spot, lenient for necrotic lesions).
   
🛠️ System Architecture
graph TD
    A[Camera Input] --> B{Thermal Check}
    B -- Safe (<82°C) --> C[YOLOv8 Inference]
    B -- Overheat --> D[Cooling Sleep Mode]
    C --> E[Post-Processing]
    E --> F[Class-Specific Thresholds]
    F --> G[Temporal Confirmation Buffer]
    G -- Confirmed --> H[Local CSV Logging]
    G -- Transient Noise --> I[Discard]

🧠 Advanced Post-Processing Logic
Class-Specific Sensitivity (The "Double Standard" Strategy)

Standard models often struggle with environmental artifacts (e.g., sunlight glare resembling white fungal spots). This engine applies tailored logic:

    High-Confidence Classes (e.g., Algal Leaf Spot): Requires conf > 0.65 to suppress glare-induced false positives.

    Low-Contrast Classes (e.g., Anthracnose): Accepts conf > 0.35 to ensure recall of subtle necrotic features in shadowed regions.

Temporal Consistency Check

To combat "flickering" detections caused by wind blowing through leaves:
IF detection_count_in_buffer >= CONFIRMATION_FRAMES (e.g., 3):
    Log "Confirmed Disease"
ELSE:
    Treat as "Environmental Noise"

📊 Telemetry & Logging

The system maintains a rigorous audit trail for post-mission analysis. All data is saved to `cycle_events.csv` and `data_for_thesis.csv`.

⏱️ Inference Latency:** Time taken (ms) for model forward pass + Non-Maximum Suppression (NMS).
🌡️ Thermal Status:** Real-time CPU core temperature and hardware throttling flags.
💻 Resource Usage:** System-wide CPU Load (%) and RAM utilization.
🎯 Detection Result:** Filtered bounding box coordinates and confidence scores (Post-NMS).

⚠️ Disclaimer & Usage Note

Proprietary Assets: The trained YOLO weights (.pt/.onnx) and the proprietary Durian dataset are not included in this repository due to IP restrictions.

Scope: This codebase serves as the deployment framework. Users must provide their own trained object detection models.

👨‍💻 Author

Lin Ding Shan Edge AI Engineer | Agricultural Robotics Researcher
