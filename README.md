# 🛣️ Smart Road Damage Detection Assistant

> Built for CSIR-CRRI Research Internship | Final Year Project | Aditya Dorwal

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![YOLOv8m](https://img.shields.io/badge/YOLOv8m-Ultralytics-orange)](https://ultralytics.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red)]([https://streamlit.io](https://smart-road-damage-detection-h.streamlit.app/))

---

## Why I Built This

India has one of the largest road networks in the world, but a lot of road inspection
is still done manually — which is slow, inconsistent, and puts inspectors at risk near
live traffic. I wanted to see if a computer vision system could automate the detection
part, even as a basic prototype.

CSIR-CRRI works directly on pavement condition assessment and road inspection tech,
so this felt like a natural fit to demonstrate during the internship application.

---

## What It Does

Upload a road photograph → the system detects potholes and surface cracks,
draws bounding boxes, estimates severity, gives an overall road condition score,
and generates a downloadable inspection report.

| Feature | Description |
|---|---|
| 📤 Image Upload | JPG / PNG / BMP |
| 🤖 Detection | YOLOv8m pretrained on RDD2022 road damage dataset |
| 🟥 Bounding Boxes | Color-coded per damage class with confidence score |
| 🚦 Road Condition | Good / Moderate / Poor / Critical (heuristic) |
| 📋 Report | Downloadable plain-text inspection report |
| ⚙️ Threshold Sliders | Confidence and IoU — model loads only once |

---

## Model

**YOLOv8m — Locally stored weights (models/best.pt)**

- Pretrained on **RDD2022** (Road Damage Dataset 2022)
- ~47,000 road images from Japan, India, Czech Republic, Norway, USA, China
- 4 damage classes: D00 Longitudinal Crack · D10 Transverse Crack · D20 Alligator Crack · D40 Pothole

> **Note on evaluation metrics:** Since I'm using pretrained weights, accuracy
> metrics (Precision, Recall, mAP) are from the original model benchmark, not
> re-evaluated here. The focus of this project was on building the inspection
> workflow around the model — not retraining it.

YOLOv8m gives a good balance between detection speed and accuracy. For a
field inspection tool, speed matters — an engineer shouldn't wait 30 seconds
per image.

---

## Road Condition Scoring

I designed a simple heuristic to convert detections into a condition label:

```
1. If any detection has Critical severity (conf >= 0.75)   -> Critical
2. If any detection has High severity (conf >= 0.55)
   OR total detections >= 5                                -> Poor
3. If total detections >= 2                                -> Moderate
4. Otherwise                                               -> Good

Severity is also weighted by bounding box area:
If a detection's area is above the median area, severity is bumped up one tier.
So a large pothole at Medium confidence becomes High severity.
```

> This is a demonstration heuristic, not a replacement for PCI (ASTM D6433) or IRC:82.

---

## Architecture

```
Image Upload (Streamlit)
       ↓
Resize to 1280px max (OpenCV)
       ↓
YOLOv8m Inference
  → Confidence filtering
  → NMS (IoU threshold)
       ↓
Result Processing (detector.py)
  → Severity assignment
  → Area-weighted bump
  → Road condition label
       ↓
Bounding Box Drawing (utils/image_utils.py)
       ↓
Report Generation (report_generator.py)
       ↓
Streamlit Dashboard
```

---

## Project Structure

```
road_damage_detection/
├── app.py                  # Streamlit dashboard
├── detector.py             # YOLOv8m wrapper + severity logic
├── report_generator.py     # Report builder
├── requirements.txt
├── README.md
├── utils/
│   ├── constants.py        # Colors, model path, class names
│   └── image_utils.py      # Preprocessing + annotation drawing
├── models/
│   └── README.txt          # Place best.pt here
└── outputs/                # Generated reports/images
```

---

## Installation

```bash
# 1. Clone
git clone https://github.com/adityadorwal/Smart-road-damage-detection.git
cd Smart-road-damage-detection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add model weights
# Place your best.pt file inside the models/ folder
# The app will not start without it
```

---

## Usage

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

Ensure `models/best.pt` is present before running.

---

## Dataset

**RDD2022 — Road Damage Dataset 2022**

| | |
|---|---|
| GitHub | https://github.com/sekilab/RoadDamageDetector |
| Format | YOLO bounding box annotations |
| Classes | D00 (Longitudinal Crack), D10 (Transverse Crack), D20 (Alligator Crack), D40 (Pothole) |

---

## Known Limitations

- Poor lighting or shadows reduce detection accuracy
- Motion blur can miss or misclassify defects
- Water reflections may cause false positives
- Extreme camera angles (< 30° from surface) degrade accuracy
- Low-resolution images (< 300×300 px) are unreliable
- Model trained on multi-country data — Indian roads may benefit from fine-tuning

---

## What I'd Add Next

- GPS geo-tagging from image EXIF data to plot damage on a map
- Batch processing for survey photo sets
- SQLite database to track defect history across inspections
- ONNX export for mobile/edge deployment
- Fine-tuning on CRRI-collected Indian road images

---
