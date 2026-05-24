"""
utils/image_utils.py - image preprocessing and bounding box drawing

Preprocessing is kept minimal on purpose:
- just resize to max 1280px on the long side
- YOLOv8 handles normalization internally
- adding filters like CLAHE or blur can actually hurt accuracy
  since the model was trained on raw field photos
"""

import cv2
import numpy as np
from typing import List, Dict, Any
from collections import Counter

from utils.constants import CLASS_COLOURS, DEFAULT_COLOUR


def preprocess_image(image_bgr: np.ndarray, max_side: int = 1280) -> np.ndarray:
    """Resize image so longest side is max_side. Never upscales."""
    h, w  = image_bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image_bgr


def draw_detections(image_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """Draw bounding boxes and labels on the image."""
    if not detections:
        return image_bgr

    h, w      = image_bgr.shape[:2]
    thickness = max(2, int(min(h, w) / 320))
    font      = cv2.FONT_HERSHEY_SIMPLEX
    fs        = max(0.45, min(h, w) / 900)
    ft        = max(1, thickness - 1)

    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        label  = det["damage_type"]
        conf   = det["confidence"]
        sev    = det.get("severity_weighted", det["severity"])
        colour = CLASS_COLOURS.get(label, DEFAULT_COLOUR)

        # main box
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), colour, thickness)

        # corner accents
        cl = max(10, (x2 - x1) // 6)
        for px, py, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(image_bgr, (px, py), (px + dx*cl, py), colour, thickness + 1)
            cv2.line(image_bgr, (px, py), (px, py + dy*cl), colour, thickness + 1)

        # label pill
        text = f"{label}  {conf:.0%}  [{sev}]"
        (tw, th), _ = cv2.getTextSize(text, font, fs, ft)
        pad = 5
        py1 = max(0, y1 - th - 2 * pad)
        py2 = max(th + 2 * pad, y1)
        cv2.rectangle(image_bgr, (x1, py1), (min(w, x1 + tw + 2*pad), py2), colour, -1)

        # auto pick text color based on background brightness
        lum     = 0.299 * colour[2] + 0.587 * colour[1] + 0.114 * colour[0]
        txt_col = (0, 0, 0) if lum > 140 else (255, 255, 255)
        cv2.putText(image_bgr, text, (x1 + pad, py2 - pad),
                    font, fs, txt_col, ft, cv2.LINE_AA)

    return _draw_legend(image_bgr, detections)


def _draw_legend(image_bgr: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Add a small damage summary box in the bottom-left corner."""
    counts = Counter(d["damage_type"] for d in detections)
    if not counts:
        return image_bgr

    h, w   = image_bgr.shape[:2]
    font   = cv2.FONT_HERSHEY_SIMPLEX
    fs     = max(0.38, min(h, w) / 1100)
    lh     = max(20, int(28 * fs / 0.4))
    pad    = 10
    bw     = 270
    bh     = (len(counts) + 1) * lh + 2 * pad
    x0, y0 = pad, h - bh - pad

    overlay = image_bgr.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + bw, y0 + bh), (15, 15, 25), -1)
    cv2.addWeighted(overlay, 0.65, image_bgr, 0.35, 0, image_bgr)

    cv2.putText(image_bgr, "DAMAGE SUMMARY", (x0 + pad, y0 + lh),
                font, fs * 0.85, (180, 220, 255), 1, cv2.LINE_AA)

    for i, (dtype, cnt) in enumerate(sorted(counts.items(), key=lambda x: -x[1]), 2):
        col = CLASS_COLOURS.get(dtype, DEFAULT_COLOUR)
        cv2.rectangle(image_bgr,
                      (x0 + pad,     y0 + i*lh - 12),
                      (x0 + pad + 8, y0 + i*lh - 2), col, -1)
        cv2.putText(image_bgr, f"  {dtype[:24]:<24} {cnt}",
                    (x0 + pad + 10, y0 + i*lh),
                    font, fs * 0.80, (220, 230, 240), 1, cv2.LINE_AA)

    return image_bgr


def to_rgb_display(image_bgr: np.ndarray, max_width: int = 700) -> np.ndarray:
    """Resize if needed and convert BGR to RGB for Streamlit."""
    h, w = image_bgr.shape[:2]
    if w > max_width:
        s = max_width / w
        image_bgr = cv2.resize(image_bgr, (int(w*s), int(h*s)),
                               interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
