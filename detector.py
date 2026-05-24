"""
detector.py - Road Damage Detector
Loads YOLOv8m weights from models/best.pt and runs inference on road images.

Classes detected:
    D00 - Longitudinal Crack
    D10 - Transverse Crack
    D20 - Alligator Crack
    D40 - Pothole

Severity scoring (my own heuristic, not PCI/ASTM):
    confidence < 0.35  -> Low
    0.35 to 0.55       -> Medium
    0.55 to 0.75       -> High
    >= 0.75            -> Critical

    If a detection's bounding box is larger than the median box size,
    severity is bumped up one level. So a large pothole at Medium
    becomes High. Only applied when 2+ detections exist.

Road condition is then the worst severity seen:
    any Critical -> Critical
    any High or 5+ detections -> Poor
    2+ detections -> Moderate
    else -> Good
"""

import numpy as np
from typing import Dict, Any, List

from utils.constants import CLASS_COLOURS, DEFAULT_COLOUR, MODEL_ID


class RoadDamageDetector:
    """Loads the YOLO model and runs road damage detection."""

    SEVERITY_BANDS = [
        ("Critical", 0.75),
        ("High",     0.55),
        ("Medium",   0.35),
        ("Low",      0.00),
    ]

    SCORE_MAP = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
    LABEL_MAP = {1: "Low", 2: "Medium", 3: "High", 4: "Critical"}

    def __init__(self):
        self.model = self._load_model()

    def _load_model(self):
        """Load YOLOv8m weights from models/best.pt."""
        try:
            from ultralytics import YOLO
            model = YOLO(MODEL_ID)
            return model
        except ImportError:
            raise ImportError(
                "ultralytics not installed. Run: pip install ultralytics"
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not load weights from '{MODEL_ID}'.\n"
                f"Make sure models/best.pt is in the project folder.\n\n"
                f"Error: {e}"
            )

    def detect(self, image_bgr: np.ndarray, conf_threshold: float = 0.30,
               iou_threshold: float = 0.45) -> Dict[str, Any]:
        """
        Run inference on a BGR image.
        Returns detections, counts, avg confidence, and road condition.
        """
        results = self.model.predict(
            source=image_bgr,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )

        detections = []
        damage_type_counts = {}

        for result in results:
            if result.boxes is None:
                continue

            boxes     = result.boxes.xyxy.cpu().numpy()
            confs     = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, class_ids)):
                x1, y1, x2, y2 = map(int, box)
                label   = result.names.get(cls_id, "Unknown")
                area_px = (x2 - x1) * (y2 - y1)

                detections.append({
                    "id":          i + 1,
                    "damage_type": label,
                    "confidence":  round(float(conf), 3),
                    "severity":    self._severity(conf),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "area_px":     area_px,
                })
                damage_type_counts[label] = damage_type_counts.get(label, 0) + 1

        total    = len(detections)
        avg_conf = float(np.mean([d["confidence"] for d in detections])) if detections else 0.0

        # apply area-based severity bump (only when 2+ detections)
        if len(detections) >= 2:
            median_area = float(np.median([d["area_px"] for d in detections]))
            for d in detections:
                d["severity_weighted"] = self._severity_weighted(
                    d["confidence"], d["area_px"], median_area
                )
        else:
            for d in detections:
                d["severity_weighted"] = d["severity"]

        return {
            "total_detections":   total,
            "detections":         detections,
            "damage_type_counts": damage_type_counts,
            "avg_confidence":     avg_conf,
            "road_condition":     self._road_condition(detections),
            "conf_threshold":     conf_threshold,
            "iou_threshold":      iou_threshold,
        }

    def _severity(self, conf: float) -> str:
        """Map confidence score to severity label."""
        for label, threshold in self.SEVERITY_BANDS:
            if conf >= threshold:
                return label
        return "Low"

    def _severity_weighted(self, conf: float, area_px: int, median_area: float) -> str:
        """Bump severity up one tier if detection area is above median."""
        base  = self._severity(conf)
        score = self.SCORE_MAP[base]
        if median_area > 0 and area_px > median_area:
            score = min(score + 1, 4)
        return self.LABEL_MAP[score]

    def _road_condition(self, detections: List[Dict]) -> str:
        """
        Overall road condition based on worst severity seen.
        This is a heuristic - not a replacement for PCI or IRC:82.
        """
        if not detections:
            return "Good"
        severities = [d.get("severity_weighted", d["severity"]) for d in detections]
        if "Critical" in severities:
            return "Critical"
        if "High" in severities or len(detections) >= 5:
            return "Poor"
        if len(detections) >= 2:
            return "Moderate"
        return "Good"

    def get_colour(self, label: str) -> tuple:
        return CLASS_COLOURS.get(label, DEFAULT_COLOUR)
