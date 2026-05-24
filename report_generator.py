"""
report_generator.py - generates plain-text inspection reports

Road condition scoring used (for reference):
    Any Critical detection (conf >= 0.75)          -> Critical
    Any High detection (conf >= 0.55) or count >= 5 -> Poor
    count >= 2                                      -> Moderate
    Otherwise                                       -> Good

This is a heuristic for demonstration, not PCI or IRC:82.
"""

from datetime import datetime
from typing import Dict, Any


class InspectionReportGenerator:

    # maintenance suggestions per condition
    MAINTENANCE_ADVICE = {
        "Critical": (
            "URGENT — Immediate repair recommended.\n"
            "  - Likely significant pavement failure or large pothole(s).\n"
            "  - Notify maintenance authority as soon as possible.\n"
            "  - Consider temporary traffic management if safety is at risk."
        ),
        "Poor": (
            "HIGH PRIORITY — Schedule repair within 1-2 weeks.\n"
            "  - Multiple defects detected; risk of rapid deterioration.\n"
            "  - Pothole patching or surface treatment required."
        ),
        "Moderate": (
            "PREVENTIVE — Schedule maintenance within 1-3 months.\n"
            "  - Crack sealing recommended to prevent water ingress.\n"
            "  - Re-inspect after monsoon season."
        ),
        "Good": (
            "ROUTINE — No immediate action required.\n"
            "  - Continue standard inspection schedule.\n"
            "  - Apply sealant at next maintenance cycle if cracks widen."
        ),
    }

    def generate(self, results: Dict[str, Any], filename: str,
                 image_size: tuple, now: datetime = None) -> str:
        """Build the inspection report from detection results."""
        if now is None:
            now = datetime.now()

        ts        = now.strftime("%Y-%m-%d  %H:%M:%S")
        report_id = now.strftime("RPT-%Y%m%d-%H%M%S")
        w, h      = image_size
        div       = "═" * 68

        lines = [
            div,
            "  ROAD DAMAGE INSPECTION REPORT",
            "  Smart Road Damage Detection System  |  AI Vision Prototype",
            div,
            f"  Report ID    : {report_id}",
            f"  Timestamp    : {ts}",
            f"  Image File   : {filename}",
            f"  Image Size   : {w} × {h} px",
            f"  Model        : YOLOv8m (locally stored weights, pretrained on RDD2022)",
            f"  Dataset      : RDD2022 (~47,000 road images, 4 damage classes)",
            div,
            "",
            "  SUMMARY",
            "  " + "─" * 64,
            f"  Total Defects Found  : {results['total_detections']}",
            f"  Road Condition       : {results['road_condition']}",
            f"  Average Confidence   : {results['avg_confidence']:.1%}",
            f"  Conf Threshold Used  : {results['conf_threshold']}",
            f"  IoU Threshold Used   : {results['iou_threshold']}",
            "",
            "  ROAD CONDITION SCORING METHOD",
            "  " + "─" * 64,
            "  Heuristic rule (for demonstration purposes):",
            "    Any Critical detection (conf >= 0.75)          -> Critical",
            "    Any High detection (conf >= 0.55) or count >= 5 -> Poor",
            "    count >= 2                                      -> Moderate",
            "    Otherwise                                       -> Good",
            "  Severity is also weighted by bounding box area relative to",
            "  the median detection area (larger defects escalate one tier).",
            "  NOTE: This is a demonstration indicator, not PCI or IRC:82.",
            "",
        ]

        # defect distribution
        counts = results.get("damage_type_counts", {})
        if counts:
            lines += [
                "  DEFECT DISTRIBUTION",
                "  " + "─" * 64,
            ]
            total = results["total_detections"]
            for dtype, cnt in sorted(counts.items(), key=lambda x: -x[1]):
                pct = cnt / total * 100 if total else 0
                bar = "█" * int(pct / 10)
                lines.append(f"  {dtype:<28}  {cnt:>3}  ({pct:4.1f}%)  {bar}")
            lines.append("")

        # per-detection table
        detections = results.get("detections", [])
        if detections:
            lines += [
                "  DETECTION DETAILS",
                "  " + "─" * 64,
                f"  {'#':<4} {'Damage Type':<28} {'Conf':>6}  {'Severity':<10}  {'Area (px²)':>10}",
                "  " + "─" * 64,
            ]
            for d in detections:
                sev = d.get("severity_weighted", d["severity"])
                lines.append(
                    f"  {d['id']:<4} {d['damage_type']:<28} "
                    f"{d['confidence']:>5.1%}  {sev:<10}  {d['area_px']:>10,}"
                )
            lines.append("")

        # recommendations
        condition = results.get("road_condition", "Good")
        advice    = self.MAINTENANCE_ADVICE.get(condition, "")
        lines += [
            "  MAINTENANCE RECOMMENDATION",
            "  " + "─" * 64,
            f"  {advice}",
            "",
        ]

        # limitations
        lines += [
            "  KNOWN LIMITATIONS",
            "  " + "─" * 64,
            "  • Poor lighting or shadows may reduce detection accuracy.",
            "  • Motion blur can miss or misclassify defects.",
            "  • Water reflections may cause false positives.",
            "  • Extreme camera angles (< 30° from surface) degrade accuracy.",
            "  • Very low-resolution images (< 300x300 px) are unreliable.",
            "  • RDD2022 is multi-country; Indian road textures may need",
            "    fine-tuning on locally collected data for best results.",
            "  • Results must be verified by a qualified road engineer.",
            "",
            div,
            "  This report is generated by an AI prototype system.",
            "  Do not use as the sole basis for maintenance decisions.",
            div,
        ]

        return "\n".join(lines)
