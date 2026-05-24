"""
utils/constants.py - shared constants for the project

MODEL NOTE:
    MODEL_ID points to local weights in models/best.pt
    These are YOLOv8m weights pretrained on RDD2022 (road damage dataset)
    Place best.pt in the models/ folder before running the app
"""

# bounding box colors per class (BGR format for OpenCV)
CLASS_COLOURS: dict = {
    "D00 Longitudinal Crack": (0,   200, 255),  # amber
    "D10 Transverse Crack":   (0,   100, 255),  # orange
    "D20 Alligator Crack":    (30,   30, 220),  # red
    "D40 Pothole":            (200,   0, 200),  # magenta
}
DEFAULT_COLOUR: tuple = (0, 200, 100)

# model path - update this if using a different weights file
MODEL_ID    = "models/best.pt"
DATASET     = "RDD2022 (Road Damage Dataset 2022)"
NUM_CLASSES = 4
CLASS_NAMES = list(CLASS_COLOURS.keys())
