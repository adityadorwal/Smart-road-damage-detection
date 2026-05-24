Place your YOLOv8m weights file here as:

    models/best.pt

This file is not included in the repository because model weight
files are large (typically 50–100 MB) and are excluded via .gitignore.

To run the app:
    1. Copy your best.pt file into this models/ folder.
    2. Run: streamlit run app.py

Model used in this project:
    YOLOv8m pretrained on RDD2022 (Road Damage Dataset 2022)
    4 damage classes: D00, D10, D20, D40
    Trained with Ultralytics YOLOv8 framework.

If you want to use a different model, update MODEL_ID in utils/constants.py.
