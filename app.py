"""
app.py - Smart Road Damage Detection Assistant
Streamlit dashboard for road inspection using YOLOv8m.

How it works:
  1. User uploads a road image
  2. Image resized to max 1280px
  3. YOLOv8m detects potholes and cracks
  4. Results shown with bounding boxes and stats
  5. Inspection report can be downloaded

Note on model caching:
  @st.cache_resource loads weights once per session.
  Thresholds are passed to detect() at runtime so adjusting
  sliders never reloads the model (was a bug in earlier version).
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from datetime import datetime

from detector import RoadDamageDetector
from report_generator import InspectionReportGenerator
from utils.image_utils import preprocess_image, draw_detections, to_rgb_display
from utils.constants import MODEL_ID

# page config
st.set_page_config(
    page_title="Road Damage Detection | CSIR-CRRI",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1321 60%, #0a1628 100%);
    color: #e8edf5;
}
.header-box {
    background: linear-gradient(90deg, #1a2744, #0f2027, #1a2744);
    border: 1px solid #2a4a7f;
    border-radius: 12px;
    padding: 26px 32px;
    margin-bottom: 20px;
    box-shadow: 0 4px 28px rgba(30,100,220,0.12);
}
.header-box h1 { color: #4fc3f7; font-size: 1.9rem; font-weight: 700; margin: 0; }
.header-box p  { color: #7a90b8; margin: 6px 0 0; font-size: 0.9rem; }
.badge {
    display: inline-block;
    background: rgba(79,195,247,0.1);
    color: #4fc3f7;
    border: 1px solid #4fc3f7;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.73rem;
    font-weight: 600;
    margin: 8px 6px 0 0;
    letter-spacing: 0.4px;
}
.kpi {
    background: linear-gradient(145deg, #131c2e, #1a2744);
    border: 1px solid #2a3f66;
    border-radius: 10px;
    padding: 18px;
    text-align: center;
}
.kpi-val  { font-size: 2rem; font-weight: 700; color: #4fc3f7; font-family: 'JetBrains Mono', monospace; }
.kpi-lbl  { font-size: 0.73rem; color: #6a80a8; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
.section-title {
    font-size: 0.85rem; font-weight: 600; color: #4fc3f7;
    border-left: 3px solid #4fc3f7; padding-left: 10px;
    text-transform: uppercase; letter-spacing: 0.8px;
    margin: 22px 0 12px;
}
.condition-Critical { color: #ff5252; font-weight: 700; font-size: 1.6rem; font-family: 'JetBrains Mono', monospace; }
.condition-Poor     { color: #ff9800; font-weight: 700; font-size: 1.6rem; font-family: 'JetBrains Mono', monospace; }
.condition-Moderate { color: #ffeb3b; font-weight: 700; font-size: 1.6rem; font-family: 'JetBrains Mono', monospace; }
.condition-Good     { color: #66bb6a; font-weight: 700; font-size: 1.6rem; font-family: 'JetBrains Mono', monospace; }
.info { background: rgba(21,101,192,0.1); border: 1px solid #1565c0; border-radius: 8px; padding: 12px 16px; color: #90caf9; font-size: 0.88rem; margin: 8px 0; }
.stButton > button { background: linear-gradient(135deg,#1565c0,#0d47a1); color: white; border: none; border-radius: 8px; font-weight: 600; }
.stDownloadButton > button { background: linear-gradient(135deg,#1b5e20,#2e7d32); color: white; border: none; border-radius: 8px; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model weights…")
def get_detector() -> RoadDamageDetector:
    """Load model once per session. Sliders pass thresholds at runtime."""
    return RoadDamageDetector()


# session state init
for k, v in {"results": None, "annotated": None, "original": None, "report": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# sidebar
with st.sidebar:
    st.markdown("## ⚙️ Detection Settings")
    st.markdown("---")

    conf = st.slider(
        "Confidence Threshold",
        min_value=0.10, max_value=0.90, value=0.30, step=0.05,
        help="Minimum confidence to accept a detection (0-1).",
    )
    iou = st.slider(
        "IoU Threshold (NMS)",
        min_value=0.10, max_value=0.90, value=0.45, step=0.05,
        help="Non-Maximum Suppression overlap threshold.",
    )

    st.markdown("---")
    st.markdown("""
    <div style='color:#6a80a8; font-size:0.82rem; line-height:1.7'>
    <b style='color:#4fc3f7'>Threshold guide:</b><br>
    ↓ Lower conf → more detections, more false positives<br>
    ↑ Higher conf → fewer detections, fewer false positives<br><br>
    This is the <b>precision-recall tradeoff</b> in object detection.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div style='color:#6a80a8; font-size:0.82rem; line-height:1.8'>
    <b style='color:#4fc3f7'>Model</b><br>
    YOLOv8m (medium variant)<br>
    Pretrained on RDD2022<br>
    Weights: models/best.pt<br><br>
    <b style='color:#4fc3f7'>Damage Classes</b><br>
    D00 — Longitudinal Crack<br>
    D10 — Transverse Crack<br>
    D20 — Alligator Crack<br>
    D40 — Pothole<br><br>
    <b style='color:#4fc3f7'>Road Condition Rule</b><br>
    Any Critical detection → Critical<br>
    Any High or ≥5 detections → Poor<br>
    ≥2 detections → Moderate<br>
    Otherwise → Good<br><br>
    <i style='color:#4a6090'>Heuristic indicator only — not a<br>
    substitute for PCI/IRC:82 assessment.</i>
    </div>
    """, unsafe_allow_html=True)


# header
st.markdown("""
<div class="header-box">
  <h1>🛣️ Smart Road Damage Detection</h1>
  <p>AI-powered road surface inspection prototype — CSIR-CRRI Research Demo</p>
  <div>
    <span class="badge">YOLOv8m</span>
    <span class="badge">RDD2022</span>
    <span class="badge">OpenCV</span>
    <span class="badge">CSIR-CRRI</span>
  </div>
</div>
""", unsafe_allow_html=True)


# image upload
st.markdown('<div class="section-title">Upload Road Image</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload a road photograph (JPG / PNG / BMP)",
    type=["jpg", "jpeg", "png", "bmp"],
)

if uploaded:
    img_pil = Image.open(uploaded).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    h_px, w_px = img_bgr.shape[:2]
    too_small = h_px < 100 or w_px < 100

    col_img, col_meta = st.columns([2, 1])
    with col_img:
        st.image(img_pil, caption="Uploaded Image", use_container_width=True)
    with col_meta:
        warning_html = (
            "<br>⚠️ <b>Image is very small (&lt;100px side).</b><br>"
            "Detection accuracy will be low."
            if too_small else ""
        )
        st.markdown(f"""
        <div class="info">
        ✅ <b>Image loaded</b><br><br>
        File: <code>{uploaded.name}</code><br>
        Size: <code>{img_pil.size[0]} × {img_pil.size[1]} px</code><br>
        File size: <code>{uploaded.size/1024:.1f} KB</code>
        {warning_html}<br><br>
        Adjust thresholds in the sidebar, then click Run Detection.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    if st.button("🔍  Run Damage Detection", type="primary"):
        with st.spinner("Running YOLOv8m inference…"):
            try:
                preprocessed = preprocess_image(img_bgr)
                detector     = get_detector()
                results      = detector.detect(
                    preprocessed,
                    conf_threshold=conf,
                    iou_threshold=iou,
                )
                annotated = draw_detections(img_bgr.copy(), results["detections"])

                now    = datetime.now()
                report = InspectionReportGenerator().generate(
                    results, uploaded.name, img_pil.size, now
                )

                st.session_state.results   = results
                st.session_state.annotated = annotated
                st.session_state.original  = img_bgr
                st.session_state.report    = report

                n = results["total_detections"]
                st.success(f"✅ Done — {n} defect{'s' if n != 1 else ''} detected.")
                st.rerun()

            except RuntimeError as e:
                st.error("Model could not be loaded.")
                st.markdown(f"""
                <div class="info">
                <b>Possible cause:</b> Model weights file not found.<br><br>
                <b>Fix:</b> Place your <code>best.pt</code> weights file inside the
                <code>models/</code> folder and restart the app.<br><br>
                <code>{e}</code>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                st.exception(e)


# results section
if st.session_state.results:
    R = st.session_state.results

    st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3)
    cond = R["road_condition"]
    with k1:
        st.markdown(f'<div class="kpi"><div class="kpi-val">{R["total_detections"]}</div>'
                    f'<div class="kpi-lbl">Defects Detected</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi"><div class="kpi-val">{R["avg_confidence"]:.0%}</div>'
                    f'<div class="kpi-lbl">Avg Confidence</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi"><div class="condition-{cond}">{cond}</div>'
                    f'<div class="kpi-lbl">Road Condition</div></div>', unsafe_allow_html=True)

    # before / after
    st.markdown('<div class="section-title">Image Analysis</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Original**")
        st.image(to_rgb_display(st.session_state.original), use_container_width=True)
    with c2:
        st.markdown("**Detected Damage**")
        st.image(to_rgb_display(st.session_state.annotated), use_container_width=True)

    # defect distribution table
    counts = R.get("damage_type_counts", {})
    if counts:
        st.markdown('<div class="section-title">Defect Distribution</div>', unsafe_allow_html=True)
        import pandas as pd
        df = pd.DataFrame(
            [(k, v, f"{v/R['total_detections']:.0%}") for k, v in sorted(counts.items(), key=lambda x: -x[1])],
            columns=["Damage Type", "Count", "Share"]
        )
        df.index = range(1, len(df) + 1)
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df.set_index("Damage Type")["Count"])

    # per-detection details
    dets = R.get("detections", [])
    if dets:
        st.markdown('<div class="section-title">Detection Details</div>', unsafe_allow_html=True)
        import pandas as pd
        df2 = pd.DataFrame([{
            "#":           d["id"],
            "Damage Type": d["damage_type"],
            "Confidence":  f"{d['confidence']:.1%}",
            "Severity":    d.get("severity_weighted", d["severity"]),
            "Area (px²)":  d["area_px"],
        } for d in dets])
        df2.index = range(1, len(df2) + 1)
        st.dataframe(df2, use_container_width=True)

    # report
    st.markdown('<div class="section-title">Inspection Report</div>', unsafe_allow_html=True)
    with st.expander("📄 View Report", expanded=False):
        st.text(st.session_state.report)

    annotated_rgb = to_rgb_display(st.session_state.annotated, max_width=9999)
    buf = io.BytesIO()
    Image.fromarray(annotated_rgb).save(buf, format="PNG")
    buf.seek(0)

    dl1, dl2, _ = st.columns([1, 1, 2])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with dl1:
        st.download_button("⬇️ Download Report", st.session_state.report,
                           file_name=f"road_report_{ts}.txt", mime="text/plain")
    with dl2:
        st.download_button("⬇️ Download Image", buf,
                           file_name=f"road_annotated_{ts}.png", mime="image/png")

# empty state
elif not uploaded:
    st.markdown("""
    <div style="text-align:center; padding:60px 20px; border:1px dashed #2a4a7f;
                border-radius:14px; background:rgba(13,24,40,0.4); margin-top:20px">
        <div style="font-size:3.5rem; margin-bottom:14px">🛣️</div>
        <div style="font-size:1.2rem; color:#4fc3f7; font-weight:600; margin-bottom:8px">
            Upload a road image to begin inspection
        </div>
        <div style="color:#4a6090; font-size:0.88rem; max-width:420px; margin:0 auto">
            Detects potholes and surface cracks using YOLOv8m trained on
            the Road Damage Dataset 2022 (RDD2022, ~47,000 images).
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;padding:16px;color:#2a3a5a;font-size:0.78rem;
            border-top:1px solid #1a2744;margin-top:40px">
    Smart Road Damage Detection · YOLOv8m + OpenCV · CSIR-CRRI Prototype
</div>
""", unsafe_allow_html=True)
