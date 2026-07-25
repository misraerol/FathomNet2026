import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time

# ---------- Page Config ----------
st.set_page_config(
    page_title="FathomNet 2026 — Marine Species Detector",
    page_icon="🐙",
    layout="wide",
)

# ---------- Custom Styling ----------
st.markdown("""
    <style>
    .main { background-color: #0B2540; }
    .stApp { background-color: #081A2E; }
    h1, h2, h3, p, span, label { color: #FFFFFF !important; }
    .stButton>button {
        background-color: #17A2A8;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        border: none;
        padding: 0.5em 1.5em;
    }
    .stButton>button:hover { background-color: #148A90; }
    div[data-testid="stFileUploader"] {
        background-color: #12304F;
        padding: 1em;
        border-radius: 10px;
    }
    .metric-card {
        background-color: #12304F;
        padding: 1em;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.title("🐙 FathomNetCLEF2026 — Marine Species Detector")
st.markdown(
    "<p style='color:#8FA3B0; font-size:16px;'>YOLOv8s fine-tuned on marine survey imagery — "
    "detects 32 species including urchins, crabs, sea fans, and more.</p>",
    unsafe_allow_html=True,
)
st.divider()

# ---------- Load Model (cached so it only loads once) ----------
@st.cache_resource
def load_model(weights_path):
    return YOLO(weights_path)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    weights_path = st.text_input(
        "Model weights path",
        value="best.pt",
        help="Path to your trained YOLOv8 .pt file (e.g. best.pt from your Kaggle training run)",
    )
    conf_threshold = st.slider("Confidence threshold", 0.1, 0.9, 0.25, 0.05)
    st.markdown("---")
    st.markdown(
        "<p style='color:#8FA3B0; font-size:13px;'>"
        "Trained on 6,439 images across 32 species, using a Positive-Unlabeled "
        "object detection dataset from Kaggle's FathomNetCLEF2026 competition."
        "</p>",
        unsafe_allow_html=True,
    )

# ---------- Main: Upload + Predict ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload an Image")
    uploaded_file = st.file_uploader(
        "Choose an underwater image...",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

with col2:
    st.subheader("🔍 Detection Results")
    if uploaded_file is not None:
        try:
            with st.spinner("Loading model..."):
                model = load_model(weights_path)

            with st.spinner("Running detection..."):
                start = time.time()
                results = model.predict(
                    source=np.array(image),
                    conf=conf_threshold,
                    verbose=False,
                )
                elapsed = time.time() - start

            r = results[0]
            annotated = r.plot()  # BGR numpy array
            st.image(annotated[..., ::-1], caption="Detections", use_container_width=True)

            # ---- Metrics row ----
            n_detections = len(r.boxes)
            m1, m2 = st.columns(2)
            m1.markdown(
                f"<div class='metric-card'><h2 style='color:#17A2A8;margin:0;'>{n_detections}</h2>"
                f"<p style='color:#8FA3B0;margin:0;'>Objects Detected</p></div>",
                unsafe_allow_html=True,
            )
            m2.markdown(
                f"<div class='metric-card'><h2 style='color:#17A2A8;margin:0;'>{elapsed:.2f}s</h2>"
                f"<p style='color:#8FA3B0;margin:0;'>Inference Time</p></div>",
                unsafe_allow_html=True,
            )

            # ---- Detected classes table ----
            if n_detections > 0:
                st.markdown("#### Detected Species")
                names = model.names
                rows = []
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    rows.append({"Species": names[cls_id], "Confidence": f"{conf:.2f}"})
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.info("No objects detected above the current confidence threshold. Try lowering it in the sidebar.")

        except FileNotFoundError:
            st.error(
                f"Couldn't find weights file at '{weights_path}'. "
                "Download best.pt from your Kaggle training run (runs/detect/.../weights/best.pt) "
                "and place it in the same folder as this app, or update the path in the sidebar."
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}")
    else:
        st.markdown(
            "<div style='background-color:#12304F; padding:3em; border-radius:10px; text-align:center;'>"
            "<p style='color:#8FA3B0;'>Upload an image on the left to see detections here.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

st.divider()
st.markdown(
    "<p style='color:#8FA3B0; font-size:12px; text-align:center;'>"
    "Built for the FathomNetCLEF2026 project — Model & Training by Mısra."
    "</p>",
    unsafe_allow_html=True,
)
