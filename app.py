"""
Satellite Land Classification — Drought Risk Detection
GreenAI Coursework | Module 5: Computer Vision for Green Technology | EfficientNetB0 + EuroSAT

Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import json
import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import traceback

st.set_page_config(
    page_title="Drought Risk Detector",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

*, html, body { box-sizing: border-box; }
[class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #0e1a0f; color: #e8ead4; }

.hero-wrap {
    padding: 48px 0 20px 0;
    border-bottom: 1px solid #2a3d2b;
    margin-bottom: 32px;
}
.hero-eyebrow {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem; font-weight: 600;
    letter-spacing: 3px; text-transform: uppercase;
    color: #7daa6a; margin-bottom: 10px;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4.5rem; line-height: 0.95;
    color: #e8ead4; margin: 0;
    text-shadow: 0 0 60px rgba(125,170,106,0.15);
}
.hero-title span { color: #7daa6a; }
.hero-sub { font-size: 0.9rem; color: #8a9e7a; margin-top: 12px; font-weight: 300; }

.upload-label {
    font-size: 0.72rem; font-weight: 600;
    letter-spacing: 2px; text-transform: uppercase;
    color: #7daa6a; margin-bottom: 8px;
}
[data-testid="stFileUploader"] {
    background: #141f15 !important;
    border: 1.5px dashed #2e4a30 !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"]:hover { border-color: #7daa6a !important; }

.stButton > button {
    background: #7daa6a !important; color: #0e1a0f !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important; font-size: 0.95rem !important;
    padding: 14px 28px !important; width: 100% !important;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: #9bc98a !important;
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(125,170,106,0.25) !important;
}

.result-card {
    border-radius: 16px; padding: 28px 32px;
    margin: 16px 0; position: relative; overflow: hidden;
}
.result-green  { background: #0f2211; border: 1px solid #27ae60; }
.result-yellow { background: #211a0a; border: 1px solid #f39c12; }
.result-orange { background: #211208; border: 1px solid #e67e22; }
.result-blue   { background: #081520; border: 1px solid #3498db; }
.result-grey   { background: #151515; border: 1px solid #636e72; }

.result-class  { font-family: 'Bebas Neue', sans-serif; font-size: 2rem; line-height: 1; margin-bottom: 4px; }
.result-interp { font-size: 0.85rem; opacity: 0.7; font-weight: 300; }
.result-conf   { font-size: 0.8rem; margin-top: 10px; opacity: 0.6; }

.info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 16px; }
.info-cell { background: #141f15; border-radius: 10px; padding: 14px 16px; border: 1px solid #2a3d2b; }
.info-cell-label { font-size: 0.65rem; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; color: #7daa6a; margin-bottom: 4px; }
.info-cell-val { font-size: 1rem; font-weight: 500; color: #e8ead4; }

[data-testid="stSidebar"] { background: #0b150c !important; border-right: 1px solid #1e2e1f !important; }
hr { border-color: #1e2e1f !important; }
[data-testid="metric-container"] { background: #141f15; border: 1px solid #2a3d2b; border-radius: 10px; padding: 14px 18px; }
[data-testid="metric-container"] label { color: #7daa6a !important; font-size: 0.75rem !important; }
.stCaption { color: #5a7050 !important; }
.stProgress > div > div { background: #7daa6a !important; }
</style>
""", unsafe_allow_html=True)


# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    # Try multiple possible locations and formats
    search_paths = [
        ('drought_model_artifacts/drought_model.keras', 'keras'),
        ('drought_model_artifacts/drought_model.h5',    'h5'),
        ('drought_model.keras',                          'keras'),
        ('drought_model.h5',                             'h5'),
    ]
    meta_paths = [
        'drought_model_artifacts/metadata.json',
        'metadata.json',
    ]

    model = None
    load_error = None

    for path, fmt in search_paths:
        if os.path.exists(path):
            try:
                size_mb = os.path.getsize(path) / 1e6
                model = tf.keras.models.load_model(path)
                break
            except Exception as e:
                load_error = f"Found {path} ({size_mb:.1f} MB) but failed to load: {e}"
                continue
        
    if model is None:
        # Build a list of what was found for the error message
        found = [p for p, _ in search_paths if os.path.exists(p)]
        if found:
            raise FileNotFoundError(f"Model files found {found} but all failed to load. Last error: {load_error}")
        else:
            cwd_files = os.listdir('.')
            raise FileNotFoundError(
                f"No model file found. Current directory: {os.getcwd()}\n"
                f"Files here: {cwd_files}\n"
                f"Searched: {[p for p,_ in search_paths]}"
            )

    meta = None
    for mp in meta_paths:
        if os.path.exists(mp):
            with open(mp) as f:
                meta = json.load(f)
            break

    if meta is None:
        # Build a minimal fallback metadata so the app still works
        meta = {
            'architecture': 'Custom CNN',
            'dataset': 'EuroSAT',
            'img_size': 64,
            'num_classes': 10,
            'class_names': [
                'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
                'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
                'River', 'SeaLake'
            ],
            'drought_interpretation': {
                'AnnualCrop': '🟡 Agricultural Land',
                'Forest': '🟢 Healthy Vegetation',
                'HerbaceousVegetation': '🟢 Healthy Vegetation',
                'Highway': '⚪ Urban / Built-up',
                'Industrial': '⚪ Urban / Built-up',
                'Pasture': '🟢 Healthy Vegetation',
                'PermanentCrop': '🟡 Agricultural Land',
                'Residential': '⚪ Urban / Built-up',
                'River': '🔵 Water Body',
                'SeaLake': '🔵 Water Body',
            },
            'drought_colors': {
                'AnnualCrop': '#f39c12', 'Forest': '#27ae60',
                'HerbaceousVegetation': '#2ecc71', 'Highway': '#95a5a6',
                'Industrial': '#7f8c8d', 'Pasture': '#1abc9c',
                'PermanentCrop': '#e67e22', 'Residential': '#bdc3c7',
                'River': '#3498db', 'SeaLake': '#2980b9',
            },
            'test_accuracy': 0.0,
            'test_auc': 0.0,
        }

    return model, meta


model_loaded = False
load_error_msg = ""

try:
    model, meta = load_model()
    CLASS_NAMES     = meta['class_names']
    DROUGHT_INTERP  = meta['drought_interpretation']
    DROUGHT_COLORS  = meta['drought_colors']
    IMG_SIZE        = meta['img_size']
    model_loaded    = True
except Exception as e:
    load_error_msg = str(e)
    model_loaded   = False
    CLASS_NAMES    = []
    DROUGHT_INTERP = {}
    DROUGHT_COLORS = {}
    IMG_SIZE       = 64


# ── Predict Function ───────────────────────────────────────────────────────────
def predict(pil_img):
    img = pil_img.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)

    # ── IMPORTANT: Your model was trained with efficientnet.preprocess_input()
    # which scales pixels to [-1, 1]. The Rescaling(1/255) layer inside the
    # model architecture was overridden by this preprocessing at training time.
    # We replicate the same preprocessing here:
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    arr = np.expand_dims(arr, 0)

    probs    = model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_cls = CLASS_NAMES[pred_idx]
    return {
        'class_index'     : pred_idx,
        'predicted_class' : pred_cls,
        'drought_label'   : DROUGHT_INTERP.get(pred_cls, pred_cls),
        'color'           : DROUGHT_COLORS.get(pred_cls, '#7daa6a'),
        'confidence'      : float(probs[pred_idx]),
        'probabilities'   : {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)}
    }


def color_to_css(color):
    mapping = {
        '#27ae60': 'result-green',  '#2ecc71': 'result-green',
        '#1abc9c': 'result-green',  '#f39c12': 'result-yellow',
        '#e67e22': 'result-orange', '#3498db': 'result-blue',
        '#2980b9': 'result-blue',   '#95a5a6': 'result-grey',
        '#7f8c8d': 'result-grey',   '#bdc3c7': 'result-grey',
    }
    return mapping.get(color, 'result-grey')


ADVISORIES = {
    'AnnualCrop'           : 'Agricultural land detected. Monitor for water stress during dry seasons.',
    'Forest'               : 'Dense forest coverage. Healthy carbon sink with no drought indicators.',
    'HerbaceousVegetation' : 'Grassland/shrubland. Check rainfall data — vulnerable to rapid drying.',
    'Highway'              : 'Infrastructure detected. Urban heat island effect may worsen nearby drought.',
    'Industrial'           : 'Industrial zone. High impervious surface — runoff risk during rain events.',
    'Pasture'              : 'Pasture land. Livestock grazing areas at risk during prolonged dry spells.',
    'PermanentCrop'        : 'Orchards/vineyards detected. High irrigation dependency — monitor water tables.',
    'Residential'          : 'Residential area. Urban greening recommended to reduce heat stress.',
    'River'                : 'Water body detected. Healthy hydrological indicator for surrounding region.',
    'SeaLake'              : 'Large water body. Strong positive indicator for regional moisture levels.',
}


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛰️ Drought Risk Detector")
    st.markdown("**GreenAI | Module 5**  \nComputer Vision for Green Technology")
    st.markdown("---")
    if model_loaded:
        st.markdown("**Model**")
        st.caption(f"Architecture: {meta.get('architecture','Custom CNN')}")
        st.caption(f"Dataset: {meta.get('dataset','EuroSAT')}")
        acc = meta.get('test_accuracy', 0)
        auc = meta.get('test_auc', 0)
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{acc*100:.1f}%")
        c2.metric("AUC", f"{auc:.3f}")
    st.markdown("---")
    st.markdown("**Land Classes**")
    for cls in CLASS_NAMES:
        interp = DROUGHT_INTERP.get(cls, cls)
        st.caption(f"• {cls} → {interp}")
    st.markdown("---")
    st.caption("EuroSAT Dataset  \nSentinel-2 Satellite Imagery  \n27,000 images · 10 classes · 64×64px")


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-eyebrow">GreenAI — Final Project</div>
    <div class="hero-title">SATELLITE<br><span>DROUGHT</span><br>DETECTOR</div>
    <div class="hero-sub">Upload a satellite or aerial land image → CNN classifies the land type → Drought risk is assessed</div>
</div>
""", unsafe_allow_html=True)


# ── Model Error Debug Block ────────────────────────────────────────────────────
if not model_loaded:
    st.error("⚠️ Model not loaded — see details below")
    with st.expander("🔍 Debug Info (click to expand)", expanded=True):
        st.markdown("**Error message:**")
        st.code(load_error_msg, language="text")
        st.markdown("**Current working directory:**")
        st.code(os.getcwd(), language="text")
        st.markdown("**Files in current directory:**")
        try:
            all_files = []
            for root, dirs, files in os.walk('.'):
                # skip hidden / venv dirs
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('venv','__pycache__','node_modules')]
                for f in files:
                    fpath = os.path.join(root, f)
                    try:
                        size = os.path.getsize(fpath)
                        all_files.append(f"{fpath}  ({size/1e6:.2f} MB)")
                    except:
                        all_files.append(fpath)
            st.code('\n'.join(all_files) if all_files else "(no files found)", language="text")
        except Exception as fe:
            st.code(str(fe))

        st.markdown("""
**✅ How to fix:**

1. In Colab, make sure Cell 23 saves both formats:
```python
model.save(f'{SAVE_DIR}/drought_model.keras')  # primary
model.save(f'{SAVE_DIR}/drought_model.h5')      # backup
```

2. Download and place files so your folder looks like:
```
your_project/
├── app.py
└── drought_model_artifacts/
    ├── drought_model.keras   ← must be 3+ MB
    └── metadata.json
```

3. The `.keras` file must be **3+ MB**. If it's only a few KB,
   the download was incomplete — re-download from Colab.
""")
    st.stop()


# ── Main Layout ────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown('<p class="upload-label">📡 Upload Satellite Image</p>', unsafe_allow_html=True)
    st.caption("Best results with Sentinel-2, Google Earth, or Landsat imagery. Regular aerial/field photos also work.")

    uploaded = st.file_uploader(
        "Drop image here",
        type=['jpg', 'jpeg', 'png', 'tif', 'tiff'],
        label_visibility='collapsed'
    )

    if uploaded:
        pil_img = Image.open(uploaded)
        st.image(pil_img, caption=f"Uploaded: {uploaded.name} | {pil_img.size[0]}×{pil_img.size[1]}px",
                 use_column_width=True)
        st.markdown("")
        predict_btn = st.button("🛰️ Classify Land & Assess Drought Risk")
    else:
        st.markdown("""
        <div style='background:#0b150c; border:1.5px dashed #2a3d2b; border-radius:12px;
                    padding:48px; text-align:center; color:#3a5c3b; margin:12px 0;'>
            <div style='font-size:2.5rem; margin-bottom:12px'>🛰️</div>
            <div style='font-size:0.9rem;'>No image uploaded yet</div>
            <div style='font-size:0.75rem; margin-top:6px; opacity:0.6;'>
                Try: Google Earth screenshot, Landsat tile, or Sentinel-2 image
            </div>
        </div>
        """, unsafe_allow_html=True)
        predict_btn = False
        pil_img     = None

with right_col:
    st.markdown('<p class="upload-label">🔍 Results</p>', unsafe_allow_html=True)

    if predict_btn and pil_img:
        with st.spinner("Running model..."):
            try:
                result = predict(pil_img)
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.code(traceback.format_exc())
                st.stop()

        idx        = result['class_index']
        pred_cls   = result['predicted_class']
        interp     = result['drought_label']
        conf       = result['confidence']
        color      = result['color']
        css_class  = color_to_css(color)

        st.markdown(f"""
        <div class="result-card {css_class}">
            <div class="result-class" style="color:{color}">{pred_cls}</div>
            <div class="result-interp">{interp}</div>
            <div class="result-conf">Confidence: {conf*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        advisory = ADVISORIES.get(pred_cls, "Land type identified.")
        st.markdown(f"""
        <div style='background:#0f1f10; border-left:3px solid {color};
                    border-radius:0 10px 10px 0; padding:14px 18px;
                    font-size:0.88rem; color:#b8c9a4; margin-bottom:16px;'>
            {advisory}
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="upload-label" style="margin-top:8px">All Class Probabilities</p>',
                    unsafe_allow_html=True)

        probs_sorted = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor('#0e1a0f')
        ax.set_facecolor('#0e1a0f')

        cls_labels = [p[0] for p in probs_sorted]
        cls_vals   = [p[1]*100 for p in probs_sorted]
        bar_colors = [color if c == pred_cls else '#2a3d2b' for c in cls_labels]

        bars = ax.barh(cls_labels, cls_vals, color=bar_colors, edgecolor='#1e2e1f', height=0.6)
        for bar, val in zip(bars, cls_vals):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', color='#8a9e7a', fontsize=8.5, fontweight='500')

        ax.set_xlim(0, 110)
        ax.set_xlabel('Probability (%)', color='#5a7050', fontsize=8)
        ax.tick_params(colors='#8a9e7a', labelsize=8)
        ax.spines[['top','right','left','bottom']].set_visible(False)
        ax.axvline(50, color='#2a3d2b', ls='--', alpha=0.6, lw=0.8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Class", pred_cls[:12])
        m2.metric("Confidence",      f"{conf*100:.1f}%")
        m3.metric("Drought Signal",  interp.split(' ', 1)[-1][:14])

    elif not predict_btn and pil_img:
        st.markdown("""
        <div style='background:#0b150c; border:1px solid #1e2e1f; border-radius:12px;
                    padding:50px; text-align:center; color:#3a5c3b;'>
            <div style='font-size:2rem; margin-bottom:8px'>⬅️</div>
            <div style='font-size:0.9rem;'>Click the button to classify</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#0b150c; border:1px solid #1e2e1f; border-radius:12px;
                    padding:50px; text-align:center; color:#3a5c3b;'>
            <div style='font-size:2rem; margin-bottom:8px'>🛰️</div>
            <div style='font-size:0.9rem;'>Upload an image to get started</div>
        </div>
        """, unsafe_allow_html=True)


# ── How It Works ───────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<p class="upload-label">How It Works</p>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
for col, step, desc in zip(
    [c1, c2, c3, c4],
    ["01 Upload", "02 Preprocess", "03 Classify", "04 Interpret"],
    [
        "Upload a satellite or aerial image of land (any resolution)",
        "Image resized to 64×64px and preprocessed with EfficientNet scaling",
        "CNN classifies into 1 of 10 EuroSAT land types",
        "Land type is mapped to drought risk context with advisory"
    ]
):
    col.markdown(f"""
    <div class="info-cell">
        <div class="info-cell-label">{step}</div>
        <div style='font-size:0.82rem; color:#8a9e7a; line-height:1.5;'>{desc}</div>
    </div>
    """, unsafe_allow_html=True)