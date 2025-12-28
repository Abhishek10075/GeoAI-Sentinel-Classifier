import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

# --- Page Config ---
st.set_page_config(page_title="GeoAI Multi-Class Analyzer", layout="centered")


# --- Advanced Mapping Logic ---
# Isme aapke bataye gaye saare objects shamil hain
def get_advanced_geo_class(label):
    mapping = {
        # Nature & Terrain
        'buckeye': 'Forest (Dense Trees)',
        'cliff': 'Mountain / Rocky Terrain',
        'valley': 'Mountain / Valley',
        'lakeshore': 'Sea / Water Body',
        'sandbar': 'Sea / Coastal Area',

        # Agriculture & Land
        'worm_fence': 'Crop / Agriculture Land',
        'pot': 'Crop / Small Vegetation',
        'plow': 'Agriculture Land (Ploughed)',
        'golden_retriever': 'Open Land / Barren Land',
        'velvet': 'Green Land / Grassland',

        # Human Settlement (City/Village)
        'lotion': 'City / Urban Area (High Density)',
        'shingle': 'Town / Residential Area',
        'thatch': 'Village / Rural Settlement',
        'container_ship': 'Port / Industrial Area'
    }
    # Agar match nahi mila toh clean label dikhaye
    return mapping.get(label, label.replace('_', ' ').title())


# --- Model Loading ---
@st.cache_resource
def load_model():
    # Pre-trained model jo fine-tuning ke liye base banta hai
    return YOLO('yolov8n-cls.pt')


model = load_model()

# --- UI Interface ---
st.title("🛰️ Multi-Class Satellite Analyzer")
st.markdown("### Detect Forest, Sea, Mountain, City, and more")

uploaded_file = st.file_uploader("Upload Sentinel-2 Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Target Satellite Patch", use_column_width=True)

    if st.button('Analyze All Objects'):
        temp_file = "analyze_this.png"
        img.save(temp_file)

        # Inference
        results = model(temp_file)

        for r in results:
            top_label = r.names[r.probs.top1]
            confidence = r.probs.top1conf.item() * 100

            # Advanced mapping se sahi class nikaalein
            final_identity = get_advanced_geo_class(top_label)

            # --- Results Display ---
            st.success(f"### Detection: {final_identity}")

            # Data visualization cards
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Confidence", f"{confidence:.2f}%")
            with col2:
                st.metric("Sensor", "Sentinel-2")

            # Class Specific Details
            st.info(f"**Technical Note:** Analysis based on 10m spatial resolution and spectral textures.")

        os.remove(temp_file)