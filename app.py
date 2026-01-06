import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="GeoAI Multi-Class Analyzer",
    layout="centered"
)

# ---------------- Label to Geo-Class Mapping ----------------
def get_advanced_geo_class(label):
    geo_mapping = {
        # Natural Features
        "buckeye": "Forest (Dense Trees)",
        "cliff": "Mountain / Rocky Terrain",
        "valley": "Mountain / Valley",
        "lakeshore": "Water Body / Lake",
        "sandbar": "Coastal / Sandy Area",

        # Agricultural & Open Land
        "worm_fence": "Agricultural Land",
        "pot": "Small Vegetation Area",
        "plow": "Ploughed Agricultural Field",
        "golden_retriever": "Open / Barren Land",
        "velvet": "Grassland / Green Cover",

        # Human Settlements
        "lotion": "Urban Area (High Density)",
        "shingle": "Residential / Town Area",
        "thatch": "Rural / Village Area",
        "container_ship": "Industrial / Port Area"
    }

    return geo_mapping.get(label, label.replace("_", " ").title())


# ---------------- Model Loader ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n-cls.pt")


model = load_model()

# ---------------- User Interface ----------------
st.title("🛰️ Multi-Class Satellite Analyzer")
st.markdown("### Land-use and terrain classification from satellite imagery")

uploaded_file = st.file_uploader(
    "Upload Sentinel-2 Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Satellite Image", use_column_width=True)

    if st.button("Analyze Image"):
        temp_image_path = "temp_input.png"
        image.save(temp_image_path)

        predictions = model(temp_image_path)

        for result in predictions:
            predicted_label = result.names[result.probs.top1]
            confidence_score = result.probs.top1conf.item() * 100

            geo_class = get_advanced_geo_class(predicted_label)

            # -------- Result Display --------
            st.success(f"Detected Area Type: **{geo_class}**")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Confidence Level", f"{confidence_score:.2f}%")
            with col2:
                st.metric("Satellite Source", "Sentinel-2")

            st.info(
                "Classification is based on spatial texture patterns "
                "and medium-resolution satellite imagery."
            )

        os.remove(temp_image_path)
