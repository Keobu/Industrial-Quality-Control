import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from src.preprocessing import IndustrialPreprocessor
from src.features import FeatureExtractor
from src.model import DefectClassifier
from src.postprocessing import DefectLocalizer

st.set_page_config(page_title="Industrial Quality Control AI", layout="wide")
st.title(" Automated Industrial Quality Control System")
st.write("Upload a component image to run the hybrid CV pipeline for anomaly detection.")

@st.cache_resource
def load_pipeline():
    preprocessor = IndustrialPreprocessor(target_size=(256, 256))
    extractor = FeatureExtractor()
    classifier = DefectClassifier()
    if os.path.exists("models/best_svm_model.joblib"):
        classifier.load_model("models/best_svm_model.joblib")
    else:
        st.warning("Pre-trained model not found. Please run 'python3 main.py' first.")
    return preprocessor, extractor, classifier

preprocessor, extractor, classifier = load_pipeline()

st.sidebar.header("Pipeline Configurations")
confidence_threshold = st.sidebar.slider("SVM Confidence Threshold", 0.0, 1.0, 0.5)

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    temp_path = "temp_uploaded_img.png"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. Input Image")
        st.image(uploaded_file, use_container_width=True)
        
    processed_img = preprocessor.pipeline(temp_path)
    
    with col2:
        st.subheader("2. Preprocessing (CLAHE)")
        st.image(processed_img, channels="GRAY", use_container_width=True)
        
    with st.spinner("Extracting Hybrid Features..."):
        class_feats = extractor.extract_classical_features(processed_img)
        deep_feats = extractor.extract_deep_features(temp_path)
        combined_feats = extractor.combine_features(class_feats, deep_feats).reshape(1, -1)
        
        prediction = classifier.predict(combined_feats)[0]
        probabilities = classifier.predict_proba(combined_feats)[0]
        
    with col3:
        st.subheader("3. Pipeline Diagnostics")
        annotated_img, defect_found = DefectLocalizer.highlight_defects(temp_path, processed_img)
        
        if prediction == 1 and probabilities[1] >= confidence_threshold:
            st.error(f" DEFECT DETECTED ({probabilities[1]*100:.2f}%)")
            if annotated_img is not None:
                annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                st.image(annotated_img_rgb, use_container_width=True, caption="Localized Anomalies")
        else:
            st.success(f" COMPONENT OK ({probabilities[0]*100:.2f}%)")
            st.image(uploaded_file, use_container_width=True, caption="Clear Surface")

    if os.path.exists(temp_path):
        os.remove(temp_path)