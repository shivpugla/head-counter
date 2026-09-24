import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

st.set_page_config(page_title="Student Head Counter", layout="wide")
st.title("🎓 Classroom Student Counter")

@st.cache_resource
def load_model():
    # Use yolov8n.pt for fast cloud inference
    return YOLO("yolov8n.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload Classroom Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        st.image(img, use_container_width=True)
    
    # Run YOLO detection (Class 0 is 'person')
    results = model(img, classes=[0], conf=0.25)
    
    # Count detections
    count = len(results[0].boxes)
    
    with col2:
        st.subheader("Detections")
        res_plot = results[0].plot()
        st.image(res_plot, use_container_width=True)
        st.metric(label="Total Students Counted", value=count)
