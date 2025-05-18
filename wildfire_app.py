import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
import time

# App Config
st.set_page_config(
    page_title="Wildfire Detection System",
    page_icon="🔥",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .stProgress > div > div > div > div {
        background-color: #ff5722;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0px;
        font-weight: bold;
        text-align: center;
    }
    .fire-detected {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #ef5350;
    }
    .no-fire {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 2px solid #81c784;
    }
    </style>
    """, unsafe_allow_html=True)

# App Header
st.title("🔥 Wildfire Detection System")
st.markdown("""
    Upload an image to detect whether it contains wildfire. 
    The AI model analyzes visual patterns to identify fire presence.
    """)

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.markdown("""
    This app uses a **CNN model** trained on:
    - 2,000+ wildfire images
    - 2,000+ normal landscape images
    """)
    st.markdown("Model Accuracy: **87%**")
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("1. Upload any landscape image")
    st.markdown("2. The model analyzes visual patterns")
    st.markdown("3. Get instant fire detection results")
    
    st.markdown("**Note:** This is a prototype. Always verify detections with human observation.")
    

# Load model (with caching)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("FFD.keras")

model = load_model()
class_names = ['No Wildfire Detected', '🚨Wildfire Detected!']

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

if uploaded_file:
    # Display original image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess and predict
    with st.spinner('Analyzing image...'):
        # Show progress bar
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            progress_bar.progress(percent_complete + 1)
        
        # Prepare image
        img = img.resize((150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Make prediction
        prediction = model.predict(img_array)
        confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
        predicted_class = int(prediction[0] > 0.5)
        
        # Display results
        with col2:
            st.subheader("Analysis Results")
            
            if predicted_class:
                st.markdown(f"""
                <div class="prediction-box fire-detected">
                    <h2>{class_names[predicted_class]}</h2>
                    <p>Confidence: {confidence*100:.1f}%</p>
                    <p>⚠️ Emergency alert suggested</p>
                </div>
                """, unsafe_allow_html=True)
                st.warning("Potential wildfire detected! Please verify and contact authorities if confirmed.")
            else:
                st.markdown(f"""
                <div class="prediction-box no-fire">
                    <h2>{class_names[predicted_class]}</h2>
                    <p>Confidence: {confidence*100:.1f}%</p>
                    <p>✅ Normal conditions</p>
                </div>
                """, unsafe_allow_html=True)
                st.success("No wildfire detected in this image.")
            
            # Show confidence meter
            st.metric(label="Model Confidence", value=f"{confidence*100:.1f}%")
            
            # Interpretation guide
            with st.expander("How to interpret results"):
                st.markdown("""
                - **>90% confidence**: Strong prediction
                - **70-90% confidence**: Moderate certainty
                - **<70% confidence**: Uncertain prediction
                """)
                st.markdown("""
                *Note: Always verify AI predictions with human observation when possible.*
                """)

    # Add some space
    st.markdown("---")
    st.markdown("### Need Help?")
    st.markdown("For emergency wildfire reporting, contact your local fire department immediately.")