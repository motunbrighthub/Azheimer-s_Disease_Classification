import os
import gdown
import zipfile
import streamlit as st

# Local file path for the model
MODEL_PATH = "alzheimers_model2.keras"


MODEL_URL = "/kaggle/input/alzheimer-model/keras/default/1/alzheimers_model2.keras"

# Download if not exists
if not os.path.exists(MODEL_PATH):
    st.info("Downloading Alzheimer model weights...")
    gdown.download(MODEL_URL, "alzheimers_model.zip", quiet=False)
    
    # Unzip
    with zipfile.ZipFile("alzheimers_model.zip", "r") as zip_ref:
        zip_ref.extractall(".")
"""
ALZHEIMER'S CLASSIFICATION - STREAMLIT WEB APP
==============================================

A user-friendly web interface for Alzheimer's disease stage prediction
from brain MRI scans using deep learning and AI-powered explanations.

Author: Adijat Oyetoke
Date: 21/01/2026
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go
import io
import os
import pandas as pd
import requests
import zipfile
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# --- Page config ---
st.set_page_config(
    page_title="Alzheimer's Disease Classifier by ADIJAT OYETOKE",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
CATEGORIES = ['ModerateDemented', 'NonDemented', 'VeryMildDemented', 'MildDemented']

CLASS_INFO = {
    'NonDemented': {'severity': 'None','description': 'No signs of dementia detected','recommendation': 'Continue regular health check-ups','color': '#2ecc71'},
    'VeryMildDemented': {'severity': 'Very Mild','description': 'Very early signs of cognitive decline','recommendation': 'Consult with a healthcare professional for monitoring','color': '#f39c12'},
    'MildDemented': {'severity': 'Mild','description': 'Mild cognitive impairment detected','recommendation': 'Medical evaluation and monitoring recommended','color': '#e67e22'},
    'ModerateDemented': {'severity': 'Moderate','description': 'Moderate dementia signs detected','recommendation': 'Immediate medical consultation strongly recommended','color': '#e74c3c'}
}

# --- YarnGPT API ---
YARN_API_KEY = os.environ.get("YARN_API_KEY")

# --- Custom CSS ---
st.markdown("""
<style>
.main-header {font-size:42px;font-weight:bold;color:#1f77b4;text-align:center;margin-bottom:10px;}
.sub-header {font-size:18px;color:#666;text-align:center;margin-bottom:30px;}
.prediction-box {padding:20px;border-radius:10px;margin:10px 0;}
.high-confidence {background-color:#d4edda;border-left:5px solid #28a745;}
.low-confidence {background-color:#fff3cd;border-left:5px solid #ffc107;}
.warning-box {background-color:#f8d7da;border-left:5px solid #dc3545;padding:15px;border-radius:5px;margin:10px 0;}
</style>
""", unsafe_allow_html=True)

# =========================
# Helper functions
# =========================

@st.cache_resource
def load_trained_model(model_path):
    """Load the trained model."""
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image_for_model(image, target_size=(128,128)):
    """Preprocess uploaded image for prediction."""
    try:
        img_array = np.array(image)
        if len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        img_resized = cv2.resize(img_array, target_size)
        img_float = img_resized.astype('float32')
        img_preprocessed = preprocess_input(img_float)
        img_batch = np.expand_dims(img_preprocessed, axis=0)
        return img_batch, img_resized
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None, None

def predict_image(model, preprocessed_image):
    """Predict class of MRI scan."""
    try:
        preds = model.predict(preprocessed_image, verbose=0)
        idx = np.argmax(preds[0])
        predicted_class = CATEGORIES[idx]
        confidence = preds[0][idx]*100
        all_probs = {CATEGORIES[i]: float(preds[0][i]*100) for i in range(len(CATEGORIES))}
        return predicted_class, confidence, all_probs
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None

def create_probability_chart(probabilities, predicted_class):
    classes = list(probabilities.keys())
    probs = list(probabilities.values())
    colors = [CLASS_INFO[c]['color'] if c==predicted_class else '#95a5a6' for c in classes]
    fig = go.Figure(data=[go.Bar(x=probs, y=classes, orientation='h', marker=dict(color=colors),
                                 text=[f'{p:.1f}%' for p in probs], textposition='outside')])
    fig.update_layout(title="Class Probability Distribution", xaxis_title="Probability (%)", yaxis_title="",
                      height=300, margin=dict(l=20,r=20,t=40,b=20), xaxis=dict(range=[0,105]))
    return fig

def create_gauge_chart(confidence):
    if confidence>=80: color="#2ecc71"
    elif confidence>=60: color="#f39c12"
    else: color="#e74c3c"
    fig = go.Figure(go.Indicator(mode="gauge+number", value=confidence, title={'text':"Confidence Level"},
                                 gauge={'axis':{'range':[0,100]}, 'bar':{'color':color},
                                        'steps':[{'range':[0,60],'color':'#ffebee'},
                                                 {'range':[60,80],'color':'#fff9e6'},
                                                 {'range':[80,100],'color':'#e8f5e9'}]}))
    fig.update_layout(height=250, margin=dict(l=20,r=20,t=40,b=20))
    return fig

def explain_with_yarngpt(stage, confidence, language="English"):
    """Get human-friendly explanation using YarnGPT."""
    if not YARN_API_KEY:
        return "YarnGPT API key not set."
    prompt = f"""
A machine learning model analyzed a brain MRI scan.
Result:
- Stage: {stage}
- Confidence: {confidence:.2f}%
Explain this in simple, empathetic language in {language}.
Include a disclaimer that this is not a medical diagnosis.
"""
    try:
        response = requests.post(
            "https://api.yarngpt.ai/v1/generate",
            headers={"Authorization": f"Bearer {YARN_API_KEY}", "Content-Type":"application/json"},
            json={"prompt": prompt, "max_tokens":300}
        )
        return response.json().get("text","No explanation received.")
    except Exception as e:
        return f"Error fetching explanation: {e}"

# =========================
# MAIN APP
# =========================
def main():
    st.markdown('<p class="main-header">🧠 Alzheimer\'s Disease Classifier</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Brain MRI Analysis with Human-Friendly Explanations</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("ℹ About")
        st.info("""
Classify brain MRI scans into:
- Non-Demented
- Very Mild Demented
- Mild Demented
- Moderate Demented

⚠️ Educational/research purposes only. Not a medical diagnosis.
""")
    
    # Load model
    MODEL_PATH = "alzheimers_model2.keras"  # Adjust path if needed
    model = load_trained_model(MODEL_PATH)
    if not model: st.stop()

    # Upload image
    st.header(" Upload Brain MRI Scan")
    uploaded_file = st.file_uploader("Choose an MRI scan (PNG/JPG/JPEG)", type=["png","jpg","jpeg"])
    
    if uploaded_file:
        image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
        st.image(image, caption="Uploaded MRI", use_column_width=True)

        # Preprocess
        preprocessed_img, _ = preprocess_image_for_model(image)
        if preprocessed_img is not None:
            predicted_class, confidence, all_probs = predict_image(model, preprocessed_img)
            if predicted_class:
                st.success(f"Predicted Stage: {predicted_class} (Confidence: {confidence:.2f}%)")
                
                # Charts
                col1,col2 = st.columns([1,2])
                with col1: st.plotly_chart(create_gauge_chart(confidence), use_container_width=True)
                with col2: st.plotly_chart(create_probability_chart(all_probs,predicted_class), use_container_width=True)
                
                # YarnGPT explanation
                with st.expander("Human-Friendly Explanation (YarnGPT)"):
                    language = st.selectbox("Select Explanation Language:",
                                            ['Nigeria English','Pidgin','Igbo','Hausa','Yoruba'])
                    explanation = explain_with_yarngpt(predicted_class, confidence, language)
                    st.write(explanation)

                # Download results
                results_text = f"Predicted Class: {predicted_class}\nConfidence: {confidence:.2f}%\n\nAll Probabilities:\n"
                for c,p in all_probs.items(): results_text += f"{c}: {p:.2f}%\n"
                results_text += "\nFor educational/research purposes only."
                st.download_button("Download Results (TXT)", data=results_text, file_name="alzheimers_results.txt", mime="text/plain")

            else:
                st.error("Prediction failed. Try another image.")
        else:
            st.error("Failed to preprocess the image. Upload a valid MRI scan.")
    else:
        st.info(" Upload a brain MRI scan to begin analysis")

if __name__ == "__main__":
    main()


