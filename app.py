import streamlit as st

# Page config sabse pehle
st.set_page_config(page_title="🌿 Weed Detection & Organic Control", layout="wide")

from ultralytics import YOLO
from PIL import Image
import torch
import types
import os
import tempfile
import requests

# Fix torch.classes RuntimeError for Streamlit
if not hasattr(torch.classes, "__path__"):
    torch.classes.__path__ = types.SimpleNamespace(_path=[])

# Set your Groq API key here
GROQ_API_KEY = "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"

# Function to get recommendations from Groq API
def get_recommendation_from_groq(weed_name):
    prompt = (
        f"List 3-4 eco-friendly methods to control the weed '{weed_name}'. "
        f"Avoid any chemical herbicides. Be concise and practical. "
        f"Also specify biological methods of weed removal and in which amount it is needed, you can also specify the proportions of ingredients needed to make a biological matter."
    )

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are an expert in sustainable farming and weed control."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error fetching response from Groq API: {e}"

# Load YOLO model
model_path = r"C:\Users\jaspi\OneDrive\Desktop\finaltest\content\ultralytics\runs\detect\train\weights\best.pt"
model = YOLO(model_path)

# Sidebar
st.sidebar.title("🌿 Weed Detection Panel")
st.sidebar.markdown("Upload a plant image to detect weeds and receive eco-friendly control suggestions.")

uploaded_file = st.sidebar.file_uploader("📤 Upload a Plant Image", type=["jpg", "jpeg", "png"])

# Main Title
st.title("🌱 Weed Detection & Organic Control")

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Detecting weeds... Please wait"):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                img.save(tmp.name)
                results = model(tmp.name)
            os.remove(tmp.name)
        except Exception as e:
            st.error(f"Image processing error: {e}")

    # Get class labels
    names = model.names
    detected_weeds = {names[int(box.cls)] for r in results for box in r.boxes}

    # Display detection results with bounding boxes
    st.subheader("🔎 Detection Results")
    detected_image = results[0].plot()
    st.image(detected_image, caption="Detected Weeds", use_container_width=True)

    if detected_weeds:
        st.subheader("🌱 Organic Methods for Weed Control")

        for weed in detected_weeds:
            with st.spinner(f"🌿 Finding eco-friendly solutions for {weed}..."):
                recommendation = get_recommendation_from_groq(weed)
            st.markdown(f"### 🌾 {weed.capitalize()}")
            st.success(recommendation)
    else:
        st.warning("No weeds detected in the uploaded image.")
else:
    st.info("Please upload an image from the sidebar to start the detection process.")
