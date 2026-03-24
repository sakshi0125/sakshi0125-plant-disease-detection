import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import random

# ✅ FIX: load model safely
model = tf.keras.models.load_model(
    "new_model.keras",
    compile=False,
    safe_mode=False
)

class_names = [
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_healthy",
    "Potato_Early_blight",
    "Potato_healthy"
]

# Page settings
st.set_page_config(page_title="Plant Disease Detector 🌱", layout="wide")

# Sidebar
st.sidebar.title("⚙️ Settings")
option = st.sidebar.radio("Choose Input Method:", ["Upload Image", "Random Sample"])

st.sidebar.markdown("---")
st.sidebar.info("This app detects plant diseases using Deep Learning 🌿")

# Main Title
st.markdown("<h1 style='text-align: center; color: green;'>🌱 Plant Disease Detection System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>AI-powered leaf disease classifier 🍃</h4>", unsafe_allow_html=True)

st.markdown("---")

uploaded_file = None

# Upload option
if option == "Upload Image":
    uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg","png","jpeg"])

# ✅ Random option (fixed)
elif option == "Random Sample":
    sample_folder = "sample_images"
    files = os.listdir(sample_folder)

    if "random_image" not in st.session_state:
        st.session_state.random_image = random.choice(files)

    if st.button("🎲 Generate Random Image"):
        new_image = random.choice(files)
        while new_image == st.session_state.random_image:
            new_image = random.choice(files)
        st.session_state.random_image = new_image

    uploaded_file = os.path.join(sample_folder, st.session_state.random_image)

# Prediction function
def predict_image(img):
    img = img.resize((128,128))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    return class_names[class_index], confidence

# Display
if uploaded_file:
    img = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="🌿 Selected Image", use_container_width=True)

    with col2:
        label, conf = predict_image(img)

        st.success(f"✅ Prediction: {label}")
        st.progress(int(conf * 100))
        st.write(f"📊 Confidence: {conf*100:.1f}%")

        if conf > 0.8:
            st.balloons()

# Footer
st.markdown(
    "<div style='margin-top: 40px; text-align:center; color:gray;'>"
    "<p style='font-size:14px;'>"
    "Developed by <b>Sakshi Palkar</b> | 🤖 Machine Learning • Deep Learning • Computer Vision"
    "</p>"
    "</div>",
    unsafe_allow_html=True
)
