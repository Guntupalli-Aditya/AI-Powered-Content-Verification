import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model

# Load trained model

model = load_model("G:/Practice/SDP/AI Image Detection/ai_detector_model.keras")  # Load using full path

# Function to preprocess image
def preprocess_image(image):
    image = cv2.resize(image, (32, 32))  # Resize to match model input size
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("🖼️ AI-Generated Image Detector")
st.write("Upload an image to check if it's AI-generated or real.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display the image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image and make prediction
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)

    # Display results
    ai_percentage = round(float(prediction[0][0]) * 100, 2)
    real_percentage = 100 - ai_percentage

    if ai_percentage > 50:
        st.markdown("### 🚨 This image is **likely AI-generated**.")
    else:
        st.markdown("### ✅ This image is **likely real**.")
