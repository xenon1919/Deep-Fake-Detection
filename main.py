import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown

# Google Drive file ID
file_id = '1_LcZWdBMXzVSYp_JQPs3Zk89Cl-H4kDn'  # Replace with your actual file ID
model_url = f'https://drive.google.com/uc?id={file_id}'

# Download the model file
gdown.download(model_url, 'xception_deepfake_image.h5', quiet=False)

# Load the pre-trained model
model = tf.keras.models.load_model("xception_deepfake_image.h5")

# Image preprocessing function
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize image to match model's input size
    img_array = np.array(img).astype("float32") / 127.5 - 1.0  # Normalize to [-1, 1]
    return np.expand_dims(img_array, axis=0)

# Streamlit app title and file uploader
st.title("Deepfake Detection - Real or Fake? 🔍")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for prediction
    input_data = preprocess_image(image)

    # Make a prediction using the model
    pred = model.predict(input_data)[0][0]
    label = "FAKE" if pred > 0.5 else "REAL"
    st.write(f"### Prediction: **{label}** ({pred:.2f})")
