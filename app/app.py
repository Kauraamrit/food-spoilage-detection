import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# PAGE CONFIGURATION
# =========================

st.set_page_config(
    page_title="Food Spoilage Detector",
    page_icon="🍎",
    layout="centered"
)

# =========================
# TITLE
# =========================

st.title("🍎 Food Spoilage Detector")

st.write(
    "Upload an image of an apple, banana, or orange "
    "to classify it as fresh or rotten."
)

# =========================
# LOAD MODEL
# =========================

MODEL_PATH = "models/food_spoilage_model.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# =========================
# CLASS NAMES
# =========================

class_names = [
    "Fresh Apple",
    "Fresh Banana",
    "Fresh Orange",
    "Rotten Apple",
    "Rotten Banana",
    "Rotten Orange"
]

# =========================
# IMAGE UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "Upload a food image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PREDICTION
# =========================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔍 Predict"):

        # Resize image
        image_resized = image.resize((224, 224))

        # Convert to NumPy array
        image_array = np.array(image_resized)

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        # Make prediction
        predictions = model.predict(image_array)

        predicted_index = np.argmax(predictions[0])

        predicted_class = class_names[predicted_index]

        confidence = predictions[0][predicted_index] * 100

        # =========================
        # DISPLAY RESULT
        # =========================

        st.subheader("Prediction Result")

        st.success(
            f"Prediction: {predicted_class}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )