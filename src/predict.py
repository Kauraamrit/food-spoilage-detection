import tensorflow as tf
import numpy as np

# =========================
# 1. LOAD TRAINED MODEL
# =========================

MODEL_PATH = "models/food_spoilage_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# 2. CLASS NAMES
# =========================

class_names = [
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges"
]

# =========================
# 3. LOAD IMAGE
# =========================

image_path = input("Enter the path of the food image: ")

image = tf.keras.utils.load_img(
    image_path,
    target_size=(224, 224)
)

image_array = tf.keras.utils.img_to_array(image)

# Add batch dimension
image_array = np.expand_dims(image_array, axis=0)

# =========================
# 4. MAKE PREDICTION
# =========================

predictions = model.predict(image_array)

predicted_index = np.argmax(predictions[0])

predicted_class = class_names[predicted_index]

confidence = predictions[0][predicted_index] * 100

# =========================
# 5. DISPLAY RESULT
# =========================

print("\n==============================")
print(" FOOD SPOILAGE PREDICTION")
print("==============================")

print(f"Prediction : {predicted_class}")
print(f"Confidence : {confidence:.2f}%")

print("==============================")