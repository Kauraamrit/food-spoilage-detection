import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =========================
# 1. SETTINGS
# =========================

MODEL_PATH = "models/food_spoilage_model.keras"
TEST_DIR = os.path.join("dataset", "test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# =========================
# 2. LOAD TRAINED MODEL
# =========================

print("Loading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# 3. LOAD TEST DATASET
# =========================

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names

print("\nTest Classes:")
print(class_names)

# =========================
# 4. TEST ACCURACY
# =========================

print("\nEvaluating model on test dataset...\n")

test_loss, test_accuracy = model.evaluate(test_ds)

print("\n==============================")
print(" TEST RESULTS")
print("==============================")

print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")

# =========================
# 5. GET PREDICTIONS
# =========================

y_true = np.concatenate([
    labels.numpy()
    for images, labels in test_ds
])

predictions = model.predict(test_ds)

y_pred = np.argmax(predictions, axis=1)

# =========================
# 6. CLASSIFICATION REPORT
# =========================

print("\n==============================")
print(" CLASSIFICATION REPORT")
print("==============================")

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names
)

print(report)

# =========================
# 7. CONFUSION MATRIX
# =========================

cm = confusion_matrix(y_true, y_pred)

print("\n==============================")
print(" CONFUSION MATRIX")
print("==============================")

print(cm)

# =========================
# 8. SAVE CONFUSION MATRIX
# =========================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(10, 8))

disp.plot(
    ax=ax,
    xticks_rotation=45
)

plt.title("Food Spoilage Detection - Confusion Matrix")
plt.tight_layout()

plt.savefig("confusion_matrix.png", dpi=300)

print("\nConfusion matrix saved as:")
print("confusion_matrix.png")

plt.show()