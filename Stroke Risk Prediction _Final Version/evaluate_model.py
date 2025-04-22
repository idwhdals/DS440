import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import load_model
import tensorflow as tf
import os

# Load model
model_path = os.path.join("Models", "BrainStrokeClassifier.keras")
model = load_model(model_path)

# Load datasets
DatasetDir = "./ct_images"

def preprocess_dataset(subfolder):
    dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DatasetDir, subfolder),
        image_size=(256, 256),
        batch_size=32
    ).map(lambda x, y: (x / 255.0, y))
    return dataset

datasets = {
    "Train": preprocess_dataset("Train"),
    "Validation": preprocess_dataset("Validation"),
    "Test": preprocess_dataset("Test")
}

# 📊 Metric calculation
for split, dataset in datasets.items():
    print(f"\n📌 Evaluating on {split} data:")

    precision_metric = Precision()
    recall_metric = Recall()
    accuracy_metric = BinaryAccuracy()
    cm = np.zeros((2, 2))

    for X_batch, y_batch in dataset.as_numpy_iterator():
        yhat = model.predict(X_batch)
        yhat_bin = (yhat > 0.5).astype(int)

        precision_metric.update_state(y_batch, yhat_bin)
        recall_metric.update_state(y_batch, yhat_bin)
        accuracy_metric.update_state(y_batch, yhat_bin)

        cm += confusion_matrix(y_batch, yhat_bin)

    # Metric Results
    precision = precision_metric.result().numpy()
    recall = recall_metric.result().numpy()
    accuracy = accuracy_metric.result().numpy()

    print(f"   🔹 Precision : {precision:.3f}")
    print(f"   🔹 Recall    : {recall:.3f}")
    print(f"   🔹 Accuracy  : {accuracy:.3f}")
    print("   🔹 Confusion Matrix:")
    print(cm.astype(int))

    # Confusion matrix plot
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm.astype(int), annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Normal", "Stroke"], yticklabels=["Normal", "Stroke"])
    plt.title(f"🧠 Confusion Matrix ({split})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()
