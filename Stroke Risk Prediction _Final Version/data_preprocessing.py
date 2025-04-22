import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

DatasetDir = "./ct_images"

# 📦 Load datasets
TrainingData = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DatasetDir, "Train"),
    image_size=(256, 256),
    batch_size=32
).map(lambda x, y: (x / 255.0, y))

TestingData = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DatasetDir, "Test"),
    image_size=(256, 256),
    batch_size=32
).map(lambda x, y: (x / 255.0, y))

ValidationData = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DatasetDir, "Validation"),
    image_size=(256, 256),
    batch_size=32
).map(lambda x, y: (x / 255.0, y))

# 🧠 Functional API model (Grad-CAM 호환)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard

inputs = Input(shape=(256, 256, 3))
x = Conv2D(16, (3, 3), activation='relu', name="conv2d_0")(inputs)
x = MaxPooling2D(name="maxpool_0")(x)
x = Conv2D(32, (3, 3), activation='relu', name="conv2d_1")(x)
x = MaxPooling2D(name="maxpool_1")(x)
x = Conv2D(16, (3, 3), activation='relu', name="conv2d_2")(x)
x = MaxPooling2D(name="maxpool_2")(x)
x = Flatten(name="flatten")(x)
x = Dense(256, activation='relu', name="dense_1")(x)
outputs = Dense(1, activation='sigmoid', name="output")(x)

Model = Model(inputs=inputs, outputs=outputs)

Model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
Model.summary()

# 🏋️‍♂️ Training
LogDir = 'logs'
History = Model.fit(
    TrainingData,
    epochs=10,
    validation_data=ValidationData,
    callbacks=[TensorBoard(log_dir=LogDir)]
)

# 📈 Visualization
plt.figure()
plt.plot(History.history['loss'], label='Loss')
plt.plot(History.history['val_loss'], label='Val Loss')
plt.title("Loss over Epochs")
plt.legend()
plt.show()

plt.figure()
plt.plot(History.history['accuracy'], label='Accuracy')
plt.plot(History.history['val_accuracy'], label='Val Accuracy')
plt.title("Accuracy over Epochs")
plt.legend()
plt.show()

# 🧪 Model Evaluation
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from sklearn.metrics import confusion_matrix
import seaborn as sns

PrecisionMetric = Precision()
RecallMetric = Recall()
AccuracyMetric = BinaryAccuracy()
cm = np.zeros((2, 2))

for X, y in TestingData.as_numpy_iterator():
    yhat = Model.predict(X)
    yhat_bin = (yhat > 0.5).astype(int)
    PrecisionMetric.update_state(y, yhat_bin)
    RecallMetric.update_state(y, yhat_bin)
    AccuracyMetric.update_state(y, yhat_bin)
    cm += confusion_matrix(y, yhat_bin)

print("📊 Confusion Matrix:\n", cm.astype(int))
sns.heatmap(cm.astype(int), annot=True, fmt="d", xticklabels=["Normal", "Stroke"], yticklabels=["Normal", "Stroke"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

print(f"✅ Precision : {PrecisionMetric.result().numpy():.3f}")
print(f"✅ Recall    : {RecallMetric.result().numpy():.3f}")
print(f"✅ Accuracy  : {AccuracyMetric.result().numpy():.3f}")

# 📌 Test with single image
import cv2

test_image_path = "./ct_images/Test/Normal/1 (24).JPG"
Img = cv2.imread(test_image_path)
Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
ResizedImg = tf.image.resize(Img, (256, 256))
Input = np.expand_dims(ResizedImg / 255.0, 0)

yhat = Model.predict(Input)
print(f"\n📌 Prediction Score: {yhat[0][0]:.3f}")
print("📌 Predicted Class:", "🧠 Stroke" if yhat > 0.5 else "✅ Normal")

# 💾 Save model (for Grad-CAM)
os.makedirs("Models", exist_ok=True)
Model.save(os.path.join("Models", "BrainStrokeClassifier.keras"))
