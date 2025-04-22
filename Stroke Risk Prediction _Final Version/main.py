from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import os
from PIL import Image
from io import BytesIO

from gradcam import make_gradcam_heatmap

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

model_path = "Models/BrainStrokeClassifier.keras"
model = load_model(model_path)

model = load_model("Models/BrainStrokeClassifier.keras")
model(np.zeros((1, 256, 256, 3)))  # <--- 꼭 이걸로 초기화! predict() 말고 __call__()


def preprocess_image(uploaded_file) -> np.ndarray:
    image = Image.open(BytesIO(uploaded_file)).convert("RGB")
    image = image.resize((256, 256))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        with open("debug_input.jpg", "wb") as f:
            f.write(contents)

        input_tensor = preprocess_image(contents)

        print("✅ Input tensor shape:", input_tensor.shape)
        print("✅ Input tensor NaNs?:", np.isnan(input_tensor).any())

        prediction = model.predict(input_tensor)
        print("✅ Raw model output:", prediction)

        pred_score = float(prediction[0][0])
        if np.isnan(pred_score):
            print("⚠️ Warning: Prediction score is NaN!")

        # Grad-CAM
        make_gradcam_heatmap(model, input_tensor, last_conv_layer_name="conv2d_2")

        result = "Stroke" if pred_score > 0.5 else "Normal"

        return JSONResponse(content={
            "prediction_score": round(pred_score, 4),
            "predicted_class": result
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
