# Brain Stroke Prediction Using CT Images


## 📝 Overview

This project provides a web-based tool to predict brain stroke from CT scan images using a deep learning model. 
The system offers visual explanations (via Grad-CAM) to increase interpretability and is designed to support 
real-time image evaluation using a FastAPI backend.


## ⚙️ Features

- Upload CT images and receive AI-generated predictions (Stroke / Normal)
- Visual explanation with Grad-CAM to highlight stroke regions
- Web-based interface for seamless interaction
- Supports real-time inference using pre-trained models


## 🗂 Project Files Explained

- **main.py**  
  → The main FastAPI server that handles requests and responses
- **gradcam.py**  
  → Generates Grad-CAM visual explanations from the model
- **evaluate_model.py**  
  → Evaluates model accuracy, precision, recall, etc.
- **data_preprocessing.py**  
  → Prepares and cleans CT image data for prediction
- **templates/index.html**  
  → The HTML page users see (uploaded via FastAPI’s Jinja2)
- **static/**  
  → Frontend resources – index.html
- **ct_images/**  
  → Contains all the CT scan images used for training, validation, and testing

pip install -r requirements.txt

python main.py
