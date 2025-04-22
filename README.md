# Brain Stroke Prediction Using CT Images

## DS440W Capstone Project


## Overview

This project provides a web-based tool to predict brain stroke from CT scan images using a deep learning model. 
The system offers visual explanations (via Grad-CAM) to increase interpretability and is designed to support 
real-time image evaluation using a FastAPI backend.


## Features

- Upload CT images and receive AI-generated predictions (Stroke / Normal)
- Visual explanation with Grad-CAM to highlight stroke regions
- Web-based interface for seamless interaction
- Supports real-time inference using pre-trained models


## Repository Structure

| Folder/File             | Description |
|-------------------------|-------------|
| `CT_Images/`            | Contains training, validation, and testing CT images *(stored externally via [Google Drive](https://drive.google.com/drive/folders/1jQNXy4npUp6VJsmkpkWIDlopd-junDMe?usp=sharing))*  
| `static/`               | Frontend UI files (html)  
| `main.py`               | FastAPI backend main entrypoint  
| `gradcam.py`            | Generates Grad-CAM heatmaps  
| `evaluate_model.py`     | Evaluates model accuracy and metrics  
| `data_preprocessing.py` | Image preprocessing and pipeline functions  
| `setup_fastapi.sh`      | Bash script to launch the app  
| `requirements.txt`      | Lists all required Python dependencies for the project  
| `.gitignore`            | Git tracking exclusions  
| `README.md`             | Project documentation (this file)



## Quick Start

# 1. Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start server
python main.py






