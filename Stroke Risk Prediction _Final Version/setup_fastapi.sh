#!/bin/zsh

echo "📁 [1] Creating virtual environment..."
python3 -m venv venv

echo "🚀 [2] Activating virtual environment..."
source venv/bin/activate

echo "📦 [3] Installing required packages..."
pip install --upgrade pip
pip install fastapi uvicorn python-multipart tensorflow pillow

echo "✅ All dependencies installed!"
echo "🌐 Launching FastAPI server (http://127.0.0.1:8000/docs)..."
uvicorn main:app --reload

