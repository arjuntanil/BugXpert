"""
This file is used for deployment on Render.
It ensures the models are loaded and ready before serving the application.
"""

import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask

# Import the app
from app import app as application

# Ensure the app static directories exist
os.makedirs('app/static/images', exist_ok=True)

# Check if models exist, if not, create them
if not os.path.exists("code_quality_knn_model.pkl"):
    print("Initializing KNN model...")
    import knn_quality_model

if __name__ == "__main__":
    # This is used when running locally only.
    # When deploying to Render, Render will use gunicorn 
    # which uses the app variable in app.py
    port = int(os.environ.get("PORT", 5000))
    application.run(host='0.0.0.0', port=port) 