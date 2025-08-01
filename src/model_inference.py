import pickle
import os
import pandas as pd
import numpy as np

# Paths to trained models
MODEL_DIR = 'trained_models'
LGBM_MODEL_PATH = os.path.join(MODEL_DIR, 'lgbm_trading_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

# Global variables for loaded model and scaler
lgbm_model = None
scaler = None

def load_ml_models():
    global lgbm_model, scaler
    try:
        with open(LGBM_MODEL_PATH, 'rb') as f:
            lgbm_model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print("ML models and scaler loaded successfully.")
        return True
    except FileNotFoundError:
        print(f"ERROR: Model files not found. Ensure '{LGBM_MODEL_PATH}' and '{SCALER_PATH}' exist.")
        print("Please run 'ML_Model_Implementation.ipynb' first to train and save them.")
        return False
    except Exception as e:
        print(f"An error occurred loading models: {e}")
        return False

def get_ml_prediction(features_df):
    if lgbm_model is None or scaler is None:
        print("Error: ML models not loaded. Cannot make prediction.")
        return None, None

    if features_df is None or features_df.empty:
        return None, None

    # Scale the single data point using the *fitted* scaler
    scaled_features = scaler.transform(features_df)
    
    # Predict using the model
    prediction = lgbm_model.predict(scaled_features)[0]
    prediction_proba = lgbm_model.predict_proba(scaled_features)[0, 1]

    return prediction, prediction_proba