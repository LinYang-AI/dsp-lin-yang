import joblib
import numpy as np
import pandas as pd
from preprocess import preprocessing

ROOT = '../'
MODELS_DIR = ROOT + 'models'
SCALER_PATH = MODELS_DIR + '/scaler.joblib'
OHE_PATH = MODELS_DIR + '/ohe.joblib'


def make_predictions(test_df: pd.DataFrame) -> np.ndarray:
    model = joblib.load(MODELS_DIR)
    X = test_df.copy()
    X_encoded = preprocessing(X, training_mode=False)
    prediction = model.predict(X_encoded)

    return prediction
