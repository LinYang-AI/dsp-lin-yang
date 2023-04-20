import joblib
import numpy as np
import pandas as pd


ROOT = '../'
MODELS_DIR = ROOT + 'models'
SCALER_PATH = MODELS_DIR + '/scaler.joblib'
OHE_PATH = MODELS_DIR + '/ohe.joblib'

# class model_inference():
#     def __init__(self, file_name, model_name):
#         self.file_name = file_name
#         self.model_name = model_name


def make_predictions(test_df: pd.DataFrame) -> np.ndarray:
    # Load encoders and scaler
    scaler = joblib.load(SCALER_PATH)
    ohe = joblib.load(OHE_PATH)
    model = joblib.load(MODELS_DIR + '/model.joblib')
    # Data preprocessing
    X = test_df.copy()

    X_num_encoded = scaler.transform(
        X[X.select_dtypes(include=['int64', 'float64']).columns.tolist()])
    X_num_encoded_df = pd.DataFrame(
        X_num_encoded, columns=X.select_dtypes(include=['int64', 'float64']).
        columns.tolist())
    X_cat_encoded = ohe.transform(
        X[X.select_dtypes(include=['object']).columns.tolist()])
    X_cat_encoded_df = pd.DataFrame.sparse.from_spmatrix(
        data=X_cat_encoded, columns=ohe.get_feature_names_out())
    X_encoded = X_num_encoded_df.join(X_cat_encoded_df)
    X_encoded.replace(np.nan, 0, inplace=True)
    # Make prediction
    prediction = model.predict(X_encoded)

    return prediction
