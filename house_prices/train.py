"""_summary_
This module contains two methods, one for calculating the root
mean squared log error, and one for training the model
Returns:
    dictionary: the module returns a dictionary of objects that
    are the model itself,mean squared error, and the root mean
    squared log error
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from preprocess import preprocessing

ROOT = '../'
MODELS_DIR = ROOT + 'models.jblib'

def compute_rmsle(y_test: np.ndarray, y_pred:
                  np.ndarray, precision: int = 2) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)


def build_model(data_df: pd.DataFrame) -> dict:

    X_train, X_test, y_train, y_test = preprocessing(data_df)

    model = GradientBoostingRegressor(learning_rate=0.01,
                                      n_estimators=100,
                                      subsample=0.8,
                                      max_depth=10,
                                      max_features='auto',
                                      random_state=42)
    # train model
    model.fit(X_train, y_train)
    # save model
    joblib.dump(model, MODELS_DIR)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmsle = compute_rmsle(y_test, y_pred)
    # return trained model
    model_matrix = {'model': model, 'mse': mse, 'rmsle': rmsle}
    return model_matrix
