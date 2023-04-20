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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from preprocess import preprocessing

# class build_model():
#     def __init__(self, X_train, y_train, learning_rate=0.1,
#                   n_estimaters = 100, subsample = 0.8,
#                   max_depth = 10, max_features = 'auto'):
#         self.X_train = X_train
#         self.y_train = y_train
#         self.learning_rate = learning_rate
#         self.n_estimaters = n_estimaters
#         self.subsample = subsample
#         self.max_depth = max_depth
#         self.max_features = max_features

# def model_training(file_name, learning_rate, n_estimaters,
#                    subsample, max_depth, max_features, random_state):

# model setup


def compute_rmsle(y_test: np.ndarray, y_pred:
                  np.ndarray, precision: int = 2) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)


def build_model(data_df: pd.DataFrame) -> dict:

    X_train, X_test, y_train, y_test = preprocessing(data_df)
    # X_train.replace(np.nan, 0, inplace=True)
    # X_test.replace(np.nan, 0, inplace=True)

    model = GradientBoostingRegressor(learning_rate=0.01,
                                      n_estimators=100,
                                      subsample=0.8,
                                      max_depth=10,
                                      max_features='auto',
                                      random_state=42)
    # train model
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmsle = compute_rmsle(y_test, y_pred)
    # return trained model
    model_matrix = {'model': model, 'mse': mse, 'rmsle': rmsle}
    return model_matrix
