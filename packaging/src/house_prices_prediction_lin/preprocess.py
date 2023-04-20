"""_summary_
This is a module that contains two functions which does the feature
engineering and preprocessing, respectively.
Returns:
List: a list of pandas DataFrames that are X_train, X_test, y_train, y_test
"""
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

ROOT = '../'
MODELS_DIR = ROOT + 'models'
SCALER_PATH = MODELS_DIR + '/scaler.joblib'
OHE_PATH = MODELS_DIR + '/ohe.joblib'


"""_summary_
This function does feature engineering
Returns:
List: a list of lists that are categorical features,
numerical features, and labels, respectively.
"""


def feature_engineering(data_df: pd.DataFrame) -> list:
    # remove columns that only have one unique value
    col_drop = data_df.columns[data_df.nunique() == 1]
    data_df.drop(col_drop, axis=1, inplace=True)
    # Seperate the numerical and categorical features from label
    CATEGORICAL_COLUMNS = data_df.select_dtypes(
        include=['object']).columns.tolist()
    NUMERICAL_COLUMNS = data_df.drop(['SalePrice'], axis=1).select_dtypes(
        include=['int64', 'float64']).columns.tolist()
    LABEL_COLUMN = data_df.drop(CATEGORICAL_COLUMNS, axis=1).drop(
        NUMERICAL_COLUMNS, axis=1).columns.tolist()
    FEATURE_LIST = [CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS, LABEL_COLUMN]

    return FEATURE_LIST


"""_summary_
This is a function that encode the data with sklearn.StandardScaler,
and sklearn.OneHotEncoder
Returns:
List: a list of pandas DataFrame
"""


def num_scaler(data: pd.DataFrame):
    scaler = StandardScaler()
    scaler.fit(data)
    numerical_features_scaled = scaler.transform(data)
    numerical_features_scaled_df = pd.DataFrame(
        data=numerical_features_scaled,
        columns=data.columns
    )
    return scaler, numerical_features_scaled_df


def ohe_encoder(data: pd.DataFrame):
    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe.fit(data)
    categorical_featuers_encoded = ohe.transform(data)
    categorical_featuers_encoded_df = pd.DataFrame(
        data=categorical_featuers_encoded,
        columns=data.columns
    )
    return ohe, categorical_featuers_encoded_df


def preprocessing(data_df: pd.DataFrame) -> list:

    FEATURE_LIST = feature_engineering(data_df)

    # Preprocessing the dataset
    scaler, numerical_features_scaled_df = num_scaler(data_df[FEATURE_LIST[1]])
    joblib.dump(scaler, SCALER_PATH)
    ohe, categorical_features_encoded_df = ohe_encoder(
        data_df[FEATURE_LIST[0]])
    joblib.dump(ohe, OHE_PATH)

    data_encoded = numerical_features_scaled_df.join(
        categorical_features_encoded_df).join(data_df[FEATURE_LIST[2]])

    X, y = data_encoded.drop(
        FEATURE_LIST[2], axis=1), data_encoded[FEATURE_LIST[2]]
    # To replace all the NaN values with 0
    X.replace(np.nan, 0, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
