"""
heart_utils.py
Reusable, leakage-safe preprocessing + feature-engineering components for the
Heart Disease Prediction project. Imported by the notebook AND the Streamlit app
so that training and deployment use exactly the same transformations.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_FEATURES = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
                "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"]
TARGET = "HeartDisease"

CATEGORICAL = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
CONTINUOUS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak", "HR_Pct_Max"]
BINARY = ["FastingBS", "Chol_Missing", "ST_Depression", "Hypertensive", "Chol_High"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless feature engineering (no statistics learned from data -> no leakage).

    * Physiologically impossible zeros in Cholesterol / RestingBP -> NaN
      (imputed later *inside* the pipeline, using training data only).
    * Chol_Missing    : flag that Cholesterol was not recorded (0 in raw data)
    * HR_Pct_Max      : MaxHR / (220 - Age)  -> % of age-predicted max heart rate achieved
    * ST_Depression   : Oldpeak >= 1 mm (clinically significant ST depression)
    * Hypertensive    : RestingBP >= 140 mmHg
    * Chol_High       : Cholesterol >= 240 mg/dl (recorded values only)
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        chol_zero = X["Cholesterol"] == 0
        bp_zero = X["RestingBP"] == 0
        X["Chol_Missing"] = chol_zero.astype(int)
        X.loc[chol_zero, "Cholesterol"] = np.nan
        X.loc[bp_zero, "RestingBP"] = np.nan
        X["HR_Pct_Max"] = X["MaxHR"] / (220 - X["Age"])
        X["ST_Depression"] = (X["Oldpeak"] >= 1).astype(int)
        X["Hypertensive"] = (X["RestingBP"].fillna(0) >= 140).astype(int)
        X["Chol_High"] = (X["Cholesterol"].fillna(0) >= 240).astype(int)
        return X


class IQRClipper(BaseEstimator, TransformerMixin):
    """Winsorise outliers to [Q1 - k*IQR, Q3 + k*IQR]; bounds learned on TRAIN data only."""

    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(getattr(X, "columns", range(np.shape(X)[1])), dtype=object)
        X = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(X, 25, axis=0), np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return self.feature_names_in_
        return np.asarray(input_features, dtype=object)


def build_preprocessor(use_fe=True, clip_factor=1.5, scale=True):
    """FeatureEngineer -> ColumnTransformer(impute, clip, scale, one-hot)."""
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if clip_factor is not None:
        num_steps.append(("clip", IQRClipper(factor=clip_factor)))
    if scale:
        num_steps.append(("scale", StandardScaler()))
    num = Pipeline(num_steps)

    cont = CONTINUOUS if use_fe else ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
    binary = BINARY if use_fe else ["FastingBS"]
    ct = ColumnTransformer(
        [("num", num, cont),
         ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False), CATEGORICAL),
         ("bin", "passthrough", binary)],
        verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    # Even without engineered features we still need the zero->NaN handling
    return Pipeline([("fe", FeatureEngineer()), ("prep", ct)])


def make_model_pipeline(estimator, **prep_kwargs):
    return Pipeline([("pre", build_preprocessor(**prep_kwargs)), ("clf", estimator)])
