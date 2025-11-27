import pandas as pd
from lightgbm import LGBMRegressor

def create_lag_features(df, date_col, value_col, lags=[1,7,14]):
    df = df.sort_values(date_col).copy()
    for l in lags:
        df[f"lag_{l}"] = df[value_col].shift(l)
    df = df.dropna()
    return df

def train_lgb(df, features, target):
    model = LGBMRegressor()
    model.fit(df[features], df[target])
    return model
