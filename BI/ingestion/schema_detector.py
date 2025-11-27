import pandas as pd
import re

def detect_schema(df):
    schema = {}
    for col in df.columns:
        col_low = str(col).lower()
        vals = df[col].dropna()
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        is_date = False
        try:
            if len(vals) > 0:
                pd.to_datetime(vals.sample(min(len(vals), 20)))
                is_date = True
        except Exception:
            is_date = False
        schema[col] = {
            "is_date": is_date,
            "is_numeric": bool(is_numeric),
            "is_currency": bool(re.search(r"amount|price|cost|revenue|total", col_low)),
        }
    return schema
