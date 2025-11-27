import pandas as pd
import io

def load_csv_from_fileobj(fileobj, **kwargs):
    fileobj.seek(0)
    # Use TextIOWrapper so pandas can read fileobj bytes
    return pd.read_csv(io.TextIOWrapper(fileobj, encoding=kwargs.pop("encoding", "utf-8")), low_memory=False, **kwargs)
