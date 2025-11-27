import pdfplumber
import pandas as pd
import io

def extract_tables_from_pdf_bytes(b):
    tables = []
    with pdfplumber.open(io.BytesIO(b)) as pdf:
        for page in pdf.pages:
            raw = page.extract_tables()
            for t in raw:
                # table rows as lists of strings; first row assumed header
                if len(t) < 2:
                    continue
                df = pd.DataFrame(t[1:], columns=t[0])
                tables.append(df)
    return tables
