# Load csv as pandas DataFrame

import pandas as pd
from helper.validate import is_nan_value
from resources.constants import EMBED_COLUMN_NAME

def load_data(file_path, embed_columns):
    chunks = []
    # Combine all column in embed_columns to one column name "embed_content"


    df = pd.read_csv(file_path)
    df[EMBED_COLUMN_NAME] = df[embed_columns].apply(lambda row: " ".join(row.values.astype(str)), axis=1)
    data_dict = df.to_dict(orient="records")

    for record in data_dict:
        # Loop through keys in record 
        for key in record:
            if is_nan_value(record[key]):
                record[key] = ""
        chunks.append(record)
    return chunks
