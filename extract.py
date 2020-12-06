import json
import pandas as pd
import numpy as np

class Extractor:
    @staticmethod
    def extract_examples(filename, metadata_filename=None, metadata_fields=None, return_X_y=False):
        """
        Extract examples as pairs of "reviewText" and "overall". By default returns a Pandas DataFrame.
        """
        data = []
        with open(filename) as f:
            for line in f:
                data.append(json.loads(line))
        df = pd.DataFrame.from_records(data)
        if metadata_filename or metadata_fields:
            raise NotImplementedError("Adding metadata to extracted examples is currently not supported.")
        df = df[["reviewText", "overall"]].dropna()
        if return_X_y:
            X = df["reviewText"].to_list()
            y = df["overall"].to_list()
            return X, y
        return df

if __name__ == "__main__":
    # example use
    filename = "AMAZON_FASHION_5.json"
    X, y = Extractor.extract_examples(filename, return_X_y=True)
