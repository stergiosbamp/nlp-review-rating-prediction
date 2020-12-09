import json

import pandas as pd


class Extractor:

    @staticmethod
    def extract_examples(filename,
                         data_fields,
                         target_field,
                         metadata_filename=None,
                         metadata_fields=None,
                         return_X_y=False):
        """Extracts examples from a JSON Lines file.

        Returns a Pandas DataFrame by default.

        Args:
            filename (str): File path of the JSON Lines file.
            data_fields (List[str]): List of fields that correspond to the
                feature variables.
            target_field (str): Field that corresponds to the target variable.
            metadata_filename (str, optional): File path of JSON Lines
                metadata file.
            metadata_fields (List[str], optional): List of fields that
                correspond to additional feature variables.
            return_X_y (bool): Whether to return a tuple of numpy.ndarray
                instead of a pandas.DataFrame.
        
        Returns:
            By defalut a pandas.DataFrame. Otherwise if return_X_y is True, a
                tuple of numpy.ndarray.
        
        """

        if metadata_filename or metadata_fields:
            raise NotImplementedError(
                ("Adding metadata to extracted examples is currently "
                 "not supported."))

        data = []
        with open(filename) as f:
            for line in f:
                data.append(json.loads(line))

        df = pd.DataFrame.from_records(data)
        df = df[data_fields + [target_field]].dropna()

        if return_X_y:
            X = df[data_fields].values
            y = df[target_field].values
            return X, y

        return df


if __name__ == "__main__":

    # example use
    filename = "AMAZON_FASHION_5.json"
    data_fields = ["reviewText"]
    target_field = "overall"
    X, y = Extractor.extract_examples(filename,
                                      data_fields,
                                      target_field,
                                      return_X_y=True)
