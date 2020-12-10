import json

import pandas as pd


class Extractor:

    @staticmethod
    def extract_examples(filename,
                         data_fields,
                         target_field,
                         metadata_filename=None,
                         metadata_fields=None,
                         drop_duplicates=False,
                         return_X_y=False):
        """Extracts examples from a JSON Lines file.

        Returns a pandas.core.frame.DataFrame object by default.

        Args:
            filename (str): File path of the JSON Lines file.
            data_fields (List[str]): List of fields that correspond to the
                feature variables.
            target_field (str): Field that corresponds to the target variable.
            metadata_filename (str, optional): File path of JSON Lines
                metadata file.
            metadata_fields (List[str], optional): List of fields that
                correspond to additional feature variables.
            drop_duplicates (bool): Whether to drop duplicate examples.
                Defaults to False.
            return_X_y (bool): Whether to return data and target as seperate
                objects. Defaults to False.

        Returns:
            pandas.core.frame.DataFrame or pandas.core.series.Series object
            depending on len(target_fields) and return_X_y flag.

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

        if drop_duplicates:
            df.drop_duplicates(inplace=True)

        if return_X_y:
            if len(data_fields) == 1:
                X = df[data_fields[0]]
            else:
                X = df[data_fields]
            y = df[target_field]
            return X, y

        return df


if __name__ == "__main__":

    # example use
    filename = "data/AMAZON_FASHION_5.json"
    data_fields = ["reviewText"]
    target_field = "overall"
    X, y = Extractor.extract_examples(filename,
                                      data_fields,
                                      target_field,
                                      drop_duplicates=True,
                                      return_X_y=True)
