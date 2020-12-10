import download
import extract


class DatasetBuilder:

    def __init__(self, dest_dir="data/"):
        self._downloader = download.Downloader(dest_dir=dest_dir)
        self._extractor = extract.Extractor()

    def get_data(self,
                 url,
                 data_fields,
                 target_field,
                 drop_duplicates=False,
                 return_X_y=False):
        """Downloads JSON List file, unzips it, and extracts examples.

        Returns a pandas.core.frame.DataFrame object by default.

        Args:
            url (str): URL of the file.
            data_fields (List[str]): List of fields that correspond to the
                feature variables.
            target_field (str): Field that corresponds to the target variable.
            drop_duplicates (bool): Whether to drop duplicate examples.
                Defaults to False.
            return_X_y (bool): Whether to return data and target as seperate
                objects. Defaults to False.

        Returns:
            pandas.core.frame.DataFrame or pandas.core.series.Series object
            depending on len(target_fields) and return_X_y flag.

        """

        self._downloader.get_unzip(url)

        filename = self._downloader._url_to_filename(url).with_suffix("")

        data = self._extractor.extract_examples(filename=filename,
                                                data_fields=data_fields,
                                                target_field=target_field,
                                                metadata_filename=None,
                                                metadata_fields=None,
                                                return_X_y=return_X_y,
                                                drop_duplicates=drop_duplicates)

        return data


if __name__ == "__main__":

    # example use
    url = 'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/AMAZON_FASHION_5.json.gz'
    data_fields = ["reviewText"]
    target_field = "overall"
    dataset_builder = DatasetBuilder()
    X, y = dataset_builder.get_data(url,
                                    data_fields,
                                    target_field,
                                    drop_duplicates=True,
                                    return_X_y=True)
