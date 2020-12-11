import gzip
import pathlib
import shutil
import urllib.request


class Downloader:
    """Class for downloading files in a specific directory.

    Attributes:
        dest_dir: The destination directory where every file is stored.

    """

    def __init__(self, dest_dir="data/"):
        self.dest_dir = pathlib.Path(dest_dir)

    def _url_to_filename(self, url):
        """Converts file url to file path.

        Args:
            url (str): URL of the file.

        """

        return self.dest_dir.joinpath(url.split("/")[-1])

    def get(self, url):
        """Gets the file corresponding to the given URL.

        Args:
            url (str): URL of the file.

        """

        filename = self._url_to_filename(url)

        if filename.exists():
            return

        if not self.dest_dir.exists():
            self.dest_dir.mkdir()

        urllib.request.urlretrieve(url, filename)

    def _unzip(self, filename):
        """Replaces the given file with an unzipped one.

        Args:
            filename (pathlib.Path): The file path.
        """

        new_filename = filename.with_suffix("")

        with gzip.open(filename, 'rb') as f_in:
            with open(new_filename, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        filename.unlink()

    def get_unzip(self, url):
        """Gets and unzips the file corresponding to the given URL.

        Keeps only the unzipped file. 

        Args:
            url (str): URL of the file.

        """

        filename = self._url_to_filename(url)

        if filename.with_suffix("").exists():
            return

        self.get(url)
        self._unzip(filename)


if __name__ == "__main__":

    # example use
    url = 'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/AMAZON_FASHION_5.json.gz'
    downloader = Downloader()
    downloader.get_unzip(url)
