import gzip
import pathlib
import shutil
import urllib.request


class Downloader:

    @staticmethod
    def get(url, dest_dir="data/"):
        """Gets the file corresponding to the given URL.

        Args:
            url (str): URL of the file.
            dest_dir (str): Destination directory of the file. Defaults to
                "data/".

        """
        dest_dir = pathlib.Path(dest_dir)
        filename = pathlib.Path(dest_dir, url.split("/")[-1])
        if not dest_dir.exists():
            dest_dir.mkdir()
        urllib.request.urlretrieve(url, filename)

    @staticmethod
    def _unzip(filename):
        """Replaces the given file with an unzipped one.

        Args:
            filename (str): The file path.
        """
        filename = pathlib.Path(filename)
        new_filename = filename.with_suffix("")
        with gzip.open(filename, 'rb') as f_in:
            with open(new_filename, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        filename.unlink()

    @staticmethod
    def get_unzip(url, dest_dir="data/"):
        """Gets and unzips the file corresponding to the given URL.

        Keeps only the unzipped file.

        Args:
            url (str): URL of the file.
            dest_dir (str): Destination directory of the file. Defaults to
                "data/".

        """
        filename = pathlib.Path(dest_dir, url.split("/")[-1])
        Downloader.get(url, dest_dir)
        Downloader._unzip(filename)


if __name__ == "__main__":

    # example use
    url = 'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/AMAZON_FASHION_5.json.gz'
    Downloader.get_unzip(url)
