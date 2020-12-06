import urllib.request
import gzip
import shutil
import os

class Downloader:
    @staticmethod
    def get(url):
        """
        Get and unzip the jsonlist file from the given url. Keep only the unzipped file.
        """
        filename = url.split("/")[-1]
        urllib.request.urlretrieve(url, filename)
        with gzip.open(filename, 'rb') as f_in:
            with open(filename.replace(".gz", ""), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(filename)

if __name__ == "__main__":
    # example use
    url = 'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/AMAZON_FASHION_5.json.gz'
    Downloader.get(url)
