import requests
from tqdm import tqdm
import os
import logging


CACHE_DIR=".datasets/"


def download_file(url, dest):
    if not os.path.isfile(dest):
        logging.info("downloading", url, "to", dest)
        with requests.get(url, stream=True) as stream:
            stream.raise_for_status()
            total_size_in_bytes= int(stream.headers.get('content-length', 0))
            chunk_size = 1024*1024
            progress_bar = tqdm(total=total_size_in_bytes, unit='B', unit_scale=True)
            with open(dest, 'wb') as fp:
                for chunk in stream.iter_content(chunk_size): 
                    progress_bar.update(len(chunk))
                    fp.write(chunk)
            progress_bar.close()


class Wikipedia(object):

    def __init__(self, date):
        self.date = date
        self.url = "https://dumps.wikimedia.org/enwiki/{}/enwiki-{}-pages-articles-multistream.xml.bz2".format(date, date)
        self.cache_dir = os.path.join(".datasets/wikipedia")
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.dump_file = os.path.join(self.cache_dir, os.path.basename(self.url))

    def metadata(self):
        return {
            "version": 1,
            "parameters": {
                "date": self.date,
                "url": self.url
            }
        }

    def preprocess(self):
        download_file(self.url, self.dump_file)


if __name__ == "__main__":
    from pprint import pprint
    wiki = Wikipedia("20210120")
    pprint(wiki.metadata())
    wiki.preprocess()

