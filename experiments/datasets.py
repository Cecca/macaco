import msgpack
import gzip
import numpy as np
import zipfile
import multiprocessing
from gensim.models.ldamulticore import LdaMulticore
from gensim.corpora.wikicorpus import WikiCorpus
from gensim.corpora.dictionary import Dictionary
import requests
from tqdm import tqdm
import os
import logging
import random

logging.getLogger().setLevel(logging.INFO)


# The local cache directory for datasets
CACHE_DIR = ".datasets/"
# The url for looking for already-preprocessed datasets
CACHE_URL = "https://www.inf.unibz.it/~ceccarello/data"


def download_file(url, dest):
    if not os.path.isfile(dest):
        logging.info("downloading %s to %s", url, dest)
        with requests.get(url, stream=True) as stream:
            stream.raise_for_status()
            total_size_in_bytes = int(stream.headers.get("content-length", 0))
            chunk_size = 1024 * 1024
            progress_bar = tqdm(total=total_size_in_bytes, unit="B", unit_scale=True)
            with open(dest, "wb") as fp:
                for chunk in stream.iter_content(chunk_size):
                    progress_bar.update(len(chunk))
                    fp.write(chunk)
            progress_bar.close()


class Dataset(object):
    """Base class providing common functionality for datasets"""

    def metadata(self):
        raise NotImplementedError()

    def preprocess(self):
        raise NotImplementedError()

    def get_path(self):
        raise NotImplementedError()

    def try_download_preprocessed(self):
        dirname, basename = os.path.split(self.get_path())
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        try:
            download_file(os.path.join(CACHE_URL, basename), self.get_path())
        except Exception as e:
            logging.warn(e)
            logging.warn("dataset not found online, computing locally")
            return False
        return True

    def __iter__(self):
        with gzip.open(self.get_path(), "rb") as fp:
            unpacker = msgpack.Unpacker(fp, raw=False)
            # Skipt the metadata
            next(unpacker)
            # Iterate through the elements
            for doc in unpacker:
                yield doc

    def num_elements(self):
        cnt = 0
        for doc in self:
            cnt += 1
        return cnt


class GloveMap(object):
    URL = "http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"
    CACHE = os.path.join(CACHE_DIR, "glove", os.path.basename(URL))

    def __init__(self, dimensions):
        assert dimensions in [50, 100, 200, 300]
        self.mapping = {}
        self.dimension = dimensions
        if not os.path.isfile(GloveMap.CACHE):
            download_file(GloveMap.URL, GloveMap.CACHE)
        with zipfile.ZipFile(GloveMap.CACHE) as zipfp:
            with zipfp.open("glove.6B.{}d.txt".format(dimensions)) as fp:
                for line in fp.readlines():
                    tokens = line.decode("utf-8").split()
                    self.mapping[tokens[0]] = np.array([float(t) for t in tokens[1:]])
                    assert self.dimension == len(tokens) - 1
        logging.info("Glove map with {} entries".format(len(self.mapping)))

    def get(self, word):
        return self.mapping.get(word)

    def map_bow(self, mapping, bow):
        vec = np.zeros(self.dimension)
        cnt = 0
        for (word_idx, count) in bow:
            wordvec = self.get(mapping[word_idx])
            if wordvec is not None:
                vec += wordvec * count
                cnt += 1
            else:
                pass
        if cnt > 0:
            return vec / np.float(cnt)
        else:
            return None


class CachedBowsCorpus(object):
    """Cache of bag of word for a corpus

    Iterating through the wiki corpus to retrieve the bag of
    words representation for each page is *extremely* slow.
    Therefore we do it once and cache it to a file, which has
    10x more throughput once cached.
    """

    def __init__(self, wiki, path, meta):
        self.meta = meta
        self.path = path
        if not os.path.isfile(path):
            logging.info("create cache of bag of words")
            progress_bar = tqdm(unit="pages")
            wiki.metadata = True
            with open(path, "wb") as fp:
                for doc in wiki:
                    progress_bar.update(1)
                    fp.write(msgpack.packb(doc))
            progress_bar.close()

    def __iter__(self):
        with open(self.path, "rb") as fp:
            progress_bar = tqdm(unit="pages")
            unpacker = msgpack.Unpacker(fp, raw=False)
            for (doc, meta) in unpacker:
                progress_bar.update(1)
                if self.meta:
                    yield (doc, meta)
                else:
                    yield doc
            progress_bar.close()


class Wikipedia(Dataset):
    version = 1

    def __init__(self, date, dimensions, topics):
        self.date = date
        self.dimensions = dimensions
        self.topics = topics
        self.url = "https://dumps.wikimedia.org/enwiki/{}/enwiki-{}-pages-articles-multistream.xml.bz2".format(
            date, date
        )
        self.cache_dir = os.path.join(".datasets/wikipedia", date)
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.dump_file = os.path.join(self.cache_dir, os.path.basename(self.url))
        self.dictionary = os.path.join(self.cache_dir, "dictionary")
        self.lda_model_path = os.path.join(
            self.cache_dir, "model-lda-{}".format(self.topics)
        )
        self.bow_cache = os.path.join(self.cache_dir, "bows.msgpack")
        self.out_fname = os.path.join(
            self.cache_dir,
            "wiki-d{}-c{}-v{}.msgpack.gz".format(
                self.dimensions, self.topics, Wikipedia.version
            ),
        )

    def get_path(self):
        return self.out_fname

    def metadata(self):
        meta = {
            "name": "Wikipedia",
            "constraint": {"transversal": {"topics": list(range(0, self.topics))}},
            "version": Wikipedia.version,
            "parameters": {
                "dimensions": self.dimensions,
                "topics": self.topics,
                "date": self.date,
            },
            "url": self.url,
        }
        return meta

    def load_lda(self, docs, dictionary):
        cores = multiprocessing.cpu_count()
        if not os.path.exists(self.lda_model_path):
            logging.info("Training LDA")
            lda = LdaMulticore(
                docs,
                id2word=dictionary,
                num_topics=self.topics,
                passes=1,
                workers=cores,
            )
            logging.info("Saving LDA")
            lda.save(self.lda_model_path)
        else:
            logging.info("Model file found, loading")
            lda = LdaMulticore.load(self.lda_model_path)
        return lda

    def preprocess(self):
        if not os.path.isfile(self.out_fname):
            self.do_preprocessing()

    def do_preprocessing(self):
        cores = multiprocessing.cpu_count()
        download_file(self.url, self.dump_file)
        if not os.path.isfile(self.dictionary):
            logging.info("Creating gensim dictionary (takes a long time)")
            wiki = WikiCorpus(self.dump_file)
            wiki.dictionary.save(self.dictionary)
        dictionary = Dictionary.load(self.dictionary)
        wiki = WikiCorpus(self.dump_file, dictionary=dictionary)
        wiki.metadata = True

        logging.info("Loading word embeddings")
        glove = GloveMap(self.dimensions)
        logging.info("Setting up LDA")
        lda = self.load_lda(
            CachedBowsCorpus(wiki, self.bow_cache, meta=False), dictionary
        )

        logging.info("Remapping vectors")
        with gzip.open(self.out_fname, "wb") as out_fp:
            header = msgpack.packb(self.metadata())
            out_fp.write(header)
            for (bow, (id, title)) in CachedBowsCorpus(wiki, self.bow_cache, meta=True):
                vector = list(glove.map_bow(dictionary, bow))
                if vector is not None:
                    topics = lda.get_document_topics(bow, minimum_probability=0.1)
                    outdata = {
                        "id": int(id),
                        "title": title,
                        "topics": [p[0] for p in topics],
                        "vector": vector,
                    }
                    encoded = msgpack.packb(outdata)
                    out_fp.write(encoded)


class SampledDataset(Dataset):
    version = 1

    def __init__(self, base, size, seed):
        self.base = base
        self.size = size
        self.seed = seed
        self.cache_dir = os.path.join(".datasets/sampled")
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)
        params_list = list(base.metadata()["parameters"].items())
        params_list.sort()
        params_str = "-".join(["{}-{}".format(k, v) for k, v in params_list])
        self.path = os.path.join(
            self.cache_dir,
            "{}-{}-sample{}-v{}.msgpack.gz".format(
                base.metadata()["name"], params_str, size, SampledDataset.version
            ),
        )

    def get_path(self):
        return self.path

    def metadata(self):
        parameters = self.base.metadata()["parameters"]
        for k in parameters:
            parameters[k] = str(parameters[k])
        parameters.update(
            {
                "size": str(self.size),
                "seed": str(self.seed),
                "base_version": str(self.base.metadata()["version"]),
            }
        )
        return {
            "name": "{}-sample-{}".format(self.base.metadata()["name"], self.size),
            "parameters": parameters,
            "version": SampledDataset.version,
        }

    def preprocess(self):
        if not os.path.isfile(self.path):
            self.base.preprocess()
            logging.info("file %s is missing", self.path)
            logging.info(
                "preprocessing sampled dataset with sample size %d from %s",
                self.size,
                self.base.metadata()["name"],
            )
            n = self.base.num_elements()
            p = min(self.size / n, 1)
            random.seed(self.seed)

            progress_bar = tqdm(total=n, unit="pages", unit_scale=False)
            with gzip.open(self.path, "wb") as out_fp:
                header = msgpack.packb(self.metadata())
                out_fp.write(header)
                for doc in self.base:
                    progress_bar.update(1)
                    if random.random() <= p:
                        out_fp.write(msgpack.packb(doc))
            progress_bar.close()


DATASETS = {"wiki-d50-c100": Wikipedia("20210120", dimensions=50, topics=100)}

# Sampled datasets
for size in [100000]:
    DATASETS["wiki-d50-c100-s100000"] = SampledDataset(
        base=DATASETS["wiki-d50-c100"], size=size, seed=12341245
    )

if __name__ == "__main__":
    dataset = DATASETS["wiki-d50-c100"]
    dataset.try_download_preprocessed()
    dataset.preprocess()
    print(dataset.get_path())