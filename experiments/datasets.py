import subprocess
import msgpack
import gzip
import zipfile
import multiprocessing
import requests
from tqdm import tqdm
import os
import logging
import random
import csv

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

    def build_metadata(self):
        raise NotImplementedError()

    def preprocess(self):
        raise NotImplementedError()

    def get_path(self):
        raise NotImplementedError()

    def metadata(self):
        current_metadata = self.build_metadata()
        if os.path.isfile(self.get_path()):
            with gzip.open(self.get_path(), "rb") as fp:
                unpacker = msgpack.Unpacker(fp)
                file_metadata = next(unpacker)
            if current_metadata != file_metadata:
                logging.warning("metadata changed wrt to the file, updating")
                # load the old data
                with gzip.open(self.get_path(), "rb") as fp:
                    unpacker = msgpack.Unpacker(fp)
                    next(unpacker)  # skip metadata
                    data = [d for d in tqdm(unpacker)]
                # overwrite the file
                with tqdm(total=len(data), unit="items") as pb:
                    with gzip.open(self.get_path(), "wb") as fp:
                        msgpack.pack(current_metadata, fp)
                        for d in data:
                            pb.update(1)
                            msgpack.pack(d, fp)

        return current_metadata

    def write_metadata(self, fp):
        msgpack.pack(self.build_metadata(), fp)

    def try_download_preprocessed(self):
        dirname, basename = os.path.split(self.get_path())
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        try:
            download_file(os.path.join(CACHE_URL, basename), self.get_path())
        except Exception as e:
            logging.warning(e)
            logging.warning("dataset not found online, computing locally")
            return False
        return True

    def __iter__(self):
        with gzip.open(self.get_path(), "rb") as fp:
            unpacker = msgpack.Unpacker(fp, raw=False)
            # Skip the metadata
            next(unpacker)
            # Iterate through the elements
            for doc in unpacker:
                yield doc

    def num_elements(self):
        cnt = 0
        for doc in self:
            cnt += 1
        return cnt


class WordEmbeddingMap(object):
    URL = "http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"
    CACHE = os.path.join(CACHE_DIR, "glove", os.path.basename(URL))

    def __init__(self, dimensions, wiki=None):
        import numpy as np

        self.mapping = {}
        if dimensions in [50, 100, 200, 300]:
            file_dimensions = dimensions
        elif dimensions < 50:
            # and then truncate
            file_dimensions = 50
        else:
            raise ValueError
        self.dimension = dimensions
        if not os.path.isfile(WordEmbeddingMap.CACHE):
            download_file(WordEmbeddingMap.URL, WordEmbeddingMap.CACHE)
        with zipfile.ZipFile(WordEmbeddingMap.CACHE) as zipfp:
            with zipfp.open("glove.6B.{}d.txt".format(file_dimensions)) as fp:
                for line in fp.readlines():
                    tokens = line.decode("utf-8").split()
                    self.mapping[tokens[0]] = np.array(
                        [float(t) for t in tokens[1 : self.dimension]]
                    )
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

    def __init__(self, date, dimensions, topics, distance="cosine"):
        self.distance = distance
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
            "wiki-d{}-c{}-v{}{}.msgpack.gz".format(
                self.dimensions,
                self.topics,
                Wikipedia.version,
                "euclidean" if distance == "euclidean" else "",
            ),
        )

    def get_cache_dir(self):
        return self.cache_dir

    def get_path(self):
        return self.out_fname

    def build_metadata(self):
        datatype = "WikiPageEuclidean" if self.distance == "euclidean" else "WikiPage"
        meta = {
            "name": "Wikipedia{}".format(
                "-euclidean" if self.distance == "euclidean" else ""
            ),
            "datatype": {datatype: None},
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
        from gensim.models.ldamulticore import LdaMulticore
        from gensim.corpora.wikicorpus import WikiCorpus
        from gensim.corpora.dictionary import Dictionary

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
        glove = WordEmbeddingMap(self.dimensions)
        logging.info("Setting up LDA")
        lda = self.load_lda(
            CachedBowsCorpus(wiki, self.bow_cache, meta=False), dictionary
        )

        logging.info("Remapping vectors")
        with gzip.open(self.out_fname, "wb") as out_fp:
            self.write_metadata(out_fp)
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

    def get_cache_dir(self):
        return self.cache_dir

    def get_path(self):
        return self.path

    def build_metadata(self):
        parameters = self.base.metadata()["parameters"]
        parameters.update(
            {
                "size": self.size,
                "seed": self.seed,
                "base_version": self.base.metadata()["version"],
            }
        )
        return {
            "name": "{}-sample-{}".format(self.base.metadata()["name"], self.size),
            "constraint": self.base.metadata()["constraint"],
            "datatype": self.base.metadata()["datatype"],
            "parameters": parameters,
            "version": SampledDataset.version,
        }

    def preprocess(self):
        if not os.path.isfile(self.path):
            self.base.try_download_preprocessed()
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
                self.write_metadata(out_fp)
                for doc in self.base:
                    progress_bar.update(1)
                    if random.random() <= p:
                        out_fp.write(msgpack.packb(doc))
            progress_bar.close()


class MusixMatch(Dataset):
    version = 1

    def __init__(self):
        self.cache = os.path.join(CACHE_DIR, "mxm")
        self.file_name = os.path.join(
            self.cache, "mxm-v{}.msgpack.gz".format(MusixMatch.version)
        )
        self.train_url = "http://millionsongdataset.com/sites/default/files/AdditionalFiles/mxm_dataset_train.txt.zip"
        self.test_url = "http://millionsongdataset.com/sites/default/files/AdditionalFiles/mxm_dataset_test.txt.zip"
        self.genres_url = "http://www.tagtraum.com/genres/msd_tagtraum_cd2.cls.zip"
        self.test_file = os.path.join(self.cache, "test.zip")
        self.train_file = os.path.join(self.cache, "train.zip")
        self.genres_file = os.path.join(self.cache, "genres.zip")

    def get_cache_dir(self):
        return self.cache

    def get_path(self):
        return self.file_name

    def build_metadata(self):
        meta = {
            "name": "MusixMatch",
            "datatype": {"Song": None},
            "constraint": {"partition": {"categories": self.genres_counts()}},
            "version": MusixMatch.version,
            "parameters": {},
        }
        return meta

    def extract_file(self, genres, fp):
        for line in tqdm(fp.readlines(), unit="songs", leave=False):
            line = line.decode("utf-8")
            if not line.startswith("#") and not line.startswith("%"):
                tokens = line.split(",")
                track_id = tokens[0]
                genre = genres.get(track_id, "Unknown")
                coordinates = []
                for token in tokens[2:]:
                    parts = token.split(":")
                    idx = int(parts[0])
                    value = int(parts[1])
                    coordinates.append((idx, value))
                yield {
                    "track_id": track_id,
                    "genre": genre,
                    "vector": {"d": 5000, "c": coordinates},
                }

    def genres(self):
        download_file(self.genres_url, self.genres_file)
        genres = dict()
        with zipfile.ZipFile(self.genres_file) as zipfp:
            with zipfp.open("msd_tagtraum_cd2.cls") as fp:
                for line in fp.readlines():
                    line = line.decode("utf-8")
                    tokens = line.split("\t")
                    track_id = tokens[0]
                    # ignore more specific genres
                    genre = tokens[1] if len(tokens) > 1 else "Unknown"
                    genres[track_id] = genre.strip()
        return genres

    def genres_counts(self):
        counts = dict()
        for genre in self.genres().values():
            if genre not in counts:
                counts[genre] = 0
            counts[genre] += 1
        return counts

    def preprocess(self):
        if not os.path.isfile(self.file_name):
            download_file(self.test_url, self.test_file)
            download_file(self.train_url, self.train_file)

            genres = self.genres()

            points = []
            with zipfile.ZipFile(self.test_file) as zipfp:
                with zipfp.open("mxm_dataset_test.txt") as fp:
                    points.extend(self.extract_file(genres, fp))
            with zipfile.ZipFile(self.train_file) as zipfp:
                with zipfp.open("mxm_dataset_train.txt") as fp:
                    points.extend(self.extract_file(genres, fp))

            with gzip.open(self.file_name, "wb") as fp:
                self.write_metadata(fp)
                for song in tqdm(points, leave=False, unit="songs"):
                    msgpack.pack(song, fp)


# The idea is to take a subset of a dataset based on the doubling dimension
# of the points it contains. Obviously this does not work, since subsampling
# changes the number of balls needed to cover any given ball, by definition.
class BoundedDifficultyDataset(Dataset):
    version = 1

    def __init__(self, base, size):
        self.base = base
        self.size = size
        self.cache_dir = os.path.join(".datasets/bounded-difficulty")
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)
        params_list = list(base.metadata()["parameters"].items())
        params_list.sort()
        params_str = "-".join(["{}-{}".format(k, v) for k, v in params_list])
        self.path = os.path.join(
            self.cache_dir,
            "{}-{}-bd{}-v{}.msgpack.gz".format(
                base.metadata()["name"],
                params_str,
                size,
                BoundedDifficultyDataset.version,
            ),
        )

    def get_path(self):
        return self.path

    def build_metadata(self):
        parameters = self.base.metadata()["parameters"]
        parameters.update(
            {
                "size": self.size,
                "base_version": self.base.metadata()["version"],
            }
        )
        return {
            "name": "{}-sample-{}".format(self.base.metadata()["name"], self.size),
            "constraint": self.base.metadata()["constraint"],
            "datatype": self.base.metadata()["datatype"],
            "parameters": parameters,
            "version": SampledDataset.version,
        }

    def preprocess(self):
        if not os.path.isfile(self.path):
            self.base.try_download_preprocessed()
            self.base.preprocess()
            doubling_dims = self.base.get_doubling_dimension()
            allowed_ids = set((pair[0] for pair in doubling_dims[: self.size]))

            n = self.base.num_elements()

            progress_bar = tqdm(total=n, unit="pages", unit_scale=False)
            with gzip.open(self.path, "wb") as out_fp:
                self.write_metadata(out_fp)
                idx = 0
                for doc in self.base:
                    progress_bar.update(1)
                    if idx in allowed_ids:
                        out_fp.write(msgpack.packb(doc))
                    idx += 1
            progress_bar.close()


DATASETS = {
    "wiki-d50-c100": Wikipedia("20210120", dimensions=50, topics=100),
    "wiki-d10-c50": Wikipedia("20210120", dimensions=10, topics=50),
    "wiki-d50-c100-eucl": Wikipedia(
        "20210120", dimensions=50, topics=100, distance="euclidean"
    ),
    "MusixMatch": MusixMatch(),
}

# Sampled datasets
for size in [100000, 50000, 10000, 1000]:
    DATASETS["wiki-d50-c100-s{}".format(size)] = SampledDataset(
        base=DATASETS["wiki-d50-c100"], size=size, seed=12341245
    )
    DATASETS["wiki-d10-c50-s{}".format(size)] = SampledDataset(
        base=DATASETS["wiki-d10-c50"], size=size, seed=12341245
    )
    DATASETS["wiki-d50-c100-s{}-eucl".format(size)] = SampledDataset(
        base=DATASETS["wiki-d50-c100-eucl"], size=size, seed=12341245
    )
    # DATASETS["MusixMatch-s{}".format(size)] = SampledDataset(
    #     base=DATASETS["MusixMatch"], size=size, seed=12341245
    # )


if __name__ == "__main__":
    dataset = DATASETS["wiki-d10-c50"]
    dataset.try_download_preprocessed()
    dataset.preprocess()
    print(dataset.metadata())
    print(dataset.get_path())
