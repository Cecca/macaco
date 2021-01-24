import msgpack
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

logging.getLogger().setLevel(logging.INFO)


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
                    self.mapping[tokens[0]] = np.array(
                        [float(t) for t in tokens[1:]])
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
                vec += (wordvec * count)
                cnt += 1
            else:
                pass
        if cnt > 0:
            return vec / np.float(cnt)
        else:
            return None


class Wikipedia(object):
    version = 1

    def __init__(self, date, dimensions, categories):
        self.date = date
        self.dimensions = dimensions
        self.categories = categories
        self.url = "https://dumps.wikimedia.org/enwiki/{}/enwiki-{}-pages-articles-multistream.xml.bz2".format(date, date)
        self.cache_dir = os.path.join(".datasets/wikipedia", date)
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.dump_file = os.path.join(self.cache_dir, os.path.basename(self.url))
        self.dictionary = os.path.join(self.cache_dir, "dictionary")
        self.lda_model_path = os.path.join(
            self.cache_dir, "model-lda-{}".format(self.categories))
        self.out_fname = os.path.join(
            self.cache_dir, "wiki-d{}-c{}-v{}.msgpack".format(
                self.dimensions,
                self.categories,
                Wikipedia.version
            )
        )

    def metadata(self):
        return {
            "version": Wikipedia.version,
            "parameters": {
                "dimensions": self.dimensions,
                "categories": self.categories,
                "date": self.date,
                "url": self.url
            }
        }

    def load_lda(self, docs, dictionary):
        cores = multiprocessing.cpu_count()
        if not os.path.exists(self.lda_model_path):
            logging.info("Training LDA")
            lda = LdaMulticore(docs, 
                               id2word=dictionary,
                               num_topics=self.categories,
                               passes=1,
                               workers = 4)
            logging.info("Saving LDA")
            lda.save(self.lda_model_path)
        else:
            logging.info("Model file found, loading")
            lda = LdaMulticore.load(self.lda_model_path)
        return lda

    def preprocess(self):
        cores = multiprocessing.cpu_count()
        download_file(self.url, self.dump_file)
        if not os.path.isfile(self.dictionary):
            logging.info("Creating gensim dictionary (takes a long time)")
            wiki = WikiCorpus(self.dump_file)
            wiki.dictionary.save(self.dictionary)
        dictionary = Dictionary.load(self.dictionary)
        wiki = WikiCorpus(self.dump_file, 
                          dictionary=dictionary)
        wiki.metadata = True

        logging.info("Loading documents into memory")
        # This is a workaround because WikiCorpus does not 
        # implement `len`, which is needed to compute the LDA
        # docs_with_meta = [d for (i, d) in enumerate(wiki)
        #                   if i < 10]
        docs_with_meta = []
        for i, d in enumerate(wiki):
            if i > 10:
                break
            docs_with_meta.append(d)
        logging.info("Copying just the bows")
        docs_no_meta = [bow for (bow, meta) in docs_with_meta]
        logging.info("...done")

        logging.info("Loading word embeddings")
        glove = GloveMap(self.dimensions)
        lda = self.load_lda(docs_no_meta, dictionary)

        progress_bar = tqdm(total=len(docs_with_meta), 
                            unit='pages', 
                            unit_scale=False)
        with open(self.out_fname, "wb") as out_fp:
            header = msgpack.packb(self.metadata())
            out_fp.write(header)
            for (bow, (id, title)) in docs_with_meta:
                vector = list(glove.map_bow(dictionary, bow))
                progress_bar.update(1)
                if vector is not None:
                    topics = lda.get_document_topics(
                        bow, minimum_probability=0.1)
                    outdata = {
                        'id': int(id),
                        'title': title,
                        'topic': [p[0] for p in topics],
                        'vector': vector
                    }
                    encoded = msgpack.packb(outdata)
                    out_fp.write(encoded)
        progress_bar.close()


if __name__ == "__main__":
    from pprint import pprint
    wiki = Wikipedia("20210120", dimensions=50, categories=100)
    pprint(wiki.metadata())
    wiki.preprocess()

