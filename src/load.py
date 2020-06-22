#!/usr/bin/env python3
import csv
import itertools
import pickle
import os

from pprint import pprint
import gensim
from gensim.models import Phrases
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from smart_open import open
from pprint import pprint

from config import Config
import time

def split(file_in, dir_out, train_size=34000):
    """ Splits dataset to train and test portions

    Args:
        file_in: raw dataset file path
        dir_out: path to dir for output
        train_size: size of train portion+1

    """
    with open(file_in) as f:
        x = csv.reader(f)
        with open(os.path.join(dir_out,'train_wiki.csv'), 'w') as f:
            for z in range(34000):
                r = next(x)
                w = csv.writer(f)
                w.writerow(r)

        with open(os.path.join(dir_out,'test_wiki.csv'), 'w') as f:
            for r in x:
                w = csv.writer(f)
                w.writerow(r)


class Dataset:
    """
    A class representing dataset.
    It is iterable over procesesed documents
    (processed might mean lemmatization, removal of some entities, etc.)

    """

    def __init__(self):
        pass

    @classmethod
    def create(cls, path, config):
        """ Creates new dataset from csv file.

        Args:
            path: (str) path to csv file
            config: Instance of Config class

        Returns:
            New `Dataset` object

        """
        c = cls()
        c.path = path
        c.c = config
        c.nlp = spacy.load("en_core_web_sm", disable=['tagger', 'parser'])
        c.titles = [' '.join([t, y]) for t, y in zip(c._index_iterator(c.c.title_col), c._index_iterator(c.c.year_col))]
        c.processed = False
        return c

    @classmethod
    def load(cls, path, config):
        """ Load preprocessed dataset form pickled file """
        c = cls()
        c.c = config
        with open(''.join([path, '.titles']), 'rb') as f:
            c.titles = pickle.load(f)#[:c.c.top]
        with open(''.join([path, '.plots']), 'rb') as f:
            c.data = pickle.load(f)[:c.c.top]
        c.processed = True
        return c

    def save(self, path):
        """ Save processed dataset to pickle """
        with open(''.join([path, '.plots']), 'wb') as f:
            pickle.dump(list(self), f)
        with open(''.join([path, '.titles']), 'wb') as f:
            pickle.dump(self.titles, f)


    def _index_iterator(self, index):
        """ Read `index`-the column from raw csv file """
        with open(self.path, encoding='UTF-8', buffering=64000) as f:
            r = csv.reader(f)
            next(r) # header
            for row in r:
                yield row[index]

    def _extract(self, w):
        if self.c.lemma:
            return w.lemma_.lower()
        else:
            return w.text.lower()

    def _process_doc(self, doc):
        if self.c.entity:
            return (t for t in doc if (t.ent_type_ != 'PERSON'))
        else:
            return doc

    def _process(self, index):
        """ Perform text processing on `index`-th column of the csv file """
        docs = self.nlp.pipe(self._index_iterator(index), batch_size=512, n_process=4)
        docs_clean = [[self._extract(w) for w in self._process_doc(doc) if (not w.is_stop and not w.is_punct and not w.like_num)] for doc in docs]
        bigram = Phrases(docs_clean, min_count=1, delimiter=b'_')

        for i,d in enumerate(docs_clean):
            out = list(d)
            for b in bigram[d]:
                if '_' in b:
                    out.append(b)
            yield out

    def __iter__(self):
        if not self.processed:
            return self._process(self.c.plot_col)
        else:
            return iter(self.data)


if __name__=='__main__':
    c = Config('config.yaml')
    # d = Dataset.create('../data.nosync/test_wiki.csv', c)
    # d.save('../models.nosync/data_test')
    d = Dataset.load('../models.nosync/data_train', c)
    # d = Dataset.create('../data.nosync/train_wiki.csv', c)
    # d.save('../models.nosync/data_train')
    # d.load('../models.nosync/data2')
    pprint(list(d)[10:20])
