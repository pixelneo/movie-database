#!/usr/bin/env python3
import csv
import itertools
import pickle
import os
from typing import List

from pprint import pprint
import gensim
from gensim.models import Phrases
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
            w = csv.writer(f)
            for z in range(train_size):
                r = next(x)
                w.writerow(r)

        with open(os.path.join(dir_out,'test_wiki.csv'), 'w') as f:
            w = csv.writer(f)
            for r in x:
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
        obj = cls()
        obj.path = path
        obj.config = config
        obj.nlp = spacy.load("en_core_web_sm", disable=['tagger', 'parser'])
        obj.titles = [' '.join([t, y]) for t, y in zip(obj._index_iterator(obj.config.title_col), obj._index_iterator(obj.config.year_col))]
        obj.processed = False
        return obj

    @classmethod
    def load(cls, path, config):
        """ Load preprocessed dataset form pickled file """
        obj = cls()
        obj.config = config
        with open(''.join([path, '.titles']), 'rb') as f:
            obj.titles = pickle.load(f)#[:obj.config.top]
        with open(''.join([path, '.plots']), 'rb') as f:
            obj.data = pickle.load(f)[:obj.config.top]
        obj.processed = True
        return obj

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
        if self.config.lemma:
            return w.lemma_.lower()
        else:
            return w.text.lower()

    def _process_person_names(self, doc):
        if self.config.entity:
            return (t for t in doc if (t.ent_type_ != 'PERSON'))
        else:
            return doc

    def _process(self, index) -> List[str]:
        """ Perform text processing on `index`-th column of the csv file """
        docs = self.nlp.pipe(self._index_iterator(index), batch_size=512, n_process=4)
        docs_clean = [[self._extract(w) for w in self._process_person_names(doc) if (not w.is_stop and not w.is_punct and not w.like_num)] for doc in docs]
        bigram = Phrases(docs_clean, min_count=1, delimiter=b'_')

        for d in docs_clean:
            out = list(d)
            for b in bigram[d]:
                if '_' in b:
                    out.append(b)
            yield out

    def __iter__(self):
        if not self.processed:
            return self._process(self.config.plot_col)
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
