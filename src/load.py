#!/usr/bin/env python3
import csv
import itertools
import pickle

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

class Dataset:
    def __init__(self):
        pass

    @classmethod
    def create(cls, path, config):
        c = cls()
        c.path = path
        c.c = config
        c.nlp = spacy.load("en_core_web_sm", disable=['tagger', 'parser'])
        c.titles = dict(((' '.join(t), i) for i, t in enumerate(c._index_iterator(c.c.title_col))))
        c.processed = False
        return c

    @classmethod
    def load(cls, path):
        c = cls()
        with open(''.join([path, '.titles']), 'rb') as f:
            c.titles = pickle.load(f)
        with open(''.join([path, '.plots']), 'rb') as f:
            c.data = pickle.load(f)
        c.processed = True 

    def save(self, path):
        with open(''.join([path, '.plots']), 'wb') as f:
            pickle.dump(list(self), f)
        with open(''.join([path, '.titles']), 'wb') as f:
            pickle.dump(self.titles, f)


    def _index_iterator(self, index):
        with open(self.path, encoding='UTF-8', buffering=31000) as f:
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
    c = Config('config.json')
    d = Dataset.create('../data.nosync/test2.csv', c)
    d.save('../models.nosync/data2')
    # d.load('../models.nosync/data2')
    pprint(list(d)[19:20])
