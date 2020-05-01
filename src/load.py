#!/usr/bin/env python3
import csv

from pprint import pprint
import gensim
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from smart_open import open

from config import Config

class Dataset:
    def __init__(self, path, config):
        self.path = path
        self.nlp = spacy.load("en_core_web_sm", disable=['tagger', 'parser', 'ner'])
        self.titles = dict(((t, i) for i, t in enumerate(self._index_iterator(1))))
        self.processed = None
        self.config = config

    def _index_iterator(self, index):
        with open(self.path, encoding='UTF-8', buffering=1024) as f:
            r = csv.reader(f)
            next(r) # header
            for row in r:
                yield row[index]

    def _extract(self, w):
        if self.config.lemma:
            return w.lemma_.lower()
        else:
            return w.text.lower()

    def _process(self):
        for doc in self.nlp.pipe(self._index_iterator(7)):
            yield [self._extract(w) for w in doc if (not w.is_stop and not w.is_punct and not w.like_num)]

    def __iter__(self):
        return self._process()



if __name__=='__main__':
    c = Config('config.json')
    d = Dataset('../data.nosync/wiki_movie_plots_deduped.csv', c)

    for i, d in enumerate(d):
        print(d[0], end=' ')
