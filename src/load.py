#!/usr/bin/env python3
import csv

from pprint import pprint
import gensim
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

from config import Config

class Dataset:
    def __init__(self, data, config):
        self.data = data
        self.nlp = spacy.load("en_core_web_sm")#, disable=['tagger', 'parser'])
        self.titles = dict(((t, i) for i, t in enumerate(self.data['Title'])))
        self.processed = None
        self.config = config

    @classmethod
    def create(cls, path, config):
        return cls(pd.read_csv(path, usecols=[0,1,7]), config)

    @classmethod
    def save(cls):
        raise NotImplementedError()

    @classmethod
    def load(cls, config=None):
        raise NotImplementedError()

    def _extract(self, w):
        o = []
        if self.config.lemma:
            o.append(w.lemma_)
        else:
            o.append(w.text)

        if self.config.entity:
            raise NotImplementedError()
            # o.append(''.join([e.label_, '_ENT']))
        return o

    def _process(self):
        flatten = lambda l: [item for sublist in l for item in sublist]
        doc_gen = self.nlp.pipe(self.data['Plot'], n_process=2)  # tokenize
        self.processed = [flatten((self._extract(w) for w in doc if (not w.is_stop and not w.is_punct and not w.like_num))) for doc in doc_gen]
        return self.processed



if __name__=='__main__':
    c = Config('config.json')
    d = Dataset.create('../data.nosync/test.csv', c)
    d._process()

    print(d.processed[0:2])
