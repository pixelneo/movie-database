#!/usr/bin/env python3
import csv

from pprint import pprint
import gensim
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

class Dataset:
    def __init__(self, data):
        self.data = data
        self.nlp = spacy.load("en_core_web_sm", disable=['tagger', 'parser', 'ner'])
        self.titles = dict(((t, i) for i, t in self.data[TODO]))
        self.processed = None

    @classmethod
    def create(cls, path):
        return cls(pd.read_csv(path, usecols=[0,1,7]))

    @classmethod
    def save(cls):
        raise NotImplementedError()

    @classmethod
    def load(cls):
        raise NotImplementedError()

    def _process(self):
        doc_gen = self.nlp.pipe(self.data['Plot'], n_process=2)  # tokenize
        self.processed = [[w.lemma_ for w in doc if (not w.is_stop and not w.is_punct and not w.like_num)] for doc in doc_gen]
        return self.processed

if __name__=='__main__':
    d = Dataset.create('../data.nosync/test.csv')
    d._process()

    print(d.processed[0:2])
