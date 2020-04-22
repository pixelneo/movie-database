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

    def _preprocess(self, lemmatize=False, stopwords=False):
        self.processed = self.nlp.pipe(self.data['Plot'])  # tokenize




if __name__=='__main__':
    d = Dataset.create('../data.nosync/test.csv')
    d._preprocess()
    print(STOP_WORDS)

    print(list(list(d.processed)[0]))
