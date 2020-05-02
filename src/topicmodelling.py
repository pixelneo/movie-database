#!/usr/bin/env python3
import itertools

import numpy as np
import gensim
from gensim.models import Phrases, LdaMulticore, LdaModel
from gensim.corpora import Dictionary
from gensim.test.utils import datapath
from pprint import pprint

from config import Config
from load import Dataset

class TopicModelling:
    """
    Class for processing data by selected method: LDA, Doc2Vec, ...
    """
    def __init__(self, dataset, config):
        self._dataset = dataset
        self.c = config

    def train_lda(self):
        docs = self._dataset
        dictionary = Dictionary(docs)
        dictionary.filter_extremes(no_below=3, no_above=0.5)
        corpus = [dictionary.doc2bow(doc) for doc in docs]
        _ = dictionary[0]
        dictionary.save('../models.nosync/lda/dict')

        alpha = np.arange(0.005, 0.05, (0.05-0.005)/self.c.lda_topics)

        print('starting LDA')
        model = LdaMulticore(
            corpus=corpus,
            workers=3,
            id2word=dictionary.id2token,
            chunksize=8192,
            alpha=alpha,
            eta='auto',
            iterations=self.c.lda_iter,
            num_topics=self.c.lda_topics,
            passes=self.c.lda_passes,
            eval_every=1000
        )
        # path = datapath('lda_model')
        path = '../models.nosync/lda/model'
        model.save(path)
        return model, corpus

    def eval_lda(self):
        path = '../models.nosync/lda/model'
        model = model.load(path)
        # TODO smtng like loop over all docs in dataset and for each find similar docs and print their titles



    def _doc2vec(self, topics):
        raise NotImplementedError()

    def _lsa(self, topics):
        raise NotImplementedError()

    def model(self, topics=None):
        if self.c.topic_method == 'lda':
            pass
        elif self.c.topic_method == 'doc2vec':
            raise NotImplementedError()
        elif self.c.topic_method == 'lsa':
            raise NotImplementedError()
        else:
            raise AttributeError('Selected topic modelling method does not exist')

if __name__=='__main__':
    c = Config('config.json')
    d = Dataset.create('../data.nosync/test2.csv', c)
    # d = Dataset.load('../models.nosync/data')
    t = TopicModelling(d, c)
    m, data = t.train_lda()
    pprint(m.top_topics(data))
    # pprint(m.get_topics().shape)

