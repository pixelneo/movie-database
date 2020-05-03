#!/usr/bin/env python3
import itertools
import logging

import numpy as np
import gensim
from gensim.models import Phrases, LdaMulticore, LdaModel
from gensim.corpora import Dictionary
from gensim.test.utils import datapath
from pprint import pprint

from config import Config
from load import Dataset

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO 

class TopicModelling:
    """
    Class for processing data by selected method: LDA, Doc2Vec, ...
    """
    def __init__(self, config):
        self.c = config

    def _prepare_lda(self, dataset):
        docs = dataset
        dictionary = Dictionary(docs)
        dictionary.filter_extremes(no_below=3, no_above=0.5)
        corpus = [dictionary.doc2bow(doc) for doc in docs]
        _ = dictionary[0]
        return corpus, dictionary

    def train_lda(self, dataset):
        corpus, dictionary = self._prepare_lda(dataset)
        dictionary.save('../models.nosync/lda/dict')

        alpha = np.arange(0.005, 0.05, (0.05-0.005)/self.c.lda_topics)

        print('starting LDA')
        model = LdaModel(
            corpus=corpus,
            # distributed=True,
            # workers=1,
            id2word=dictionary.id2token,
            chunksize=4000,
            alpha='auto', #alpha,
            eta='auto',
            iterations=self.c.lda_iter,
            num_topics=self.c.lda_topics,
            passes=self.c.lda_passes,
            eval_every=5000
        )
        # path = datapath('lda_model')
        path = '../models.nosync/lda/model'
        model.save(path)
        return model, corpus

    def eval_lda(self, dataset):
        path = '../models.nosync/lda/model'
        model = LdaMulticore.load(path)
        corpus, dictionary = self._prepare_lda(dataset)
        x = model.log_perplexity(corpus)
        print(x)
        print(model.alpha)
        print(model.eta)
        exit()
        # TODO smtng like loop over all docs in dataset and for each find similar docs and print their titles
        # print(corpus)
        # exit()
        for i, d in enumerate(corpus):
            for j, s in model.get_document_topics(d):
                print(dictionary.id2token[j], s, end=' ')
            print()




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
    # d = Dataset.create('../data.nosync/train_wiki.csv', c)
    # d.save('../models.nosync/data_train')
    # d2 = Dataset.create('../data.nosync/test_wiki.csv', c)
    # d2.save('../models.nosync/data_test')
    print('datasets done')
    d = Dataset.load('../models.nosync/data_train', c)
    # d2 = Dataset.load('../models.nosync/data_test', c)
    t = TopicModelling(c)
    m, data = t.train_lda(d)
    # t.eval_lda(d2)

