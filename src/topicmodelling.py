#!/usr/bin/env python3
import itertools
import logging

import numpy as np
import gensim
from gensim.models import Phrases, LdaMulticore, LdaModel, LsiModel
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.corpora import Dictionary
from gensim.test.utils import datapath
import gensim.downloader as api
from pprint import pprint

from config import Config
from load import Dataset

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO 


class TopicModelling:
    """
    Abstract class for processing data by selected method: LDA, Doc2Vec, ...
    """
    def __init__(self, config):
        self.c = config

    @classmethod
    def create(cls, config):
        if config.method == 'lda':
            return LdaModelling(config)
        elif config.method == 'doc2vec':
            return Doc2VecModelling(config)
        elif config.method == 'lsa':
            return LsiModelling(config)
        else:
            raise AttributeError('Selected topic modelling method does not exist')

    def _prepare(self, dataset):
        docs = dataset
        dictionary = Dictionary(docs)
        dictionary.filter_extremes(no_below=2, no_above=0.5)
        corpus = [dictionary.doc2bow(doc) for doc in docs]
        _ = dictionary[0]
        return corpus, dictionary

    def train(self, dataset):
        raise NotImplementedError()

    def eval(self, dataset):
        raise NotImplementedError()

    def infer(self, dataset):
        raise NotImplementedError()


class LdaModelling(TopicModelling):
    def __init__(self, config):
        super().__init__(config)

    def train(self, dataset):
        corpus, dictionary = self._prepare(dataset)
        dictionary.save('../models.nosync/lda/dict')

        print('starting LDA')
        model = LdaMulticore(
            corpus=corpus,
            # distributed=True,
            workers=3,
            id2word=dictionary.id2token,
            chunksize=4000,
            alpha=self.c.alpha, # optimized alpha
            eta='auto',
            iterations=self.c.lda_iter,
            num_topics=self.c.lda_topics,
            passes=self.c.lda_passes,
            eval_every=5000
        )
        path = '../models.nosync/lda/model'
        model.save(path)
        return model, corpus

    def eval(self, dataset):
        path = '../models.nosync/lda/model'
        model = LdaMulticore.load(path)
        corpus, dictionary = self._prepare(dataset)
        x = model.log_perplexity(corpus)
        print(x)
        for i, (d, t) in enumerate(zip(corpus, dataset.titles)):
            print(t)
            for j, s in model.get_document_topics(d):
                print(dictionary.id2token[j], end=' ')
            print('\n')

    def infer(self, dataset):
        path = '../models.nosync/lda/model'
        model = LdaMulticore.load(path)
        corpus, dictionary = self._prepare(dataset)
        return corpus, model

class Doc2VecModelling(TopicModelling):
    def __init__(self, config):
        super().__init__(config)

    def train(self, dataset):
        # corpus, dictionary = self._prepare(dataset)
        data = [TaggedDocument(doc, [i]) for i, doc in enumerate(dataset)]
        print('starting Doc2Vec')
        model = Doc2Vec(data, epochs=20, window=10, vector_size=50, workers=4)
        path = '../models.nosync/doc2vec/model'
        model.save(path)
        return data, model

    def infer(self, dataset):
        data = [TaggedDocument(doc, [i]) for i, doc in enumerate(dataset)]
        path = '../models.nosync/doc2vec/model'
        model = Doc2Vec.load(path)
        # model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        return data, model

class LsiModelling(TopicModelling):
    def __init__(self, config):
        super().__init__(config)

    def train(self, dataset):
        corpus, dictionary = self._prepare(dataset)
        dictionary.save('../models.nosync/lsa/dict')

        print('starting LSA')
        model = LsiModel(
            corpus=corpus,
            id2word=dictionary.id2token,
            num_topics=self.c.lsa_topics
        )
        path = '../models.nosync/lsa/model'
        model.save(path)
        return model, corpus

    def infer(self, dataset):
        path = '../models.nosync/lsa/model'
        model = LsiModel.load(path)
        corpus, dictionary = self._prepare(dataset)
        return corpus, model






if __name__=='__main__':
    c = Config('config.yaml')
    # d = Dataset.create('../data.nosync/train_wiki.csv', c)
    # d.save('../models.nosync/data_train')
    # d2 = Dataset.create('../data.nosync/test_wiki.csv', c)
    # d2.save('../models.nosync/data_test')
    print('datasets done')
    # d = Dataset.load('../models.nosync/data_train', c)
    d2 = Dataset.load('../models.nosync/data_test', c)
    t = TopicModelling.create(c)
    t.infer(d2)
    # m, data = t.train_lda(d)

