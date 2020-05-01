#!/usr/bin/env bash
import gensim

from config import Config

class TopicModelling:
    def __init__(self, dataset, config):
        self._data = dataset
        self.c = config

    def _lda(self, topics):
        # TODO do this
        pass

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


