#!/usr/bin/env python3

import numpy as np
import gensim
from gensim.models import Phrases, LdaMulticore, LdaModel
from gensim.corpora import Dictionary
from gensim.test.utils import datapath
from pprint import pprint

from config import Config
from load import Dataset

class TopicModelling:
    def __init__(self, dataset, config):
        self._dataset = dataset
        self.c = config

    def lda(self):
        # preprocess
        docs = self._dataset
        # bigram = Phrases(docs, min_count=2, delimiter=b'*')
        # for i_d, doc in enumerate(docs):
            # for b in bigram[doc]:
                # if '*' in  b:
                    # docs[i_d].append(b)
        dictionary = Dictionary(docs)
        dictionary.filter_extremes(no_below=2, no_above=0.3)
        corpus = [dictionary.doc2bow(doc) for doc in docs]
        _ = dictionary[0]

        alpha = np.arange(0.01, 0.1, (0.1-0.01)/self.c.lda_topics)

        print('starting LDA')
        model = LdaMulticore(
            corpus=corpus,
            workers=3,
            id2word=dictionary.id2token,
            chunksize=8192,
            alpha=alpha,
            eta=0.001,
            iterations=self.c.lda_iter,
            num_topics=self.c.lda_topics,
            passes=self.c.lda_passes,
            eval_every=None
        )
        path = datapath('lda_model')
        model.save(path)
        return model, corpus


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
    d = Dataset('../data.nosync/wiki_movie_plots_deduped.csv', c)
    t = TopicModelling(d, c)
    m, data = t.lda()
    pprint(m.top_topics(data))

