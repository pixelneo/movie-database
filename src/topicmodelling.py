#!/usr/bin/env python3

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
        data = self._dataset.processed
        bigram = Phrases(data, min_count=2, delimiter=b'*')
        for i_d in range(len(data)):
            for b in bigram[data[i_d]]:
                if '*' in  b:
                    data[i_d].append(b)
        dictionary = Dictionary(data)
        dictionary.filter_extremes(no_below=2, no_above=0.3)
        corpus = [dictionary.doc2bow(doc) for doc in data]
        _ = dictionary[0]

        model = LdaModel(
            corpus=corpus,
            # workers=2,
            id2word=dictionary.id2token,
            chunksize=2048,
            alpha='auto',
            eta='auto',
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
    d = Dataset.create('../data.nosync/test2.csv', c)
    d.process()
    t = TopicModelling(d, c)
    m, data = t.lda()
    pprint(m.top_topics(data)[:5])

