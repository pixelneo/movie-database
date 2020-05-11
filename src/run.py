#!/usr/bin/env python3
from topicmodelling import TopicModelling
from load import *
from config import Config
from clustering import Cluster

c = Config('config.yaml')

def create():
    split('../data.nosync/wiki_movie_plots_deduped.csv', '../data.nosync', c.train_size)
    d = Dataset.create('../data.nosync/train_wiki.csv', c)
    d.save('../models.nosync/data_train')
    d2 = Dataset.create('../data.nosync/test_wiki.csv', c)
    d2.save('../models.nosync/data_test')

def train():
    d = Dataset.load('../models.nosync/data_train', c)
    t = TopicModelling(c)
    m, data = t.train_lda(d)  #TODO fix

def eval():
    d2 = Dataset.load('../models.nosync/data_test', c)
    t = TopicModelling(c)
    t.eval_lda(d2)  #TODO fix

def inf():
    d2 = Dataset.load('../models.nosync/data_train', c)
    t = TopicModelling(c)
    corpus, model = t.inf_lda(d2)  #TODO fix
    cl = Cluster(c)
    cl.find_similar(d2, corpus, model)

def test():
    raise NotImplementedError()
    d2 = Dataset.load('../models.nosync/data_test', c)
    t = TopicModelling(c)
    t.inf_lda(d2)



runs = {
    "create": create,
    "train": train,
    "eval": eval,
    "inf": inf,
    "test": test,
}
if __name__=='__main__':
    import sys
    runs[sys.argv[1]]()

