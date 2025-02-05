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
    t = TopicModelling.create(c)
    m, data = t.train(d)

def inf():
    d2 = Dataset.load('../models.nosync/data_train', c)
    t = TopicModelling.create(c)
    corpus, model = t.infer(d2)  #TODO fix
    cl = Cluster(c)
    for t, titles in cl.find_similar(d2, corpus, model):
        print('{}: {}'.format(t, ', '.join(titles)))
        print('---\n')

def test():
    raise NotImplementedError()
    d2 = Dataset.load('../models.nosync/data_test', c)
    t = TopicModelling.create(c)
    t.infer(d2)

def reproduce(method):
    c.reload(method)
    create()
    train()
    inf()

def help():
    print(runs.keys())

runs = {
    "create": create,
    "train": train,
    "eval": eval,
    "inf": inf,
    "test": test,
    "reproduce": reproduce,
    "help": help,
}
if __name__=='__main__':
    import sys
    if len(sys.argv) == 2:
        runs[sys.argv[1]]()
    elif len(sys.argv) == 3:
        runs[sys.argv[1]](sys.argv[2])

