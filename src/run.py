#!/usr/bin/env python3
from topicmodelling import TopicModelling
from load import *
from config import Config

c = Config('config.json')

def create():
    split('../data.nosync/wiki_movie_plots_deduped.csv', '../data.nosync', c.train_size)
    d = Dataset.create('../data.nosync/train_wiki.csv', c)
    d.save('../models.nosync/data_train')
    d2 = Dataset.create('../data.nosync/test_wiki.csv', c)
    d2.save('../models.nosync/data_test')

def train():
    d = Dataset.load('../models.nosync/data_train', c)
    t = TopicModelling(c)
    m, data = t.train_lda(d)

def eval():
    d2 = Dataset.load('../models.nosync/data_test', c)
    t = TopicModelling(c)
    t.eval_lda(d2)

if __name__=='__main__':
    import sys
    if sys.argv[1] == 'create':
        create()
    elif sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'eval':
        eval()
    else:
        raise ValueError('wrong argument')

