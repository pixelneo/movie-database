#!/usr/bin/env bash
import pickle

from sklearn.cluster import AgglomerativeClustering, KMeans
from gensim import similarities
import numpy as np

class Cluster:
    def __init__(self, config):
        self.config = config
        self.types = {
        }

    def find_similar(self, dataset, corpus, model):
        if self.config.method == 'lda' or self.config.method == 'lsa':
            matrix = model[corpus]
            index = similarities.MatrixSimilarity(matrix)
            index.save('../models.nosync/index')
            sim = index[matrix]
            sim = np.array(sim)
            for t, s in zip(dataset.titles, sim):
                ind = np.argsort(s)[::-1]
                titles = []
                for it2 in ind[1:6]:
                    titles.append(dataset.titles[it2])
                yield (t, titles)

        elif self.config.method == 'doc2vec':
            for doc, title in zip(dataset, dataset.titles):
                vec = model.infer_vector(doc)
                sim = model.docvecs.most_similar([vec], topn=6)
                titles = []
                for t,s in sim:
                    titles.append(dataset.titles[t])
                yield (t, titles)


    def train_cluster(self, dataset, matrix):
        """ Deprecated """
        model = self.types[self.config.cluster_type](self.config.clusters)
        mapping = model.fit_predict(matrix)
        clusters = [None]*self.config.clusters
        for i, m in enumerate(mapping):
            if not clusters[m]:
                clusters[m] = list()
            clusters[m].append(i)

        for c in clusters:
            print(len(c))
            for movie in c:
                print(dataset.titles[movie])
            print('-----\n\n')




    def inf_cluster(self, dataset, matrix):
        """ Deprecated """
        with open('../models.nosync/cluster', 'rb') as f:
            model = pickle.load(f)
        model.predict(matrix)



