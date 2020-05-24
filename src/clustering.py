#!/usr/bin/env bash
import pickle

from sklearn.cluster import AgglomerativeClustering, KMeans
from gensim import similarities
import numpy as np

class Cluster:
    def __init__(self, config):
        self.c = config
        self.types = {
            'hiearch': lambda x: AgglomerativeClustering(n_clusters=x, affinity='cosine', linkage='average'),
            'kmeans': lambda x: KMeans(n_clusters=x, n_jobs=4)
        }

    def find_similar(self, dataset, corpus, model):
        matrix = model[corpus]
        index = similarities.MatrixSimilarity(matrix)
        index.save('../models.nosync/index')
        # index = similarities.MatrixSimilarity.load('../models.nosync/index')
        sim = index[matrix]
        sim = np.array(sim)
        for t, s in zip(dataset.titles, sim):
            print(t)
            ind = np.argsort(s)[::-1]
            for it2 in ind[1:6]:
                print(dataset.titles[it2], end=', ')
            print('\n-----\n')



    def train_cluster(self, dataset, matrix):
        """ Deprecated """
        model = self.types[self.c.cluster_type](self.c.clusters)
        # model = model.fit(matrix)
        # with open('../models.nosync/cluster', 'wb') as f:
            # pickle.dump(model.get_params(), f)
        mapping = model.fit_predict(matrix)
        clusters = [None]*self.c.clusters
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
        with open('../models.nosync/cluster', 'rb') as f:
            model = pickle.load(f)
        model.predict(matrix)



