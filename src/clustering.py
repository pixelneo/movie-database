#!/usr/bin/env bash
import pickle

from sklearn.cluster import AgglomerativeClustering, KMeans

class Cluster:
    def __init__(self, config):
        self.c = config
        self.types = {
            'hiearch': lambda x: AgglomerativeClustering(n_clusters=x, distance_threshold=self.c.dist_tres, affinity='cosine', linkage='average'),
            'kmeans': lambda x: KMeans(n_clusters=x, n_jobs=2)
        }

    def train_cluster(self, dataset, matrix):
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



