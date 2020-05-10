#!/usr/bin/env bash

class Cluster:
    def __init__(self, config):
        self.c = config

    def train_cluster(self, dataset, matrix):
        model = AgglomerativeClustering(n_clusters=self.c.clusters, distance_threshold=self.c.dist_tres)
