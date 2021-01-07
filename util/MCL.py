#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import numpy as np
import pandas as pd

class MarkvCluster(object):
    def __init__(self, data, expand_factor=2, inflate_factor=2.0, mult_factor=2.0, max_loop=200):
        super(MarkvCluster, self).__init__()
        M = np.corrcoef(data)
        M[M < 0] = 0
        for i in range(len(M)):
            M[i][i] = 0
        import networkx as nx
        G = nx.from_numpy_matrix(M)
        self.M, self.clusters = self.networkx_mcl(G, expand_factor=expand_factor, inflate_factor=inflate_factor,
                                                  max_loop=max_loop, mult_factor=mult_factor)
        self.cluster_array = self.get_array()

    def get_array(self):
        array = []
        for key in self.clusters:
            for value in self.clusters[key]:
                array.append([value, key])
        df = pd.DataFrame(np.array(array), columns=['Sample', 'Cluster'])
        df = df.sort_values(by='Sample', ascending=True)
        return df.Cluster.values

    def networkx_mcl(self, G, expand_factor=2, inflate_factor=2, max_loop=10, mult_factor=1):
        import networkx as nx
        A = nx.adjacency_matrix(G)
        return self.mcl(np.array(A.todense()), expand_factor, inflate_factor, max_loop, mult_factor)

    def mcl(self, M, expand_factor=2, inflate_factor=2, max_loop=10, mult_factor=1):
        M = self.add_diag(M, mult_factor)
        M = self.normalize(M)
        for i in range(max_loop):
            # logging.info("loop %s" % i)
            M = self.inflate(M, inflate_factor)
            M = self.expand(M, expand_factor)
            if self.stop(M, i): break

        clusters = self.get_clusters(M)
        return M, clusters

    def add_diag(self, A, mult_factor):
        return A + mult_factor * np.identity(A.shape[0])

    def normalize(self, A):
        column_sums = A.sum(axis=0)
        new_matrix = A / column_sums[np.newaxis, :]
        return new_matrix

    def inflate(self, A, inflate_factor):
        return self.normalize(np.power(A, inflate_factor))

    def expand(self, A, expand_factor):
        return np.linalg.matrix_power(A, expand_factor)

    def stop(self, M, i):
        if i % 5 == 4:
            m = np.max(M ** 2 - M) - np.min(M ** 2 - M)
            if m == 0:
                # logging.info("Stop at iteration %s" % i)
                return True
        return False

    def get_clusters(self, A):
        clusters = []
        for i, r in enumerate((A > 0).tolist()):
            if r[i]:
                clusters.append(A[i, :] > 0)
        clust_map = {}
        for cn, c in enumerate(clusters):
            for x in [i for i, x in enumerate(c) if x]:
                clust_map[cn] = clust_map.get(cn, []) + [x]
        return clust_map


if __name__ == '__main__':
    data = np.random.rand(32, 5)
    print(data)
    mcl = MarkvCluster(data)
    print(mcl.cluster_array)
