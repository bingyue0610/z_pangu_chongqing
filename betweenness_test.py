# coding: utf-8

import pandas as pd
import networkx as nx
import numpy as np
from functools import partial
import random
from heapq import heappush, heappop
from itertools import count
import copy


def get_sample_graph():
    sample_list1 = [[1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 4]]
    tmp_test_g1 = nx.Graph()
    tmp_test_g1.add_edges_from(sample_list1)
    tmp_test_g1.add_node(7)
    nx.shortest_path(tmp_test_g1, 1, 4)
    return tmp_test_g1


def read_edges_to_graph(edges_location):
    tmp_edges_df = pd.read_csv(edges_location, dtype={'num1': str, 'num2': str})
    tmp_edges_list = np.array(tmp_edges_df)
    tmp_g1 = nx.Graph()
    tmp_g1.add_edges_from(tmp_edges_list)
    return tmp_g1


def single_source_shortest_path_basic(G, one_node):
    S = []
    P = {}
    for v in G:
        P[v] = []
    sigma = dict.fromkeys(G, 0.0)
    D = {}
    sigma[one_node] = 1.0
    D[one_node] = 0
    Q = [one_node]
    while Q:
        v = Q.pop(0)
        S.append(v)
        Dv = D[v]
        sigmav = sigma[v]
        for w in G[v]:
            if w not in D:
                Q.append(w)
                D[w] = Dv + 1
            if D[w] == Dv + 1:
                sigma[w] += sigmav
                P[w].append(v)
    return S, P, sigma


def accumulate_basic(betweenness, S, P, sigma, one_node):
    delta = dict.fromkeys(S, 0.0)
    while S:
        w = S.pop()
        coeff = (1.0 + delta[w]) / sigma[w]
        for v in P[w]:
            delta[v] += sigma[v] * coeff
        if w != one_node:
            if delta[w] > 0:
                if w in betweenness:
                    betweenness[w] += delta[w]
                else:
                    betweenness[w] = 0
                    betweenness[w] += delta[w]
    return betweenness


def single_node_betweenness_centrality(G, one_node):
    # betweenness = dict.fromkeys(G, 0.0)
    betweenness = {}
    S, P, sigma = single_source_shortest_path_basic(G, one_node)
    betweenness = accumulate_basic(betweenness, S, P, sigma, one_node)
    return betweenness


def two_node_betweenness_centrality(G, node1, node2):
    try:
        tmp_list = list(nx.all_shortest_paths(G, node1, node2))
        betweenness = {}
        tmp_len = len(tmp_list)
        for tmp_sublist in tmp_list:
            for key in tmp_sublist:
                if key != node1 and key != node2 and key not in betweenness:
                    betweenness[key] = 1.0 / tmp_len
                elif key != node1 and key != node2 and key in betweenness:
                    betweenness[key] += 1.0 / tmp_len
    except:
        betweenness = {}
    return betweenness


def rescale_result(betweenness, len_graph):
    scale = 1.0 / ((len_graph - 1) * (len_graph - 2))
    for v in betweenness:
        betweenness[v] *= scale
    return betweenness

if __name__ == '__main__':
    test_g1 = get_sample_graph()
    # test_g = read_edges_to_graph('relations.csv')
    ss = single_source_shortest_path_basic(test_g1, 1)
    for i in range(7 - 1):
        for j in range(i, 7):
            print(test_g1.nodes()[i+1], test_g1.nodes()[j+1], two_node_betweenness_centrality(test_g1, test_g1.nodes()[i+1], test_g1.nodes()[j+1]))



    # '15288432394'
    # nx.betweenness_centrality(test_g1)
    # {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2, 6: 0.2}
    """
    import time
    time0 = time.time()
    ss = single_source_shortest_path_basic(test_g, '13547737755')
    time1 = time.time()
    print time1 - time0
    26.3949999809
    """
    """
    import time
    time0 = time.time()
    ss = nx.all_shortest_paths(test_g, '15884090680',  '18783626359')
    time1 = time.time()
    print time1 - time0
    """


