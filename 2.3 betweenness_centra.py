import pandas as pd
import numpy as np
import networkx as nx
import random
from heapq import heappush, heappop
from itertools import count
import time

def get_some_test_data_with_weights():
    tmp_list = []
    tmp_list.append(['a', 'b', 1])
    tmp_list.append(['a', 'c', 1])
    tmp_list.append(['b', 'c', 1])
    return tmp_list

def get_facebook_test_data():
    tmp_df = pd.read_table('facebook_combined.txt', names=['start', 'end'], sep=' ')
    tmp_array = np.array(tmp_df)
    return list(tmp_array)


def generate_undirected_graph(data_list, ifweighted=True):
    """
    无向有权图，or 无向无权图
    :param ifweighted: 是否制作带有权重的图
    :param data_list:     data_list = [[node1, node2, weight1], ...]
    :return:
    """
    tmp_g = nx.Graph()
    if ifweighted is False:
        tmp_g.add_edges_from(data_list)
    elif ifweighted is True:
        tmp_g.add_weighted_edges_from(data_list)
    return tmp_g

# 最大联通分量A 不用重写good!!!!!!!!!!!!
def get_max_sub_component(a_graph):
    largest_components = max(nx.connected_components(a_graph), key=len)
    tmp_largest_sub = a_graph.subgraph(largest_components)
    return tmp_largest_sub

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
    print('a')
    test_data = get_facebook_test_data()
    test_g = generate_undirected_graph(test_data, ifweighted=False)
    largest_sub = get_max_sub_component(test_g)
    local_betweenness_centra = single_node_betweenness_centrality(largest_sub, 1)
