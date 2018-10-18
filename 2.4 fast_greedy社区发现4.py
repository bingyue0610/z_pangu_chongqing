"""
第四版更新内容：全局快速+边
"""


import pandas as pd
import numpy as np
import networkx as nx
import random
import time
import copy
from itertools import product


def get_some_test_data_with_weights():
    tmp_list = []
    tmp_list.append([1, 2, 1])
    tmp_list.append([1, 3, 1])
    tmp_list.append([2, 3, 1])
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
    sample_list1 = [[1, 2], [1, 3], [1, 4], [1, 5], [1, 6],
                    [2, 3], [2, 4], [2, 5], [2, 6],
                    [3, 4], [3, 5], [3, 6],
                    [4, 5], [4, 6],
                    [5, 6]]
    tmp_test_g1 = nx.Graph()
    tmp_test_g1.add_edges_from(sample_list1)
    tmp_test_g1.add_node(7)
    nx.shortest_path(tmp_test_g1, 1, 4)
    return tmp_test_g1

def get_sample_graph_100n_1000m():
    random.seed(5)
    tmp_list = []
    for i in range(1000):
        tmp_list.append([str(random.randint(0, 100)), str(random.randint(0,100))])
    tmp_test_g = nx.Graph()
    tmp_test_g.add_edges_from(tmp_list)
    return tmp_test_g


def get_sample_graph_20n_100m():
    random.seed(5)
    tmp_list = []
    for i in range(130):
        tmp_list.append([str(random.randint(0, 20)), str(random.randint(0, 20))])
    tmp_test_g = nx.Graph()
    tmp_test_g.add_edges_from(tmp_list)
    return tmp_test_g


def fast_greedy_from_seperate_to_whole(a_graph):
    start_time = time.time()
    adjacency_dict = get_adjacency_dict(a_graph)
    for key in adjacency_dict:
        for sub_key in adjacency_dict[key]:
            if key == sub_key:
                print(key, sub_key)
    degree_dict = get_degree_dict(adjacency_dict)
    new_g = nx.Graph()
    new_g.add_nodes_from(list(a_graph.nodes))
    # print('new_g', tuple(nx.connected_components(new_g)))
    # print('new_g_edges', new_g.edges)
    left_edges = list(a_graph.edges)
    count = 0
    tmp_file = open('fast_greedy_community_detection_n20_m100_with_new_modularity_with_full_iter_fast_edge_drop.txt', 'w')
    while len(left_edges) > 0:
        added_edge = left_edges[0]
        max_modularity_community = tuple(nx.connected_components(new_g))
        max_new_community_modularity = new_community_and_new_modularity(a_graph, new_g, added_edge, adjacency_dict, degree_dict)[1]
        for edge in left_edges:
            tmp_community, tmp_new_modularity = new_community_and_new_modularity(a_graph, new_g, edge, adjacency_dict, degree_dict)
            if tmp_new_modularity > max_new_community_modularity:
                max_new_community_modularity = tmp_new_modularity
                added_edge = edge
                max_modularity_community = tmp_community
        # print('added_edge11111111:', added_edge)
        # print('community_modularity1111111111111:', max_new_community_modularity)
        # print('community11111111111111111', max_modularity_community)
        new_g.add_edge(*added_edge)
        left_edges.remove(added_edge)
        # 更新加边机制
        add_edges = []
        for c in sorted(nx.connected_components(new_g), key=len, reverse=True):
            if len(c) > 1:
                sub_graph_nodes = list(c)
                partial_sub_graph = a_graph.subgraph(sub_graph_nodes)
                add_edges += list(partial_sub_graph.edges)

        set_add_edges = copy.deepcopy(set(add_edges))
        for i in set(add_edges):
            set_add_edges.add((i[1], i[0]))

        new_g.add_edges_from(list(set_add_edges))
        left_edges = list(set(left_edges) - set_add_edges)


        tmp_file.write(str(max_new_community_modularity))
        tmp_file.write('\t')
        tmp_file.write(str(max_modularity_community))
        tmp_file.write('\n')

        count += 1
        if count % 10 == 1:
            print('just after:', count)
        # print('added_new_g_edges00000000000000', new_g.edges)
        # print('added_new_g000000000000', tuple(nx.connected_components(new_g)))
        # print('left_edges0000000000000', left_edges)

    end_time = time.time()
    tmp_file.write(str(end_time - start_time))
    tmp_file.close()
    print('time_spent:', end_time - start_time)
    pass

def new_community_and_new_modularity(a_graph, a_sub_graph, a_edge, a_adjacency_dict, a_degree_dict):
    a_new_graph = copy.deepcopy(a_sub_graph)
    a_new_graph.add_edge(a_edge[0], a_edge[1])
    # print('g_edges2222222222222', a_new_graph.edges)
    tmp_new_community = tuple(nx.connected_components(a_new_graph))
    tmp_modularity = personalized_modulariy_undirected_nonweighted_graph(a_graph, a_new_graph, a_adjacency_dict, a_degree_dict)
    return tmp_new_community, tmp_modularity

def get_adjacency_dict(a_graph):
    """
    返回adjacency_dict, 邻接矩阵dict，一阶
    :param a_graph: 一个无向、无权图
    :return:
    """
    start_time = time.time()
    adjacency_dict = {}
    for edge in a_graph.edges:
        if edge[0] not in adjacency_dict:
            adjacency_dict[edge[0]] = {}
            adjacency_dict[edge[0]][edge[1]] = 1
        else:
            adjacency_dict[edge[0]][edge[1]] = 1

        if edge[1] not in adjacency_dict:
            adjacency_dict[edge[1]] = {}
            adjacency_dict[edge[1]][edge[0]] = 1
        else:
            adjacency_dict[edge[1]][edge[0]] = 1
    end_time = time.time()
    print('time used on making adjacency dict is:', end_time - start_time)
    return adjacency_dict


def get_degree_dict(a_adjacency_dict):
    """
    一个无向图的节点度数dict。此处不分出度入度。
    :param a_adjacency_dict:
    :return:
    """
    start_time = time.time()
    degree_dict = {}
    for key in a_adjacency_dict:
        degree_dict[key] = sum(a_adjacency_dict[key].values())
    end_time = time.time()
    print('time used on making degree dict is:', end_time - start_time)
    return degree_dict


def personalized_modulariy_undirected_nonweighted_graph(a_graph, a_sub_graph, a_adjacency_dict, a_degree_dict):
    """
    不包括self loop
    :param a_graph:
    :param a_sub_graph:
    :param a_adjacency_dict:
    :param a_degree_dict:
    :return:
    """
    tmp_new_community = tuple(nx.connected_components(a_sub_graph))
    m = float(a_graph.size())
    norm = 1/ (2* m)

    def val(u, v):
        try:
            a_ij = float(a_adjacency_dict[u][v])
        except KeyError:
            a_ij = 0
        k_i = float(a_degree_dict[u])
        k_j = float(a_degree_dict[v])
        return a_ij - k_i * k_j * norm

    Q = sum(val(u, v) for c in tmp_new_community for u, v in product(c, repeat=2))
    return Q * norm

if __name__ == '__main__':
    print('a')
    # test_data = get_facebook_test_data()
    # test_g = generate_undirected_graph(test_data, ifweighted=False)
    test_g = get_sample_graph_100n_1000m()
    # test_g = get_sample_graph_20n_100m()
    largest_sub = get_max_sub_component(test_g)
    print('lenth of largest_sub :', len(largest_sub))
    print('lenth of test_g is :', len(test_g))
    print('lenth of largest_sub edges is :', len(largest_sub.edges))
    print('between a and b, the community detection is on')
    t1 = time.time()
    fast_greedy_from_seperate_to_whole(test_g)
    t2 = time.time()
    print(t2 - t1)
    print('b')

