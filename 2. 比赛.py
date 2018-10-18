import pandas as pd
import numpy as np
import networkx as nx
import random
import time


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

# density 不用重写good!!!!!!!!!!!!
def get_density(a_graph):
    tmp_density = nx.density(a_graph)
    return tmp_density

# radius 需要重写 good!!!!!!!!!!
def get_radius(a_graph):
    tmp_start_time = time.time()
    tmp_radius = nx.radius(a_graph)
    tmp_end_time = time.time()
    print(tmp_end_time - tmp_start_time)
    return tmp_radius

# diameter 需要重写 good!!!!!!!!!!
def get_diameter(a_graph):
    tmp_start_time = time.time()
    tmp_diameter = nx.diameter(a_graph)
    tmp_end_time = time.time()
    print(tmp_end_time - tmp_start_time)
    return tmp_diameter

# degree centraliy 不需要重写 good!!!!!!!!!!
def get_degree_centrality(a_graph):
    tmp_degree_centrality = nx.degree_centrality(a_graph)
    return tmp_degree_centrality

# closenness centrality 需要重写 good!!!!!!!!
def get_closeness_centrality(a_graph):
    tmp_start_time = time.time()
    tmp_closeness = nx.closeness_centrality(a_graph)
    tmp_end_time = time.time()
    print(tmp_end_time - tmp_start_time)
    return tmp_closeness

# betweenness centrality 需要重写 good!!!!!!!!!
def get_betweenness_centrality(a_graph):
    tmp_betwen = nx.betweenness_centrality(a_graph)
    return tmp_betwen

# girvan_newman 需要重写 good!!!!!!!!!!!写了一个fast greedy的算法
def get_girvan_newman_communities(a_graph):
    nx.community.girvan_newman(a_graph)
    return None
# done
def get_modularities(a_graph):
    nx.community.modularity()
    return None

if __name__ == '__main__':
    print('a')
    test_data = get_facebook_test_data()
    test_g = generate_undirected_graph(test_data, ifweighted=False)
    largest_sub = get_max_sub_component(test_g)
    density = get_density(largest_sub)
    radius = get_radius(largest_sub)
    diameter = get_diameter(largest_sub)
    degree_centra = get_degree_centrality(largest_sub)
    closeness_centra = get_closeness_centrality(largest_sub)  # 1768.3649997711182
    betweenness_centra = get_betweenness_centrality(largest_sub)
