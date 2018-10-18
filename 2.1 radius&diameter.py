import pandas as pd
import numpy as np
import networkx as nx
import random
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

# 此处可做分布式。将nodes 分成多份的node_list， 和G（图）一起传入。最后收集最大，最小值后
# 再比较最大值中的最大值，和最小值中的最小值。
def cal_diameter_and_radius(G, node_list):
    e = {}
    for node in node_list:
        length = nx.single_source_shortest_path_length(G, node)
        e[node] = max(length.values())
    tmp_diameter = max(e.values())
    tmp_radius = min(e.values())
    return tmp_diameter, tmp_radius

if __name__ == '__main__':
    print('a')
    test_data = get_facebook_test_data()
    test_g = generate_undirected_graph(test_data, ifweighted=False)
    largest_sub = get_max_sub_component(test_g)
    diameter, radius = cal_diameter_and_radius(largest_sub, largest_sub.nodes)
