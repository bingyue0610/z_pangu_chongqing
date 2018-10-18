import networkx as nx
import random
import time

def get_some_test_data():
    tmp_list = []
    tmp_list.append([1, 2, 1])
    tmp_list.append([1, 3, 1])
    tmp_list.append([2, 3, 1])
    return tmp_list

def bihuan1(data_list, low_bound, high_bound):
    """
    无向图，找到绝对闭环1
    data_list = [[node1, node2, weight1], ...]
    :param high_bound: 边数上限
    :param low_bound: 边数上限
    :param data_list:
    :return:
    """
    tmp_g = nx.Graph()
    tmp_g.add_weighted_edges_from(data_list)

    nodes_list = []
    for sub in nx.connected_components(tmp_g):
        tmp_size = len(sub)
        if low_bound < tmp_size < high_bound:
            tmp_subgraph_edges = len(tmp_g.subgraph(sub).edges())
            if tmp_size * (tmp_size - 1) == 2 * tmp_subgraph_edges:
                nodes_list.append(list(tmp_g.subgraph(sub).nodes()))
    return nodes_list

if __name__ == '__main__':
    print('a')
    data = get_some_test_data()
    nodes = bihuan1(data, 2, 11)