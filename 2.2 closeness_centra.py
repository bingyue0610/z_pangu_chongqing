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
# 再将收集的结果合并。
def house_special_closeness_centrality(G, node_list, wf_improved=False, reverse=False):
    if G.is_directed() and not reverse:
        path_length = nx.single_target_shortest_path_length
    else:
        path_length = nx.single_source_shortest_path_length

    nodes = node_list
    closeness_centrality = {}
    for n in nodes:
        sp = dict(path_length(G, n))
        totsp = sum(sp.values())
        if totsp > 0.0 and len(G) > 1:
            closeness_centrality[n] = (len(sp) - 1.0) / totsp
            # normalize to number of nodes-1 in connected part
            if wf_improved:
                s = (len(sp) - 1.0) / (len(G) - 1)
                closeness_centrality[n] *= s
        else:
            closeness_centrality[n] = 0.0
    return closeness_centrality


if __name__ == '__main__':
    print('a')
    test_data = get_facebook_test_data()
    test_g = generate_undirected_graph(test_data, ifweighted=False)
    largest_sub = get_max_sub_component(test_g)
    closeness_centra = house_special_closeness_centrality(largest_sub, largest_sub.nodes)
