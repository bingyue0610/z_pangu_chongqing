"""
1. 读图
2. 计算：
密度
聚类系数
原图节点数，边数
最大联通子图节点数，边数
3.写入同一个txt
"""

import pandas as pd
import numpy as np
import networkx as nx
import random
import time
import copy

def get_sample_graph_n_nodes_m_deges(n, m):
    random.seed(5)
    tmp_list = []
    for i in range(m):
        tmp_list.append([str(random.randint(0, n)), str(random.randint(0, n))])
    tmp_test_g = nx.Graph()
    tmp_test_g.add_edges_from(tmp_list)
    return tmp_test_g


def load_data_graph(file_path):
    if 'csv' in file_path:
        tmp_df = pd.read_csv(str(file_path))
    else:
        tmp_df = pd.read_excel(str(file_path))

    data_array = np.array(tmp_df)
    tmp_g = nx.Graph()
    tmp_g.add_edges_from(data_array)
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

# average_clustering
def get_clustering_coefficeint(a_graph):
    return nx.average_clustering(a_graph)

def all_go(file_path_read, file_path_write):
    test_graph = load_data_graph(file_path_read)
    nodes_test_g = len(test_graph.nodes)
    edges_test_g = len(test_graph.edges)
    largest_sub = get_max_sub_component(test_graph)
    nodes_larg_sub = len(largest_sub.nodes)
    edges_larg_sub = len(largest_sub.edges)
    density = get_density(largest_sub)
    clustering = get_clustering_coefficeint(largest_sub)

    tmp_file = open(str(file_path_write), 'w')
    tmp_file.write('all_nodes:', nodes_test_g)
    tmp_file.write('\n')
    tmp_file.write('all_edges:', edges_test_g)
    tmp_file.write('\n')
    tmp_file.write('largest_sub_ndoes:', nodes_larg_sub)
    tmp_file.write('\n')
    tmp_file.write('largest_sub_edges:', edges_larg_sub)
    tmp_file.write('\n')
    tmp_file.write('density:', density)
    tmp_file.write('\n')
    tmp_file.write('clustering_coefficient:', clustering)
    tmp_file.write('\n')
    tmp_file.close()

if __name__ == '__main__':
    all_go('read_data.csv', 'write_path.txt')
