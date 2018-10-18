import pandas as pd
import numpy as np
import networkx as nx
import random
import time
import copy

def get_sample_graph_100n_1000m(n, m):
    random.seed(5)
    tmp_list = []
    for i in range(m):
        tmp_list.append([str(random.randint(0, n)), str(random.randint(0, n))])
    tmp_test_g = nx.Graph()
    tmp_test_g.add_edges_from(tmp_list)
    return tmp_test_g

# 最大联通分量A 不用重写good!!!!!!!!!!!!
def get_max_sub_component(a_graph):
    largest_components = max(nx.connected_components(a_graph), key=len)
    tmp_largest_sub = a_graph.subgraph(largest_components)
    return tmp_largest_sub

# clustering_coeffcient
def get_clustering_coefficeint(graph):
    nodes_cluastering_dict = nx.clustering(graph)
    cluastering = nx.average_clustering(largest_sub)
    tmp_list = []
    for key in nodes_cluastering_dict:
        tmp_list.append([key, nodes_cluastering_dict[key], cluastering])
    df0 = pd.DataFrame(tmp_list, columns=['nodesid', 'nodes_clustering', 'average_clustering'])
    return df0

# lpa 社区发现
def lpa():
    nx.community.asyn_lpa_communities()
    nx.community.mudularities()
    pass

# density
def get_density(a_graph):
    tmp_density = nx.density(a_graph)
    return tmp_density



if __name__ == '__main__':
    test_g = get_sample_graph_100n_1000m(10000, 10000)
    largest_sub = get_max_sub_component(test_g)


    print(len(test_g))
    print(len(largest_sub))
    len(test_g)
    len(largest_sub)


    # ai = nx.community.asyn_lpa_communities(largest_sub)


    time1 = time.time()
    df_clustering = get_clustering_coefficeint(test_g)
    df_clustering.to_csv('clustering.csv', index=False)
    time2 = time.time()
    print(time2 - time1)
    print(df_clustering.head())
    # 写入graphml
    nx.readwrite.graphml.write_graphml(test_g, 'test_g')

