import pandas as pd
import numpy as np
import networkx as nx
import random
import time
import copy


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
        tmp_list.append([str(random.randint(0, 100)), str(random.randint(0, 100))])
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
    new_g = nx.Graph()
    new_g.add_nodes_from(list(a_graph.nodes))
    # print('new_g', tuple(nx.connected_components(new_g)))
    # print('new_g_edges', new_g.edges)
    left_edges = list(a_graph.edges)
    count = 0
    tmp_file = open('fast_greedy_community_detection_n20_m100.txt', 'w')
    while len(left_edges) > 0:
        added_edge = left_edges[0]
        max_modularity_community = tuple(nx.connected_components(new_g))
        max_new_community_modularity = new_community_and_new_modularity(a_graph, new_g, added_edge)[1]
        for edge in left_edges:
            tmp_community, tmp_new_modularity = new_community_and_new_modularity(a_graph, new_g, edge)
            if tmp_new_modularity > max_new_community_modularity:
                max_new_community_modularity = tmp_new_modularity
                added_edge = edge
                max_modularity_community = tmp_community
        # print('added_edge11111111:', added_edge)
        # print('community_modularity1111111111111:', max_new_community_modularity)
        # print('community11111111111111111', max_modularity_community)
        new_g.add_edge(*added_edge)
        left_edges.remove(added_edge)
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


def new_community_and_new_modularity(a_graph, a_sub_graph, a_edge):
    a_new_graph = copy.deepcopy(a_sub_graph)
    a_new_graph.add_edge(a_edge[0], a_edge[1])
    # print('g_edges2222222222222', a_new_graph.edges)
    tmp_new_community = tuple(nx.connected_components(a_new_graph))
    tmp_modularity = nx.community.modularity(a_graph, tmp_new_community)
    return tmp_new_community, tmp_modularity


if __name__ == '__main__':
    print('a')
    # test_data = get_facebook_test_data()
    # test_g = generate_undirected_graph(test_data, ifweighted=False)
    # test_g = get_sample_graph_100n_1000m()
    test_g = get_sample_graph_20n_100m()
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
