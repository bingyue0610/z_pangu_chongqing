import numpy as np
import pandas as pd
import json
import networkx as nx
from flask import Flask, jsonify, request
import random

app = Flask(__name__)


@app.route('/', methods=['GET'])
def test():
    return jsonify({'message': 'it works'})


# 0. 获取数据&返回
"""
1. nodes_dict
2. links_dict
3. graph_type: 包含两个参数'digraph','graph'。分别表示有向图与无向图。
4. method: 包含调用的模型方法:
degree_centrality
closeness_centrality
betweenness_centrality
lpa_detection
girvan_newman_detection
connected_components
density
radius
diameter
"""
@app.route('/models', methods=['POST'])
def method1():
    nodes_dict = request.json['nodes']
    links_dict = request.json['links']
    graph_type = request.json['graph_type']
    method = request.json['method']
    print(nodes_dict)
    print(links_dict)
    print(graph_type)
    print(method)
    relations = get_relation_data(nodes_dict)
    print(relations)
    test_g = get_graph(relations, 'graph')
    print(type(test_g))
    print(test_g.edges)

    # method = 'degree_centrality' 返回：{图内节点id: 分数}
    if method == 'degree_centrality':
        degree_centra = cal_degree_centrality(test_g)
        return jsonify(degree_centra)

    # method = 'closeness_centrality' 返回：{图内节点id: 分数}
    if method == 'closeness_centrality':
        closeness_centra = get_closeness_centrality(test_g)
        return jsonify(closeness_centra)

    # method = 'betweenness_centrality' 返回：{图内节点id: 分数}
    if method == 'betweenness_centrality':
        betweenness_centra = get_betweenness_centrality(test_g)
        return jsonify(betweenness_centra)

    # method = 'lpa_detection' 返回：{社区内的节点id： 社区编号}
    if method == 'lpa_detection':
        if nx.is_directed(test_g):
            return jsonify({'error_mesage': "have to be a un_directed graph"})
        else:
            lpa_commu = cal_lpa_community_detection(test_g)
        return jsonify(lpa_commu)

    # method = 'girvan_newman_detection' 返回：{社区内的节点id： 社区编号}
    if method == 'girvan_newman_detection':
        gn_commu = get_girvan_newman_communities(test_g)
        return jsonify(gn_commu)

    # method = 'connected_components' 返回：{最大联通量节点id：1}
    if method == 'connected_components':
        if nx.is_directed(test_g):
            return jsonify({'error_mesage': "have to be a un_directed graph"})
        else:
            largest_component = get_max_sub_component(test_g)
        return jsonify(largest_component)

    # method = 'density' 返回：{'density': 分数}
    if method == 'density':
        density = get_density(test_g)
        return jsonify({'density': density})

    # method = 'radius' 返回：{'connected_components'：{图内节点id：所属联通量编号}， 'components_radius'：{联通两编号：分数}}
    if method == 'radius':
        if nx.is_directed(test_g):
            return jsonify({'error_mesage': "have to be a un_directed graph"})
        else:
            connected_components, components_radius = get_radius(test_g)
        return jsonify({'connected_components': connected_components, 'components_radius': components_radius})

    # method = 'diameter' 返回：{'connected_components'：{图内节点id：所属联通量编号}， 'components_diameter'：{联通两编号：分数}}
    if method == 'diameter':
        if nx.is_directed(test_g):
            return jsonify({'error_mesage': "have to be a un_directed graph"})
        else:
            connected_components, components_diameter = get_diameter(test_g)
        return jsonify({'connected_components': connected_components, 'components_diameter': components_diameter})

    return jsonify({'good': '1'})
    # return jsonify(some_df.to_json(orient='index'))


# 1. 得到关系三元组，有向，startId指向endId。
def get_relation_data(a_nodes_dict):
    relation_list = []
    for a_relation in a_nodes_dict:
        tmp_relation = [str(a_relation['startId']), str(a_relation['endId'])]
        relation_list.append(tmp_relation)
    return relation_list

# 2. 根据数据关系作图。强调为有向图或者无向图。
def get_graph(a_relation_list, a_graph_type):
    if a_graph_type == 'graph':
        a_graph = nx.Graph()
    elif a_graph_type == 'digraph':
        a_graph = nx.DiGraph()

    a_graph.add_edges_from(a_relation_list)
    return a_graph

# 3. degree centraliy
def cal_degree_centrality(a_graph):
    tmp_degree_centrality = nx.degree_centrality(a_graph)
    return tmp_degree_centrality

# 4. closenness centrality
def get_closeness_centrality(a_graph):
    tmp_closeness = nx.closeness_centrality(a_graph)
    return tmp_closeness

# 5. betweenness centrality
def get_betweenness_centrality(a_graph):
    tmp_betwen = nx.betweenness_centrality(a_graph)
    return tmp_betwen

# 6. LPA return a dict
def cal_lpa_community_detection(a_graph):
    tmp_lpa_community = nx.community.label_propagation_communities(a_graph)
    communities_label_dict = {}
    count_label = 1
    for a_community in tmp_lpa_community:
        for node_id in a_community:
            communities_label_dict[node_id] = count_label
        count_label += 1
    return communities_label_dict

# 7. girvan_newman
def get_girvan_newman_communities(a_graph):
    tmp_gn_community = nx.community.girvan_newman(a_graph)

    largest_modularity = -100
    for a_communities in tmp_gn_community:
        tmp_modularity = nx.community.modularity(a_graph, a_communities)
        if tmp_modularity > largest_modularity:
            largest_modularity = tmp_modularity
            best_community = a_communities

    communities_label_dict = {}
    count_label = 1
    for a_community in best_community:
        for node_id in a_community:
            communities_label_dict[node_id] = count_label
        count_label += 1
    return communities_label_dict

# 8. connected_components
def get_max_sub_component(a_graph):
    largest_components = max(nx.connected_components(a_graph), key=len)
    largest_components_dict = {}
    for node in largest_components:
        largest_components_dict[node] = 1
    return largest_components_dict

# 9. density
def get_density(a_graph):
    tmp_density = nx.density(a_graph)
    return tmp_density

# 10. radius
def get_radius(a_graph):
    connected_components = nx.connected_components(a_graph)
    connected_components_dict = {}
    radius_dict = {}
    label_count = 1
    for a_sub_component in connected_components:
        tmp_sub = a_graph.subgraph(a_sub_component)
        for node in tmp_sub:
            connected_components_dict[node] = label_count
        tmp_radius = nx.radius(tmp_sub)
        radius_dict[str(label_count)] = tmp_radius
        label_count += 1
    return connected_components_dict, radius_dict

# 11. diameter
def get_diameter(a_graph):
    connected_components = nx.connected_components(a_graph)
    connected_components_dict = {}
    diameter_dict = {}
    label_count = 1
    for a_sub_component in connected_components:
        tmp_sub = a_graph.subgraph(a_sub_component)
        for node in tmp_sub:
            connected_components_dict[node] = label_count
        tmp_diameter = nx.diameter(tmp_sub)
        diameter_dict[str(label_count)] = tmp_diameter
        label_count += 1
    return connected_components_dict, diameter_dict

# 0000 test_g
def sample_graph(graph_type='graph'):
    random.seed(5)
    tmp_list = []
    for i in range(100):
        tmp_list.append([str(random.randint(0, 100)), str(random.randint(0, 100))])
    if graph_type == 'graph':
        tmp_test_g = nx.Graph()
    elif graph_type == 'digraph':
        tmp_test_g = nx.DiGraph()
    tmp_test_g.add_edges_from(tmp_list)
    return tmp_test_g

if __name__ == '__main__':
    app.run(debug=True, port=7222)

    test_di_g = sample_graph('digraph')
    test_undi_g = sample_graph('graph')
    # di_degree = cal_degree_centrality(test_di_g)
    # undi_degree = cal_degree_centrality(test_undi_g)
    # di_close = get_closeness_centrality(test_di_g)
    # undi_close = get_closeness_centrality(test_undi_g)
    # di_bet = get_betweenness_centrality(test_di_g)
    # undi_bet = get_betweenness_centrality(test_undi_g)
    # di_lpa = cal_lpa_community_detection(test_di_g)
    # undi_lpa = cal_lpa_community_detection(test_undi_g)
    # di_gn = get_girvan_newman_communities(test_di_g)
    # undi_gn = get_girvan_newman_communities(test_undi_g)
    # di_largest_component = get_max_sub_component(test_di_g)
    # undi_largest_component = get_max_sub_component(test_undi_g)
    # di_density = get_density(test_di_g)
    # undi_density = get_density(test_undi_g)
    # di_radius = get_radius(test_di_g)
    # undi_radius = get_radius(test_undi_g)
    # di_diameter = get_diameter(test_di_g)
    # undi_diameter = get_diameter(test_undi_g)




