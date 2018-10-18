import networkx as nx
import random
import time


# test = max(nx.connected_components(g),key=len)

# test = list(nx.connected_components(g))
# g.add_edge(7,1)
# g.number_of_nodes()
# g.number_of_edges()
def test_sub():
    g = nx.Graph()
    g.clear()
    g.add_edge(1, 2)
    g.add_edge(3, 2)
    g.add_edge(3, 1)
    g.add_edge(4, 5)
    g.add_edge(4, 6)
    g.add_edge(4, 7)
    g.add_edge(4, 8)
    g.add_edge(5, 6)
    g.add_edge(5, 7)
    g.add_edge(5, 8)
    g.add_edge(6, 7)
    g.add_edge(6, 8)
    g.add_edge(7, 8)

    return g


def create_big_gragh():
    g = nx.Graph()
    for i in range(0, 1000):
        if i % 100 == 1:
            print(i)
        g.add_edge(random.randint(100, 500), random.randint(100, 500))
    return g


def create_file():
    fb = open()
    for i in range(0, 10000000):
        fb.write(','.join(
            [str(random.randint(10000000000, 20000000000)), str(random.randint(10000000000, 20000000000))]) + '\n')


if __name__ == '__main__':

    start = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(start)
    # create_file()

    g = create_big_gragh()
    # g = test_sub()
    #
    print('node:%s' % g.number_of_nodes())
    print('edges:%s' % g.number_of_edges())

    start = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(start)
    res = {}
    num = 0
    for sub in nx.connected_components(g):
        size = len(sub)

        if 2 < size < 11000:
            num += 1
            subgraph_edges = len(g.subgraph(sub).edges())
            if size * (size - 1) == 2 * subgraph_edges:
                if str(size) in res:
                    res[str(size)] += 1
                else:
                    res[str(size)] = 1

    print(res)
    print(num)
    end = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(end)

    # if len(sub)==5:
    # print sub
    # num += 1
    # print num
