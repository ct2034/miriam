#!/usr/bin/python
import sys
import networkx as nx
import heapq
import copy

DEBUG = True
DEBUG_verbose = False

# Get arguments
filename = sys.argv[1]
if DEBUG:
    print
"Read file from" + filename

# Relabel the nodes to use the GEXF node "label" attribute instead of the
# node"id" attribute as the NetworkX node label.
G = nx.read_gexf(filename, relabel=True)
G_leftNodes = copy.deepcopy(G)
T = G.nodes()
list.sort(T)
g = [-1 for i in
     range(len(T))]  # g[]: the path length from root node to the specific
node in the graph G. Initial values are - 1.
f = [-1 for i in range(len(T))]
# 1 means visited; -1 means unvisited
visit_status = [-1 for i in range(len(T))]
# parent[]: store the names of parent nodes of nodes T[]
parent = [T[0] for i in range(len(T))]

# set information about the start node
g[int(T[0])] = 0
heap = []
heapq.heappush(heap, (0, T[0]))

count = 0  # variable for debug
while heap:
    [estimate_min_path, current_node] = heapq.heappop(heap)
    neighbors = G.neighbors(current_node)

    # check if the node has been visited. -1 means unvisited; otherwise visited
    if visit_status[int(current_node)] == -1:
        visit_status[int(current_node)] = 1
        if DEBUG:
            print
        str(count) + "th loop: node " + current_node + ", visited " + str(
            g[int(current_node)]) + ", estimate total path " +
        str(estimate_min_path)
        count = count + 1  # debug variable
        heap = []  # clear old data in heap
        # the graph consists of unvisited nodes
        G_leftNodes.remove_node(current_node)

        for neighbor in neighbors:
            neighbor_index = int(neighbor)
            edge_weight = float(G.edge[current_node][neighbor]['weight'])
            if visit_status[neighbor_index] == 1:
                continue
            if visit_status[neighbor_index] == -1:
                # subG_leftNodes = G_leftNodes
                subG_leftNodes = copy.deepcopy(G_leftNodes)
                subG_leftNodes.remove_node(neighbor)
                # Computing mst
                mst = nx.minimum_spanning_tree(subG_leftNodes, weight='weight')
                edges_mst = mst.edges(data=True)
                # edges_mst = nx.minimum_spanning_edges(subG_leftNodes, weight
                # = 'weight', data = True)
                mst_sum = 0
                for edge_node1, edge_node2, edge_data in edges_mst:
                    edge_weight_mst = edge_data['weight']
                    mst_sum += edge_weight_mst
                # connect mst with neighbor and T[o]
                nodes_mst = mst.nodes(data=False)
                min_neighbor2mst = sys.maxint
                min_root2mst = sys.maxint
                if not nodes_mst:
                    min_neighbor2mst = G[neighbor][T[0]]['weight']
                    min_root2mst = 0
                for node in nodes_mst:
                    if min_neighbor2mst > G[neighbor][node]['weight']:
                        min_neighbor2mst = G[neighbor][node]['weight']
                    if min_root2mst > G[T[0]][node]['weight']:
                        min_root2mst = G[T[0]][node]['weight']
                h = mst_sum + min_neighbor2mst + min_root2mst
                g[neighbor_index] = g[int(current_node)] + edge_weight
                f[neighbor_index] = g[int(current_node)] + edge_weight + h
                if DEBUG_verbose:
                    print
                    "g:" + str(g[neighbor_index]) + "\t h:" + \
                        str(h) + "\tf:" + str(f[neighbor_index])
                parent[neighbor_index] = current_node
                heapq.heappush(heap, [f[neighbor_index], neighbor])
    else:
        if (DEBUG):
            print
        "error (from DEBUG): found visited node" + current_node

# Return back to the source node
parent[int(T[0])] = current_node
g[int(T[0])] = g[int(current_node)] + G[current_node][T[0]][
    'weight']
# g[0] store the path from the last unvisited node to source node

# print detailed results in DEBUG mode
if DEBUG:
    print
    "current node: " + current_node
    for path_length in g:
        print
        path_length

path_array = ["0" for i in range(len(T))]
node_counter = len(T) - 1
path_array[node_counter] = current_node
while node_counter > 0:
    path_array[node_counter - 1] = parent[int(path_array[node_counter])] + ' '
    node_counter -= 1
if DEBUG:
    print
path_array

# output results
print
"Tour: " + ''.join(path_array)
print
"Cost: " + str(g[int(T[0])])
