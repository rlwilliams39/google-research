import os
import sys
import numpy as np
import random
import torch
import torch.optim as optim
import networkx as nx

def get_node_map(nodelist, shift=0):
    node_map = {}
    for i, x in enumerate(nodelist):
        node_map[x + shift] = i + shift
    return node_map


def apply_order(G, nodelist, order_only):
    if order_only:
        return nodelist
    node_map = get_node_map(nodelist)
    g = nx.relabel_nodes(G, node_map)
    return g

def order_tree(G, leaves_last = True): 
    n = len(G)
    leaves = sorted([x for x in G.nodes() if G.degree(x)==1])
    nodes = sorted([x for x in G.nodes() if x not in leaves])
    
    npl = [node for node in nx.single_source_dijkstra(G, 0)[0]]
    
    if leaves_last:
        npl_n = [node for node in npl if node in nodes]
        npl_l = [node for node in npl if node in leaves]
        npl = npl_n + npl_l
    
    reorder = dict()
    for k in range(n):
        reorder[npl[k]] = k
    new_G = nx.relabel_nodes(G, mapping = reorder)
    return new_G


def get_graph_data(G, node_order, leaves_last = True, order_only=False):
    G = G.to_undirected()
    out_list = []
    orig_node_labels = sorted(list(G.nodes()))
    orig_map = {}
    for i, x in enumerate(orig_node_labels):
        orig_map[x] = i
    G = nx.relabel_nodes(G, orig_map)
    
    if node_order == 'default':
        out_list.append(apply_order(G, list(range(len(G))), order_only))
    
    elif node_order == 'DFS' or node_order == 'BFS':
            ### BFS & DFS from largest-degree node
            CGs = [G.subgraph(c) for c in nx.connected_components(G)]
            
            # rank connected componets from large to small size
            CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
            
            node_list_bfs = []
            node_list_dfs = []
            
            for ii in range(len(CGs)):
                node_degree_list = [(n, d) for n, d in CGs[ii].degree()]
                degree_sequence = sorted(
                    node_degree_list, key=lambda tt: tt[1], reverse=True)
                bfs_tree = nx.bfs_tree(CGs[ii], source=degree_sequence[0][0])
                node_list_bfs += list(bfs_tree.nodes())
                dfs_tree = nx.dfs_tree(CGs[ii], source=degree_sequence[0][0])
                node_list_dfs += list(dfs_tree.nodes())
            
            if node_order == 'BFS':
                node_list_bfs[0], node_list_bfs[1] = node_list_bfs[1], node_list_bfs[0]
                out_list.append(apply_order(G, node_list_bfs, order_only))
            if node_order == 'DFS':
                node_list_dfs[0], node_list_dfs[1] = node_list_dfs[1], node_list_dfs[0]
                out_list.append(apply_order(G, node_list_dfs, order_only))
    
    else: 
        if node_order == "time":
            out_list.append(order_tree(G, leaves_last))
    
    if len(out_list) == 0:
        out_list = [apply_order(G, list(range(len(G))), order_only)]
    
    return out_list