# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# pylint: skip-file

import os
import sys
import networkx as nx
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def get_node_map(nodelist, shift=0):
    node_map = {}
    for i, x in enumerate(nodelist):
        node_map[x + shift] = i + shift
    return node_map


def apply_order(G, nodelist, order_only, leaflist = None):
    if order_only:
        return nodelist
    
    if leaflist is not None:
        #nodes then leaves
        #nodelist = nodelist + leaflist
        #leaves then nodes
        nodelist = leaflist + nodelist + [0]
    
    node_map = get_node_map(nodelist)
    g = nx.relabel_nodes(G, node_map)
    
    e_list = []
    #for e in g.edges():
    #    if e[0] > e[1]:
    #        e_list.append((e[1], e[0]))
    #    else:
    #        e_list.append(e)
    
    for node1, node2, data in g.edges.data():
        if node1 > node2:
            e_list.append((node2, node1, data['weight']))
        else:
            e_list.append((node1, node2, data['weight']))
            
    g = nx.Graph()
    g.add_nodes_from(list(range(len(G))))
    g.add_weighted_edges_from(sorted(e_list))
    return g



def get_graph_data(G, node_order, order_only=False):
    G = G.to_undirected()
    
    skip = True
    if skip:
        return G
    
    out_list = []
    orig_node_labels = sorted(list(G.nodes()))
    orig_map = {}
    for i, x in enumerate(orig_node_labels):
        orig_map[x] = i
    G = nx.relabel_nodes(G, orig_map)
    
    if node_order == 'default':
        out_list.append(apply_order(G, list(range(len(G))), order_only))
    
    if node_order == 'by_time': 
        def order_tree(G, find_leaves= False):
            leaves = [x for x in G.nodes() if G.degree(x)==1 and x!=0]
            nodes = [x for x in G.nodes() if x not in leaves and x!=0]
            both = leaves+nodes
            dist = []
        
            def length_iterator(path):
                if len(path) == 0 or path == "None":
                    return(0)
                else:
                    for node in path:
                        z = G.nodes[node]['length']
                        path.pop(0)
                        return(z + length_iterator(path))
        
            if find_leaves:
                Z = leaves
        
            else:
                Z = nodes
            
            for val in Z:
                if val == 0:
                    path = "None"
                else: 
                    i=0
                    for y in nx.all_simple_paths(G, source=0, target=val):
                        if i == 1:
                            print("ERROR: MULTIPLE PATHS DETECTED")
                            break
                        path = y 
                        i = i+1     
                dist.append(length_iterator(path))
            A = np.array(dist)
            B = np.array(Z)
            inds = A.argsort()
            ordered = list(B[inds])
            return(ordered)
        
        node_list_time = order_tree(G)
        leaf_list_time = [x for x in G.nodes() if G.degree(x)==1 and x!=0]
        leaf_list_time = sorted(leaf_list_time)
        out_list.append(apply_order(G, node_list_time, order_only, leaflist = leaf_list_time))
    
    else:
        if node_order == 'DFS' or node_order == 'BFS':
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
                #print(node_list_bfs)
                node_list_bfs[0], node_list_bfs[1] = node_list_bfs[1], node_list_bfs[0]
                #print(node_list_bfs)
                out_list.append(apply_order(G, node_list_bfs, order_only))
            if node_order == 'DFS':
                node_list_dfs[0], node_list_dfs[1] = node_list_dfs[1], node_list_dfs[0]
                #print(node_list_dfs)
                out_list.append(apply_order(G, node_list_dfs, order_only))
    
    if len(out_list) == 0:
        out_list = [apply_order(G, list(range(len(G))), order_only)]
    
    return out_list


def get_rand_grid(n_nodes, n_d=5):
    graphs = []
    for i in range(n_nodes - n_d, n_nodes + n_d):
        for j in range(n_nodes - n_d, n_nodes + n_d):
            graphs.append(nx.grid_2d_graph(i, j))
    return graphs


def gen_connected(g_type, min_n, max_n, **kwargs):
    n_tried = 0
    while n_tried < 100:
        n_tried += 1
        cur_n = np.random.randint(max_n - min_n + 1) + min_n
        if g_type == 'erdos_renyi':
            g = nx.erdos_renyi_graph(n=cur_n, p=kwargs['er_p'])
        else:
            raise NotImplementedError

        g_idx = max(nx.connected_components(g), key=len)
        gcc = g.subgraph(list(g_idx))
        # generate another graph if this one has fewer nodes than min_n
        if nx.number_of_nodes(gcc) < min_n:
            continue

        max_idx = max(gcc.nodes())
        if max_idx != nx.number_of_nodes(gcc) - 1:
            idx_map = {}
            for idx in gcc.nodes():
                t = len(idx_map)
                idx_map[idx] = t

            g = nx.Graph()
            g.add_nodes_from(range(0, nx.number_of_nodes(gcc)))
            for edge in gcc.edges():
                g.add_edge(idx_map[edge[0]], idx_map[edge[1]])
            gcc = g
        max_idx = max(gcc.nodes())
        assert max_idx == nx.number_of_nodes(gcc) - 1

        # check number of nodes in induced subgraph
        if len(gcc) < min_n or len(gcc) > max_n:
            continue
        return gcc
    print('too many rejections in sampling, please check the hyper params')
    sys.exit()


def get_er_graph(n_nodes, p):
    n_min = n_nodes - 5
    n_max = n_nodes + 10

    graphs = [gen_connected('erdos_renyi', n_min, n_max, er_p=p) for _ in tqdm(range(100))]
    return graphs


def create_graphs(graph_type, noise=10.0, seed=1234):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(dir_path, '../../data')
    ### load datasets
    graphs = []
    # synthetic graphs
    if 'grid' in graph_type and graph_type != 'grid':
        n_nodes = int(graph_type[4:]) if graph_type != 'grid' else 15
        graphs = get_rand_grid(n_nodes)
    elif graph_type.startswith('er'):
        _, n_nodes, p = graph_type.split('-')
        n_nodes = int(n_nodes)
        p = float(p)
        graphs = get_er_graph(n_nodes, p)
    elif 'ba' in graph_type:
        n_nodes = int(graph_type[2:]) if graph_type != 'ba' else 100
        for i in range(n_nodes - 50, n_nodes + 50):
            graphs.append(nx.barabasi_albert_graph(i, 2))
    elif 'yeast' in graph_type:
        #cwd = os.getcwd()
        os.chdir("/content/drive/MyDrive/Projects/Data/Bigg-Data")
        graphs = nx.read_gpickle("Yeast.dat")
        os.chdir(cwd)
    else:
        # ADD GRAN to your pythonpath
        from utils.data_helper import create_graphs as gran_create
        graphs = gran_create(graph_type, data_dir, noise, seed)

    return graphs
