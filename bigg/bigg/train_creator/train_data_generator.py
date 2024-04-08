import networkx as nx
import numpy as np
import torch
import random
from torch import nn
from torch.nn.parameter import Parameter
import os
import scipy
from bigg.train_creator.data_util import *

def tree_generator(n):
    '''
    Generates a random bifurcating tree w/ n nodes
    Args:
        n: number of leaves
    '''
    g = nx.Graph()
    for j in range(n - 1):
        if j == 0:
            g.add_edges_from([(0, 1), (0, 2)])
        else:
            sample_set = [k for k in g.nodes() if g.degree(k) == 1]
            selected_node = random.sample(sample_set, 1).pop()
            g.add_edges_from([(selected_node, 2*j+1), (selected_node, 2*j+2)])
    return g


def graph_generator(n = 5, num_graphs = 5000, constant_topology = False,
                    constant_weights = False, mu_weight = 10, scale = 1, weighted = True):
    '''
    Generates requested number of bifurcating trees
    Args:
    	n: number of leaves
    	num_graphs: number of requested graphs
    	constant_topology: if True, all graphs are topologically identical
    	constant_weights: if True, all weights across all graphs are identical
    	mu_weight: mean weight 
    '''
    graphs = []

    if constant_topology:
        g1 = tree_generator(n)
        g = order_tree(g1)

    for _ in range(num_graphs):
        if not constant_topology:
            g1 = tree_generator(n)
            g = order_tree(g1)
        
            if weighted:
                mu = np.random.uniform(7, 13)
                weights = np.random.gamma(scale*mu*mu, 1/scale * 1/mu, 2 * n + 1) * (1 - constant_weights) + 10 * constant_weights
                
                weighted_edge_list = []
                for (n1,n2),w in zip(g.edges(), weights):
                    weighted_edge_list.append((n1, n2, w))
                
                g = nx.Graph()
                g.add_weighted_edges_from(weighted_edge_list)
        
        nodes_dict = dict()
        for node in g.nodes():
            nodes_dict[node] = node / 10
        nx.set_node_attributes(g, nodes_dict, name = 'length')

        graphs.append(g)
    return graphs

def get_rand_lobster(n, p1, p2, num_graphs, min_nodes = 1, max_nodes = 9999, weighted = False):
    num_nodes = n
    p1 = p1
    p2 = p2
    #num_graphs = 100
    #local = False
    #dist = "Uniform"
    #loc = 1
    #scale = 2
    min_nodes = min_nodes
    max_nodes = max_nodes
    
    graphs = []
    for _ in range(num_graphs):
        x = nx.random_lobster(num_nodes, p1, p2)
        while len(x) not in range(min_nodes, max_nodes + 1):
            x = nx.random_lobster(num_nodes, p1, p2)
        
        graphs.append(x)
        
        if weighted:
            edge_dict = []
            for (n1, n2) in x.edges():
                #if local:
                #     loc = (n1 + n2) / 2 
                #     scale = abs(n1 - n2 + 1) / 2
                #w = weight_distributions(dist = dist, loc = loc, scale = scale)
                w = scipy.stats.norm.rvs(loc = 1, scale = 0.5, size = 1)
                w = np.log(np.exp(w) + 1)
                edge_dict.append((n1, n2, w))
            graphs[-1].add_weighted_edges_from(edge_dict) 
    return graphs

def adj_vec(g, as_torch = True, weighted = True, normalize = False):
    '''
    Transforms nx graph into adjacency vector
    Args:
        as_torch: if True, returns tensor object.
        weighted: if True, returns weighted adjacency vector.
    '''
    n = len(g)
    if normalize:
       #Lstar = nx.normalized_laplacian_matrix(g).todense()
       #print(Lstar)
       #A = np.identity(n) - Lstar
       Astar = (nx.adjacency_matrix(g).todense() > 0) + 0
       deg = np.diag([y**-0.5 for _, y in g.degree()])
       out = np.matmul(deg, Astar)
       A = np.matmul(out, deg)
    
    else:
        A = nx.adjacency_matrix(g).todense()
    
    U = A[np.triu_indices(n, k = 0)]
    k = len(U)
    
    if not weighted and not normalize:
        U = (U > 0) + 0
    
    if as_torch:
        U = torch.tensor(U, dtype=torch.float32)
        U = U.reshape([1 , k])[0]
#    print(U)
    return U


def train_data_creator(n = 5, num_graphs = 5000, as_torch = True,
                       constant_topology = False, constant_weights = False,
                       weighted = True, scale = 1):
    '''
    Creates a set of training graphs
    Args:
        n: number of leaves
        num_graphs: number of requested graphs
        as_torch: if True, returns tensor object.
        constant_topology: if True, all graphs are topologically identical
        constant_weights: if True, all weights across all graphs are identical
        mu_weight: mean weight 
    '''
    graphs = graph_generator(n, num_graphs, constant_topology, constant_weights, scale)
    adj_vecs = []
    for g in graphs:
        U = adj_vec(g, as_torch, weighted)
        adj_vecs.append(U)
    return adj_vecs


def graph_from_adj(vec):
    '''
    Transforms (weighted) adjacency vector into (weighted) graph
    Args:
        vec: adjacency vector to be converted. Can be weighted or unweighted.
    '''
    vec = np.array(vec)
    k = len(vec)
    n = int((2*k + .25)**0.5 + 0.5)
    tri = np.zeros((n, n))
    tri[np.triu_indices(n, 0)] = vec
    tri = np.transpose(tri)
    tri[np.triu_indices(n, 0)] = vec   
    g = nx.from_numpy_array(tri)
    return g


## NEED TO INCORPORATE WEIGHTED GRAPHS.......
def order_tree(G, by_path = True): 
    n = len(G)
    leaves = sorted([x for x in G.nodes() if G.degree(x)==1])
    nodes = sorted([x for x in G.nodes() if x not in leaves])
    npl = nodes + leaves
    if by_path:
        npl = [node for node in nx.shortest_path(G, 0)]
    reorder = dict()
    #print(npl)
    for k in range(n):
        reorder[npl[k]] = k
    new_G = nx.relabel_nodes(G, mapping = reorder)
    return new_G