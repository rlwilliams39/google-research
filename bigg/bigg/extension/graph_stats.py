from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cmd
from copyreg import pickle
# pylint: skip-file

import os
import sys
import pickle as cp
import networkx as nx
import numpy as np
import random
from tqdm import tqdm
import torch
import torch.optim as optim
from collections import OrderedDict
from bigg.common.configs import cmd_args, set_device
from bigg.extension.customized_models import BiggWithEdgeLen
from bigg.model.tree_clib.tree_lib import setup_treelib, TreeLib
from bigg.experiments.train_utils import sqrtn_forward_backward, get_node_dist
from scipy.stats.distributions import chi2


def graph_stat_gen(graphs):#, train, test, kind = None):
    lognormal = False
    softplus2 = False
    print("lognormal? ", lognormal)
    print("softplus2? ", softplus2)
    results = B5_stats(graphs, transform = True)
    print(results)
    
    collect_graphs = [train, graphs, test]
    if kind is None:
        return 0
    if kind == "GroupA1.dat":
        results = A1_stats(collect_graphs)
        return 0
        
    if kind == "GroupB5.dat":
        results = B5_stats(collect_graphs, transform = True)
        return 0
    
    kind = "GroupB5-1.dat"
    
    if kind in ["GroupB5-1.dat", "GroupB5-2.dat", "GroupB5-N5.dat", "GroupB5-N25.dat", "GroupB5-N50.dat", "GroupB5-N100.dat", "GroupB5-N25_NEW.dat"]:
       results = B5_stats(collect_graphs, transform = False)
       return results
       
    if kind == "Yeast.dat":
        result = Yeast_stats(collect_graphs)
        return 0
    else:
        return 0
    return 0

def Yeast_stats(graphs):
    for idx in range(2):
        cur_graphs = graphs[idx]
        n = len(cur_graphs)
        k = 0
        if idx == 0:
            print("TRAINING -- SANITY CHECK")
        else:
            print("RESULTS LOADING")
        for T in cur_graphs:
            if nx.is_tree(T):
                k += 1
        print("Proportion of Trees Produced: ", k / n)
    return 0

def A1_stats(graphs):
    test_graphs = graphs[2]
    for idx in range(2):
        cur_graphs = graphs[idx]
        if idx == 0:
            print("TRAINING GRAPHS:")
            print("SKIPPING TRAINING SET")
            continue
        else:
            print("GENERATED GRAPHS:")
        
        dist = np.round(dist_met(cur_graphs, test_graphs, N = 100000, swap = (idx != 0), scale = True), 3)
        w_list = [] #List of weights grouped by order:
        within_var = []
        bad_topology = 0
        
        for T in cur_graphs:
            weights = []
            skip = False
            for (n1, n2, w) in T.edges(data=True):
                if (n1, n2) not in [(1, 0), (0, 2), (1,3), (1, 4), (0, 1)]:
                    bad_topology += 1
                    skip = True
                else:
                    weights.append(w['weight'])
            if len(weights) != 4:
                bad_topology += 1
                skip = True
            if not skip:
                within_var.append(np.std(weights))
                for i in range(len(weights)): 
                    w_list.append(weights[i])
        
        med = np.round(np.median(w_list), 4)
        mt = np.round(np.mean(w_list), 4)
        st = np.round(np.std(w_list), 4)
        n = len(cur_graphs)
        lo = np.round(np.mean(w_list) - 1.96 * st / n**0.5, 4)
        up = np.round(np.mean(w_list) + 1.96 * st / n**0.5, 4)
        p = np.round(bad_topology / n, 4)
        results = [dist, 1-p, med, mt, lo, up, st, np.round(np.mean(within_var)**0.5, 4)]
        if idx == 0:
            print("results = [dist, 1-p, med, mt, lo, up, st, np.round(np.mean(within_var), 4)]")
        print(results)
    return 0


def B5_stats(graphs, transform = False):
    #test_graphs = graphs[2]
    for idx in range(1):
        cur_graphs = graphs#[idx]
        
        #if idx == 0:
        #    print("TRAINING GRAPHS:")
        #    k = len(cur_graphs[0].edges())
        #else:
        #    print("GENERATED GRAPHS:")
        
        
        dist = 0 #np.round(dist_met(cur_graphs, test_graphs, N = 100000, swap = (idx != 0), scale = True), 3)
        weights = []
        tree_var = []
        tree_mean = []
        good_graphs = []
        
        correct = 0
        good_graphs = []
        
        for T in graphs:
            if nx.is_tree(T):
                leaves = [n for n in T.nodes() if T.degree(n) == 1]
                internal = [n for n in T.nodes() if T.degree(n) == 3]
                root = [n for n in T.nodes() if T.degree(n) == 2]
            if 2*len(leaves) - 1 == len(T) and len(leaves) == len(internal) + 2 and len(root) == 1 and len(leaves) + len(internal)+ len(root) == len(T):
                correct += 1
                good_graphs.append(T)        
        
        for T in good_graphs:    
            good_graphs.append(T)
            T_weights = []
            for (n1, n2, w) in T.edges(data = True):
                if transform:
                    t = np.log(np.exp(w['weight']) - 1)
                else:
                    t = w['weight']
                T_weights.append(t)
                weights.append(t)
            tree_var.append(np.var(T_weights, ddof = 1))
            tree_mean.append(np.mean(T_weights))
        
        xbar = np.mean(weights)
        s = np.std(weights, ddof = 1)
        n = len(weights)
        
        mu_lo = np.round(xbar - 1.96 * s / n**0.5, 3)
        mu_up = np.round(xbar + 1.96 * s / n**0.5, 3)
        
        s_lo = np.round(s * (n-1)**0.5 * (1/chi2.ppf(0.975, df = n-1))**0.5, 3)
        s_up = np.round(s * (n-1)**0.5 * (1/chi2.ppf(0.025, df = n-1))**0.5, 3)
        
        mean_tree_var = np.mean(tree_var)
        tree_var_lo = mean_tree_var - 1.96 * np.std(tree_var, ddof = 1) / len(tree_var)**0.5
        tree_var_up = mean_tree_var + 1.96 * np.std(tree_var, ddof = 1) / len(tree_var)**0.5
        
        #print("NUMBER SKIPPED: ", num_skip)
        print("NUMBER CORRECT TREE: ", correct)
        print("dist, mu-hat, mu_lo, mu_up, s, s_lo, s_up, mean_tree_var, tree_var_lo, tree_var_up")
        results = [dist, xbar, mu_lo, mu_up, s, s_lo, s_up, mean_tree_var**0.5, tree_var_lo**0.5, tree_var_up**0.5]
        print(np.round(results, 3))
    return good_graphs

def dist_met(train, test, N = 10000, swap = True, scale = False):
    n = len(train)
    m = len(test)
    assert min(n,m) > 0
    s = 0
    skip = 0
    for i in range(N):
        kn = np.random.randint(0, n)
        km = np.random.randint(0, m)
        train_g =  nx.to_numpy_array(train[kn])
        test_g = nx.to_numpy_array(test[km])
        if swap:
            train_g[:, [1, 0]] = train_g[:, [0, 1]]
            train_g[[1, 0], :] = train_g[[0, 1], :]
        if train_g.shape == test_g.shape:
            k = 1 + scale * (len(np.nonzero(train_g)[0])/2 - 1)
            s += np.sqrt(np.sum(np.square(train_g-test_g)) / k)
        else:
            skip += 1
    assert N > skip
    return(0.5 * s / (N - skip))

def feature_fixer(graphs, root = np.nan):#1e-6):
    ## Input: graphs with weighted edges
    ## Output: graphs with weighted nodes, weights corresponding to child edge
    new_graphs = []
    for g in graphs:
        g_new = nx.Graph()
        g_new.add_nodes_from(sorted(g.nodes()))
        edges = []
        for (e1, e2, w) in g.edges(data = True):
            g_new.add_edge(e1, e2)
            g_new.nodes[e2]['length'] = w['weight']
        g_new.nodes[0]['length'] = root ##np.nan (will change to NAN soon)
        new_graphs.append(g_new)
    return new_graphs
    
def reconstructor(graphs):
    ## Input: graphs with weighted nodes
    ## Output: graphs with weighted edges
    new_graphs = []
    for g in graphs:
        g_new = nx.Graph()
        g_new.add_nodes_from(sorted(g.nodes()))
        weighted_edges = []
        for (e1, e2) in g.edges():
            w = g.nodes[e2]['length']
            g_new.add_weighted_edges_from([(e1, e2, w)])
        new_graphs.append(g_new)
    return new_graphs