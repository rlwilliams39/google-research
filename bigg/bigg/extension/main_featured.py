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


def dist_met(train, test, N = 10000, swap = True, scale = False):
    n = len(train)
    m = len(test)
    assert min(n,m) > 0
    s = 0
    skip = 0
    for i in range(N):
        kn = np.random.randint(0, n)
        km = np.random.randint(0, m)
        train_g =  nx.to_numpy_matrix(train[kn])
        test_g = nx.to_numpy_matrix(test[km])
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

def get_node_feats(g):
    length = []
    for i, (idx, feat) in enumerate(g.nodes(data=True)):
        assert i == idx
        length.append(feat['length'])
    return np.expand_dims(np.array(length, dtype=np.float32), axis=1)


def get_edge_feats(g):
    edges = sorted(g.edges(data=True), key=lambda x: x[0] * len(g) + x[1])
    weights = [x[2]['weight'] for x in edges]
    return np.expand_dims(np.array(weights, dtype=np.float32), axis=1)


def debug_model(model, graph, node_feats, edge_feats):
    ll, _ = model.forward_train([0], node_feats=node_feats, edge_feats=edge_feats)
    print(ll)

    edges = []
    for e in graph.edges():
        if e[1] > e[0]:
            e = (e[1], e[0])
        edges.append(e)
    edges = sorted(edges)
    ll, _, _, _, _ = model(len(graph), edges, node_feats=node_feats, edge_feats=edge_feats)
    print(ll)
    import sys
    sys.exit()


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    set_device(cmd_args.gpu)
    setup_treelib(cmd_args)
    assert cmd_args.blksize < 0  # assume graph is not that large, otherwise model parallelism is needed

    #with open(os.path.join(cmd_args.data_dir, 'Group202A.dat'), 'rb') as f:
    #    train_graphs = cp.load(f)
    #train_graphs = nx.read_gpickle('/content/drive/MyDrive/Projects/Data/Bigg-Data/Yeast.dat')    
    #train_graphs = nx.readwrite.read_gpickle()
    
    
    path = os.path.join(cmd_args.data_dir, '%s-graphs.pkl' % 'train')
    with open(path, 'rb') as f:
        train_graphs = cp.load(f)
    
    ########################################################################
    has_node_feats = False
    if has_node_feats:
        train_graphs = feature_fixer(train_graphs)
        print(train_graphs[0].nodes(data=True))
    ########################################################################
    
    [TreeLib.InsertGraph(g) for g in train_graphs]

    max_num_nodes = max([len(gg.nodes) for gg in train_graphs])
    cmd_args.max_num_nodes = max_num_nodes
    print('# graphs', len(train_graphs), 'max # nodes', max_num_nodes)
    #print(train_graphs[0].edges(data=True))
    
    #list_node_feats = [torch.from_numpy(get_node_feats(g)).to(cmd_args.device) for g in train_graphs]    
    list_edge_feats = [torch.from_numpy(get_edge_feats(g)).to(cmd_args.device) for g in train_graphs]
    

    model = BiggWithEdgeLen(cmd_args).to(cmd_args.device)
    
    
    if cmd_args.model_dump is not None and os.path.isfile(cmd_args.model_dump):
        print('loading from', cmd_args.model_dump)
        model.load_state_dict(torch.load(cmd_args.model_dump))
    
    #########################################################################################################
    if cmd_args.phase != 'train':
        # get num nodes dist
        print("Now generating sampled graphs...")
        num_node_dist = get_node_dist(train_graphs)
        
        path = os.path.join(cmd_args.data_dir, '%s-graphs.pkl' % 'test')
        with open(path, 'rb') as f:
            gt_graphs = cp.load(f)
        print('# gt graphs', len(gt_graphs))
        gen_graphs = []
        with torch.no_grad():
            for _ in tqdm(range(cmd_args.num_test_gen)):
                num_nodes = np.argmax(np.random.multinomial(1, num_node_dist)) 
                _, pred_edges, _, pred_node_feats, pred_edge_feats = model(num_nodes)
                
                if has_node_feats:
                    pred_g = nx.Graph()
                    pred_g.add_nodes_from(range(num_nodes))
                    print(pred_node_feats)
                    sys.exit()
                
                else:
                    weighted_edges = []
                    for e, w in zip(pred_edges, pred_edge_feats):
                        assert e[0] > e[1]
                        w = w.item()
                        w = np.round(w, 4)
                        edge = (e[0], e[1], w)
                        weighted_edges.append(edge)
                    pred_g = nx.Graph()
                    pred_g.add_weighted_edges_from(weighted_edges)
                    gen_graphs.append(pred_g)
        counter = 0
        for g in gen_graphs:
            if counter <= 100:
                print("edges:", g.edges(data=True))
                counter += 1
        print('saving graphs')
        with open(cmd_args.model_dump + '.graphs-%s' % str(cmd_args.greedy_frac), 'wb') as f:
            cp.dump(gen_graphs, f, cp.HIGHEST_PROTOCOL)
        print('graph generation complete')
        
        sum_stats = True
        skip_train = False
        if sum_stats:
            print("Generating Summary Statistics...")
            collect_graphs = [train_graphs, gen_graphs]
            names = ['TRAINING', 'GEN GRAPHS']
            for idx in range(len(collect_graphs)):
                if skip_train and 1-idx:
                    print("SKIPPING TRAIN")
                    print("dist, mu-hat, mu_lo, mu_up, s, s_lo, s_up, mean_tree_var, tree_var_lo, tree_var_up")
                    print("[0.667 1.506 1.503 1.51  1.031 1.029 1.034 0.25  0.249 0.251]")
                    continue
                print(names[idx])
                graphs = collect_graphs[idx]
                
                assert len(graphs) > 0
                dist = np.round(dist_met(graphs, gt_graphs, N = 200000, swap = bool(idx), scale = True), 3)
                
                weights = []
                tree_var = []
                tree_mean = []
                num_skip = 0
                for T in graphs:
                    T_weights = []
                    if len(T.edges()) != 6:#4:
                        num_skip += 1
                        continue
                    for (n1, n2, w) in T.edges(data = True):
                        #t = np.log(np.exp(w['weight']) - 1)
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
                
                #xbar = np.mean(n1_weights)
                #s = np.std(n1_weights, ddof = 1)
                #n = len(n1_weights)
                
                #n1_lo = np.round(xbar - 1.96 * s / n**0.5, 3)
                #n1_up = np.round(xbar + 1.96 * s / n**0.5, 3)
                
                #slo = np.round(s * (n-1)**0.5 * (1/chi2.ppf(0.975, df = n-1))**0.5, 3)
                #sup = np.round(s * (n-1)**0.5 * (1/chi2.ppf(0.025, df = n-1))**0.5, 3)
                
                #print(T_within_var)
                #within_var_mean = np.mean(T_within_var)
                #print(within_var_mean)
                #within_lo = within_var_mean - 1.96 * np.std(T_within_var, ddof = 1) / len(T_within_var)**0.5
                #within_up = within_var_mean + 1.96 * np.std(T_within_var, ddof = 1) / len(T_within_var)**0.5
                
                #within_sd = np.round(within_var_mean**0.5, 3)
                #wlo = np.round(np.sqrt(within_lo), 3)
                #wup = np.round(np.sqrt(within_up), 3)
                
                #within_sd = np.round(within_sd, 3)
    
                #xbar = np.round(xbar, 4)
                #s = np.round(s, 4)
                print("NUMBER SKIPPED: ", num_skip)
                print("dist, mu-hat, mu_lo, mu_up, s, s_lo, s_up, mean_tree_var, tree_var_lo, tree_var_up")
                results = [dist, xbar, mu_lo, mu_up, s, s_lo, s_up, mean_tree_var**0.5, tree_var_lo**0.5, tree_var_up**0.5]
                print(np.round(results, 3))
        sys.exit()
    #########################################################################################################
    
    # debug_model(model, train_graphs[0], list_node_feats[0], list_edge_feats[0])
    serialized = False

    optimizer = optim.Adam(model.parameters(), lr=cmd_args.learning_rate, weight_decay=1e-4)
    indices = list(range(len(train_graphs)))
    
    if cmd_args.epoch_load is None:
        cmd_args.epoch_load = 0
    for epoch in range(cmd_args.epoch_load, cmd_args.num_epochs):
        pbar = tqdm(range(cmd_args.epoch_save))

        optimizer.zero_grad()
        for idx in pbar:
            random.shuffle(indices)
            batch_indices = indices[:cmd_args.batch_size]
            num_nodes = sum([len(train_graphs[i]) for i in batch_indices])

            #node_feats = torch.cat([list_node_feats[i] for i in batch_indices], dim=0)
            #node_feats = node_feats[1:]
            
            #print(node_feats)
            #sys.exit()
            edge_feats = torch.cat([list_edge_feats[i] for i in batch_indices], dim=0)
            
            
            if serialized:
                for ind in batch_indices:
                    g = train_graphs[ind]
                    n = len(g)
                    m = len(g.edges())
                    
                    ### Obtaining edge list 
                    edgelist = []
                    for e in g.edges():
                        if e[0] < e[1]:
                            e = (e[1], e[0])
                        edgelist.append((e[0], e[1]))
                    edgelist.sort(key = lambda x: x[0])
                    
                    ### Obtaining weights list
                    weightdict = dict()
                    for node1, node2, data in g.edges(data=True):
                        if node1 < node2:
                            e = (node2, node1)
                        else:
                            e = (node1, node2)
                        weightdict[e] = data['weight']
                    
                    ### Compute log likelihood, loss
                    ll, _, _, _ = model(node_end = n, edge_list = edgelist, weights = weightdict)
            else:
                ll, _ = model.forward_train(batch_indices, node_feats=None, edge_feats = edge_feats)
            loss = -ll / num_nodes
            loss.backward()
            loss = loss.item()

            if (idx + 1) % cmd_args.accum_grad == 0:
                if cmd_args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cmd_args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
            pbar.set_description('epoch %.2f, loss: %.4f' % (epoch + (idx + 1) / cmd_args.epoch_save, loss))
        
        print('saving')
        torch.save(model.state_dict(), os.path.join(cmd_args.save_dir, 'epoch-%d.ckpt' % (epoch + 1)))
        #_, pred_edges, _, pred_node_feats, pred_edge_feats = model(len(train_graphs[0]))
        #print(pred_edges)
        #print(pred_node_feats)
        #print(pred_edge_feats)
    print("Model training complete.")
        
        
        
        
                
        
        
        