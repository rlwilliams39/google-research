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
#from bigg.extension.lin_mod import EdgeWeightLinearModel
from bigg.model.tree_clib.tree_lib import setup_treelib, TreeLib
from bigg.experiments.train_utils import sqrtn_forward_backward, get_node_dist
from scipy.stats.distributions import chi2
from bigg.extension.graph_stats import *
#from bigg.extension.alternative_model import *
from bigg.train_creator.train_data_generator import *


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
    has_node_feats = False

    #with open(os.path.join(cmd_args.data_dir, 'Group202A.dat'), 'rb') as f:
    #    train_graphs = cp.load(f)
    #train_graphs = nx.read_gpickle('/content/drive/MyDrive/Projects/Data/Bigg-Data/Yeast.dat')    
    #train_graphs = nx.readwrite.read_gpickle()
    
    
    #path = os.path.join(cmd_args.data_dir, '%s-graphs.pkl' % 'train')
    #print(path)
    #with open(path, 'rb') as f:
    #    train_graphs = cp.load(f)
    
    train_graphs = graph_generator(n = 3, num_graphs = 10, constant_topology = False, constant_weights = False, mu_weight = 10, scale = 1, weighted = True)
    for i in range(10):
        print(train_graphs[i].edges(data=True))
    
    [TreeLib.InsertGraph(g) for g in train_graphs]

    max_num_nodes = max([len(gg.nodes) for gg in train_graphs])
    cmd_args.max_num_nodes = max_num_nodes
    print('# graphs', len(train_graphs), 'max # nodes', max_num_nodes)
    if max_num_nodes < 100:
        print(train_graphs[0].edges(data=True))
    
    #list_node_feats = [torch.from_numpy(get_node_feats(g)).to(cmd_args.device) for g in train_graphs] 
    list_edge_feats = [torch.from_numpy(get_edge_feats(g)).to(cmd_args.device) for g in train_graphs]
    

    model = BiggWithEdgeLen(cmd_args).to(cmd_args.device)
    #print("ARGUMENT", cmd_args.lin_model)
    ### LINEAR MODEL
    if cmd_args.lin_model:
        lin_model = EdgeWeightLinearModel(cmd_args)
        if cmd_args.phase != 'train':
            with open(cmd_args.save_dir + 'lin_model.pkl', 'rb') as f:
                lin_model = cp.load(f)    
    ###
    
    if cmd_args.model_dump is not None and os.path.isfile(cmd_args.model_dump):
        print('loading from', cmd_args.model_dump)
        model.load_state_dict(torch.load(cmd_args.model_dump))

    
    #########################################################################################################
    if cmd_args.phase != 'train':
        # get num nodes dist
        print("Now generating sampled graphs...")
        num_node_dist = get_node_dist(train_graphs)
        
        
        path = os.path.join(cmd_args.data_dir, '%s-graphs.pkl' % 'test')
        #with open(path, 'rb') as f:
        #    gt_graphs = cp.load(f)
        #print('# gt graphs', len(gt_graphs))
        gen_graphs = []
        with torch.no_grad():
            for _ in tqdm(range(cmd_args.num_test_gen)):
                num_nodes = np.argmax(np.random.multinomial(1, num_node_dist)) 
                _, pred_edges, _, pred_node_feats, pred_edge_feats = model(num_nodes)
                #pred_edge_feats = weight_generator(cmd_args.weight_gen_type, pred_edges, pred_edge_feats, train_graphs)
                
                if cmd_args.has_edge_feats:
                    weighted_edges = []
                    for e, w in zip(pred_edges, pred_edge_feats):
                        #print("e: ", e)
                        assert e[0] > e[1]
                        #if cmd_args.alt_edge_feats is None:
                        w = w.item()
                        w = np.round(w, 4)
                        edge = (e[1], e[0], w)
                        #print("edge:", edge)
                        weighted_edges.append(edge)
                    #print("weighted edges: ", weighted_edges)
                    pred_g = nx.Graph()
                    pred_g.add_weighted_edges_from(weighted_edges)
                    gen_graphs.append(pred_g)
                
                else:
                    pred_g = nx.Graph()
                    pred_g.add_edges_from(pred_edges)
                    gen_graphs.append(pred_g)
         
        counter = 0
        for g in gen_graphs:
            if counter <= 50:
                print("edges:", g.edges(data=True))
                counter += 1
        
        if False: #cmd_args.has_edge_feats:
            print("Generating Statistics for ", cmd_args.file_name)
            final_graphs = graph_stat_gen(gen_graphs, train_graphs, gt_graphs, kind = cmd_args.file_name)
            print("final_g len: ", len(final_graphs))
        
        else:
            print("Testing for Tree Structures...")
            trees = 0
            for T in gen_graphs:
                if nx.is_tree(T):
                    leaves = [n for n in T.nodes() if T.degree(n) == 1]
                    internal = [n for n in T.nodes() if T.degree(n) == 3]
                    root = [n for n in T.nodes() if T.degree(n) == 2]
                    if 2*len(leaves) - 1 == len(T) and len(leaves) == len(internal) + 2 and len(root) == 1 and len(leaves) + len(internal)+ len(root) == len(T):
                        trees += 1
            print("Number of Trees: ", trees)
            print("Out of....: ", len(gen_graphs))
            final_graphs = gen_graphs
        
        print('saving graphs')
        with open(cmd_args.model_dump + '.graphs-%s' % str(cmd_args.greedy_frac), 'wb') as f:
            cp.dump(final_graphs, f, cp.HIGHEST_PROTOCOL)
        print('graph generation complete')
        
        sys.exit()
    #########################################################################################################
    
    # debug_model(model, train_graphs[0], list_node_feats[0], list_edge_feats[0])
    serialized = False

    optimizer = optim.Adam(model.parameters(), lr=cmd_args.learning_rate, weight_decay=1e-4)
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9) ##added
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
                
                if cmd_args.lin_model:
                    for idx in batch_indices:
                        g = train_graphs[idx]
                        #weights = list_edge_feats[idx]
                        W = nx.adjacency_matrix(g).todense()
                        
                        weights = W[np.triu_indices(len(W), k = 1)]
                        weights = np.array([max(10**-9, w) for w in weights])
                        weights = np.log(np.exp(weights)-1)
                        ### Do SVD on Adjacency Matrix: take ___ columns of U from __ largest SVD
                        ### Multivariate Linear Regression................................
                        
                        
                        n = len(g)
                        feature_matrix = np.zeros(shape = (int(n*(n-1)/2), n))
                        
                        row = 0
                        for i in range(n):
                            for k in range(i+1, n):
                                feature_matrix[row][i] = 1  
                                feature_matrix[row][k] = 1 
                                row += 1
                        
                        #Y = XB + E
                        #nx1 nxp px1 nx1
                        
                        #n = # graphs
                        
                        #Y   = XB + E
                        #nxd = nx
                        
                        lin_model.train(feature_matrix, weights)
                
            
            loss = -ll / num_nodes
            loss.backward()
            loss = loss.item()

            if (idx + 1) % cmd_args.accum_grad == 0:
                if cmd_args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cmd_args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
            pbar.set_description('epoch %.2f, loss: %.4f' % (epoch + (idx + 1) / cmd_args.epoch_save, loss))
        #scheduler.step()
        print('epoch complete')
        cur = epoch+1
        if cur % 10 == 0 or cur == cmd_args.num_epochs: #save every 10th / last epoch
            print('saving epoch')
            torch.save(model.state_dict(), os.path.join(cmd_args.save_dir, 'epoch-%d.ckpt' % (epoch + 1)))
            if cmd_args.lin_model:
                with open(cmd_args.save_dir + 'lin_model.pkl', 'wb') as f:
                    cp.dump(lin_model, f, cp.HIGHEST_PROTOCOL)
        #_, pred_edges, _, pred_node_feats, pred_edge_feats = model(len(train_graphs[0]))
        #print(pred_edges)
        #print(pred_node_feats)
        #print(pred_edge_feats)
    print("Model training complete.")
    
    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    