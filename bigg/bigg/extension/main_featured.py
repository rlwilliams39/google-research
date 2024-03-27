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
from bigg.extension.eval_.graph_stats import *
from bigg.extension.eval_.mmd import *
from bigg.extension.eval_.mmd_stats import *
from bigg.train_creator.train_data_generator import *
from bigg.train_creator.data_util import *


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
    
    path = os.path.join(cmd_args.data_dir, '%s-graphs.pkl' % 'train')
    print(path)
    #with open(path, 'rb') as f:
    #    train_graphs = cp.load(f)
    
    ## Try with this:
    train_graphs_gen = graph_generator(n = cmd_args.leaves, num_graphs = 1000, constant_topology = False, constant_weights = False, mu_weight = 10, scale = 1, weighted = True)
    train_graphs = []
    for g in train_graphs_gen:
        cano_g = get_graph_data(g, node_order = 'time', leaves_last = False, order_only = False)
        train_graphs += cano_g
    print(train_graphs[0].edges(data=True))
    
    #[TreeLib.InsertGraph(g) for g in train_graphs]
    n = int(cmd_args.leaves - 1) ## number of internal nodes + root
    m = int(cmd_args.leaves) ## number of leaves
    #[TreeLib.InsertGraph(g, bipart_stats=(n, m)) for g in train_graphs]
    [TreeLib.InsertGraph(g) for g in train_graphs]

    max_num_nodes = max([len(gg.nodes) for gg in train_graphs])
    cmd_args.max_num_nodes = max_num_nodes
    print('# graphs', len(train_graphs), 'max # nodes', max_num_nodes)
    
    list_node_feats = None #[torch.from_numpy(get_node_feats(g)).to(cmd_args.device) for g in train_graphs] 
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
        
        #with open(path, 'rb') as f:
        #    gt_graphs = cp.load(f)
        #print('# gt graphs', len(gt_graphs))
        gt_graphs = train_graphs[0:10]
        if cmd_args.g_type == "tree":
            degree_list = [gt_graphs[0].degree(i) for i in range(n)]
            lb_lst = degree_list
            up_lst = degree_list
            col_rng = (0, int(2*m-1))
        
        else:
            lb_list = None
            up_list = None
            col_rng = None
        
        #gt_graphs = None
        gen_graphs = []
        with torch.no_grad():
            for _ in tqdm(range(cmd_args.num_test_gen)):
                if cmd_args.g_type != "tree":
                    num_nodes = np.argmax(np.random.multinomial(1, num_node_dist)) 
                _, pred_edges, _, pred_node_feats, pred_edge_feats = model(n, lb_list=lb_lst, ub_list=up_lst, col_range=col_rng, display=cmd_args.display, num_nodes = 19)
                
                if cmd_args.has_edge_feats:
                    weighted_edges = []
                    for e, w in zip(pred_edges, pred_edge_feats):
                        assert e[0] > e[1]
                        w = w.item()
                        w = np.round(w, 4)
                        edge = (e[1], e[0], w)
                        weighted_edges.append(edge)
                    pred_g = nx.Graph()
                    pred_g.add_weighted_edges_from(weighted_edges)
                    gen_graphs.append(pred_g)
                
                else:
                    pred_g = nx.Graph()
                    fixed_edges = []
                    for e in pred_edges:
                        w = 1.0
                        edge = (e[0], e[1], w)
                        fixed_edges.append(edge)
                    pred_g.add_weighted_edges_from(fixed_edges)
                    gen_graphs.append(pred_g)
        
        for idx in range(10):
            print("edges: ", gen_graphs[idx].edges(data=True))
        
        print(cmd_args.g_type)
        get_graph_stats(gen_graphs, gt_graphs, cmd_args.g_type, cmd_args.has_edge_feats)
        
        print('saving graphs')
        with open(cmd_args.model_dump + '.graphs-%s' % str(cmd_args.greedy_frac), 'wb') as f:
            cp.dump(gen_graphs, f, cp.HIGHEST_PROTOCOL)
        print('graph generation complete')
        
        sys.exit()
    #########################################################################################################
    
    #debug_model(model, train_graphs[0], None, list_edge_feats[0])
    print("Serialized? ", cmd_args.serialized)
    print("Alt Update?", cmd_args.alt_update)

    optimizer = optim.Adam(model.parameters(), lr=cmd_args.learning_rate, weight_decay=1e-4)
    indices = list(range(len(train_graphs)))
    
    if cmd_args.epoch_load is None:
        cmd_args.epoch_load = 0
    
    prev = datetime.now()
    for epoch in range(cmd_args.epoch_load, cmd_args.num_epochs):
        pbar = tqdm(range(cmd_args.epoch_save))

        optimizer.zero_grad()
        for idx in pbar:
            random.shuffle(indices)
            batch_indices = indices[:cmd_args.batch_size]
            num_nodes = sum([len(train_graphs[i]) for i in batch_indices])

            node_feats = None #torch.cat([list_node_feats[i] for i in batch_indices], dim=0)
            edge_feats = torch.cat([list_edge_feats[i] for i in batch_indices], dim=0)
            
            if cmd_args.serialized:
                ll = 0
                for ind in batch_indices:
                    g = train_graphs[ind]
                    n = len(g)
                    
                    ### Obtaining edge list 
                    edgelist = []
                    for e in g.edges():
                        if e[0] < e[1]:
                            e = (e[1], e[0])
                        edgelist.append((e[0], e[1]))
                    edgelist.sort(key = lambda x: x[0])
                    
                    ### Compute log likelihood, loss
                    ll_i, _, _, _, _ = model.forward(node_end = n, edge_list = edgelist, edge_feats = list_edge_feats[ind])
                    ll = ll_i + ll
            else:
                if cmd_args.g_type == "tree":
                    list_col_rnges = [(0, int(2*m-1)) for i in batch_indices]
                else:
                    list_col_rnges = None
                    
                ll, _ = model.forward_train(batch_indices, node_feats = node_feats, edge_feats = edge_feats, list_col_ranges = list_col_rnges)
            loss = -ll / num_nodes
            loss.backward()
            loss = loss.item()

            if (idx + 1) % cmd_args.accum_grad == 0:
                if cmd_args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cmd_args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
            pbar.set_description('epoch %.2f, loss: %.4f' % (epoch + (idx + 1) / cmd_args.epoch_save, loss))
        
        print('epoch complete')
        cur = epoch + 1
        if cur % 10 == 0 or cur == cmd_args.num_epochs: #save every 10th / last epoch
            print('saving epoch')
            torch.save(model.state_dict(), os.path.join(cmd_args.save_dir, 'epoch-%d.ckpt' % (epoch + 1)))
            #if cmd_args.lin_model:
            #    with open(cmd_args.save_dir + 'lin_model.pkl', 'wb') as f:
            #        cp.dump(lin_model, f, cp.HIGHEST_PROTOCOL)
    elapsed = datetime.now() - prev
    print("Time elapsed during training: ", elapsed)
    print("Model training complete.")
    
    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    