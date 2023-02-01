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

from bigg.model.tree_model import RecurTreeGen
import torch
from bigg.common.pytorch_util import glorot_uniform, MLP
import torch.nn as nn
import numpy as np

# pylint: skip-file


class BiggWithEdgeLen(RecurTreeGen):

    def __init__(self, args):
        super().__init__(args)
        self.edgelen_encoding = MLP(1, [2 * args.embed_dim, args.embed_dim])
        self.nodelen_encoding = MLP(1, [2 * args.embed_dim, args.embed_dim])
        self.nodelen_pred = MLP(args.embed_dim, [2 * args.embed_dim, 1])
        self.edgelen_pred = MLP(args.embed_dim, [2 * args.embed_dim, args.embed_dim, 2]) ## Changed
        self.node_state_update = nn.LSTMCell(args.embed_dim, args.embed_dim)
        self.edge_state_update = nn.LSTMCell(args.embed_dim, args.embed_dim) ## ADDED

    # to be customized
    def embed_node_feats(self, node_feats):
        return self.nodelen_encoding(node_feats)

    def embed_edge_feats(self, edge_feats):
        return self.edgelen_encoding(edge_feats)

    def predict_node_feats(self, state, node_feats=None):
        """
        Args:
            state: tuple of (h=N x embed_dim, c=N x embed_dim), the current state
            node_feats: N x feat_dim or None
        Returns:
            new_state,
            likelihood of node_feats under current state,
            and, if node_feats is None, then return the prediction of node_feats
            else return the node_feats as it is
        """
        h, _ = state
        pred_node_len = self.nodelen_pred(h)
        state_update = self.embed_node_feats(pred_node_len) if node_feats is None else self.embed_node_feats(node_feats)
        new_state = self.node_state_update(state_update, state)
        if node_feats is None:
            ll = 0
            node_feats = pred_node_len
        else:
            ll = -(node_feats - pred_node_len) ** 2
            ll = torch.sum(ll)
        return new_state, ll, node_feats

    def predict_edge_feats(self, state, edge_feats=None):
        """
        Args:
            state: tuple of (h=N x embed_dim, c=N x embed_dim), the current state
            edge_feats: N x feat_dim or None
        Returns:
            likelihood of edge_feats under current state,
            and, if edge_feats is None, then return the prediction of edge_feats
            else return the edge_feats as it is
        """
        h, _ = state
        params = self.edgelen_pred(h)
        
        ## GAMMA:
        params = torch.nn.functional.softplus(params, beta = 1)
        
        #print("features", edge_feats)
        #print("parameters", params)
        
        if edge_feats is None:
            ll = 0
            #edge_feats = None
            ##pred_mean = params[0][0].item()
            #pred_lvar = params[0][1]
            #pred_var = torch.add(torch.nn.functional.softplus(pred_lvar, beta = 1), 1e-6).item()
            #pred_var = torch.nn.functional.softplus(pred_lvar, beta = 1)
            #edge_feats = torch.FloatTensor([[np.random.normal(pred_mean, pred_var**0.5)]])
            #edge_feats = torch.exp(edge_feats)
            
            ## GAMMA DISTRIBUTION DRAW
            pred_a = params[0][0].item()
            pred_b = params[0][1].item()
            edge_feats = torch.FloatTensor([[np.random.gamma(pred_a, pred_b)]])
            #edge_feats = torch.log(edge_feats)
            
        else:
            ### Update log likelihood with weight prediction
            ### https://stackoverflow.com/questions/66091226/runtimeerror-expected-all-tensors-to-be-on-the-same-device-but-found-at-least
            ### NOTE: find more efficient way of doing this
            
            
            logw_obs = torch.log(edge_feats)
            k = len(params)
            y = torch.tensor([0]).repeat(k).to('cuda')
            z = 1 - y
            
            ## MEAN AND VARIANCE OF LOGNORMAL
            #mean = params.gather(1, y.view(-1, 1)).squeeze()
            #lvar = params.gather(1, z.view(-1, 1)).squeeze()
            #var = torch.add(torch.nn.functional.softplus(lvar, beta = 1), 1e-6)
            #var = torch.nn.functional.softplus(lvar, beta = 1)
            #print(mean)
            #print(lvar)
            
            ## diff_sq = (mu - logw)^2
            #diff_sq = torch.square(torch.sub(mean, logw_obs))
            
            ## diff_sq2 = v^-1*diff_sq
            #diff_sq2 = torch.div(diff_sq, var)
            
            ## log_var = log(v)
            #log_var = torch.log(var)
            
            ## add to ll
            #ll = - torch.mul(log_var, 0.5) - torch.mul(diff_sq2, 0.5) - logw_obs - 0.5 * np.log(2*np.pi)
            #ll = torch.sum(ll)
            
            ## ALPHA AND BETA OF GAMMA
            a = params.gather(1, y.view(-1, 1)).squeeze()
            b = params.gather(1, z.view(-1, 1)).squeeze()
            
            ll = torch.mul(torch.sub(a, 1), logw_obs) - torch.div(edge_feats, b) - torch.mul(a, torch.log(b)) - torch.lgamma(a)
            ll = torch.sum(ll)
            
        #state_update = self.embed_edge_feats(torch.log(edge_feats)) 
        #state = self.edge_state_update(state_update, state)
        #return new_state, ll, edge_feats
        return ll, edge_feats
