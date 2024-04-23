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
        self.edgelen_encoding = MLP(1, [args.embed_dim // 4, args.embed_dim])
        #self.edgelen_encoding_c = MLP(1, [2 * args.embed_dim, args.embed_dim])
        #self.edgelen_encoding_test = nn.LSTMCell(1, args.embed_dim)
        #self.edgelen_encodingLSTM = nn.LSTMCell(1, args.embed_dim)
        self.nodelen_encoding = MLP(1, [2 * args.embed_dim, args.embed_dim])
        self.nodelen_pred = MLP(args.embed_dim, [2 * args.embed_dim, 2])
        self.edgelen_mean = MLP(args.embed_dim, [2 * args.embed_dim, 1]) ## Changed
        self.edgelen_lvar = MLP(args.embed_dim, [2 * args.embed_dim, 1]) ## Changed
        self.node_state_update = nn.LSTMCell(args.embed_dim, args.embed_dim)
        #self.combine_states = nn.Linear(2 * args.embed_dim, args.embed_dim)

    # to be customized
    def embed_node_feats(self, node_feats):
        return self.nodelen_encoding(node_feats)

    def embed_edge_feats(self, edge_feats):
        return self.edgelen_encoding(edge_feats)
    
    def embed_edge_feats_c(self, edge_feats):
        return self.edgelen_encoding_c(edge_feats)
    
    def combine_states(self, state):
        return (self.combine_states(state[0]), self.combine_states(state[1]))

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
        #print(node_feats)
        h, _ = state
        params = self.nodelen_pred(h)
        
        #pred_node_len = self.nodelen_pred(h)
        #state_update = self.embed_node_feats(pred_node_len) if node_feats is None else self.embed_node_feats(node_feats)
        #new_state = self.node_state_update(state_update, state)
        if node_feats is None:
            ll = 0
            pred_mean = params[0][0].item()
            pred_lvar = params[0][1]
            pred_var = torch.add(torch.nn.functional.softplus(pred_lvar, beta = 1), 1e-6).item()
            node_feats = torch.FloatTensor([[np.random.normal(pred_mean, pred_var**0.5)]])
            pred_node_length = torch.exp(node_feats)
            node_feats = pred_node_length
        else:
            ### Update log likelihood with weight prediction
            ### https://stackoverflow.com/questions/66091226/runtimeerror-expected-all-tensors-to-be-on-the-same-device-but-found-at-least
            ### NOTE: find more efficient way of doing this
            
            
            logw_obs = torch.log(node_feats)
            k = len(params)
            y = torch.tensor([0]).repeat(k).to('cuda')
            #y = torch.tensor([0]).repeat(k)
            z = 1 - y
            
            print(params)
            ## MEAN AND VARIANCE OF LOGNORMAL
            mean = params.gather(1, y.view(-1, 1)).squeeze()
            lvar = params.gather(1, z.view(-1, 1)).squeeze()
            var = torch.add(torch.nn.functional.softplus(lvar, beta = 1), 1e-9)
            
            ## diff_sq = (mu - logw)^2
            diff_sq = torch.square(torch.sub(mean, logw_obs))
            
            ## diff_sq2 = v^-1*diff_sq
            diff_sq2 = torch.div(diff_sq, var)
            
            log_var = torch.log(var)
            
            ## add to ll
            ll = - torch.mul(log_var, 0.5) - torch.mul(diff_sq2, 0.5) - logw_obs - 0.5 * np.log(2*np.pi)
            ll = torch.sum(ll)
        
        state_update = self.embed_node_feats(torch.log(node_feats)) if node_feats is None else self.embed_node_feats(torch.log(node_feats))
        new_state = self.node_state_update(state_update, state)
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
        mus, lvars = self.edgelen_mean(h), self.edgelen_lvar(h)
        
        if edge_feats is None:
            ll = 0
            pred_mean = mus
            pred_lvar = lvars
            ## Try exponentiation instead of softplus for ar...
            pred_sd = torch.exp(0.5 * pred_lvar)
            edge_feats = torch.normal(pred_mean, pred_sd)
            edge_feats = torch.nn.functional.softplus(edge_feats)
            
        else:
            ### Update log likelihood with weight prediction
            
            ### Trying with softplus parameterization...
            edge_feats_invsp = torch.log(torch.special.expm1(edge_feats))
            
            ## MEAN AND VARIANCE OF LOGNORMAL
            var = torch.exp(lvars) #torch.nn.functional.softplus(lvars, beta = b)
            
            ## diff_sq = (mu - softminusw)^2
            diff_sq = torch.square(torch.sub(mus, edge_feats_invsp))
            
            ## diff_sq2 = v^-1*diff_sq
            diff_sq2 = torch.div(diff_sq, var)
            
            ## add to ll
            ll = - torch.mul(lvars, 0.5) - torch.mul(diff_sq2, 0.5) + edge_feats - edge_feats_invsp - 0.5 * np.log(2*np.pi)
            ll = torch.sum(ll)
        return ll, edge_feats
