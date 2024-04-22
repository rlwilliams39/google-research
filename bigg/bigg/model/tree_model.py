# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from collections import defaultdict
from torch.nn.parameter import Parameter
from bigg.common.pytorch_util import glorot_uniform, MLP, BinaryTreeLSTMCell
from tqdm import tqdm
from bigg.model.util import AdjNode, ColAutomata, AdjRow
from bigg.model.tree_clib.tree_lib import TreeLib
from bigg.torch_ops import multi_index_select, PosEncoding
from functools import partial

def hc_multi_select(ids_from, ids_to, h_froms, c_froms):
    h_vecs = multi_index_select(ids_from,
                                ids_to,
                                *h_froms)
    c_vecs = multi_index_select(ids_from,
                                ids_to,
                                *c_froms)
    return h_vecs, c_vecs


def tree_state_select(h_bot, c_bot, h_buf, c_buf, fn_all_ids):
    bot_froms, bot_tos, prev_froms, prev_tos = fn_all_ids()
    if h_buf is None or prev_tos is None:
        h_vecs = multi_index_select([bot_froms], [bot_tos], h_bot)
        c_vecs = multi_index_select([bot_froms], [bot_tos], c_bot)
    elif h_bot is None or bot_tos is None:
        h_vecs = multi_index_select([prev_froms], [prev_tos], h_buf)
        c_vecs = multi_index_select([prev_froms], [prev_tos], c_buf)
    else:
        h_vecs, c_vecs = hc_multi_select([bot_froms, prev_froms],
                                         [bot_tos, prev_tos],
                                         [h_bot, h_buf], [c_bot, c_buf])
    return h_vecs, c_vecs


def batch_tree_lstm2(h_bot, c_bot, h_buf, c_buf, fn_all_ids, cell):
    h_list = []
    c_list = []
    for i in range(2):
        h_vecs, c_vecs = tree_state_select(h_bot, c_bot, h_buf, c_buf, lambda : fn_all_ids(i))
        h_list.append(h_vecs)
        c_list.append(c_vecs)
    return cell((h_list[0], c_list[0]), (h_list[1], c_list[1]))


def selective_update_hc(h, c, zero_one, feats, embedding = None, alt_update = False):
    #### Here, I want to update using the weights LSTM. And then only for those that are 1.
    nz_idx = torch.tensor(np.nonzero(zero_one)[0]).to(h.device)
    if alt_update:
        local_edge_feats_h = scatter(feats[0], nz_idx, dim=0, dim_size=h.shape[0]) ## ADDED
        local_edge_feats_c = scatter(feats[1], nz_idx, dim=0, dim_size=c.shape[0]) ## ADDED
    else:
        local_edge_feats = scatter(feats, nz_idx, dim=0, dim_size=h.shape[0])
        
    zero_one = torch.tensor(zero_one, dtype=torch.bool).to(h.device).unsqueeze(1)
    
    if embedding is not None:
        new_h, new_c = embedding(local_edge_feats, (h, c))
        h = torch.where(zero_one, new_h, h)
        c = torch.where(zero_one, new_c, c)
    
    else:
        if alt_update:
            h = torch.where(zero_one, local_edge_feats_h, h)
            c = torch.where(zero_one, local_edge_feats_c, c)
        
        else:
            h = torch.where(zero_one, local_edge_feats, h)
            c = torch.where(zero_one, local_edge_feats, c)
    
    return h, c

def featured_batch_tree_lstm2(edge_feats, is_rch, h_bot, c_bot, h_buf, c_buf, fn_all_ids, cell, t_lch=None, t_rch=None, cell_node=None, embedding=None, alt_update=False):
    new_ids = [list(fn_all_ids(0)), list(fn_all_ids(1))]
    lch_isleaf, rch_isleaf = new_ids[0][0], new_ids[1][0]
    new_ids[0][0] = new_ids[1][0] = None
    is_leaf = [lch_isleaf, rch_isleaf]
    if edge_feats is not None:
        if alt_update:
            edge_feats_h = [edge_feats[0][~is_rch], edge_feats[0][is_rch]] ### CHANGED
            edge_feats_c = [edge_feats[1][~is_rch], edge_feats[1][is_rch]] ### CHANGED
        else:
            edge_feats = [edge_feats[~is_rch], edge_feats[is_rch]]
        assert np.sum(is_rch) == np.sum(rch_isleaf)
    node_feats = [t_lch, t_rch]
    h_list = []
    c_list = []
    for i in range(2):
        leaf_check = is_leaf[i]
        local_hbot, local_cbot = h_bot[leaf_check], c_bot[leaf_check]
        if edge_feats is not None:
            if alt_update:
                local_hbot, local_cbot = selective_update_hc(local_hbot, local_cbot, leaf_check, (edge_feats_h[i], edge_feats_c[i]), embedding, alt_update = True)
            else:
                local_hbot, local_cbot = selective_update_hc(local_hbot, local_cbot, leaf_check, edge_feats[i], embedding)
        if cell_node is not None:
            local_hbot, local_cbot = cell_node(node_feats[i], (local_hbot, local_cbot))
        h_vecs, c_vecs = tree_state_select(local_hbot, local_cbot, h_buf, c_buf, lambda : new_ids[i])
        h_list.append(h_vecs)
        c_list.append(c_vecs)
    return cell((h_list[0], c_list[0]), (h_list[1], c_list[1]))


def batch_tree_lstm3(h_bot, c_bot, h_buf, c_buf, h_past, c_past, fn_all_ids, cell):
    if h_past is None:
        return batch_tree_lstm2(h_bot, c_bot, h_buf, c_buf, lambda i: fn_all_ids(i)[:-2], cell)
    elif h_bot is None:
        return batch_tree_lstm2(h_buf, c_buf, h_past, c_past, lambda i: fn_all_ids(i)[2:], cell)
    elif h_buf is None:
        return batch_tree_lstm2(h_bot, c_bot, h_past, c_past, lambda i: fn_all_ids(i)[0, 1, 4, 5], cell)
    else:
        h_list = []
        c_list = []
        for i in range(2):
            bot_froms, bot_tos, prev_froms, prev_tos, past_froms, past_tos = fn_all_ids(i)
            h_vecs, c_vecs = hc_multi_select([bot_froms, prev_froms, past_froms],
                                             [bot_tos, prev_tos, past_tos],
                                             [h_bot, h_buf, h_past],
                                             [c_bot, c_buf, c_past])
            h_list.append(h_vecs)
            c_list.append(c_vecs)
        return cell((h_list[0], c_list[0]), (h_list[1], c_list[1]))


def featured_batch_tree_lstm3(feat_dict, h_bot, c_bot, h_buf, c_buf, h_past, c_past, fn_all_ids, cell, cell_node, embedding = None, alt_update = False):
    edge_feats = is_rch = None
    t_lch = t_rch = None
    if 'edge' in feat_dict:
        edge_feats, is_rch = feat_dict['edge']
    if 'node' in feat_dict:
        t_lch, t_rch = feat_dict['node']
    if h_past is None:
        return featured_batch_tree_lstm2(edge_feats, is_rch, h_bot, c_bot, h_buf, c_buf, lambda i: fn_all_ids(i)[:-2], cell, t_lch, t_rch, cell_node, embedding = embedding, alt_update = alt_update)
    elif h_bot is None:
        return batch_tree_lstm2(h_buf, c_buf, h_past, c_past, lambda i: fn_all_ids(i)[2:], cell)
    elif h_buf is None:
        return featured_batch_tree_lstm2(edge_feats, is_rch, h_bot, c_bot, h_past, c_past, lambda i: fn_all_ids(i)[0, 1, 4, 5], cell, t_lch, t_rch, cell_node, embedding = embedding, alt_update = alt_update)
    else:
        raise NotImplementedError  #TODO: handle model parallelism with features


class FenwickTree(nn.Module):
    def __init__(self, args):
        super(FenwickTree, self).__init__()
        self.has_edge_feats = args.has_edge_feats
        self.has_node_feats = args.has_node_feats
        self.init_h0 = Parameter(torch.Tensor(1, args.embed_dim))
        self.init_c0 = Parameter(torch.Tensor(1, args.embed_dim))
        glorot_uniform(self)
        if self.has_node_feats:
            self.node_feat_update = nn.LSTMCell(args.embed_dim, args.embed_dim)
        self.merge_cell = BinaryTreeLSTMCell(args.embed_dim)
        self.summary_cell = BinaryTreeLSTMCell(args.embed_dim)
        self.alt_update = args.alt_update
        if args.pos_enc:
            self.pos_enc = PosEncoding(args.embed_dim, args.device, args.pos_base)
        else:
            self.pos_enc = lambda x: 0

    def reset(self, list_states=[]):
        self.list_states = []
        for l in list_states:
            t = []
            for e in l:
                t.append(e)
            self.list_states.append(t)

    def append_state(self, state, level):
        if level >= len(self.list_states):
            num_aug = level - len(self.list_states) + 1
            for i in range(num_aug):
                self.list_states.append([])
        self.list_states[level].append(state)

    def forward(self, new_state=None):
        if new_state is None:
            if len(self.list_states) == 0:
                return (self.init_h0, self.init_c0)
        else:
            self.append_state(new_state, 0)
        pos = 0
        while pos < len(self.list_states):
            if len(self.list_states[pos]) >= 2:
                #print("Hello 1")
                lch_state, rch_state = self.list_states[pos]  # assert the length is 2
                new_state = self.merge_cell(lch_state, rch_state)
                #print("lch_state", lch_state)
                #print("rch_State", rch_state)
                #print(new_state)
                self.list_states[pos] = []
                self.append_state(new_state, pos + 1)
            pos += 1
        state = None
        for pos in range(len(self.list_states)):
            #print("Hello 2")
            if len(self.list_states[pos]) == 0:
                continue
            cur_state = self.list_states[pos][0]
            if state is None:
                state = cur_state
            else:
                #print("state A: ", state)
                #print("state B: ", cur_state)
                state = self.summary_cell(state, cur_state)
                #print("resulting state: ", state)
        #print("returned state: ", state)
        return state

    def forward_train(self, h_bot, c_bot, h_buf0, c_buf0, prev_rowsum_h, prrev_rowsum_c, embedding = None):
        # embed row tree
        tree_agg_ids = TreeLib.PrepareRowEmbed()
        row_embeds = [(self.init_h0, self.init_c0)]
        if self.has_edge_feats or self.has_node_feats:
            feat_dict = c_bot
            if 'node' in feat_dict:
                node_feats, is_tree_trivial, t_lch, t_rch = feat_dict['node']
                sel_feat = node_feats[is_tree_trivial]
                feat_dict['node'] = (sel_feat[t_lch], sel_feat[t_rch])
            h_bot, c_bot = h_bot
        if h_bot is not None:
            row_embeds.append((h_bot, c_bot))
        if prev_rowsum_h is not None:
            row_embeds.append((prev_rowsum_h, prrev_rowsum_c))
        if h_buf0 is not None:
            row_embeds.append((h_buf0, c_buf0))

        for i, all_ids in enumerate(tree_agg_ids):
            fn_ids = lambda x: all_ids[x]
            lstm_func = batch_tree_lstm3
            if i == 0 and (self.has_edge_feats or self.has_node_feats):
                lstm_func = featured_batch_tree_lstm3
            lstm_func = partial(lstm_func, h_buf=row_embeds[-1][0], c_buf=row_embeds[-1][1],
                                h_past=prev_rowsum_h, c_past=prrev_rowsum_c, fn_all_ids=fn_ids, cell=self.merge_cell)
            if i == 0:
                if self.has_edge_feats or self.has_node_feats:
                    new_states = lstm_func(feat_dict, h_bot, c_bot, cell_node=None if not self.has_node_feats else self.node_feat_update, embedding = embedding, alt_update = self.alt_update)
                else:
                    new_states = lstm_func(h_bot, c_bot)
            else:
                new_states = lstm_func(None, None)
            row_embeds.append(new_states)
        h_list, c_list = zip(*row_embeds)
        joint_h = torch.cat(h_list, dim=0)
        joint_c = torch.cat(c_list, dim=0)

        # get history representation
        init_select, all_ids, last_tos, next_ids, pos_info = TreeLib.PrepareRowSummary()
        cur_state = (joint_h[init_select], joint_c[init_select])
        if self.has_node_feats:
            base_nodes, _ = TreeLib.GetFenwickBase()
            if len(base_nodes):
                needs_base_nodes = (init_select >= 1) & (init_select <= 2)
                sub_states = (cur_state[0][needs_base_nodes], cur_state[1][needs_base_nodes])
                sub_states = self.node_feat_update(node_feats[base_nodes], sub_states)
                nz_idx = torch.tensor(np.nonzero(needs_base_nodes)[0]).to(node_feats.device)
                new_cur = [scatter(x, nz_idx, dim=0, dim_size=init_select.shape[0]) for x in sub_states]
                needs_base_nodes = torch.tensor(needs_base_nodes, dtype=torch.bool).to(node_feats.device).unsqueeze(1)
                cur_state = [torch.where(needs_base_nodes, new_cur[i], cur_state[i]) for i in range(2)]
                cur_state = tuple(cur_state)
        ret_state = (joint_h[next_ids], joint_c[next_ids])
        hist_rnn_states = []
        hist_froms = []
        hist_tos = []
        for i, (done_from, done_to, proceed_from, proceed_input) in enumerate(all_ids):
            hist_froms.append(done_from)
            hist_tos.append(done_to)
            hist_rnn_states.append(cur_state)

            next_input = joint_h[proceed_input], joint_c[proceed_input]
            sub_state = cur_state[0][proceed_from], cur_state[1][proceed_from]
            cur_state = self.summary_cell(sub_state, next_input)
        hist_rnn_states.append(cur_state)
        hist_froms.append(None)
        hist_tos.append(last_tos)
        hist_h_list, hist_c_list = zip(*hist_rnn_states)
        pos_embed = self.pos_enc(pos_info)
        row_h = multi_index_select(hist_froms, hist_tos, *hist_h_list) + pos_embed
        row_c = multi_index_select(hist_froms, hist_tos, *hist_c_list) + pos_embed
        return (row_h, row_c), ret_state


class BitsRepNet(nn.Module):
    def __init__(self, args):
        super(BitsRepNet, self).__init__()
        self.bits_compress = args.bits_compress
        self.out_dim = args.embed_dim
        assert self.out_dim >= self.bits_compress
        self.device = args.device

    def forward(self, on_bits, n_cols):
        h = torch.zeros(1, self.out_dim).to(self.device)
        h[0, :n_cols] = -1.0
        h[0, on_bits] = 1.0

        return h, h


class RecurTreeGen(nn.Module):

    # to be customized
    def embed_node_feats(self, node_feats):
        raise NotImplementedError

    def embed_edge_feats(self, edge_feats):
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    def __init__(self, args):
        super(RecurTreeGen, self).__init__()
        
        self.alt_update = args.alt_update
        print("Alternate Update? ", self.alt_update)
        self.first_idea = args.first_idea
        print("First idea? ", self.first_idea)
        self.directed = args.directed
        self.self_loop = args.self_loop
        self.bits_compress = args.bits_compress
        self.has_edge_feats = args.has_edge_feats
        self.has_node_feats = args.has_node_feats
        if self.has_edge_feats:
            assert self.bits_compress == 0
        self.greedy_frac = args.greedy_frac
        self.share_param = args.share_param
        if not self.bits_compress:
            self.leaf_h0 = Parameter(torch.Tensor(1, args.embed_dim))
            self.leaf_c0 = Parameter(torch.Tensor(1, args.embed_dim))
            self.empty_h0 = Parameter(torch.Tensor(1, args.embed_dim))
            self.empty_c0 = Parameter(torch.Tensor(1, args.embed_dim))

        self.topdown_left_embed = Parameter(torch.Tensor(2, args.embed_dim))
        self.topdown_right_embed = Parameter(torch.Tensor(2, args.embed_dim))
        glorot_uniform(self)

        if self.bits_compress > 0:
            self.bit_rep_net = BitsRepNet(args)

        if self.share_param:
            self.m_l2r_cell = BinaryTreeLSTMCell(args.embed_dim)
            self.lr2p_cell = BinaryTreeLSTMCell(args.embed_dim)
            self.pred_has_ch = MLP(args.embed_dim, [2 * args.embed_dim, 1])
            self.m_pred_has_left = MLP(args.embed_dim, [2 * args.embed_dim, 1])
            self.m_pred_has_right = MLP(args.embed_dim, [2 * args.embed_dim, 1])
            self.m_cell_topdown = nn.LSTMCell(args.embed_dim, args.embed_dim)
            self.m_cell_topright = nn.LSTMCell(args.embed_dim, args.embed_dim)
            self.merge_weight = BinaryTreeLSTMCell(args.embed_dim) ### ADDED


        else:
            fn_pred = lambda: MLP(args.embed_dim, [2 * args.embed_dim, 1])
            fn_tree_cell = lambda: BinaryTreeLSTMCell(args.embed_dim)
            fn_lstm_cell = lambda: nn.LSTMCell(args.embed_dim, args.embed_dim)
            num_params = int(np.ceil(np.log2(args.max_num_nodes))) + 1
            self.pred_has_ch = fn_pred()

            pred_modules = [[] for _ in range(2)]
            tree_cell_modules = []
            lstm_cell_modules = [[] for _ in range(2)]
            for _ in range(num_params):
                for i in range(2):
                    pred_modules[i].append(fn_pred())
                    lstm_cell_modules[i].append(fn_lstm_cell())
                tree_cell_modules.append(fn_tree_cell())

            self.has_left_modules, self.has_right_modules = [nn.ModuleList(l) for l in pred_modules]
            self.l2r_modules= nn.ModuleList(tree_cell_modules)
            self.cell_topdown_modules, self.cell_topright_modules = [nn.ModuleList(l) for l in lstm_cell_modules]
            self.lr2p_cell = fn_tree_cell()
        self.row_tree = FenwickTree(args)

        if args.tree_pos_enc:
            self.tree_pos_enc = PosEncoding(args.embed_dim, args.device, args.pos_base, bias=np.pi / 4)
        else:
            self.tree_pos_enc = lambda x: 0

    def cell_topdown(self, x, y, lv):
        cell = self.m_cell_topdown if self.share_param else self.cell_topdown_modules[lv]
        return cell(x, y)

    def cell_topright(self, x, y, lv):
        cell = self.m_cell_topright if self.share_param else self.cell_topright_modules[lv]
        return cell(x, y)

    def l2r_cell(self, x, y, lv):
        cell = self.m_l2r_cell if self.share_param else self.l2r_modules[lv]
        return cell(x, y)

    def pred_has_left(self, x, lv):
        mlp = self.m_pred_has_left if self.share_param else self.has_left_modules[lv]
        return mlp(x)

    def pred_has_right(self, x, lv):
        mlp = self.m_pred_has_right if self.share_param else self.has_right_modules[lv]
        return mlp(x)

    def get_empty_state(self):
        if self.bits_compress:
            return self.bit_rep_net([], 1)
        else:
            return (self.empty_h0, self.empty_c0)

    def get_prob_fix(self, prob):
        p = prob * (1 - self.greedy_frac)
        if prob >= 0.5:
            p += self.greedy_frac
        return p

    def weight_state_update(self, inpt, weight_state, H):
        #N = embedded_edge_feats.shape[0]
        #batch_size = 10
        #num_trees = int(N / batch_size)
        #return_tensor = torch.zeros(embedded_edge_feats.shape)
        N = inpt.shape[0]
        inpt = torch.split(inpt, int(N / self.batch_size))
        inpt = torch.cat(inpt, dim=1)
        num_edges = inpt.shape[0]
        outpt_h = torch.zeros((num_edges, self.batch_size * H)).to(inpt.device)
        outpt_c = torch.zeros((num_edges, self.batch_size * H)).to(inpt.device)
        for idx in range(num_edges):
            cur_edges = torch.split(inpt[idx], self.batch_size)
            cur_edges = inpt[idx].unsqueeze(-1)
            #print(cur_edges)
            #print(cur_edges.shape)
            edge_embed = self.embed_edge_feats(cur_edges)
            #print(edge_embed)
            #print(edge_embed.shape)
            #print(weight_state[0].squeeze(0))
            #print(weight_state[0].squeeze(0).shape)
            weight_state = self.merge_weight((edge_embed, edge_embed), weight_state)
            outpt_h[idx] = weight_state[0].flatten()
            outpt_c[idx] = weight_state[1].flatten()
        outpt_h = torch.split(outpt_h, H, dim=1)
        outpt_h = torch.cat(outpt_h, dim = 0)
        outpt_c = torch.split(outpt_c, H, dim=1)
        outpt_c = torch.cat(outpt_c, dim = 0)
        return outpt_h, outpt_c  

    def gen_row(self, ll, state, tree_node, col_sm, lb, ub, edge_feats=None, weight_state=None):
        assert lb <= ub
        if tree_node.is_root:
            prob_has_edge = torch.sigmoid(self.pred_has_ch(state[0]))

            if col_sm.supervised:
                has_edge = len(col_sm.indices) > 0
            else:
                has_edge = np.random.rand() < self.get_prob_fix(prob_has_edge.item())
                if ub == 0:
                    has_edge = False
                if tree_node.n_cols <= 0:
                    has_edge = False
                if lb:
                    has_edge = True
            if has_edge:
                ll = ll + torch.log(prob_has_edge)
            else:
                ll = ll + torch.log(1 - prob_has_edge)
            tree_node.has_edge = has_edge
        else:
            assert ub > 0
            tree_node.has_edge = True

        if not tree_node.has_edge:  # an empty tree
            return ll, self.get_empty_state(), 0, None, weight_state

        if tree_node.is_leaf:
            tree_node.bits_rep = [0]
            col_sm.add_edge(tree_node.col_range[0])
            if self.bits_compress:
                return ll, self.bit_rep_net(tree_node.bits_rep, tree_node.n_cols), 1, None, weight_state
            else:
                if self.has_edge_feats:
                    cur_feats = edge_feats[col_sm.pos - 1].unsqueeze(0) if col_sm.supervised else None
                    edge_ll, cur_feats = self.predict_edge_feats(state, cur_feats)
                    ll = ll + edge_ll
                    if self.alt_update:
                        if weight_state is not None:
                            #print(cur_feats)
                            edge_embed = self.embed_edge_feats(cur_feats)
                            edge_embed = self.merge_weight((edge_embed, edge_embed), weight_state)
                            weight_state = edge_embed
                        else:    
                            edge_embed = self.embed_edge_feats(cur_feats)
                            edge_embed = self.merge_weight((edge_embed, edge_embed), (self.leaf_h0, self.leaf_c0))
                        #edge_embed_h = self.embed_edge_feats(cur_feats)
                        #edge_embed_c = self.embed_edge_feats_c(cur_feats)
                        #edge_embed = (edge_embed_h, edge_embed_c)
                        return ll, edge_embed, 1, cur_feats, weight_state
                    
                    else:
                        edge_embed = self.embed_edge_feats(cur_feats)
                        return ll, (edge_embed, edge_embed), 1, cur_feats, weight_state
                else:
                    return ll, (self.leaf_h0, self.leaf_c0), 1, None, None
        else:
            tree_node.split()

            mid = (tree_node.col_range[0] + tree_node.col_range[1]) // 2
            left_prob = torch.sigmoid(self.pred_has_left(state[0], tree_node.depth))

            if col_sm.supervised:
                has_left = col_sm.next_edge < mid
            else:
                has_left = np.random.rand() < self.get_prob_fix(left_prob.item())
                if ub == 0:
                    has_left = False
                if lb > tree_node.rch.n_cols:
                    has_left = True
            ll = ll + (torch.log(left_prob) if has_left else torch.log(1 - left_prob))
            left_pos = self.tree_pos_enc([tree_node.lch.n_cols])
            state = self.cell_topdown(self.topdown_left_embed[[int(has_left)]] + left_pos, state, tree_node.depth)
            pred_edge_feats = []
            if has_left:
                lub = min(tree_node.lch.n_cols, ub)
                llb = max(0, lb - tree_node.rch.n_cols)
                ll, left_state, num_left, left_edge_feats, weight_state = self.gen_row(ll, state, tree_node.lch, col_sm, llb, lub, edge_feats, weight_state)
                pred_edge_feats.append(left_edge_feats)
            else:
                left_state = self.get_empty_state()
                num_left = 0

            right_pos = self.tree_pos_enc([tree_node.rch.n_cols])
            topdown_state = self.l2r_cell(state, (left_state[0] + right_pos, left_state[1] + right_pos), tree_node.depth)
            rlb = max(0, lb - num_left)
            rub = min(tree_node.rch.n_cols, ub - num_left)
            if not has_left:
                has_right = True
            else:
                right_prob = torch.sigmoid(self.pred_has_right(topdown_state[0], tree_node.depth))
                if col_sm.supervised:
                    has_right = col_sm.has_edge(mid, tree_node.col_range[1])
                else:
                    has_right = np.random.rand() < self.get_prob_fix(right_prob.item())
                    if rub == 0:
                        has_right = False
                    if rlb:
                        has_right = True
                ll = ll + (torch.log(right_prob) if has_right else torch.log(1 - right_prob))

            topdown_state = self.cell_topright(self.topdown_right_embed[[int(has_right)]], topdown_state, tree_node.depth)

            if has_right:  # has edge in right child
                ll, right_state, num_right, right_edge_feats, weight_state = self.gen_row(ll, topdown_state, tree_node.rch, col_sm, rlb, rub, edge_feats, weight_state)
                pred_edge_feats.append(right_edge_feats)
            else:
                right_state = self.get_empty_state()
                num_right = 0
            if tree_node.col_range[1] - tree_node.col_range[0] <= self.bits_compress:
                summary_state = self.bit_rep_net(tree_node.bits_rep, tree_node.n_cols)
            else:
                summary_state = self.lr2p_cell(left_state, right_state)
            if self.has_edge_feats:
                edge_feats = torch.cat(pred_edge_feats, dim=0)
            return ll, summary_state, num_left + num_right, edge_feats, weight_state

    def forward(self, node_end, edge_list=None, node_feats=None, edge_feats=None, node_start=0, list_states=[], lb_list=None, ub_list=None, col_range=None, num_nodes=None, display=False):
        pos = 0
        total_ll = 0.0
        edges = []
        self.row_tree.reset(list_states)
        controller_state = self.row_tree()
        if num_nodes is None:
            num_nodes = node_end
        pbar = range(node_start, node_end)
        if display:
            pbar = tqdm(pbar)
        list_pred_node_feats = []
        list_pred_edge_feats = []
        weight_state = (self.leaf_h0, self.leaf_c0)
        for i in pbar:
            if edge_list is None:
                col_sm = ColAutomata(supervised=False)
            else:
                indices = []
                while pos < len(edge_list) and i == edge_list[pos][0]:
                    indices.append(edge_list[pos][1])
                    pos += 1
                indices.sort()
                col_sm = ColAutomata(supervised=True, indices=indices)
            
            #col_range = (i+1, num_nodes)

            cur_row = AdjRow(i, self.directed, self.self_loop, col_range= col_range)
            lb = 0 if lb_list is None else lb_list[i]
            ub = cur_row.root.n_cols if ub_list is None else ub_list[i]
            #k = np.sum(np.array(list_col_indices) == i)
            #lb -= k
            #ub -= k
            cur_pos_embed = self.row_tree.pos_enc([num_nodes - i])
            controller_state = [x + cur_pos_embed for x in controller_state]
            if self.has_node_feats:
                target_node_feats = None if node_feats is None else node_feats[[i]]
                controller_state, ll_node, target_node_feats = self.predict_node_feats(controller_state, target_node_feats)
                total_ll = total_ll + ll_node
                list_pred_node_feats.append(target_node_feats)
            if self.has_edge_feats:
                target_edge_feats = None if edge_feats is None else edge_feats[len(edges) : len(edges) + len(col_sm)]
            else:
                target_edge_feats = None
            ll, cur_state, _, target_edge_feats, weight_state = self.gen_row(0, controller_state, cur_row.root, col_sm, lb, ub, target_edge_feats, weight_state = weight_state)
            if target_edge_feats is not None and target_edge_feats.shape[0]:
                list_pred_edge_feats.append(target_edge_feats)
            if self.has_node_feats:
                target_feat_embed = self.embed_node_feats(torch.log(target_node_feats))
                cur_state = self.row_tree.node_feat_update(target_feat_embed, cur_state)
            #new_indices = [x for x in col_sm.indices]
            #list_col_indices += new_indices
            #print(lb)
            #print(col_sm.indices)
            #print(ub)
            assert lb <= len(col_sm.indices) <= ub
            controller_state = self.row_tree(cur_state)
            new_edges = [(i, x) for x in col_sm.indices]
            edges += [(i, x) for x in col_sm.indices]
            total_ll = total_ll + ll

        if self.has_node_feats:
            node_feats = torch.cat(list_pred_node_feats, dim=0)
        if self.has_edge_feats:
            edge_feats = torch.cat(list_pred_edge_feats, dim=0)
        return total_ll, edges, self.row_tree.list_states, node_feats, edge_feats

    def binary_ll(self, pred_logits, np_label, need_label=False, reduction='sum'):
        pred_logits = pred_logits.view(-1, 1)
        label = torch.tensor(np_label, dtype=torch.float32).to(pred_logits.device).view(-1, 1)
        loss = F.binary_cross_entropy_with_logits(pred_logits, label, reduction=reduction)
        if need_label:
            return -loss, label
        return -loss

    def forward_row_trees(self, graph_ids, node_feats=None, edge_feats=None, list_node_starts=None, num_nodes=-1, list_col_ranges=None):
        TreeLib.PrepareMiniBatch(graph_ids, list_node_starts, num_nodes, list_col_ranges)
        # embed trees
        all_ids = TreeLib.PrepareTreeEmbed()
        if self.has_node_feats:
            node_feats = self.embed_node_feats(torch.log(node_feats))
        if self.has_edge_feats:
            edge_feats_embed = self.embed_edge_feats(edge_feats)
            if self.alt_update:
                #E = edge_feats.shape[0] ## ADDED
                #edge_feats = self.merge_weight((edge_feats, edge_feats), (self.leaf_h0.repeat(E, 1), self.leaf_c0.repeat(E, 1))) ## ADDED
                H = edge_feats_embed.shape[1]
                outpt_h, outpt_c = self.weight_state_update(edge_feats, (self.leaf_h0.repeat(self.batch_size, 1), self.leaf_c0.repeat(self.batch_size, 1)), H)
                edge_feats = (outpt_h, outpt_c)
                #edge_feats_c = self.embed_edge_feats_c(edge_feats)
                #edge_feats = (edge_feats, edge_feats_c)
            else:
                edge_feats = edge_feats_embed

        if not self.bits_compress:
            h_bot = torch.cat([self.empty_h0, self.leaf_h0], dim=0)
            c_bot = torch.cat([self.empty_c0, self.leaf_c0], dim=0)
            fn_hc_bot = lambda d: (h_bot, c_bot)
        else:
            binary_embeds, base_feat = TreeLib.PrepareBinary()
            fn_hc_bot = lambda d: (binary_embeds[d], binary_embeds[d]) if d < len(binary_embeds) else base_feat
        max_level = len(all_ids) - 1
        h_buf_list = [None] * (len(all_ids) + 1)
        c_buf_list = [None] * (len(all_ids) + 1)

        for d in range(len(all_ids) - 1, -1, -1):
            fn_ids = lambda i: all_ids[d][i]
            if d == max_level:
                h_buf = c_buf = None
            else:
                h_buf = h_buf_list[d + 1]
                c_buf = c_buf_list[d + 1]
            h_bot, c_bot = fn_hc_bot(d + 1)
            if self.has_edge_feats:
                edge_idx, is_rch = TreeLib.GetEdgeAndLR(d + 1)
                if self.alt_update:
                    local_edge_feats_h, local_edge_feats_c = (edge_feats[0][edge_idx], edge_feats[1][edge_idx]) ### CHANGED
                    local_edge_feats = (local_edge_feats_h, local_edge_feats_c)
                else:
                    local_edge_feats = edge_feats[edge_idx] ### CHANGED
                new_h, new_c = featured_batch_tree_lstm2(local_edge_feats, is_rch, h_bot, c_bot, h_buf, c_buf, fn_ids, self.lr2p_cell, alt_update = self.alt_update)
            else:
                new_h, new_c = batch_tree_lstm2(h_bot, c_bot, h_buf, c_buf, fn_ids, self.lr2p_cell)
            h_buf_list[d] = new_h
            c_buf_list[d] = new_c
        hc_bot = fn_hc_bot(0)
        feat_dict = {}
        if self.has_edge_feats:
            edge_idx, is_rch = TreeLib.GetEdgeAndLR(0)
            if self.alt_update:
                local_edge_feats_h = edge_feats[0][edge_idx] ## ADDED
                local_edge_feats_c = edge_feats[1][edge_idx] ## ADDED 
                local_edge_feats = (local_edge_feats_h, local_edge_feats_c) ## ADDED
            else:
                local_edge_feats = edge_feats[edge_idx]
            feat_dict['edge'] = (local_edge_feats, is_rch)
        if self.has_node_feats:
            is_tree_trivial = TreeLib.GetIsTreeTrivial()
            new_h, new_c = self.row_tree.node_feat_update(node_feats[~is_tree_trivial], (new_h, new_c))
            h_buf_list[0] = new_h
            c_buf_list[0] = new_c
            t_lch, t_rch = TreeLib.GetTrivialNodes()
            feat_dict['node'] = (node_feats, is_tree_trivial, t_lch, t_rch)
        if len(feat_dict):
            hc_bot = (hc_bot, feat_dict)
        return hc_bot, fn_hc_bot, h_buf_list, c_buf_list

    def forward_row_summaries(self, graph_ids, node_feats=None, edge_feats=None,
                             list_node_starts=None, num_nodes=-1, prev_rowsum_states=[None, None], list_col_ranges=None):
        hc_bot, _, h_buf_list, c_buf_list = self.forward_row_trees(graph_ids, node_feats, edge_feats,
                                                                   list_node_starts, num_nodes, list_col_ranges)
        row_states, next_states = self.row_tree.forward_train(*hc_bot, h_buf_list[0], c_buf_list[0], *prev_rowsum_states)
        return row_states, next_states

    def forward_train(self, graph_ids, node_feats=None, edge_feats=None,
                      list_node_starts=None, num_nodes=-1, prev_rowsum_states=[None, None], list_col_ranges=None):
        ll = 0.0
        self.batch_size = len(graph_ids)
        hc_bot, fn_hc_bot, h_buf_list, c_buf_list = self.forward_row_trees(graph_ids, node_feats, edge_feats,
                                                                           list_node_starts, num_nodes, list_col_ranges)
        #self.batch_size = len(graph_ids)
        row_states, next_states = self.row_tree.forward_train(*hc_bot, h_buf_list[0], c_buf_list[0], *prev_rowsum_states)
        if self.has_node_feats:
            row_states, ll_node_feats, _ = self.predict_node_feats(row_states, node_feats)
            ll = ll + ll_node_feats
        if self.has_edge_feats:
            edge_feats_embed = self.embed_edge_feats(edge_feats)
            if self.alt_update:
                H = edge_feats_embed.shape[1]
                outpt_h, outpt_c = self.weight_state_update(edge_feats, (self.leaf_h0.repeat(self.batch_size, 1), self.leaf_c0.repeat(self.batch_size, 1)), H)
                edge_feats_embed = (outpt_h, outpt_c)
                #E = edge_feats_embed.shape[0]
                #edge_feats_embed = self.merge_weight((edge_feats_embed, edge_feats_embed), (self.leaf_h0.repeat(E, 1), self.leaf_c0.repeat(E, 1)))
                #edge_feats_embed_c = self.embed_edge_feats_c(edge_feats)
                #edge_feats_embed = (edge_feats_embed, edge_feats_embed_c)
        logit_has_edge = self.pred_has_ch(row_states[0])
        has_ch, _ = TreeLib.GetChLabel(0, dtype=bool)
        ll = ll + self.binary_ll(logit_has_edge, has_ch)
        cur_states = (row_states[0][has_ch], row_states[1][has_ch])

        lv = 0
        while True:
            is_nonleaf = TreeLib.QueryNonLeaf(lv)
            if self.has_edge_feats:
                edge_of_lv = TreeLib.GetEdgeOf(lv)
                edge_state = (cur_states[0][~is_nonleaf], cur_states[1][~is_nonleaf])
                target_feats = edge_feats[edge_of_lv]
                edge_ll, _ = self.predict_edge_feats(edge_state, target_feats)
                ll = ll + edge_ll
            if is_nonleaf is None or np.sum(is_nonleaf) == 0:
                break
            cur_states = (cur_states[0][is_nonleaf], cur_states[1][is_nonleaf])
            left_logits = self.pred_has_left(cur_states[0], lv)
            has_left, num_left = TreeLib.GetChLabel(-1, lv)
            left_update = self.topdown_left_embed[has_left] + self.tree_pos_enc(num_left)
            left_ll, float_has_left = self.binary_ll(left_logits, has_left, need_label=True, reduction='sum')
            ll = ll + left_ll

            cur_states = self.cell_topdown(left_update, cur_states, lv)

            left_ids = TreeLib.GetLeftRootStates(lv)
            h_bot, c_bot = fn_hc_bot(lv + 1)
            if lv + 1 < len(h_buf_list):
                h_next_buf, c_next_buf = h_buf_list[lv + 1], c_buf_list[lv + 1]
            else:
                h_next_buf = c_next_buf = None
            if self.has_edge_feats:
                edge_idx, is_rch = TreeLib.GetEdgeAndLR(lv + 1)
                if self.alt_update:
                    left_feats = (edge_feats_embed[0][edge_idx[~is_rch]], edge_feats_embed[1][edge_idx[~is_rch]]) ## ADDED
                else:
                    left_feats = edge_feats_embed[edge_idx[~is_rch]]
                h_bot, c_bot = h_bot[left_ids[0]], c_bot[left_ids[0]]
                h_bot, c_bot = selective_update_hc(h_bot, c_bot, left_ids[0], left_feats, alt_update = self.alt_update)
                left_ids = tuple([None] + list(left_ids[1:]))

            left_subtree_states = tree_state_select(h_bot, c_bot,
                                                    h_next_buf, c_next_buf,
                                                    lambda: left_ids)

            has_right, num_right = TreeLib.GetChLabel(1, lv)
            right_pos = self.tree_pos_enc(num_right)
            left_subtree_states = [x + right_pos for x in left_subtree_states]
            topdown_state = self.l2r_cell(cur_states, left_subtree_states, lv)

            right_logits = self.pred_has_right(topdown_state[0], lv)
            right_update = self.topdown_right_embed[has_right]
            topdown_state = self.cell_topright(right_update, topdown_state, lv)
            right_ll = self.binary_ll(right_logits, has_right, reduction='none') * float_has_left
            ll = ll + torch.sum(right_ll)
            lr_ids = TreeLib.GetLeftRightSelect(lv, np.sum(has_left), np.sum(has_right))
            new_states = []
            for i in range(2):
                new_s = multi_index_select([lr_ids[0], lr_ids[2]], [lr_ids[1], lr_ids[3]],
                                            cur_states[i], topdown_state[i])
                new_states.append(new_s)
            cur_states = tuple(new_states)
            lv += 1
        return ll, next_states





































