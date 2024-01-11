"""
Reference: https://github.com/gusye1234/LightGCN-PyTorch
"""
import world
import torch
from torch import nn
import torch.nn.functional as F
import os
import numpy as np

from collections import Counter


class RGCN(nn.Module):
    def __init__(self, config: dict, dataset):
        super(RGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config["latent_dim_rec"]
        self.n_layers = self.config["n_layers"]
        self.keep_prob = self.config["keep_prob"]
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim
        )
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim
        )

        nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
        world.LOGGER.info("use xavier initilizer")
        # nn.init.normal_(self.embedding_user.weight, std=0.1)
        # nn.init.normal_(self.embedding_item.weight, std=0.1)
        # world.LOGGER.info("use NORMAL distribution initilizer")

        self.f = nn.Sigmoid()
        self.emb_weights = self.config["emb_weight"]
        self.temp = self.config["temp"]

        self.mlp_drop = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.latent_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(self.config["mlp_dropout"]),
            nn.Linear(self.latent_dim, 1, bias=False),
        )
        self.anchors = self.get_anchors()
        world.LOGGER.info(f"anchors: {self.anchors}")
        torch.save(self.anchors, os.path.join(world.FOLDER_PATH, f"anchor.pt"))

        self.Graph = self.dataset.get_sparse_graph()
        world.LOGGER.info(f"use normalized Graph(dropout:{self.config['dropout']})")

    def dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def dropout(self, keep_prob):
        graph = self.dropout_x(self.Graph, keep_prob)
        return graph

    def get_users_rating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def get_embedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, user_emb_0, pos_emb_0, neg_emb_0) = self.get_embedding(
            users.long(), pos.long(), neg.long()
        )
        reg_loss = (
            (1 / 2)
            * (user_emb_0.norm(2).pow(2) + pos_emb_0.norm(2).pow(2) + neg_emb_0.norm(2).pow(2))
            / float(len(users))
        )
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def get_anchors(self):
        R = self.dataset.user_item_net.todense()
        # C = R.T @ R ## item based
        C = R @ R.T  ## user based
        num_anchors = self.config["groups"]
        world.LOGGER.info(f"Clustering(group:{num_anchors}) starts")
        anchors = []
        while len(anchors) < num_anchors:
            degree = C.sum(axis=1)
            if degree.sum() == 0:
                break
            anchor = degree.argmax()
            C[anchor, :] = 0
            C[:, anchor] = 0
            anchors.append(anchor)
        coverage = (
            np.hstack([R[anchor, :] for anchor in anchors]).sum(axis=1) > 0
        ).sum() / R.shape[0]
        world.LOGGER.info(f"anchor covers {coverage} users")
        anchors = np.array(anchors)
        # anchors += self.dataset.n_user
        return anchors

    def get_local_graph(self, g_row, g_col, g_values, membership):
        local_node_list, local_graph_list, local_mask_list = [], [], []
        for c in range(self.config["groups"]):
            c_nodes = membership[c, :].nonzero().squeeze().tolist()
            if type(c_nodes) != list or len(c_nodes) <= 1:
                continue
            # local graph
            row_mask = torch.isin(g_row, torch.tensor([c_nodes]).to(self.config["device"]))
            col_mask = torch.isin(g_col, torch.tensor([c_nodes]).to(self.config["device"]))
            mask = row_mask & col_mask
            mapper = dict(zip(c_nodes, range(len(c_nodes))))
            l_row = [mapper[i] for i in g_row[mask].tolist()]
            l_col = [mapper[i] for i in g_col[mask].tolist()]
            l_values = torch.ones_like(g_values[mask])
            local_graph = torch.sparse.FloatTensor(
                torch.tensor([l_row, l_col]).long().to(self.config["device"]),
                l_values,
                torch.Size((len(c_nodes), len(c_nodes))),
            )
            # normalize
            l_row_sum = Counter(l_row)
            d_inv = [
                np.power(l_row_sum[i], -0.5) if i in l_row_sum else 0 for i in range(len(c_nodes))
            ]
            d_mat = torch.sparse.FloatTensor(
                torch.arange(len(c_nodes)).unsqueeze(0).repeat(2, 1).to(self.config["device"]),
                torch.tensor(d_inv).float().to(self.config["device"]),
                local_graph.shape,
            )
            norm_local_graph = torch.sparse.mm(d_mat, local_graph)
            norm_local_graph = torch.sparse.mm(norm_local_graph, d_mat)
            local_node_list.append(c_nodes)
            local_graph_list.append(norm_local_graph)
            local_mask_list.append(mask)
        return (local_node_list, local_graph_list, local_mask_list)

    def clustering(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        if self.config["dropout"] and self.training:
            g_droped = self.dropout(self.keep_prob)
        else:
            g_droped = self.Graph
        g_row = g_droped.coalesce().indices()[0]
        g_col = g_droped.coalesce().indices()[1]
        g_values = g_droped.coalesce().values()
        anchor_emb = all_emb[self.anchors]
        anchor_sims = torch.mm(anchor_emb, all_emb.t())
        anchor_sims = F.softmax(anchor_sims, dim=0)
        membership = anchor_sims > (1 / self.config["groups"])
        self.anchor_sims = anchor_sims.detach()

        self.local_node_list, self.local_graph_list, self.local_mask_list = self.get_local_graph(
            g_row, g_col, g_values, membership
        )

    def computer(self):
        '''
        Below code is an example of applying ReducedGCN to LightGCN.
            Reference: https://github.com/gusye1234/LightGCN-PyTorch
        If you want to apply ReducedGCN to other GCN-based model,
        you may fix the message passing and final representation part below.
        '''
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.config["dropout"] and self.training:
            g_droped = self.dropout(self.keep_prob)
        else:
            g_droped = self.Graph
        g_row = g_droped.detach().coalesce().indices()[0]
        g_col = g_droped.detach().coalesce().indices()[1]
        g_values = g_droped.detach().coalesce().values()

        for layer in range(self.n_layers):
            src = all_emb[g_row]
            trg = all_emb[g_col]
            drop = self.f(self.mlp_drop(torch.cat([src, trg], dim=1)).squeeze())
            g_values = g_values * torch.exp(-(drop * ((layer + 1) / self.temp)))
            G = torch.sparse.FloatTensor(
                torch.stack([g_row, g_col]), g_values, torch.Size(g_droped.shape)
            )
            # message passing for global community
            global_emb = torch.sparse.mm(G, all_emb)

            local_emb = torch.zeros(global_emb.shape).to(self.config["device"])
            for c, (local_nodes, local_graph, local_mask) in enumerate(
                zip(self.local_node_list, self.local_graph_list, self.local_mask_list)
            ):
                l_row = local_graph.detach().coalesce().indices()[0]
                l_col = local_graph.detach().coalesce().indices()[1]
                l_values = local_graph.detach().coalesce().values()
                l_drop = drop[local_mask]
                l_values = l_values * torch.exp(-(l_drop * ((layer + 1) / self.temp)))
                localG = torch.sparse.FloatTensor(
                    torch.stack([l_row, l_col]), l_values, torch.Size(local_graph.shape)
                )
                c_emb = all_emb[local_nodes, :]
                
                # message passing for sub-communities
                c_emb = torch.sparse.mm(localG, c_emb)
                
                c_emb = c_emb * self.anchor_sims[c, :][local_nodes].unsqueeze(1)
                local_emb[local_nodes, :] = local_emb[local_nodes, :] + c_emb
                self.local_graph_list[c] = localG

            all_emb = self.emb_weights[0] * global_emb + self.emb_weights[1] * local_emb
            embs.append(all_emb)
        
        # final representation
        embs = torch.stack(embs, dim=1)
        out = torch.mean(embs, dim=1)
        users, items = torch.split(out, [self.num_users, self.num_items])
        
        return users, items
