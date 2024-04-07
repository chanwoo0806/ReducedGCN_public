"""
Reference: https://github.com/gusye1234/LightGCN-PyTorch
"""
import world
import torch
from torch import nn
import torch.nn.functional as F
import os
import numpy as np
import scipy.sparse as sp
CHUNK_SIZE_FOR_SPMM = 1000000

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


class RGCF(nn.Module):
    def __init__(self, config: dict, dataset):
        super(RGCF, self).__init__()
        self.config = config
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config["latent_dim_rec"]
        self.n_layers = self.config["n_layers"]

        # define layers
        self.user_linear = torch.nn.Linear(in_features=self.num_items, out_features=self.latent_dim, bias=False)
        self.item_linear = torch.nn.Linear(in_features=self.num_users, out_features=self.latent_dim, bias=False)
        nn.init.xavier_uniform_(self.user_linear.weight, gain=1)
        nn.init.xavier_uniform_(self.item_linear.weight, gain=1)
        world.LOGGER.info("use xavier initilizer")
        
        # generate intermediate data
        self.interaction_matrix = self.dataset.interaction_matrix
        self.adj_matrix = self.get_adj_mat(self.interaction_matrix)
        self.norm_adj_matrix = self.get_norm_mat(self.adj_matrix).to(self.config['device'])

        # self.reg_weight = self.config['reg_weight'] # equalt to self.config['decay']
        self.prune_threshold = self.config['prune_threshold']
        self.MIM_weight = self.config['MIM_weight']
        self.tau = self.config['tau']
        self.aug_ratio = self.config['aug_ratio']
        self.pool_multi = 10
        
        self.for_learning_adj()

        self.f = nn.Sigmoid()
        # self.Graph = self.dataset.get_sparse_graph()
        # world.LOGGER.info("use normalized Graph)")

    # Generate Adjacency Matrix
    def get_adj_mat(self, inter_M, data=None):
        if data is None:
            data = [1] * inter_M.data
        inter_M_t = inter_M.transpose()
        A = sp.dok_matrix((self.num_users + self.num_items, self.num_users + self.num_items), dtype=np.float32)
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.num_users), data))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.num_users, inter_M_t.col), data)))
        A._update(data_dict)  # dok_matrix
        return A

    def get_norm_mat(self, A):
        r""" A_{hat} = D^{-0.5} \times A \times D^{-0.5} """
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.stack([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape)).coalesce()
        return SparseL

    def for_learning_adj(self):
        self.adj_indices = self.norm_adj_matrix.indices()
        self.adj_shape = self.norm_adj_matrix.shape
        self.adj = self.norm_adj_matrix

        inter_data = torch.FloatTensor(self.interaction_matrix.data).to(self.config['device'])
        inter_user = torch.LongTensor(self.interaction_matrix.row).to(self.config['device'])
        inter_item = torch.LongTensor(self.interaction_matrix.col).to(self.config['device'])
        inter_mask = torch.stack([inter_user, inter_item], dim=0)

        self.inter_spTensor = torch.sparse.FloatTensor(inter_mask, inter_data, self.interaction_matrix.shape).coalesce()
        self.inter_spTensor_t = self.inter_spTensor.t().coalesce()

        self.inter_indices = self.inter_spTensor.indices()
        self.inter_shape = self.inter_spTensor.shape
    
    # User/Item Embedding
    # Returns: torch.FloatTensor: The embedding tensor of all user, shape: [num_users, embedding_size]
    def get_all_user_embedding(self):
        all_user_embedding = torch.sparse.mm(self.inter_spTensor, self.user_linear.weight.t())
        return all_user_embedding

    def get_all_item_embedding(self):
        all_item_embedding = torch.sparse.mm(self.inter_spTensor_t, self.item_linear.weight.t())
        return all_item_embedding
    
    # For Graph Denoising
    def sp_cos_sim(self, a, b, eps=1e-8, CHUNK_SIZE=CHUNK_SIZE_FOR_SPMM):
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))

        L = self.inter_indices.shape[1]
        sims = torch.zeros(L, dtype=a.dtype).to(self.config['device'])
        for idx in range(0, L, CHUNK_SIZE):
            batch_indices = self.inter_indices[:, idx:idx + CHUNK_SIZE]

            a_batch = torch.index_select(a_norm, 0, batch_indices[0, :])
            b_batch = torch.index_select(b_norm, 0, batch_indices[1, :])

            dot_prods = torch.mul(a_batch, b_batch).sum(1)
            sims[idx:idx + CHUNK_SIZE] = dot_prods

        return torch.sparse_coo_tensor(self.inter_indices, sims, size=self.interaction_matrix.shape,
                                       dtype=sims.dtype).coalesce()
    def get_sim_mat(self):
        user_feature = self.get_all_user_embedding().to(self.config['device'])
        item_feature = self.get_all_item_embedding().to(self.config['device'])
        sim_inter = self.sp_cos_sim(user_feature, item_feature)
        return sim_inter

    def inter2adj(self, inter):
        inter_t = inter.t().coalesce()
        data = inter.values()
        data_t = inter_t.values()
        adj_data = torch.cat([data, data_t], dim=0)
        adj = torch.sparse.FloatTensor(self.adj_indices, adj_data, self.adj_shape).to(self.config['device']).coalesce()
        return adj

    def get_sim_adj(self, pruning):
        sim_mat = self.get_sim_mat()
        sim_adj = self.inter2adj(sim_mat)

        # pruning
        sim_value = torch.div(torch.add(sim_adj.values(), 1), 2)
        pruned_sim_value = torch.where(sim_value < pruning, torch.zeros_like(sim_value),
                                       sim_value) if pruning > 0 else sim_value
        pruned_sim_adj = torch.sparse.FloatTensor(sim_adj.indices(), pruned_sim_value, self.adj_shape).coalesce()
        self.pruned_sim_adj = pruned_sim_adj

        # normalize
        pruned_sim_indices = pruned_sim_adj.indices()
        diags = torch.sparse.sum(pruned_sim_adj, dim=1).to_dense() + 1e-7
        diags = torch.pow(diags, -1)
        diag_lookup = diags[pruned_sim_indices[0, :]]

        pruned_sim_adj_value = pruned_sim_adj.values()
        normal_sim_value = torch.mul(pruned_sim_adj_value, diag_lookup)
        normal_sim_adj = torch.sparse.FloatTensor(pruned_sim_indices, normal_sim_value,
                                                  self.adj_shape).to(self.config['device']).coalesce()

        return normal_sim_adj

    # For Augmentation
    def cal_cos_sim(self, u_idx, i_idx, eps=1e-8, CHUNK_SIZE=CHUNK_SIZE_FOR_SPMM):
        user_feature = self.get_all_user_embedding().to(self.config['device'])
        item_feature = self.get_all_item_embedding().to(self.config['device'])

        L = u_idx.shape[0]
        sims = torch.zeros(L, dtype=user_feature.dtype).to(self.config['device'])
        for idx in range(0, L, CHUNK_SIZE):
            a_batch = torch.index_select(user_feature, 0, u_idx[idx:idx + CHUNK_SIZE])
            b_batch = torch.index_select(item_feature, 0, i_idx[idx:idx + CHUNK_SIZE])
            dot_prods = torch.mul(a_batch, b_batch).sum(1)
            sims[idx:idx + CHUNK_SIZE] = dot_prods
        return sims
    
    def get_aug_adj(self, adj):
        # random sampling
        aug_user = torch.from_numpy(np.random.choice(self.num_users,
                                                     int(adj._nnz() * self.aug_ratio * 0.5 * self.pool_multi))).to(self.config['device'])
        aug_item = torch.from_numpy(np.random.choice(self.num_items,
                                                     int(adj._nnz() * self.aug_ratio * 0.5 * self.pool_multi))).to(self.config['device'])
        # consider reliability
        cos_sim = self.cal_cos_sim(aug_user, aug_item)
        val, idx = torch.topk(cos_sim, int(adj._nnz() * self.aug_ratio * 0.5))
        aug_user = aug_user[idx]
        aug_item = aug_item[idx]

        aug_indices = torch.stack([aug_user, aug_item + self.num_users], dim=0)
        aug_value = torch.ones_like(aug_user) * torch.median(adj.values())
        sub_aug = torch.sparse.FloatTensor(aug_indices, aug_value, adj.shape).coalesce()
        aug = sub_aug + sub_aug.t()
        aug_adj = (adj + aug).coalesce()

        aug_adj_indices = aug_adj.indices()
        diags = torch.sparse.sum(aug_adj, dim=1).to_dense() + 1e-7
        diags = torch.pow(diags, -0.5)
        diag_lookup = diags[aug_adj_indices[0, :]]

        value_DA = diag_lookup.mul(aug_adj.values())
        normal_aug_value = value_DA.mul(diag_lookup)
        normal_aug_adj = torch.sparse.FloatTensor(aug_adj_indices, normal_aug_value,
                                                  self.norm_adj_matrix.shape).to(self.config['device']).coalesce()
        return normal_aug_adj

    # For normal graph conv
    def computer(self):
        user_embeddings = self.get_all_user_embedding()
        item_embeddings = self.get_all_item_embedding()
        all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        embeddings_list = [all_embeddings]

        self.adj = self.norm_adj_matrix if self.prune_threshold < 0.0 else self.get_sim_adj(self.prune_threshold)
        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.adj, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.num_users, self.num_items])
        return user_all_embeddings, item_all_embeddings
    
    def get_embedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.get_all_user_embedding()[users]
        pos_emb_ego = self.get_all_item_embedding()[pos_items]
        neg_emb_ego = self.get_all_item_embedding()[neg_items]
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def get_users_rating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    # For self-supervision loss
    def computer_ssl(self):
        user_embeddings = self.get_all_user_embedding()
        item_embeddings = self.get_all_item_embedding()
        all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        embeddings_list = [all_embeddings]

        self.aug_adj = self.get_aug_adj(self.adj.detach())
        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.aug_adj, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.num_users, self.num_items])
        return user_all_embeddings, item_all_embeddings

    def ssl_triple_loss(self, z1: torch.Tensor, z2: torch.Tensor, all_emb: torch.Tensor):
        norm_emb1 = F.normalize(z1)
        norm_emb2 = F.normalize(z2)
        norm_all_emb = F.normalize(all_emb)
        pos_score = torch.mul(norm_emb1, norm_emb2).sum(dim=1)
        ttl_score = torch.matmul(norm_emb1, norm_all_emb.transpose(0, 1))
        pos_score = torch.exp(pos_score / self.tau)
        ttl_score = torch.exp(ttl_score / self.tau).sum(dim=1)

        ssl_loss = -torch.log(pos_score / ttl_score).sum()
        return ssl_loss
    
    # Actually, not BPR Loss but Total Loss
    def bpr_loss(self, users, pos, neg):
        # obtain embedding
        (users_emb, pos_emb, neg_emb, user_emb_0, pos_emb_0, neg_emb_0) = self.get_embedding(
            users.long(), pos.long(), neg.long()
        )
        
        # calculate L2 reg
        reg_loss = (
            (1 / 2)
            * (user_emb_0.norm(2).pow(2) + pos_emb_0.norm(2).pow(2) + neg_emb_0.norm(2).pow(2))
            / float(len(users))
        )
        
        # calculate BPR Loss
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        # calculate agreement
        if self.MIM_weight > 0.:
            aug_user_all_embeddings, _ = self.computer_ssl()
            aug_u_embeddings = aug_user_all_embeddings[users]
            mutual_info = self.ssl_triple_loss(users_emb, aug_u_embeddings, aug_user_all_embeddings)
            loss += self.MIM_weight * mutual_info
            
        return loss, reg_loss