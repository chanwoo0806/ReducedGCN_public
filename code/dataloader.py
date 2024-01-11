"""
Reference: https://github.com/gusye1234/LightGCN-PyTorch
"""
from os.path import join
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from time import time
from tqdm import tqdm


class Loader(Dataset):
    def __init__(self, config=world.config, path=join(world.DATA_PATH, "gowalla")):
        world.LOGGER.info(f"loading [{path}]")
        self.group = config["groups"]
        self.mode_dict = {"train": 0, "test": 1}
        self.mode = self.mode_dict["train"]
        self.n_user = 0
        self.m_item = 0
        self.path = path

        train_file = join(path, "train.txt")
        test_file = join(path, "test.txt")
        (
            self.train_unique_users,
            self.train_user,
            self.train_item,
            m_item_train,
            n_user_train,
            self.train_data_size,
        ) = self.load_data(train_file)
        (
            self.test_unique_users,
            self.test_user,
            self.test_item,
            m_item_test,
            n_user_test,
            self.test_data_size,
        ) = self.load_data(test_file)
        self.m_item = max(m_item_train, m_item_test) + 1
        self.n_user = max(n_user_train, n_user_test) + 1

        world.LOGGER.info(f"{self.train_data_size} interactions for training")
        world.LOGGER.info(f"{self.test_data_size} interactions for testing")
        world.LOGGER.info(
            f"{world.config['dataset']} Sparsity : {(self.train_data_size + self.test_data_size) / self.n_users / self.m_items}"
        )
        world.LOGGER.info(f"number of users : {self.n_users}")
        world.LOGGER.info(f"number of items : {self.m_items}")

        # (users,items), bipartite graph
        self.Graph = None
        self.user_item_net = csr_matrix(
            (np.ones(len(self.train_user)), (self.train_user, self.train_item)),
            shape=(self.n_user, self.m_item),
        )
        self.users_D = np.array(self.user_item_net.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.0] = 1
        self.items_D = np.array(self.user_item_net.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.0] = 1.0
        # pre-calculate
        self._all_pos = self.get_user_pos_items(list(range(self.n_user)))
        self.test_dict_ = self.build_test()
        world.LOGGER.info(f"{world.config['dataset']} is ready to go")

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def test_dict(self):
        return self.test_dict_

    @property
    def all_pos(self):
        return self._all_pos

    def load_data(self, filepath):
        m_item, n_user, data_size = 0, 0, 0
        unique_users, user, item = [], [], []
        with open(filepath) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip("\n").split(" ")
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    unique_users.append(uid)
                    user.extend([uid] * len(items))
                    item.extend(items)
                    m_item = max(m_item, max(items))
                    n_user = max(n_user, uid)
                    data_size += len(items)
        unique_users = np.array(unique_users)
        user = np.array(user)
        item = np.array(item)
        return (unique_users, user, item, m_item, n_user, data_size)

    def _remove_zeros_from_sparse_tensor(self, X):
        X = X.coalesce()
        indices = X._indices()
        values = X._values()
        non_zero_mask = values != 0
        new_indices = indices[:, non_zero_mask]
        new_values = values[non_zero_mask]
        new_X = torch.sparse_coo_tensor(new_indices, new_values, X.shape)
        return new_X

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def get_sparse_graph(self):
        world.LOGGER.info("loading adjacency matrix")
        if self.Graph is None:
            try:
                self.Graph = torch.load(self.path + "/s_pre_adj_mat_tensor.pt")
                world.LOGGER.info("successfully loaded...")
            except:
                world.LOGGER.info("generating adjacency matrix")
                start = time()
                adj_mat = sp.dok_matrix(
                    (self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32
                )
                adj_mat = adj_mat.tolil()
                R = self.user_item_net.tolil()
                adj_mat[: self.n_users, self.n_users :] = R
                adj_mat[self.n_users :, : self.n_users] = R.T
                adj_mat = adj_mat.todok()

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.0
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()

                self.Graph = self._convert_sp_mat_to_sp_tensor(
                    norm_adj
                )  # D^(-1/2) A D^(1/2) -> (A+I)
                torch.save(self.Graph, self.path + "/s_pre_adj_mat_tensor.pt")
                end = time()
                world.LOGGER.info(f"costing {end-start}s, saved norm_mat...")

            self.Graph = self.Graph.coalesce().to(world.config["device"])
            world.LOGGER.info("don't split the matrix")

        return self.Graph

    def build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.test_item):
            user = self.test_user[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def get_user_item_feedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        return np.array(self.user_item_net[users, items]).astype("uint8").reshape((-1,))

    def get_user_pos_items(self, users):
        posItems = []
        for user in users:
            posItems.append(self.user_item_net[user].nonzero()[1])
        return posItems
