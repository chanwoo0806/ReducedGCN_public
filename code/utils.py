"""
Reference: https://github.com/gusye1234/LightGCN-PyTorch
"""
import world
import torch
from torch import nn, optim
import numpy as np
from time import time
from sklearn.metrics import roc_auc_score
import os
import pandas as pd
import datetime

try:
    from cppimport import imp_from_filepath
    from os.path import join, dirname

    path = join(dirname(__file__), "sources/sampling.cpp")
    sampling = imp_from_filepath(path)
    sample_ext = True
    world.LOGGER.info("Cpp extension loaded well")
except:
    world.LOGGER.info("Cpp extension not loaded")
    sample_ext = False


class BPRLoss:
    def __init__(self, recmodel, config: dict):
        self.model = recmodel
        self.weight_decay = config["decay"]
        self.lr = config["lr"]
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)
        world.LOGGER.info(f"use BPRLoss")

    def stage_one(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss * self.weight_decay
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        torch.cuda.empty_cache()

        return loss.cpu().item()


class GroupDataset:
    def __init__(self, group, num_users):
        indices = group.coalesce().indices()
        mask = indices[0] < num_users
        indices = indices[:, mask]
        indices[1] -= num_users

        self.users = indices[0].unique()
        self.n_users = self.users.numel()
        self.items = indices[1].unique()
        self.m_items = self.items.numel()

        pos_dict = (
            pd.DataFrame(indices.T, columns=["user_id", "item_id"])
            .groupby("user_id")["item_id"]
            .apply(np.array)
            .to_dict()
        )
        self.all_pos = [np.array([]) for _ in range(num_users)]
        for k, v in pos_dict.items():
            self.all_pos[k] = v


def uniform_sample_original(dataset, neg_ratio=1):
    all_pos = dataset.all_pos
    if sample_ext == False:
        S = uniform_sample_original_python(dataset)
    else:
        S = sampling.sample_negative(
            dataset.n_users, dataset.m_items, dataset.train_data_size, all_pos, neg_ratio
        )
    return S


def uniform_sample_original_python(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    # s = datetime.datetime.now()

    user_num = dataset.train_data_size
    users = np.random.randint(0, dataset.n_users, user_num)
    all_pos = dataset.all_pos
    S = []
    sample_time1 = 0.0
    sample_time2 = 0.0
    for i, user in enumerate(users):
        start = time()
        posForUser = all_pos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        end = time()
        sample_time1 += end - start

    # e = datetime.datetime.now()
    # t = (e - s).seconds
    # world.LOGGER.info(f"sampler: {t} seconds")
    return np.array(S)


def uniform_sample_from_group(dataset, train_size):
    users = np.random.choice(dataset.users.numpy(), size=train_size, replace=True)
    all_pos = dataset.all_pos
    S = []
    for i, user in enumerate(users):
        posForUser = all_pos[user]
        if len(posForUser) == 0:
            continue
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        end = time()
    return np.array(S)


# ===================end samplers==========================
# =====================utils====================================


def getFileName():
    if world.config["model"] == "mf":
        file = f"mf-{world.config['dataset']}-d{world.config['latent_dim_rec']}.pth.tar"
    elif world.config["model"] == "lgn":
        file = f"lgn-{world.config['dataset']}-l{world.config['n_layers']}-d{world.config['latent_dim_rec']}.pth.tar"
    elif world.config["model"] == "impgcn":
        file = f"impgcn-{world.config['dataset']}-l{world.config['n_layers']}-d{world.config['latent_dim_rec']}-g{world.config['groups']}.pth.tar"
    elif world.config["model"] == "localgcn":
        file = f"localgcn-{world.config['dataset']}-l{world.config['n_layers']}-d{world.config['latent_dim_rec']}-g{world.config['groups']}.pth.tar"

    return os.path.join(world.FILE_PATH, file)


def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get("batch_size", world.config["bpr_batch_size"])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i : i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i : i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get("indices", False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError("All inputs to shuffle must have " "the same length.")

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """

    from time import time

    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get("name"):
            timer.NAMED_TAPE[kwargs["name"]] = (
                timer.NAMED_TAPE[kwargs["name"]] if timer.NAMED_TAPE.get(kwargs["name"]) else 0.0
            )
            self.named = kwargs["name"]
            if kwargs.get("group"):
                # TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)
            
# ====================Metrics==============================
# =========================================================
def recall_precision_at_K(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {"recall": recall, "precision": precis}


def NDCG_at_K(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1.0 / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1.0 / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.0] = 1.0
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.0
    return np.sum(ndcg)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype("float")


# ====================end Metrics=============================
# =========================================================
