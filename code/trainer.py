"""
Reference: https://github.com/gusye1234/LightGCN-PyTorch
"""
import world
import numpy as np
import torch
import utils
from utils import timer
from tqdm import tqdm
import multiprocessing
import os
import itertools
import pandas as pd


def BPR_train_original(dataset, model, bpr, epoch, neg_k=1, w=None):
    model.train()
    with timer(name="Sample"):
        S = utils.uniform_sample_original(dataset)

    users = torch.Tensor(S[:, 0]).long()
    pos_items = torch.Tensor(S[:, 1]).long()
    neg_items = torch.Tensor(S[:, 2]).long()

    users = users.to(world.config["device"])
    pos_items = pos_items.to(world.config["device"])
    neg_items = neg_items.to(world.config["device"])
    users, pos_items, neg_items = utils.shuffle(users, pos_items, neg_items)
    total_batch = len(users) // world.config["bpr_batch_size"] + 1
    aver_loss = 0.0
    train_loader = enumerate(
        tqdm(
            utils.minibatch(users, pos_items, neg_items, batch_size=world.config["bpr_batch_size"]),
            total=total_batch,
        )
    )

    model.clustering()
    for batch_i, (batch_users, batch_pos, batch_neg) in train_loader:
        cri = bpr.stage_one(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if world.config["tensorboard"] and w:
            w.add_scalar(
                f"BPRLoss/BPR",
                cri,
                epoch * int(len(users) / world.config["bpr_batch_size"]) + batch_i,
            )

    save_dict = {
        "last_epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": bpr.opt.state_dict(),
        # "scheduler_state_dict": scheduler.state_dict(),
    }
    save_model(world.FOLDER_PATH, save_dict)

    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    results = f"loss{aver_loss:.3f}-{time_info}"
    return results, save_dict


def test_one_batch(X):
    sorted_items = X[0].numpy()
    gt = X[1]
    r = utils.getLabel(gt, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.config["topks"]:
        ret = utils.recall_precision_at_K(gt, r, k)
        pre.append(ret["precision"])
        recall.append(ret["recall"])
        ndcg.append(utils.NDCG_at_K(gt, r, k))
    return {"recall": np.array(recall), "precision": np.array(pre), "ndcg": np.array(ndcg)}


def evaluate(
    dataset, model, epoch, patience, best_score, save_dict=None, w=None, multicore=0, analysis=None
):
    u_batch_size = world.config["test_u_batch_size"]
    test_dict: dict = dataset.test_dict

    # eval mode with no dropout
    model = model.eval()
    max_K = max(world.config["topks"])
    best_save_dir = os.path.join(world.FOLDER_PATH, "best")

    if multicore == 1:
        cores = multiprocessing.cpu_count() // 2
        pool = multiprocessing.Pool(cores)
    results = {
        "precision": np.zeros(len(world.config["topks"])),
        "recall": np.zeros(len(world.config["topks"])),
        "ndcg": np.zeros(len(world.config["topks"])),
    }

    with torch.no_grad():
        users = list(test_dict.keys())
        users_list = []
        rating_list = []
        gt_list = []
        total_batch = len(users) // u_batch_size + 1
        test_loader = tqdm(utils.minibatch(users, batch_size=u_batch_size), total=total_batch)
        for batch_users in test_loader:
            all_pos = dataset.get_user_pos_items(batch_users)
            gt = [test_dict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.config["device"])

            rating = model.get_users_rating(batch_users_gpu)

            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(all_pos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            gt_list.append(gt)
        assert total_batch == len(users_list)

        X = zip(rating_list, gt_list)
        if analysis:
            user_ratings = {}
            us = list(itertools.chain.from_iterable(users_list))
            g = list(itertools.chain.from_iterable(gt_list))
            r = torch.cat(rating_list, dim=0).tolist()
            user_ratings["users"] = us
            user_ratings["label"] = g
            user_ratings["pred"] = r
            df = pd.DataFrame(user_ratings)
            df.to_parquet(world.ANALYSIS_PATH)

        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size / len(users))
        for result in pre_results:
            results["recall"] += result["recall"]
            results["precision"] += result["precision"]
            results["ndcg"] += result["ndcg"]
        results["recall"] /= float(len(users))
        results["precision"] /= float(len(users))
        results["ndcg"] /= float(len(users))
        if world.config["tensorboard"] and w:
            for i, k in enumerate(world.config["topks"]):
                w.add_scalar(f"Test/Recall_{k}", results["recall"][i], epoch)
                w.add_scalar(f"Test/Precision_{k}", results["precision"][i], epoch)
                w.add_scalar(f"Test/NDCG_{k}", results["ndcg"][i], epoch)
        if multicore == 1:
            pool.close()

        score = results["ndcg"][world.config["topks"].index(20)]
        if save_dict and score > best_score:
            best_score = score
            patience = 0
            world.LOGGER.info("Save best model")
            save_model(best_save_dir, save_dict)
        else:
            patience += 1

        return results, best_score, patience


def save_model(save_dir, save_dict):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_save_path = os.path.join(save_dir, "model.pt")
    chkpoint_save_path = os.path.join(save_dir, "checkpoint.pt")
    torch.save(save_dict["model_state_dict"], model_save_path)
    torch.save(save_dict, chkpoint_save_path)
