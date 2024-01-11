import world
import utils
import torch
from tensorboardX import SummaryWriter
import trainer
import os
import dataloader
import model
import sys
import datetime


def load_model(model_name):
    models = {"rgcn": model.RGCN}
    return models[model_name]


def load_dataset(dataset_path):
    return dataloader.Loader(path=dataset_path)


if __name__ == "__main__":
    world.LOGGER.info(world.LOGO)
    world.LOGGER.info(f">>FOLDER: {world.FOLDER_PATH}")
    world.LOGGER.info(f">>DEVICE: {world.config['device']}")
    with open(os.path.join(world.FOLDER_PATH, "config.txt"), "w") as f:
        f.writelines([f"{k}: {v}\n" for k, v in world.config.items()])

    world.LOGGER.info(f">>DATASET: {world.config['dataset']}")
    dataset = load_dataset(dataset_path=os.path.join(world.DATA_PATH, world.config["dataset"]))

    world.LOGGER.info(f">>MODEL: {world.config['model']}")
    model = load_model(world.config["model"])(world.config, dataset)
    model = model.to(world.config["device"])

    # init tensorboard
    if world.config["tensorboard"]:
        w: SummaryWriter = SummaryWriter(world.FOLDER_PATH)
    else:
        w = None
        world.LOGGER.info("not enable tensorflowboard")

    bpr = utils.BPRLoss(model, world.config)
    start_epoch = 0
    # load pretrained model weights
    if world.config["load"]:
        try:
            chkpoint_path = f"{world.FOLDER_PATH}/checkpoint.pt"
            chkpoint = torch.load(chkpoint_path)
            start_epoch = chkpoint["last_epoch"] + 1
            model.load_state_dict(chkpoint["model_state_dict"])
            bpr.opt.load_state_dict(chkpoint["optimizer_state_dict"])
            world.LOGGER.info(f"loaded model weights from {chkpoint_path}")
        except FileNotFoundError:
            world.LOGGER.info(f"{chkpoint_path} not exists, start from beginning")

    try:
        patience = 0
        best_score = 0.0
        for epoch in range(start_epoch, world.config["epochs"]):
            train_info, save_dict = trainer.BPR_train_original(
                dataset, model, bpr, epoch, neg_k=world.config["neg_k"], w=w
            )
            world.LOGGER.info(f'EPOCH[{epoch+1}/{world.config["epochs"]}] {train_info}')

            if epoch % world.config["eval_step"] == 0:
                world.LOGGER.info("[TEST]")
                test_info, best_score, patience = trainer.evaluate(
                    dataset,
                    model,
                    epoch,
                    patience,
                    best_score,
                    save_dict,
                    w,
                    world.config["multicore"],
                )
                world.LOGGER.info(test_info)
                if world.config["early_stop"] and patience >= world.config["early_stop"]:
                    world.LOGGER.info(f"Early stopping at epoch {epoch+1}/{world.config['epochs']}")
                    break
    finally:
        if world.config["tensorboard"]:
            w.close()
