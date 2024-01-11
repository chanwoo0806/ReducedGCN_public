"""
Reference: https://github.com/gusye1234/LightGCN-PyTorch
"""

import os
import torch
from parse import parse_args
import time
import logging
import sys
from warnings import simplefilter


simplefilter(action="ignore", category=FutureWarning)
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def mkdir_if_not_exist(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)


def get_logger(dataset_name, log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-5s [%(filename)s:%(lineno)d] %(message)s"
    )

    streaming_handler = logging.StreamHandler()
    streaming_handler.setFormatter(formatter)
    filename = f"{dataset_name}.log"
    file_handler = logging.FileHandler(os.path.join(log_dir, filename))
    file_handler.setFormatter(formatter)
    logger.addHandler(streaming_handler)
    logger.addHandler(file_handler)
    return logger


def set_config():
    config = {}
    config["model"] = args.model
    config["dataset"] = args.dataset
    config["bpr_batch_size"] = args.bpr_batch
    config["test_u_batch_size"] = args.testbatch
    config["latent_dim_rec"] = args.recdim
    config["n_layers"] = args.layer
    config["lr"] = args.lr
    config["decay"] = args.decay
    config["early_stop"] = args.early_stop
    config["eval_step"] = args.eval_step
    config["epochs"] = args.epochs
    config["topks"] = args.topks
    config["tensorboard"] = args.tensorboard
    config["neg_k"] = args.neg_k
    config["multicore"] = args.multicore
    config["bigdata"] = False
    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config["load"] = args.load
    config["path"] = args.path
    config["comment"] = args.comment

    ## dropout
    config["mlp_dropout"] = args.mlp_dropout
    config["dropout"] = args.dropout
    config["keep_prob"] = args.keepprob

    ## RGCN
    config["emb_weight"] = args.emb_weight
    config["temp"] = args.temp
    config["groups"] = args.groups

    all_dataset = ["yelp2018", "ml-1m", "ks10"]
    if args.dataset not in all_dataset:
        raise NotImplementedError(f"Haven't supported {args.dataset} yet!, try {all_dataset}")
    all_models = ["rgcn"]
    if args.model not in all_models:
        raise NotImplementedError(f"Haven't supported {args.model} yet!, try {all_models}")
    return config


args = parse_args()
config = set_config()

LOGO = r"""
   ____    U _____ u ____     _   _    ____ U _____ u ____     ____     ____  _   _     
U |  _"\ u \| ___"|/|  _"\ U |"|u| |U /"___|\| ___"|/|  _"\ U /"___|uU /"___|| \ |"|    
 \| |_) |/  |  _|" /| | | | \| |\| |\| | u   |  _|" /| | | |\| |  _ /\| | u <|  \| |>   
  |  _ <    | |___ U| |_| |\ | |_| | | |/__  | |___ U| |_| |\| |_| |  | |/__U| |\  |u   
  |_| \_\   |_____| |____/ u<<\___/   \____| |_____| |____/ u \____|   \____||_| \_|    
  //   \\_  <<   >>  |||_  (__) )(   _// \\  <<   >>  |||_    _)(|_   _// \\ ||   \\,-. 
 (__)  (__)(__) (__)(__)_)     (__) (__)(__)(__) (__)(__)_)  (__)__) (__)(__)(_")  (_/  
"""
ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
CODE_PATH = os.path.join(ROOT_PATH, "code")
DATA_PATH = os.path.join(ROOT_PATH, "data")
BOARD_PATH = os.path.join(CODE_PATH, "runs")
if args.load:
    FOLDER_PATH = os.path.join(
        BOARD_PATH, os.path.join(args.dataset, os.path.join(args.model, args.load))
    )
else:
    FOLDER_PATH = (
        os.path.join(
            BOARD_PATH,
            os.path.join(args.dataset, os.path.join(args.model, time.strftime("%m-%d-%Hh%Mm%Ss"))),
        )
        + f"-l{args.layer}-d{args.recdim}-g{args.groups}-{args.comment}"
    )
LOGGER = get_logger(args.dataset, FOLDER_PATH)

sys.path.append(os.path.join(CODE_PATH, "sources"))
mkdir_if_not_exist(FOLDER_PATH)
