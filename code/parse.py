"""
Reference: https://github.com/gusye1234/LightGCN-PyTorch
"""
import argparse
import yaml


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", type=str, default=None, help="Path to the configuration file."
    )
    parser.add_argument("--model", type=str, default="rgcn")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ks10",
        help="available datasets: [yelp2018, ml-1m, ks10]",
    )
    parser.add_argument(
        "--bpr_batch",
        type=int,
        default=16384,
        help="the batch size for bpr loss training procedure",
    )
    parser.add_argument(
        "--testbatch", type=int, default=1024, help="the batch size of users for testing"
    )
    parser.add_argument("--recdim", type=int, default=64, help="the embedding size of lightGCN")
    parser.add_argument("--layer", type=int, default=5, help="the layer num of lightGCN")
    parser.add_argument("--lr", type=float, default=1e-2, help="the learning rate")
    parser.add_argument(
        "--decay", type=float, default=1e-4, help="the weight decay for l2 normalizaton"
    )
    parser.add_argument("--early_stop", type=int, default=50)
    parser.add_argument("--eval_step", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--topks", type=str, default="1,5,10,20", help="@k test list")
    parser.add_argument("--tensorboard", type=int, default=1, help="enable tensorboard")
    parser.add_argument("--neg_k", type=int, default=1)
    parser.add_argument(
        "--multicore", type=int, default=0, help="whether we use multiprocessing or not in test"
    )

    parser.add_argument(
        "--load", type=str, default=None, help="whether we load the pretrained weight or not"
    )
    parser.add_argument("--path", type=str, default="./checkpoints", help="path to save weights")
    parser.add_argument("--comment", type=str, default="debug")

    # dropout
    parser.add_argument(
        "--mlp_dropout", type=float, default=0.0, help="dropout probability of MLP layer"
    )
    parser.add_argument("--dropout", type=int, default=0, help="using the dropout or not")
    parser.add_argument(
        "--keepprob",
        type=float,
        default=0.8,
        help="probability of keeping edges when using dropout",
    )

    # ReducedGCN
    parser.add_argument(
        "--emb_weight", type=str, default="1.5,0.5", help="weight ratio of global & local embedding"
    )
    parser.add_argument("--temp", type=float, default=1.0, help="temperature")
    parser.add_argument("--groups", type=int, default=64, help="Number of sub-community")

    args = parser.parse_args()

    def str_to_list(arg, elem_type):
        return [elem_type(x) for x in arg.split(",")]

    args.topks = str_to_list(args.topks, int)
    args.emb_weight = str_to_list(args.emb_weight, float)
    if args.config_file:
        config = load_config(args.config_file)
        for option, value in config.items():
            setattr(args, option, value)
    return args
