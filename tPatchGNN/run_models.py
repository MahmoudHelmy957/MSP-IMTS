import sys
sys.path.append("..")

import os
import time
import datetime
import argparse
import numpy as np
import pandas as pd
import random
from random import SystemRandom
from sklearn import model_selection
import re
import torch
import torch.nn as nn
import torch.optim as optim

import lib.utils as utils
from lib.parse_datasets import parse_datasets
from model.tPatchGNN import *
from lib.evaluation import compute_all_losses

parser = argparse.ArgumentParser("IMTS Forecasting")

############################# multi scale ########################
parser.add_argument(
    "--multi_scales",
    type=str,
    default="",
    help='Comma list of patch sizes in hours, e.g. "2,8,24". Empty = single-scale.',
)
parser.add_argument(
    "--multi_strides",
    type=str,
    default="",
    help="Comma list of strides in hours. Empty = same as multi_scales.",
)
parser.add_argument(
    "--fusion",
    type=str,
    default="concat",
    choices=["concat", "scale_attn"],
    help="Fusion method for multi-scale.",
)
parser.add_argument(
    "--disable_fusion_mixer",
    action="store_true",
    help="Disable the NodeMixer fusion block and fall back to identity.",
)
parser.add_argument(
    "--fusion_mixer_hidden_mult",
    type=float,
    default=2.0,
    help="Expansion ratio for the fusion NodeMixer MLP.",
)
parser.add_argument(
    "--fusion_mixer_dropout",
    type=float,
    default=0.1,
    help="Dropout probability used inside the fusion NodeMixer block.",
)
parser.add_argument(
    "--fusion_track_stats",
    action="store_true",
    help="Track RMS deltas introduced by the fusion block during training/eval.",
)
################################################

parser.add_argument("--state", type=str, default="def")
parser.add_argument("-n", type=int, default=int(1e8), help="Size of the dataset")
parser.add_argument("--hop", type=int, default=1, help="hops in GNN")
parser.add_argument("--nhead", type=int, default=1, help="heads in Transformer")
parser.add_argument("--tf_layer", type=int, default=1, help="# of layer in Transformer")
parser.add_argument("--nlayer", type=int, default=1, help="# of layer in TSmodel")
parser.add_argument("--epoch", type=int, default=1000, help="training epochs")
parser.add_argument("--patience", type=int, default=10, help="patience for early stop")
parser.add_argument(
    "--history",
    type=int,
    default=24,
    help="number of hours (months for ushcn and ms for activity) as historical window",
)
parser.add_argument(
    "--model",
    type=str,
    default="tPatchGNN",
    help="Model name identifier used in logs and checkpoints.",
)

parser.add_argument(
    "-ps", "--patch_size", type=float, default=24, help="window size for a patch"
)
parser.add_argument(
    "--stride", type=float, default=24, help="period stride for patch sliding"
)
parser.add_argument(
    "--logmode", type=str, default="a", help="File mode of logging."
)

parser.add_argument("--lr", type=float, default=1e-3, help="Starting learning rate.")
parser.add_argument("--w_decay", type=float, default=0.0, help="weight decay.")
parser.add_argument("-b", "--batch_size", type=int, default=32)

parser.add_argument(
    "--save", type=str, default="experiments/", help="Path for save checkpoints"
)
parser.add_argument(
    "--load",
    type=str,
    default=None,
    help="ID of the experiment to load for evaluation. If None, run a new experiment.",
)
parser.add_argument("--seed", type=int, default=1, help="Random seed")
parser.add_argument(
    "--dataset",
    type=str,
    default="physionet",
    help="Dataset to load. Available: physionet, mimic, ushcn",
)
# Model hyperparameters (required by tPatchGNN and MultiScaleTPatchGNN)
parser.add_argument("--hid_dim", type=int, default=32, help="Hidden dimension of node embeddings.")
parser.add_argument("--te_dim", type=int, default=10, help="Temporal encoding dimension.")
parser.add_argument("--node_dim", type=int, default=10, help="Node embedding dimension.")
parser.add_argument("--gpu", type=str, default="0", help="GPU id to use (e.g., '0' or '0,1').")
parser.add_argument("--cpu", action="store_true", help="Force CPU execution even if CUDA is available.")
parser.add_argument(
    "--outlayer",
    type=str,
    default="fc",
    help="Type of output layer to use in tPatchGNN (e.g., 'fc', 'linear', 'mlp').",
)


# value 0 means using original time granularity, Value 1 means quantization by 1 hour
if __name__ == "__main__":
    args = parser.parse_args()

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        args.device = torch.device("cpu")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Compute number of patches per sequence
    import math
    args.npatch = int(math.ceil((args.history - args.patch_size) / args.stride)) + 1

    experimentID = int(SystemRandom().random() * 100000)
    ckpt_path = os.path.join(args.save, f"experiment_{experimentID}.ckpt")

    input_command = sys.argv
    ind = [i for i, val in enumerate(input_command) if val == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind + 2) :]
    input_command = " ".join(input_command)

    ##################################################################
    data_obj = parse_datasets(args, patch_ts=True)
    input_dim = data_obj["input_dim"]

    from copy import deepcopy
    use_ms = args.multi_scales not in (None, "", [])
    args.ndim = input_dim

    if use_ms:
        from model.multiscale_tpatchgnn import MultiScaleTPatchGNN

        first_batch = utils.get_next_batch(data_obj["train_dataloader"])
        fusion_init_stats = None
        fusion_identity = False

        print(
            f"[MS] use_ms={use_ms} fusion={args.fusion} "
            f"scales={args.multi_scales or 'single-scale'} "
            f"strides={(args.multi_strides or args.multi_scales) or 'same-as-scales'} "
            f"npatches_per_scale={list(map(int, first_batch['npatches']))}"
        )

        submodels = []
        for M_k in first_batch["npatches"]:
            sub_args = deepcopy(args)
            sub_args.npatch = int(M_k)
            submodels.append(
                tPatchGNN(sub_args, supports=None, dropout=0).to(args.device)
            )

        model = MultiScaleTPatchGNN(
            submodels=submodels,
            te_dim=args.te_dim,
            proj_dim=args.hid_dim,
            fusion=args.fusion,
            use_node_mixer=not args.disable_fusion_mixer,
            node_mixer_kwargs=dict(
                hidden_mult=args.fusion_mixer_hidden_mult,
                drop=args.fusion_mixer_dropout,
            ),
            track_fusion_stats=args.fusion_track_stats,
        ).to(args.device)

        fusion_init_stats = model.fusion_parameter_stats()
        fusion_identity = isinstance(model.fusion_block, nn.Identity)
    else:
        from model.tpatchgnn_with_mixer import tPatchGNN_WithMixer

        base = tPatchGNN(args).to(args.device)
        
        model = tPatchGNN_WithMixer(
            base_model=base,
            hidden_mult=args.fusion_mixer_hidden_mult,
            drop=args.fusion_mixer_dropout,
        ).to(args.device)

    ##################################################################

    if args.n < 12000:
        args.state = "debug"
        log_path = f"logs/{args.dataset}_{args.model}_{args.state}.log"
    else:
        log_path = (
            f"logs/{args.dataset}_{args.model}_{args.state}_"
            f"{args.patch_size}patch_{args.stride}stride_{args.nlayer}layer_{args.lr}lr.log"
        )

    if not os.path.exists("logs/"):
        utils.makedirs("logs/")
    logger = utils.get_logger(
        logpath=log_path, filepath=os.path.abspath(__file__), mode=args.logmode
    )
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info(input_command)
    logger.info(args)

    if use_ms and fusion_init_stats is not None:
        logger.info(
            "Fusion block: %s (tokens=%s, fused_dim=%s)",
            fusion_init_stats.get("fusion_type"),
            fusion_init_stats.get("n_tokens"),
            fusion_init_stats.get("fused_dim"),
        )
        if fusion_identity:
            logger.info(
                "Fusion mixer disabled; concatenated features flow directly to the decoder."
            )

    # ====== TRAINING LOOP (paste below the logger setup) ======
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
    num_batches = data_obj["n_train_batches"]
    best_val_mse = float("inf")
    best_iter = -1
    test_res = None
    
    for itr in range(args.epoch):
        st = time.time()
    
        # ---- Train ----
        model.train()
        for _ in range(num_batches):
            optimizer.zero_grad()
            batch = utils.get_next_batch(data_obj["train_dataloader"])
            train_res = compute_all_losses(model, batch)  # provided by model.tPatchGNN import *
            loss = train_res["loss"] if isinstance(train_res["loss"], torch.Tensor) else torch.tensor(train_res["loss"])
            loss.backward()
            optimizer.step()
    
        # ---- Validate ----
        model.eval()
        with torch.no_grad():
            val_res = evaluation(model, data_obj["val_dataloader"], data_obj["n_val_batches"])  # provided in repo
    
        # ---- Track best & evaluate on test when improved ----
        if val_res["mse"] < best_val_mse:
            best_val_mse = val_res["mse"]
            best_iter = itr
            with torch.no_grad():
                test_res = evaluation(model, data_obj["test_dataloader"], data_obj["n_test_batches"])
            # save best checkpoint
            try:
                torch.save({"model_state": model.state_dict(), "args": vars(args), "iter": itr}, ckpt_path)
            except Exception as e:
                logger.warning(f"Could not save checkpoint to {ckpt_path}: {e}")
    
        # ---- Optional fusion stats each epoch ----
        if use_ms and getattr(model, "_track_fusion_stats", False):
            fusion_stats = model.fusion_stats(reset=True)
            logger.info(
                "Fusion stats - batches: %d, Δ_rms: %.6f, input_rms: %.6f, Δ/input: %.4f, max|Δ|: %.6f",
                int(fusion_stats.get("tracked_batches", 0)),
                fusion_stats.get("delta_rms_mean", 0.0),
                fusion_stats.get("input_rms_mean", 0.0),
                fusion_stats.get("delta_to_input_ratio", 0.0),
                fusion_stats.get("delta_abs_max", 0.0),
            )
    
        # ---- Logging ----
        logger.info("- Epoch %03d, ExpID %s", itr, experimentID)
        logger.info(
            "Train - Loss (one batch): %.5f",
            train_res["loss"].item() if isinstance(train_res["loss"], torch.Tensor) else float(train_res["loss"]),
        )
        logger.info(
            "Val   - Loss, MSE, RMSE, MAE, MAPE: %.5f, %.5f, %.5f, %.5f, %.2f%%",
            val_res["loss"], val_res["mse"], val_res["rmse"], val_res["mae"], val_res["mape"] * 100,
        )
        if test_res is not None:
            logger.info(
                "Test  - Best epoch, Loss, MSE, RMSE, MAE, MAPE: %d, %.5f, %.5f, %.5f, %.5f, %.2f%%",
                best_iter, test_res["loss"], test_res["mse"], test_res["rmse"], test_res["mae"], test_res["mape"] * 100,
            )
        logger.info("Time spent: %.2fs", time.time() - st)
    
        # ---- Early stopping ----
        if (itr - best_iter) >= args.patience:
            logger.info("Early stopping at epoch %d (best epoch %d).", itr, best_iter)
            break
    # ====== END TRAINING LOOP ======
