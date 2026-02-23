# lib/cli_args.py

import argparse


def build_single_scale_parser():
    """
    CLI parser for single-scale forecasting experiments.
    Used by RunModelsSingle.py and optionally runmodels.py.
    """

    parser = argparse.ArgumentParser("IMTS Forecasting (Single-Scale)")

    # ---------------- Dataset / Model ----------------
    parser.add_argument("--state", type=str, default="def")
    parser.add_argument("-n", type=int, default=int(1e8), help="Size of the dataset")
    parser.add_argument("--dataset", type=str, default="physionet", help="Dataset to load.")
    parser.add_argument("--model", type=str, default="tPatchGNN", help="Model name")
    parser.add_argument("--outlayer", type=str, default="Linear", help="Output layer name")

    # ---------------- Architecture ----------------
    parser.add_argument("--hop", type=int, default=1, help="hops in GNN")
    parser.add_argument("--nhead", type=int, default=1, help="heads in Transformer")
    parser.add_argument("--tf_layer", type=int, default=1, help="# of layer in Transformer")
    parser.add_argument("--nlayer", type=int, default=1, help="# of layer in TSmodel")

    parser.add_argument("-hd", "--hid_dim", type=int, default=64, help="Hidden dim")
    parser.add_argument("-td", "--te_dim", type=int, default=10, help="Time enc dim")
    parser.add_argument("-nd", "--node_dim", type=int, default=10, help="Node dim")

    # ---------------- Training ----------------
    parser.add_argument("--epoch", type=int, default=1000, help="training epochs")
    parser.add_argument("--patience", type=int, default=10, help="patience for early stop")
    parser.add_argument("--lr", type=float, default=1e-3, help="Starting learning rate.")
    parser.add_argument("--w_decay", type=float, default=0.0, help="weight decay.")
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1, help="Random seed")

    # ---------------- Time-Series ----------------
    parser.add_argument("--history", type=int, default=24, help="historical window")
    parser.add_argument("-ps", "--patch_size", type=float, default=24, help="window size for a patch")
    parser.add_argument("--stride", type=float, default=24, help="period stride for patch sliding")
    parser.add_argument("--normalization", type=int, default=0,
                        help="0 = per-channel, 1 = global scalar normalization")
    # ---------------- Postprocessing ----------------
    parser.add_argument("--denorm_test_pred", type=int, default=0,
                        help="Denormalize test targets and predictions. 1 = True, 0 = False")

    # ---------------- Loss Type ----------------
    parser.add_argument("--global_loss", type=int, default=1,
                        help="Loss computation type: 1 = global loss, 0 = per-dimension loss")

    # ---------------- Logging ----------------
    parser.add_argument("--logmode", type=str, default="a", help="File mode of logging (a/w).")
    parser.add_argument("--save", type=str, default="experiments/", help="Path for save checkpoints")
    parser.add_argument("--load", type=str, default=None, help="Experiment ID to load; if None, create new.")
    parser.add_argument("--gpu", type=str, default="0", help="which gpu to use.")
    parser.add_argument("--data_sanity_batches", type=int, default=3,
                        help="batches to sample per split for sanity logs.")

    # ---------------- Quantization ----------------
    parser.add_argument("--quantization", type=float, default=0.0,
                        help="Quantization on the physionet dataset.")

    # ---------------- Error / EXACT ----------------
    parser.add_argument("--topk_err", type=int, default=10,
                        help="Top-K largest abs errors to log on TEST.")

    parser.add_argument("--exact_abs_tol", type=float, default=1e-5,
                        help="Absolute tolerance for EXACT points.")
    parser.add_argument("--exact_rel_tol", type=float, default=0.05,
                        help="Relative tolerance for EXACT points.")

    return parser