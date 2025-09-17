import os
import sys
sys.path.append("..")

import time
import datetime
import argparse
import numpy as np
import pandas as pd
import random
from random import SystemRandom
from sklearn import model_selection
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim

import lib.utils as utils
from lib.DataPreprocessing import parse_datasets
from Models.tPatchGNN import *
from Models.MultiScalePatching import *

parser = argparse.ArgumentParser('IMTS Forecasting')

############################# multi scale ########################
parser.add_argument('--multi_scales', type=str, default='', help='Comma list of patch sizes in hours, e.g. "2,8,24". Empty = single-scale.')
parser.add_argument('--multi_strides', type=str, default='', help='Comma list of strides in hours. Empty = same as multi_scales.')
parser.add_argument('--fusion', type=str, default='concat', choices=['concat','scale_attn'], help='Fusion method for multi-scale.')
################################################

parser.add_argument('--state', type=str, default='def')
parser.add_argument('-n',  type=int, default=int(1e8), help="Size of the dataset")
parser.add_argument('--hop', type=int, default=1, help="hops in GNN")
parser.add_argument('--nhead', type=int, default=1, help="heads in Transformer")
parser.add_argument('--tf_layer', type=int, default=1, help="# of layer in Transformer")
parser.add_argument('--nlayer', type=int, default=1, help="# of layer in TSmodel")
parser.add_argument('--epoch', type=int, default=1000, help="training epoches")
parser.add_argument('--patience', type=int, default=10, help="patience for early stop")
parser.add_argument('--history', type=int, default=24, help="number of hours (months for ushcn and ms for activity) as historical window")
parser.add_argument('-ps', '--patch_size', type=float, default=24, help="window size for a patch")
parser.add_argument('--stride', type=float, default=24, help="period stride for patch sliding")
parser.add_argument('--logmode', type=str, default="a", help='File mode of logging.')

parser.add_argument('--lr',  type=float, default=1e-3, help="Starting learning rate.")
parser.add_argument('--w_decay', type=float, default=0.0, help="weight decay.")
parser.add_argument('-b', '--batch_size', type=int, default=32)

parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('--seed', type=int, default=1, help="Random seed")
parser.add_argument('--dataset', type=str, default='physionet', help="Dataset to load. Available: physionet, mimic, ushcn")

# value 0 means using original time granularity, Value 1 means quantization by 1 hour, 
# value 0.1 means quantization by 0.1 hour = 6 min, value 0.016 means quantization by 0.016 hour = 1 min
parser.add_argument('--quantization', type=float, default=0.0, help="Quantization on the physionet dataset.")
parser.add_argument('--model', type=str, default='tPatchGNN', help="Model name")
parser.add_argument('--outlayer', type=str, default='Linear', help="Model name")
parser.add_argument('-hd', '--hid_dim', type=int, default=64, help="Number of units per hidden layer")
parser.add_argument('-td', '--te_dim', type=int, default=10, help="Number of units for time encoding")
parser.add_argument('-nd', '--node_dim', type=int, default=10, help="Number of units for node vectors")
parser.add_argument('--gpu', type=str, default='0', help='which gpu to use.')


args = parser.parse_args()
args.npatch = int(np.ceil((args.history - args.patch_size) / args.stride)) + 1 # (window size for a patch)




os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
file_name = os.path.basename(__file__)[:-3]
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.PID = os.getpid()
print("PID, device:", args.PID, args.device)

#####################################################################################################

if __name__ == '__main__':
	utils.setup_seed(args.seed)

	experimentID = args.load
	if experimentID is None:
		# Make a new experiment ID
		experimentID = int(SystemRandom().random()*100000)
	ckpt_path = os.path.join(args.save, "experiment_" + str(experimentID) + '.ckpt')
	
	input_command = sys.argv
	ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
	if len(ind) == 1:
		ind = ind[0]
		input_command = input_command[:ind] + input_command[(ind+2):]
	input_command = " ".join(input_command)

	# utils.makedirs("results/")

	##################################################################
	data_obj = parse_datasets(args, patch_ts=True)
	input_dim = data_obj["input_dim"]
	
	### Model setting ###
	# args.ndim = input_dim
	# model = tPatchGNN(args).to(args.device)

	### multi scale ###
	
	use_ms = (args.multi_scales not in (None, "", []))
	args.ndim = input_dim

	if use_ms:
		# Prime a batch to get per-scale npatches
		first_batch = utils.get_next_batch(data_obj["train_dataloader"])
		submodels = []
		for M_k in first_batch["npatches"]:
			sub_args = deepcopy(args)
			sub_args.npatch = int(M_k)               # Linear(hid_dim * M_k -> hid_dim)
			submodels.append(tPatchGNN(sub_args, supports=None, dropout=0).to(args.device))
		model = MultiScalePatching(
			submodels=submodels,
			te_dim=args.te_dim,
			proj_dim=args.hid_dim,                   # project concat back to hid_dim
			fusion=args.fusion
		).to(args.device)
	else:
		model = tPatchGNN(args).to(args.device)

	##################################################################
	
	# # Load checkpoint and evaluate the model
	# if args.load is not None:
	# 	utils.get_ckpt_model(ckpt_path, model, args.device)
	# 	exit()

	##################################################################

	if(args.n < 12000):
		args.state = "debug"
		log_path = "logs/{}_{}_{}.log".format(args.dataset, args.model, args.state)
	else:
		log_path = "logs/{}_{}_{}_{}patch_{}stride_{}layer_{}lr.log". \
			format(args.dataset, args.model, args.state, args.patch_size, args.stride, args.nlayer, args.lr)
	
	if not os.path.exists("logs/"):
		utils.makedirs("logs/")
	logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__), mode=args.logmode)
	logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
	logger.info(input_command)
	logger.info(args)


	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	num_batches = data_obj["n_train_batches"]
	print("n_train_batches:", num_batches)

	def _masked_metrics(pred, tgt, msk):
		diff = (pred - tgt)[msk.bool()]
		mse = (diff ** 2).mean().item()
		mae = diff.abs().mean().item()
		rmse = np.sqrt(mse)
		tgt_safe = tgt[msk.bool()].abs()
		mape = float(torch.mean((diff.abs() / torch.clamp(tgt_safe, min=1e-8))).item())
		return dict(loss=mse, mse=mse, rmse=rmse, mae=mae, mape=mape)

	best_val_mse = np.inf
	test_res = None
	best_iter = 0

	for itr in range(args.epoch):
		st = time.time()
		model.train()
		for _ in range(num_batches):
			optimizer.zero_grad()
			batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
			if use_ms:
				out = model(batch_dict["X_list"], batch_dict["tt_list"], batch_dict["mk_list"], batch_dict["tp_to_predict"])
				pred = out[0]  # (B, Lp, N)
				tgt  = batch_dict["data_to_predict"]
				msk  = batch_dict["mask_predicted_data"]
				loss = ((pred - tgt)[msk.bool()] ** 2).mean()
				loss.backward()
				optimizer.step()
				train_res = {"loss": loss.detach()}
			else:
				train_res = compute_all_losses(model, batch_dict)   # original path
				train_res["loss"].backward()
				optimizer.step()

		model.eval()
		with torch.no_grad():
			if use_ms:
				# VAL
				val_logs = []
				for _ in range(data_obj["n_val_batches"]):
					b = utils.get_next_batch(data_obj["val_dataloader"])
					out = model(b["X_list"], b["tt_list"], b["mk_list"], b["tp_to_predict"])
					pred = out[0]; tgt = b["data_to_predict"]; msk = b["mask_predicted_data"]
					val_logs.append(_masked_metrics(pred, tgt, msk))
				val_res = {k: float(np.mean([d[k] for d in val_logs])) for k in val_logs[0].keys()}

				if val_res["mse"] < best_val_mse:
					best_val_mse = val_res["mse"]; best_iter = itr
					test_logs = []
					for _ in range(data_obj["n_test_batches"]):
						b = utils.get_next_batch(data_obj["test_dataloader"])
						out = model(b["X_list"], b["tt_list"], b["mk_list"], b["tp_to_predict"])
						pred = out[0]; tgt = b["data_to_predict"]; msk = b["mask_predicted_data"]
						test_logs.append(_masked_metrics(pred, tgt, msk))
					test_res = {k: float(np.mean([d[k] for d in test_logs])) for k in test_logs[0].keys()}
			else:
				val_res  = evaluation(model, data_obj["val_dataloader"],  data_obj["n_val_batches"])
				if val_res["mse"] < best_val_mse:
					best_val_mse = val_res["mse"]; best_iter = itr
					test_res = evaluation(model, data_obj["test_dataloader"], data_obj["n_test_batches"])

			logger.info('- Epoch {:03d}, ExpID {}'.format(itr, experimentID))
			logger.info("Train - Loss (one batch): {:.5f}".format(
				train_res["loss"].item() if isinstance(train_res["loss"], torch.Tensor) else train_res["loss"]))
			logger.info("Val - Loss, MSE, RMSE, MAE, MAPE: {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.2f}%"
				.format(val_res["loss"], val_res["mse"], val_res["rmse"], val_res["mae"], val_res["mape"]*100))
			if test_res is not None:
				logger.info("Test - Best epoch, Loss, MSE, RMSE, MAE, MAPE: {}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.2f}%"
					.format(best_iter, test_res["loss"], test_res["mse"], test_res["rmse"], test_res["mae"], test_res["mape"]*100))
			logger.info("Time spent: {:.2f}s".format(time.time()-st))

		if (itr - best_iter) >= args.patience:
			print("Exp has been early stopped!")
			sys.exit(0)




