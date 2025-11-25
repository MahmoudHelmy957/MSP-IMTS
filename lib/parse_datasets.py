import os
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import re

import lib.utils as utils
from torch.distributions import uniform

from torch.utils.data import DataLoader
from lib.physionet import *
from lib.ushcn import *
from lib.mimic import MIMIC
from lib.person_activity import *
from sklearn import model_selection


#####################################################################################################
def parse_datasets(args, patch_ts=False, length_stat=False):

	device = args.device
	dataset_name = args.dataset
	

	##################################################################
	### PhysioNet dataset ### 
	### MIMIC dataset ###
	##################################################################
	### PhysioNet dataset ### 
	### MIMIC dataset ###
	if dataset_name in ["physionet", "mimic"]:

		### list of tuples (record_id, tt, vals, mask) ###
		if dataset_name == "physionet":
			total_dataset = PhysioNet('../data/physionet', quantization = args.quantization,
											download=False, n_samples = args.n, device = device)
		elif dataset_name == "mimic":
			total_dataset = MIMIC('../data/mimic/', n_samples = args.n, device = device)

		# Shuffle and split
		seen_data, test_data = model_selection.train_test_split(total_dataset, train_size= 0.8, random_state = 42, shuffle = True)
		train_data, val_data = model_selection.train_test_split(seen_data, train_size= 0.75, random_state = 42, shuffle = False)
		print("Dataset n_samples:", len(total_dataset), len(train_data), len(val_data), len(test_data))
		# --- Sanity: make sure splits are disjoint ---
		train_ids = {rid for rid, _, _, _ in train_data}
		val_ids   = {rid for rid, _, _, _ in val_data}
		test_ids  = {rid for rid, _, _, _ in test_data}
		print("overlap train∩val:", len(train_ids & val_ids),
			"train∩test:", len(train_ids & test_ids),
			"val∩test:", len(val_ids & test_ids))
		# ---------------------------------------------
		test_record_ids = [record_id for record_id, tt, vals, mask in test_data]
		print("Test record ids (first 20):", test_record_ids[:20])
		print("Test record ids (last 20):", test_record_ids[-20:])

		record_id, tt, vals, mask = train_data[0]

		input_dim = vals.size(-1)

		batch_size = min(min(len(seen_data), args.batch_size), args.n)
		data_min, data_max, time_max = get_data_min_max(seen_data, device) # (n_dim,), (n_dim,)

		# ------------------------ CHANGED BLOCK START ------------------------
		# Choose collate (multi-scale vs single-scale)
		use_ms = hasattr(args, "multi_scales") and args.multi_scales not in (None, "", [])
		if use_ms:
			from lib.physionet import patch_variable_time_collate_fn_ms
			import re
			# parse --multi_scales / --multi_strides as hours
			scales_hours = [float(x) for x in re.split(r"[,\s]+", args.multi_scales.strip()) if x]
			if getattr(args, "multi_strides", None) in (None, "", []):
				strides_hours = scales_hours[:]
			else:
				strides_hours = [float(x) for x in re.split(r"[,\s]+", args.multi_strides.strip()) if x]
				assert len(scales_hours) == len(strides_hours), "multi_scales and multi_strides length mismatch"

			# bind args & stats so the collate only needs (batch)
			collate_fn_ms = lambda batch: patch_variable_time_collate_fn_ms(
				batch, args, device=device,
				data_min=data_min, data_max=data_max, time_max=time_max,
				scales_hours=scales_hours, strides_hours=strides_hours, history_hours=float(args.history)
			)

			train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn_ms)
			val_dataloader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn_ms)
			test_dataloader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn_ms)
			# --- Sanity: pull one val batch and check masks ---
			try:
				_one = next(iter(val_dataloader))
				# Expect lists of length K
				print("[MS] batch keys:", list(_one.keys()))
				print("[MS] #scales:", len(_one["X_list"]))
				print("[MS] per-scale shapes:",
					[tuple(x.shape) for x in _one["X_list"]])  # (B, M_k, L, N) each
				print("[MS] data_to_predict:", tuple(_one["data_to_predict"].shape))
				print("[MS] mask_predicted_data sum:",
					_one["mask_predicted_data"].sum().item())
				print("[MS] tt_list mins/maxes:",
          			[(float(t.min()), float(t.max())) for t in _one["tt_list"]])
				print("[MS] tp_to_predict min/max:",
          			float(_one["tp_to_predict"].min()), float(_one["tp_to_predict"].max()))	
			except Exception as e:
				print("[MS] sanity batch failed:", repr(e))
			# -------------------------------------------------

		else:
			# original single-scale path
			if(patch_ts):
				collate_fn = patch_variable_time_collate_fn
			else:
				collate_fn = variable_time_collate_fn

			train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=True, 
				collate_fn= lambda batch: collate_fn(batch, args, device, data_type = "train",
					data_min = data_min, data_max = data_max, time_max = time_max))
			val_dataloader = DataLoader(val_data, batch_size= batch_size, shuffle=False, 
				collate_fn= lambda batch: collate_fn(batch, args, device, data_type = "val",
					data_min = data_min, data_max = data_max, time_max = time_max))
			test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle=False, 
				collate_fn= lambda batch: collate_fn(batch, args, device, data_type = "test",
					data_min = data_min, data_max = data_max, time_max = time_max))
			# --- Sanity: pull one val batch and check masks ---
			try:
				_one = next(iter(val_dataloader))
				print("[SS] observed_data:", tuple(_one["observed_data"].shape))
				print("[SS] data_to_predict:", tuple(_one["data_to_predict"].shape))
				print("[SS] mask_predicted_data sum:",
					_one["mask_predicted_data"].sum().item())
			except Exception as e:
				print("[SS] sanity batch failed:", repr(e))
			# -------------------------------------------------

		# ------------------------- CHANGED BLOCK END -------------------------

		data_objects = {
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"val_dataloader": utils.inf_generator(val_dataloader),
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_val_batches": len(val_dataloader),
					"n_test_batches": len(test_dataloader),
					# "attr": total_dataset.params, #optional
					"data_max": data_max, #optional
					"data_min": data_min,
					"time_max": time_max
					} #optional

		if(length_stat):
			max_input_len, max_pred_len, median_len = get_seq_length(args, total_dataset)
			data_objects["max_input_len"] = max_input_len.item()
			data_objects["max_pred_len"] = max_pred_len.item()
			data_objects["median_len"] = median_len.item()
			print(data_objects["max_input_len"], data_objects["max_pred_len"], data_objects["median_len"])

		return data_objects

	##################################################################

	# if dataset_name in ["physionet", "mimic"]:

	# 	### list of tuples (record_id, tt, vals, mask) ###
	# 	if dataset_name == "physionet":
	# 		total_dataset = PhysioNet('../data/physionet', quantization = args.quantization,
	# 										download=False, n_samples = args.n, device = device)
	# 	elif dataset_name == "mimic":
	# 		total_dataset = MIMIC('../data/mimic/', n_samples = args.n, device = device)

	# 	# Shuffle and split
	# 	seen_data, test_data = model_selection.train_test_split(total_dataset, train_size= 0.8, random_state = 42, shuffle = True)
	# 	train_data, val_data = model_selection.train_test_split(seen_data, train_size= 0.75, random_state = 42, shuffle = False)
	# 	print("Dataset n_samples:", len(total_dataset), len(train_data), len(val_data), len(test_data))
	# 	test_record_ids = [record_id for record_id, tt, vals, mask in test_data]
	# 	print("Test record ids (first 20):", test_record_ids[:20])
	# 	print("Test record ids (last 20):", test_record_ids[-20:])

	# 	record_id, tt, vals, mask = train_data[0]

	# 	input_dim = vals.size(-1)

	# 	batch_size = min(min(len(seen_data), args.batch_size), args.n)
	# 	data_min, data_max, time_max = get_data_min_max(seen_data, device) # (n_dim,), (n_dim,)

	# 	if(patch_ts):
	# 		collate_fn = patch_variable_time_collate_fn
	# 	else:
	# 		collate_fn = variable_time_collate_fn

	# 	train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=True, 
	# 		collate_fn= lambda batch: collate_fn(batch, args, device, data_type = "train",
	# 			data_min = data_min, data_max = data_max, time_max = time_max))
	# 	val_dataloader = DataLoader(val_data, batch_size= batch_size, shuffle=False, 
	# 		collate_fn= lambda batch: collate_fn(batch, args, device, data_type = "val",
	# 			data_min = data_min, data_max = data_max, time_max = time_max))
	# 	test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle=False, 
	# 		collate_fn= lambda batch: collate_fn(batch, args, device, data_type = "test",
	# 			data_min = data_min, data_max = data_max, time_max = time_max))

	# 	data_objects = {
	# 				"train_dataloader": utils.inf_generator(train_dataloader), 
	# 				"val_dataloader": utils.inf_generator(val_dataloader),
	# 				"test_dataloader": utils.inf_generator(test_dataloader),
	# 				"input_dim": input_dim,
	# 				"n_train_batches": len(train_dataloader),
	# 				"n_val_batches": len(val_dataloader),
	# 				"n_test_batches": len(test_dataloader),
	# 				# "attr": total_dataset.params, #optional
	# 				"data_max": data_max, #optional
	# 				"data_min": data_min,
	# 				"time_max": time_max
	# 				} #optional

	# 	if(length_stat):
	# 		max_input_len, max_pred_len, median_len = get_seq_length(args, total_dataset)
	# 		data_objects["max_input_len"] = max_input_len.item()
	# 		data_objects["max_pred_len"] = max_pred_len.item()
	# 		data_objects["median_len"] = median_len.item()
	# 		print(data_objects["max_input_len"], data_objects["max_pred_len"], data_objects["median_len"])

	# 	return data_objects

	##################################################################
	### USHCN dataset ###
	elif dataset_name == "ushcn":
		# Paper setup
		args.n_months = 48       # 48 months in the raw series
		args.pred_window = 1     # predict 1 month ahead

		# Load
		total_dataset = USHCN('../data/ushcn/', n_samples=args.n, device=device)

		# Split
		seen_data, test_data = model_selection.train_test_split(
			total_dataset, train_size=0.8, random_state=42, shuffle=True
		)
		train_data, val_data = model_selection.train_test_split(
			seen_data, train_size=0.75, random_state=42, shuffle=False
		)
		print("Dataset n_samples:", len(total_dataset), len(train_data), len(val_data), len(test_data))
		test_record_ids = [record_id for record_id, tt, vals, mask in test_data]
		print("Test record ids (first 20):", test_record_ids[:20])
		print("Test record ids (last 20):", test_record_ids[-20:])

		# Stats
		record_id, tt, vals, mask = train_data[0]
		input_dim = vals.size(-1)
		data_min, data_max, time_max = get_data_min_max(seen_data, device)  # (n_dim,), (n_dim,)

		# Pre-slice into rolling windows based on args.history & pred_window
		train_data = USHCN_time_chunk(train_data, args, device)
		val_data   = USHCN_time_chunk(val_data,   args, device)
		test_data  = USHCN_time_chunk(test_data,  args, device)
		batch_size = args.batch_size
		print("Dataset n_samples after time split:", len(train_data)+len(val_data)+len(test_data),
			len(train_data), len(val_data), len(test_data))

		# ---- Single-scale vs Multi-scale switch ----
		use_ms = hasattr(args, "multi_scales") and args.multi_scales not in (None, "", [])

		if use_ms:
			# inside the USHCN dataset block, MS path
			from lib.physionet import patch_variable_time_collate_fn_ms
			import re

			scales = [float(x) for x in re.split(r"[,\s]+", args.multi_scales.strip()) if x]
			if getattr(args, "multi_strides", None) in (None, "", []):
				strides = scales[:]
			else:
				strides = [float(x) for x in re.split(r"[,\s]+", args.multi_strides.strip()) if x]
				assert len(scales) == len(strides), "multi_scales and multi_strides length mismatch"

			def collate_fn_ms(batch):
				# Convert (rid, tt_rel, vals, mask, t_bias) -> absolute months (0..48)
				batch4 = []
				for rid, tt_rel, vals, mask, t_bias in batch:
					tt_abs = tt_rel + t_bias              # absolute months
					batch4.append((rid, tt_abs, vals, mask))

				# IMPORTANT:
				#  - time_max must match the baseline (48.0), so the model sees the *same* time range as SS.
				#  - history_hours still drives MS windowing inside the collate.
				return patch_variable_time_collate_fn_ms(
					batch4, args, device=device,
					data_min=data_min, data_max=data_max,
					time_max=time_max,                   # <- keep baseline 48.0 here
					scales_hours=scales, strides_hours=strides,
					history_hours=float(args.history)    # <- used only to define MS bins
				)



			train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn_ms)
			val_dataloader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn_ms)
			test_dataloader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn_ms)


			# Optional sanity pull
			try:
				_one = next(iter(val_dataloader))
				print("[USHCN-MS] batch keys:", list(_one.keys()))
				print("[USHCN-MS] #scales:", len(_one["X_list"]))
				print("[USHCN-MS] per-scale shapes:", [tuple(x.shape) for x in _one["X_list"]])  # (B, M_k, L, N)
				print("[USHCN-MS] data_to_predict:", tuple(_one["data_to_predict"].shape))
				print("[USHCN-MS] mask_predicted_data sum:", _one["mask_predicted_data"].sum().item())
				print("[USHCN-MS] tt_list mins/maxes:",
          [(float(t.min()), float(t.max())) for t in _one["tt_list"]])
				print("[USHCN-MS] tp_to_predict min/max:",
          float(_one["tp_to_predict"].min()), float(_one["tp_to_predict"].max()))
			except Exception as e:
				print("[USHCN-MS] sanity batch failed:", repr(e))

		else:
			# Original single-scale path (paper baseline)
			if patch_ts:
				collate_fn = USHCN_patch_variable_time_collate_fn
			else:
				collate_fn = USHCN_variable_time_collate_fn

			train_dataloader = DataLoader(
				train_data, batch_size=batch_size, shuffle=True,
				collate_fn=lambda batch: collate_fn(batch, args, device, time_max=time_max)
			)
			val_dataloader = DataLoader(
				val_data, batch_size=batch_size, shuffle=False,
				collate_fn=lambda batch: collate_fn(batch, args, device, time_max=time_max)
			)
			test_dataloader = DataLoader(
				test_data, batch_size=batch_size, shuffle=False,
				collate_fn=lambda batch: collate_fn(batch, args, device, time_max=time_max)
			)

			# Optional sanity pull
			try:
				_one = next(iter(val_dataloader))
				print("[USHCN-SS] observed_data:", tuple(_one["observed_data"].shape))
				print("[USHCN-SS] data_to_predict:", tuple(_one["data_to_predict"].shape))
				print("[USHCN-SS] mask_predicted_data sum:", _one["mask_predicted_data"].sum().item())
			except Exception as e:
				print("[USHCN-SS] sanity batch failed:", repr(e))

		data_objects = {
			"train_dataloader": utils.inf_generator(train_dataloader),
			"val_dataloader":   utils.inf_generator(val_dataloader),
			"test_dataloader":  utils.inf_generator(test_dataloader),
			"input_dim": input_dim,
			"n_train_batches": len(train_dataloader),
			"n_val_batches":   len(val_dataloader),
			"n_test_batches":  len(test_dataloader),
			"data_max": data_max,
			"data_min": data_min,
			"time_max": time_max
		}
		if length_stat:
			max_input_len, max_pred_len, median_len = USHCN_get_seq_length(args, train_data+val_data+test_data)
			data_objects["max_input_len"]  = max_input_len.item()
			data_objects["max_pred_len"]   = max_pred_len.item()
			data_objects["median_len"]     = median_len.item()
			print(data_objects["max_input_len"], data_objects["max_pred_len"], data_objects["median_len"])

		return data_objects

		

	##################################################################
    ### Activity dataset ###
	elif dataset_name == "activity":
		# 1. Configuration
		# Determine the window size (chunk size) in milliseconds
		if not hasattr(args, 'pred_window') or args.pred_window is None:
			args.pred_window = 1000  # Default 1000ms

		# 2. Load Raw Data
		total_dataset = PersonActivity('../data/activity/', n_samples=args.n,
									download=True, device=device)

		# 3. Split (Record-based to prevent leakage)
		seen_data, test_data = model_selection.train_test_split(
			total_dataset, train_size=0.8, random_state=42, shuffle=True
		)
		train_data, val_data = model_selection.train_test_split(
			seen_data, train_size=0.75, random_state=42, shuffle=False
		)
		print("Dataset n_samples (Raw):", len(total_dataset), len(train_data), len(val_data), len(test_data))

		# 4. Chunking (Convert long streams to 'Patient' windows)
		# We chunk BEFORE calculating stats to ensure stats match the actual model input
		print(f"Chunking activity data with window: {args.pred_window}...")
		train_data = Activity_time_chunk(train_data, args, device)
		val_data   = Activity_time_chunk(val_data,   args, device)
		test_data  = Activity_time_chunk(test_data,  args, device)
		print("Dataset n_samples (Chunked):", len(train_data), len(val_data), len(test_data))

		# 5. Statistics Calculation (Global Scalar Strategy)
		# CRITICAL: We use global scalar min/max to preserve relative magnitude between XYZ axes.
		def get_global_stats(dataset):
			all_vals = []
			all_tt = []
			for _, tt, vals, _ in dataset:
				all_vals.append(vals)
				all_tt.append(tt)
				
			# Concatenate all chunks
			all_vals = torch.cat(all_vals, dim=0)
			all_tt = torch.cat(all_tt, dim=0)
				
			# SCALAR min/max (vs vector min/max used in PhysioNet)
			return torch.min(all_vals), torch.max(all_vals), torch.max(all_tt)

		# Compute on Train+Val only
		data_min, data_max, time_max = get_global_stats(train_data + val_data)
			
		# Get input dim from first sample
		_, _, vals, _ = train_data[0]
		input_dim = vals.size(-1)

		print(f"Stats -> Input Dim: {input_dim}, Time Max: {time_max.item()}")
		print(f"Global Data Min: {data_min}, Global Data Max: {data_max}")

		# 6. Data Loaders
		batch_size = args.batch_size
		use_ms = hasattr(args, "multi_scales") and args.multi_scales not in (None, "", [])

		if use_ms:
			from lib.physionet import patch_variable_time_collate_fn_ms
			import re

			# Parse scales (assuming args provided are compatible with 'ms' units)
			scales = [float(x) for x in re.split(r"[,\s]+", args.multi_scales.strip()) if x]
			if getattr(args, "multi_strides", None) in (None, "", []):
				strides = scales[:]
			else:
				strides = [float(x) for x in re.split(r"[,\s]+", args.multi_strides.strip()) if x]

			# History is the full duration of the chunk
			history_window = float(time_max.item())
				
			print(f"[ACT-MS] Configuration: Scales={scales}, Strides={strides}, History={history_window}")

			collate_fn_ms = lambda batch: patch_variable_time_collate_fn_ms(
				batch, args, device=device,
				data_min=data_min, data_max=data_max, time_max=time_max,
				scales_hours=scales, strides_hours=strides, history_hours=history_window
			)

			train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn_ms)
			val_dataloader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn_ms)
			test_dataloader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn_ms)
			
		else:
			# Single Scale (Standard)
			if patch_ts:
				collate_fn = patch_variable_time_collate_fn
			else:
				collate_fn = variable_time_collate_fn

			train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, 
				collate_fn=lambda batch: collate_fn(batch, args, device, data_type="train",
					data_min=data_min, data_max=data_max, time_max=time_max))
			val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, 
				collate_fn=lambda batch: collate_fn(batch, args, device, data_type="val",
					data_min=data_min, data_max=data_max, time_max=time_max))
			test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, 
				collate_fn=lambda batch: collate_fn(batch, args, device, data_type="test",
					data_min=data_min, data_max=data_max, time_max=time_max))

		data_objects = {
			"train_dataloader": utils.inf_generator(train_dataloader), 
			"val_dataloader": utils.inf_generator(val_dataloader),
			"test_dataloader": utils.inf_generator(test_dataloader),
			"input_dim": input_dim,
			"n_train_batches": len(train_dataloader),
			"n_val_batches": len(val_dataloader),
			"n_test_batches": len(test_dataloader),
			"data_max": data_max,
			"data_min": data_min,
			"time_max": time_max
		}
		return data_objects