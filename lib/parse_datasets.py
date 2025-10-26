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
		args.n_months = 48 # 48 monthes
		args.pred_window = 1 # predict future one month

		### list of tuples (record_id, tt, vals, mask) ###
		total_dataset = USHCN('../data/ushcn/', n_samples = args.n, device = device)

		# Shuffle and split
		seen_data, test_data = model_selection.train_test_split(total_dataset, train_size= 0.8, random_state = 42, shuffle = True)
		train_data, val_data = model_selection.train_test_split(seen_data, train_size= 0.75, random_state = 42, shuffle = False)
		print("Dataset n_samples:", len(total_dataset), len(train_data), len(val_data), len(test_data))
		test_record_ids = [record_id for record_id, tt, vals, mask in test_data]
		print("Test record ids (first 20):", test_record_ids[:20])
		print("Test record ids (last 20):", test_record_ids[-20:])

		record_id, tt, vals, mask = train_data[0]

		input_dim = vals.size(-1)

		data_min, data_max, time_max = get_data_min_max(seen_data, device) # (n_dim,), (n_dim,)

		if(patch_ts):
			collate_fn = USHCN_patch_variable_time_collate_fn
		else:
			collate_fn = USHCN_variable_time_collate_fn

		train_data = USHCN_time_chunk(train_data, args, device)
		val_data = USHCN_time_chunk(val_data, args, device)
		test_data = USHCN_time_chunk(test_data, args, device)
		batch_size = args.batch_size
		print("Dataset n_samples after time split:", len(train_data)+len(val_data)+len(test_data),\
			len(train_data), len(val_data), len(test_data))
		train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=True, 
			collate_fn= lambda batch: collate_fn(batch, args, device, time_max = time_max))
		val_dataloader = DataLoader(val_data, batch_size= batch_size, shuffle=False, 
			collate_fn= lambda batch: collate_fn(batch, args, device, time_max = time_max))
		test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle=False, 
			collate_fn= lambda batch: collate_fn(batch, args, device, time_max = time_max))

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
			max_input_len, max_pred_len, median_len = USHCN_get_seq_length(args, train_data+val_data+test_data)
			data_objects["max_input_len"] = max_input_len.item()
			data_objects["max_pred_len"] = max_pred_len.item()
			data_objects["median_len"] = median_len.item()
			# data_objects["batch_size"] = args.batch_size * (args.n_months - args.pred_window + 1 - args.history)
			print(data_objects["max_input_len"], data_objects["max_pred_len"], data_objects["median_len"])

		return data_objects
		

	##################################################################
	### Activity dataset ###
	elif dataset_name == "activity":
        # Units here are milliseconds
        args.pred_window = 1000  # predict future 1000 ms

        total_dataset = PersonActivity('../data/activity/', n_samples=args.n, download=True, device=device)

        # Shuffle and split
        seen_data, test_data = model_selection.train_test_split(total_dataset, train_size=0.8, random_state=42, shuffle=True)
        train_data, val_data = model_selection.train_test_split(seen_data, train_size=0.75, random_state=42, shuffle=False)
        print("Dataset n_samples:", len(total_dataset), len(train_data), len(val_data), len(test_data))
        test_record_ids = [record_id for record_id, tt, vals, mask in test_data]
        print("Test record ids (first 20):", test_record_ids[:20])
        print("Test record ids (last 20):", test_record_ids[-20:])

        record_id, tt, vals, mask = train_data[0]
        input_dim = vals.size(-1)

        # stats
        data_min, data_max, _ = get_data_min_max(seen_data, device)
        time_max = torch.tensor(args.history + args.pred_window, device=device)  # ms
        print('manual set time_max:', time_max)

        # --- Multi-scale vs single-scale switch ---
        import re
        use_ms = hasattr(args, "multi_scales") and args.multi_scales not in (None, "", [])

        if use_ms:
            # KEEP time chunking for Activity even in MS mode
            train_data = Activity_time_chunk(train_data, args, device)
            val_data   = Activity_time_chunk(val_data,   args, device)
            test_data  = Activity_time_chunk(test_data,  args, device)

            # Reuse generic MS collate (unit-agnostic). Param names say "hours", but here they are ms.
            from lib.physionet import patch_variable_time_collate_fn_ms

            scales_ms = [float(x) for x in re.split(r"[,\s]+", args.multi_scales.strip()) if x]
            if getattr(args, "multi_strides", None) in (None, "", []):
                strides_ms = scales_ms[:]
            else:
                strides_ms = [float(x) for x in re.split(r"[,\s]+", args.multi_strides.strip()) if x]
                assert len(scales_ms) == len(strides_ms), "multi_scales and multi_strides length mismatch"

            collate_fn_ms = lambda batch: patch_variable_time_collate_fn_ms(
                batch, args, device=device,
                data_min=data_min, data_max=data_max, time_max=time_max,
                # names say "*hours*" but we pass milliseconds; the function is unit-agnostic
                scales_hours=scales_ms, strides_hours=strides_ms, history_hours=float(args.history)
            )

            batch_size = args.batch_size
            train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn_ms)
            val_dataloader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn_ms)
            test_dataloader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn_ms)

            # (optional) quick sanity pull
            try:
                _one = next(iter(val_dataloader))
                print("[MS] batch keys:", list(_one.keys()))
                print("[MS] #scales:", len(_one["X_list"]))
                print("[MS] per-scale shapes:", [tuple(x.shape) for x in _one["X_list"]])
                print("[MS] data_to_predict:", tuple(_one["data_to_predict"].shape))
                print("[MS] mask_predicted_data sum:", _one["mask_predicted_data"].sum().item())
            except Exception as e:
                print("[MS] sanity batch failed:", repr(e))

        else:
            # Original single-scale pipeline (pre-slice)
            if patch_ts:
                collate_fn = patch_variable_time_collate_fn
            else:
                collate_fn = variable_time_collate_fn

            train_data = Activity_time_chunk(train_data, args, device)
            val_data   = Activity_time_chunk(val_data,   args, device)
            test_data  = Activity_time_chunk(test_data,  args, device)

            batch_size = args.batch_size
            print("Dataset n_samples after time split:", len(train_data)+len(val_data)+len(test_data),
                  len(train_data), len(val_data), len(test_data))

            train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                collate_fn=lambda batch: collate_fn(batch, args, device, data_type="train",
                    data_min=data_min, data_max=data_max, time_max=time_max))
            val_dataloader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False,
                collate_fn=lambda batch: collate_fn(batch, args, device, data_type="val",
                    data_min=data_min, data_max=data_max, time_max=time_max))
            test_dataloader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False,
                collate_fn=lambda batch: collate_fn(batch, args, device, data_type="test",
                    data_min=data_min, data_max=data_max, time_max=time_max))

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
        return data_objects


	
