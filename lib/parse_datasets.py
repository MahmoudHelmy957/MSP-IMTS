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

    # ---------------------------
    # Helper: safe min/max prints
    # ---------------------------
    def _safe_minmax(name, tensor):
        try:
            if tensor is None:
                print(f"[RANGE] {name}: None")
                return
            if not torch.is_tensor(tensor):
                print(f"[RANGE] {name}: not a tensor ({type(tensor)})")
                return
            if tensor.numel() == 0:
                print(f"[RANGE] {name}: empty")
                return
            print(
                f"[RANGE] {name}: min={float(tensor.min()):.6f} max={float(tensor.max()):.6f} "
                f"dtype={tensor.dtype} shape={tuple(tensor.shape)}"
            )
        except Exception as e:
            print(f"[RANGE] {name}: failed ({repr(e)})")

    def _print_batch_ranges(batch, prefix=""):
        # This matches both SS and MS batch dicts you use
        if not isinstance(batch, dict):
            print(prefix + "[RANGE] batch is not dict:", type(batch))
            return

        # Targets
        _safe_minmax(prefix + "data_to_predict", batch.get("data_to_predict", None))
        _safe_minmax(prefix + "mask_predicted_data", batch.get("mask_predicted_data", None))
        _safe_minmax(prefix + "tp_to_predict", batch.get("tp_to_predict", None))

        # SS inputs
        _safe_minmax(prefix + "observed_data", batch.get("observed_data", None))
        _safe_minmax(prefix + "observed_mask", batch.get("observed_mask", None))
        _safe_minmax(prefix + "observed_tp", batch.get("observed_tp", None))

        # MS inputs
        X_list = batch.get("X_list", None)
        tt_list = batch.get("tt_list", None)
        mk_list = batch.get("mk_list", None)
        if isinstance(X_list, list) and len(X_list) > 0:
            for k, x in enumerate(X_list):
                _safe_minmax(prefix + f"X_list[{k}]", x)
        if isinstance(tt_list, list) and len(tt_list) > 0:
            for k, t in enumerate(tt_list):
                _safe_minmax(prefix + f"tt_list[{k}]", t)
        if isinstance(mk_list, list) and len(mk_list) > 0:
            for k, m in enumerate(mk_list):
                _safe_minmax(prefix + f"mk_list[{k}]", m)

    ##################################################################
    ### PhysioNet dataset ###
    ### MIMIC dataset ###
    ##################################################################
    if dataset_name in ["physionet", "mimic"]:

        ### list of tuples (record_id, tt, vals, mask) ###
        if dataset_name == "physionet":
            print("[parse_datasets] ENTERED physionet branch")
            total_dataset = PhysioNet(
                "../dataIR/physionet/",
                quantization=args.quantization,
                download=False,
                n_samples=args.n,
                device=device,
            )
        elif dataset_name == "mimic":
            total_dataset = MIMIC("../dataIR/mimic/", n_samples=args.n, device=device)

        # Shuffle and split
        seen_data, test_data = model_selection.train_test_split(
            total_dataset, train_size=0.8, random_state=42, shuffle=True
        )
        train_data, val_data = model_selection.train_test_split(
            seen_data, train_size=0.75, random_state=42, shuffle=False
        )

        print("Dataset n_samples:", len(total_dataset), len(train_data), len(val_data), len(test_data))

        # --- Sanity: make sure splits are disjoint ---
        train_ids = {rid for rid, _, _, _ in train_data}
        val_ids = {rid for rid, _, _, _ in val_data}
        test_ids = {rid for rid, _, _, _ in test_data}
        print(
            "overlap train∩val:", len(train_ids & val_ids),
            "train∩test:", len(train_ids & test_ids),
            "val∩test:", len(val_ids & test_ids),
        )
        # ---------------------------------------------

        test_record_ids = [record_id for record_id, tt, vals, mask in test_data]
        print("Test record ids (first 20):", test_record_ids[:20])
        print("Test record ids (last 20):", test_record_ids[-20:])

        record_id, tt, vals, mask = train_data[0]
        input_dim = vals.size(-1)

        batch_size = min(min(len(seen_data), args.batch_size), args.n)
        data_min, data_max, time_max = get_data_min_max(seen_data, device)  # (n_dim,), (n_dim,)

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

            # --- Sanity: pull one val batch and check masks + RANGES ---
            try:
                _one = next(iter(val_dataloader))
                print("[MS] batch keys:", list(_one.keys()))
                print("[MS] #scales:", len(_one["X_list"]))
                print("[MS] per-scale shapes:", [tuple(x.shape) for x in _one["X_list"]])  # (B, M_k, L, N) each
                print("[MS] data_to_predict:", tuple(_one["data_to_predict"].shape))
                print("[MS] mask_predicted_data sum:", _one["mask_predicted_data"].sum().item())
                print("[MS] tt_list mins/maxes:", [(float(t.min()), float(t.max())) for t in _one["tt_list"]])
                print("[MS] tp_to_predict min/max:", float(_one["tp_to_predict"].min()), float(_one["tp_to_predict"].max()))
                _print_batch_ranges(_one, prefix="[MS] ")
            except Exception as e:
                print("[MS] sanity batch failed:", repr(e))
            # -------------------------------------------------

        else:
            # original single-scale path
            if patch_ts:
                collate_fn = patch_variable_time_collate_fn
            else:
                collate_fn = variable_time_collate_fn

            train_dataloader = DataLoader(
                train_data, batch_size=batch_size, shuffle=True,
                collate_fn=lambda batch: collate_fn(
                    batch, args, device, data_type="train",
                    data_min=data_min, data_max=data_max, time_max=time_max
                )
            )
            val_dataloader = DataLoader(
                val_data, batch_size=batch_size, shuffle=False,
                collate_fn=lambda batch: collate_fn(
                    batch, args, device, data_type="val",
                    data_min=data_min, data_max=data_max, time_max=time_max
                )
            )
            test_dataloader = DataLoader(
                test_data, batch_size=batch_size, shuffle=False,
                collate_fn=lambda batch: collate_fn(
                    batch, args, device, data_type="test",
                    data_min=data_min, data_max=data_max, time_max=time_max
                )
            )

            # --- Sanity: pull one val batch and check masks + RANGES ---
            try:
                _one = next(iter(val_dataloader))
                print("[SS] observed_data:", tuple(_one["observed_data"].shape))
                print("[SS] data_to_predict:", tuple(_one["data_to_predict"].shape))
                print("[SS] mask_predicted_data sum:", _one["mask_predicted_data"].sum().item())
                _print_batch_ranges(_one, prefix="[SS] ")
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
            "data_max": data_max,
            "data_min": data_min,
            "time_max": time_max,
        }

        if length_stat:
            max_input_len, max_pred_len, median_len = get_seq_length(args, total_dataset)
            data_objects["max_input_len"] = max_input_len.item()
            data_objects["max_pred_len"] = max_pred_len.item()
            data_objects["median_len"] = median_len.item()
            print(data_objects["max_input_len"], data_objects["max_pred_len"], data_objects["median_len"])

        return data_objects

    ##################################################################
    ### USHCN dataset ###
    ##################################################################
    elif dataset_name == "ushcn":
        # Paper setup
        args.n_months = 48       # 48 months in the raw series
        args.pred_window = 1     # predict 1 month ahead

        # Load
        total_dataset = USHCN("../dataIR/ushcn/", n_samples=args.n, device=device)

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
        print(
            "Dataset n_samples after time split:",
            len(train_data) + len(val_data) + len(test_data),
            len(train_data), len(val_data), len(test_data),
        )

        # ---- Single-scale vs Multi-scale switch ----
        use_ms = hasattr(args, "multi_scales") and args.multi_scales not in (None, "", [])

        if use_ms:
            from lib.physionet import patch_variable_time_collate_fn_ms
            import re

            scales = [float(x) for x in re.split(r"[,\s]+", args.multi_scales.strip()) if x]
            if getattr(args, "multi_strides", None) in (None, "", []):
                strides = scales[:]
            else:
                strides = [float(x) for x in re.split(r"[,\s]+", args.multi_strides.strip()) if x]
                assert len(scales) == len(strides), "multi_scales and multi_strides length mismatch"

            def collate_fn_ms(batch):
                # Use local times (tt_rel) for each chunk
                batch4 = []
                local_time_max = 0.0

                for rid, tt_rel, vals, mask, t_bias in batch:
                    batch4.append((rid, tt_rel, vals, mask))
                    if tt_rel.numel() > 0:
                        local_time_max = max(local_time_max, float(tt_rel.max()))

                # Fallback if everything is empty (shouldn't normally happen)
                if local_time_max <= 0.0:
                    local_time_max = float(args.history + getattr(args, "pred_window", 0))

                # IMPORTANT: keep time_max as a tensor (matches original expectations)
                local_time_max_tensor = torch.tensor(
                    local_time_max,
                    device=device,
                    dtype=time_max.dtype,
                )

                return patch_variable_time_collate_fn_ms(
                    batch4, args, device=device,
                    data_min=data_min, data_max=data_max,
                    time_max=local_time_max_tensor,   # local, batch-dependent
                    scales_hours=scales, strides_hours=strides,
                    history_hours=float(args.history),
                )

            train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn_ms)
            val_dataloader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn_ms)
            test_dataloader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn_ms)

            # Optional sanity pull + RANGES
            try:
                _one = next(iter(val_dataloader))
                print("[USHCN-MS] batch keys:", list(_one.keys()))
                print("[USHCN-MS] #scales:", len(_one["X_list"]))
                print("[USHCN-MS] per-scale shapes:", [tuple(x.shape) for x in _one["X_list"]])
                print("[USHCN-MS] data_to_predict:", tuple(_one["data_to_predict"].shape))
                print("[USHCN-MS] mask_predicted_data sum:", _one["mask_predicted_data"].sum().item())
                print("[USHCN-MS] tt_list mins/maxes:", [(float(t.min()), float(t.max())) for t in _one["tt_list"]])
                print("[USHCN-MS] tp_to_predict min/max:", float(_one["tp_to_predict"].min()), float(_one["tp_to_predict"].max()))
                _print_batch_ranges(_one, prefix="[USHCN-MS] ")
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
                collate_fn=lambda batch: collate_fn(batch, args, device, time_max=time_max),
            )
            val_dataloader = DataLoader(
                val_data, batch_size=batch_size, shuffle=False,
                collate_fn=lambda batch: collate_fn(batch, args, device, time_max=time_max),
            )
            test_dataloader = DataLoader(
                test_data, batch_size=batch_size, shuffle=False,
                collate_fn=lambda batch: collate_fn(batch, args, device, time_max=time_max),
            )

            # Optional sanity pull + RANGES
            try:
                _one = next(iter(val_dataloader))
                print("[USHCN-SS] observed_data:", tuple(_one["observed_data"].shape))
                print("[USHCN-SS] data_to_predict:", tuple(_one["data_to_predict"].shape))
                print("[USHCN-SS] mask_predicted_data sum:", _one["mask_predicted_data"].sum().item())
                _print_batch_ranges(_one, prefix="[USHCN-SS] ")
            except Exception as e:
                print("[USHCN-SS] sanity batch failed:", repr(e))

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
            "time_max": time_max,
        }

        if length_stat:
            max_input_len, max_pred_len, median_len = USHCN_get_seq_length(args, train_data + val_data + test_data)
            data_objects["max_input_len"] = max_input_len.item()
            data_objects["max_pred_len"] = max_pred_len.item()
            data_objects["median_len"] = median_len.item()
            print(data_objects["max_input_len"], data_objects["max_pred_len"], data_objects["median_len"])

        return data_objects

    ##################################################################
    ### Activity dataset ###
    ##################################################################
    elif dataset_name == "activity":
        # 0) Normalization flag (default: per-dim)
        # normalization == 1  -> GLOBAL scalar min/max
        # normalization != 1  -> PER-DIM vector min/max (current behavior)
        if not hasattr(args, "normalization") or args.normalization is None:
            args.normalization = 0

        # 1) Configuration
        if not hasattr(args, "pred_window") or args.pred_window is None:
            args.pred_window = 1000  # Default 1000ms

        # 2) Load Raw Data
        total_dataset = PersonActivity(
            "../dataIR/activity/",
            n_samples=args.n,
            download=True,
            device=device,
        )

        # 3) Split (Record-based to prevent leakage)
        seen_data, test_data = model_selection.train_test_split(
            total_dataset, train_size=0.8, random_state=42, shuffle=True
        )
        train_data, val_data = model_selection.train_test_split(
            seen_data, train_size=0.75, random_state=42, shuffle=False
        )
        print(
            "[ACTIVITY][SPLIT] "
            f"raw_total={len(total_dataset)} train={len(train_data)} val={len(val_data)} test={len(test_data)}"
        )

        # 4) Chunking
        print(f"[ACTIVITY][CHUNK] pred_window_ms={args.pred_window}")
        train_data = Activity_time_chunk(train_data, args, device)
        val_data   = Activity_time_chunk(val_data,   args, device)
        test_data  = Activity_time_chunk(test_data,  args, device)
        print(
            "[ACTIVITY][CHUNK] "
            f"chunked_train={len(train_data)} chunked_val={len(val_data)} chunked_test={len(test_data)}"
        )

        # Get input dim from first sample
        _, _, vals0, _ = train_data[0]
        input_dim = vals0.size(-1)

        # 5) Statistics Calculation
        def _fmt_tensor_stats_1d(x: torch.Tensor, k: int = 5) -> str:
            x_cpu = x.detach().cpu()
            return (
                f"min={float(x_cpu.min()):.6f}, "
                f"max={float(x_cpu.max()):.6f}, "
                f"mean={float(x_cpu.mean()):.6f}, "
                f"first{min(k, x_cpu.numel())}="
                f"{[round(v, 6) for v in x_cpu[:k].tolist()]}"
            )

        def _fmt_scalar(x: torch.Tensor) -> str:
            return f"{float(x.detach().cpu().item()):.6f}"

        # ---- PER-DIM stats (vector min/max) ----
        def get_channel_stats(dataset, input_dim: int, device: torch.device):
            dmin = torch.full((input_dim,), float("inf"), device=device)
            dmax = torch.full((input_dim,), float("-inf"), device=device)
            tmax = torch.tensor(float("-inf"), device=device)

            for _, tt, vals, _ in dataset:
                if vals.device != device:
                    vals = vals.to(device)
                if tt.device != device:
                    tt = tt.to(device)

                vmin = vals.min(dim=0).values
                vmax = vals.max(dim=0).values

                dmin = torch.minimum(dmin, vmin)
                dmax = torch.maximum(dmax, vmax)
                tmax = torch.maximum(tmax, tt.max())

            if not torch.isfinite(dmin).all() or not torch.isfinite(dmax).all() or not torch.isfinite(tmax).all():
                raise ValueError("[ACTIVITY][STATS] Non-finite min/max/time_max computed. Check chunking/data.")
            return dmin, dmax, tmax

        # ---- GLOBAL stats (scalar min/max) ----
        def get_global_stats(dataset, device: torch.device):
            gmin = torch.tensor(float("inf"), device=device)
            gmax = torch.tensor(float("-inf"), device=device)
            tmax = torch.tensor(float("-inf"), device=device)

            for _, tt, vals, _ in dataset:
                if vals.device != device:
                    vals = vals.to(device)
                if tt.device != device:
                    tt = tt.to(device)

                gmin = torch.minimum(gmin, vals.min())
                gmax = torch.maximum(gmax, vals.max())
                tmax = torch.maximum(tmax, tt.max())

            if not torch.isfinite(gmin) or not torch.isfinite(gmax) or not torch.isfinite(tmax):
                raise ValueError("[ACTIVITY][STATS] Non-finite global min/max/time_max computed. Check chunking/data.")
            return gmin, gmax, tmax

        # Compute on Train+Val only (avoid test leakage)
        stats_source = train_data + val_data

        if int(args.normalization) == 1:
            # GLOBAL normalization -> scalar min/max, then broadcast to (D,)
            gmin, gmax, time_max = get_global_stats(stats_source, device)
            data_min = gmin.repeat(input_dim)   # shape (D,)
            data_max = gmax.repeat(input_dim)   # shape (D,)

            print("[ACTIVITY][STATS] normalization=GLOBAL (scalar -> broadcast to channels)")
            print("[ACTIVITY][STATS] " f"input_dim={input_dim} | time_max={float(time_max):.6f}")
            print("[ACTIVITY][STATS][GLOBAL_MIN] " + _fmt_scalar(gmin))
            print("[ACTIVITY][STATS][GLOBAL_MAX] " + _fmt_scalar(gmax))
            print("[ACTIVITY][STATS][DATA_MIN_VEC] " + _fmt_tensor_stats_1d(data_min))
            print("[ACTIVITY][STATS][DATA_MAX_VEC] " + _fmt_tensor_stats_1d(data_max))
        else:
            # PER-DIM normalization -> vector min/max
            data_min, data_max, time_max = get_channel_stats(stats_source, input_dim, device)

            print("[ACTIVITY][STATS] normalization=PER_DIM (per-channel)")
            print("[ACTIVITY][STATS] " f"input_dim={input_dim} | time_max={float(time_max):.6f}")
            print("[ACTIVITY][STATS][DATA_MIN] " + _fmt_tensor_stats_1d(data_min))
            print("[ACTIVITY][STATS][DATA_MAX] " + _fmt_tensor_stats_1d(data_max))

        # 6) Data Loaders (UNCHANGED)
        batch_size = args.batch_size
        use_ms = hasattr(args, "multi_scales") and args.multi_scales not in (None, "", [])

        if use_ms:
            from lib.physionet import patch_variable_time_collate_fn_ms
            import re

            scales = [float(x) for x in re.split(r"[,\s]+", args.multi_scales.strip()) if x]
            if getattr(args, "multi_strides", None) in (None, "", []):
                strides = scales[:]
            else:
                strides = [float(x) for x in re.split(r"[,\s]+", args.multi_strides.strip()) if x]
                assert len(scales) == len(strides), "[ACTIVITY][MS] multi_scales and multi_strides length mismatch"

            history_window = float(args.history)

            print(
                "[ACTIVITY][MS][CFG] "
                f"scales={scales} strides={strides} history_window={history_window:.6f}"
            )

            collate_fn_ms = lambda batch: patch_variable_time_collate_fn_ms(
                batch, args, device=device,
                data_min=data_min, data_max=data_max, time_max=time_max,
                scales_hours=scales, strides_hours=strides, history_hours=history_window
            )

            train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn_ms)
            val_dataloader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn_ms)
            test_dataloader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn_ms)

        else:
            if patch_ts:
                collate_fn = patch_variable_time_collate_fn
            else:
                collate_fn = variable_time_collate_fn

            train_dataloader = DataLoader(
                train_data, batch_size=batch_size, shuffle=True,
                collate_fn=lambda batch: collate_fn(
                    batch, args, device, data_type="train",
                    data_min=data_min, data_max=data_max, time_max=time_max
                )
            )
            val_dataloader = DataLoader(
                val_data, batch_size=batch_size, shuffle=False,
                collate_fn=lambda batch: collate_fn(
                    batch, args, device, data_type="val",
                    data_min=data_min, data_max=data_max, time_max=time_max
                )
            )
            test_dataloader = DataLoader(
                test_data, batch_size=batch_size, shuffle=False,
                collate_fn=lambda batch: collate_fn(
                    batch, args, device, data_type="test",
                    data_min=data_min, data_max=data_max, time_max=time_max
                )
            )

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
            "time_max": time_max,
        }
        return data_objects
