import os
import re
import numpy as np

LOG_DIR = "/home/solgi/MSP-IMTS/logs"
pattern = r"Test\s+-\s+Best epoch.*Loss,\s*MSE,\s*RMSE,\s*MAE,\s*MAPE:\s*(\d+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)"

results = []
print("Collecting metrics...\n")

for seed in [1, 2, 3, 4, 5]:   # skip seed 6
    filename = f"activity_single_mixer_seeds_279838_{seed}.err"  # adjust JOBID!
    path = os.path.join(LOG_DIR, filename)

    if not os.path.exists(path):
        print(f"[WARN] Missing file: {path}")
        continue

    with open(path, "r") as f:
        text = f.read()

    matches = re.findall(pattern, text)
    if not matches:
        print(f"[WARN] No test results found in {path}")
        continue

    # take last test result
    epoch, mse, rmse, mae, mape = matches[-1]
    mse = float(mse); rmse = float(rmse); mae = float(mae); mape = float(mape)
    results.append((mse, rmse, mae, mape))
    print(f"Seed {seed}: MSE={mse:.5f}, RMSE={rmse:.5f}, MAE={mae:.5f}, MAPE={mape:.2f}%")

# Compute averages
if results:
    arr = np.array(results)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)

    print("\n================= AVERAGE OVER SEEDS 1–5 =================")
    print(f"MSE  = {mean[0]:.5f} ± {std[0]:.5f}")
    print(f"RMSE = {mean[1]:.5f} ± {std[1]:.5f}")
    print(f"MAE  = {mean[2]:.5f} ± {std[2]:.5f}")
    print(f"MAPE = {mean[3]:.5f} ± {std[3]:.5f}")


#Seed 1: MSE=0.00389, RMSE=0.00389, MAE=0.06235, MAPE=0.04%
#Seed 2: MSE=0.00312, RMSE=0.00312, MAE=0.05589, MAPE=0.04%
#Seed 3: MSE=0.00291, RMSE=0.00291, MAE=0.05399, MAPE=0.03%
#Seed 4: MSE=0.00281, RMSE=0.00281, MAE=0.05304, MAPE=0.03%
#Seed 5: MSE=0.00291, RMSE=0.00291, MAE=0.05393, MAPE=0.03%
#
#================= AVERAGE OVER SEEDS 1–5 =================
#MSE  = 0.00313 ± 0.00039
#RMSE = 0.00313 ± 0.00039
#MAE  = 0.05584 ± 0.00339
#MAPE = 0.03596 ± 0.00431