import os 
import json
import numpy as np
results_dir = "./run_log/"

list_files = os.listdir(results_dir)

tps_ar = []
for i in list_files:
    with open(results_dir + i) as f:
        data = json.load(f)
    tps_ar += [j["tps"] for j in data]

print("Mean Tokens per second : ", np.mean(tps_ar))