from datasets import load_dataset
import json
dataset = load_dataset("tatsu-lab/alpaca", cache_dir = "./cache")

samples = []

for i in dataset["train"]:
    if i["input"] == "":
        samples.append(i["instruction"])
print(len(samples))
for i in range(20):
    with open(f"./benchmark_data/alpaca_instructions_{i}.json", "w") as f:
        json.dump(samples[i*1000: (i+1)*1000], f)