import requests
from multiprocessing import Pool
import json
import time
def post_request(query = "Hola"):
    url = "http://localhost:1521/tesing"

    payload = {
        "query": query
    }

    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers).json()

    print(response["response"])
    print(response["tps"])
    return response



def work_log(work_):
    outputs = []
    with open(f"./benchmark_data/alpaca_instructions_{work_}.json") as f:
        data = json.load(f)
    for i in data[:100]:
        outputs.append(post_request(i))
    with open(f"./run_log/results_{work_}.json", "w") as f:
        json.dump(outputs, f, ensure_ascii = False)

def pool_handler():
    p = Pool(5)
    work = (0, 1, 2, 3, 4)
    p.map(work_log, work)

pool_handler()