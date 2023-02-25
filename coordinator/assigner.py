import random
from typing import List, Set
import time
import math
from collections import deque
from flask import Flask, request
import redis
import os
import requests

REDIS_HOST = os.environ["REDIS_HOST"]
REDIS_PORT = os.environ["REDIS_PORT"]
REDIS_DB = int(os.environ["REDIS_DB"])
REDIS_PASSWORD = os.environ["REDIS_PASSWORD"]
PORT = os.environ.get("PORT", 5005)
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, password=REDIS_PASSWORD)

def generate_shard():
    return r.srandmember('hosts')

app = Flask(__name__)

@app.get("/generate_shard")
def handle_generate_shard():
    model = request.args.get("model")
    shard = generate_shard()
    r.sadd(f'model:{model}:shard', shard)
    return {"shard": shard.decode("UTF-8")}

@app.get("/update_model")
def handle_update_model():
    model = request.args.get("model")
    # TODO: should this be in redis or some other event queue
    for model_host in r.smembers(f'model:{model}:shard'):
        hostname = model_host.decode("UTF-8")
        requests.get(f'http://{hostname}/evict?model={model}')
    return "updated"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=PORT)