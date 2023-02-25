import redis
import os
import hmac
import random
from flask import Flask, request
import functools
import hashlib

REDIS_HOST = os.environ["REDIS_HOST"]
REDIS_PORT = os.environ["REDIS_PORT"]
REDIS_DB = int(os.environ["REDIS_DB"])
REDIS_PASSWORD = os.environ["REDIS_PASSWORD"]
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, password=REDIS_PASSWORD)

app = Flask(__name__)

def validate_token_access(token: str, model_uuid: str) -> bool:
    corresponding_uuid = r.get(f'token:{token}')
    if not corresponding_uuid:
        return None
    corresponding_uuid = corresponding_uuid.decode("UTF-8")
    return corresponding_uuid == model_uuid # TODO: make secure against timing attacks

def get_inference_server(model_uuid: str) -> str:
    num_hosts = int(r.get(f'model:{model_uuid}:num_hosts').decode("UTF-8"))
    host_num = random.randrange(num_hosts)
    return r.get(f'model:{model_uuid}:{host_num}').decode("UTF-8")

def sha256(s: str) -> str:
    return hashlib.sha256(bytes(s, "UTF-8")).hexdigest()

@app.route("/inference")
def process_inference():
    model = request.args.get("model", "")
    token = sha256(request.args.get("token", ""))
    print(f"token sha256 is {token}")
    if not validate_token_access(token, model):
        return "Permission denied"
    upstream = get_inference_server(model)
    print(f"selected upstream {upstream}")
    return upstream