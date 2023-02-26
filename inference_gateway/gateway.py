import redis
import os
import hmac
import random
from flask import Flask, request
import functools
import hashlib
import requests
from flask_cors import CORS

REDIS_HOST = os.environ["REDIS_HOST"]
REDIS_PORT = os.environ["REDIS_PORT"]
REDIS_DB = int(os.environ["REDIS_DB"])
REDIS_PASSWORD = os.environ["REDIS_PASSWORD"]
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, password=REDIS_PASSWORD)

app = Flask(__name__)
CORS(app)

print("test print")

def validate_token_access(token: str, model_uuid: str) -> bool:
    corresponding_uuid = r.get(f'token:{token}')
    if not corresponding_uuid:
        return None
    corresponding_uuid = corresponding_uuid.decode("UTF-8")
    return corresponding_uuid == model_uuid # TODO: make secure against timing attacks

def get_inference_server(model_uuid: str) -> str:
    return r.srandmember(f'model:{model_uuid}:shard').decode("UTF-8")

def sha256(s: str) -> str:
    return hashlib.sha256(bytes(s, "UTF-8")).hexdigest()

@app.post("/inference")
def process_inference():
    print("hello")
    data = request.json
    if not data:
        return "bad request"
    model = data.get("model", "")
   
    #token = sha256(data.get("token", ""))
    # if not validate_token_access(token, model):
    #     return "Permission denied"
    # del data["token"]
    print("got down here")
    upstream = get_inference_server(model)
    print(f"chose {upstream}")
    # r.publish('inference', model)
    res = requests.post(f'http://{upstream}/inference', json=data)
    print ("finisehd response")
    return res.json()