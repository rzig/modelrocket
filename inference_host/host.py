from multiprocessing import Process
from flask import Flask, request
import onnxruntime as onnx
import os
from lru import LRU
from typing import Any, Dict
import numpy as np
import boto3
import os
import time

def cleanup_session(model, session):
    del session

MAX_SESSIONS = os.environ.get("MAX_SESSIONS", 10)
PORT = os.environ.get("PORT", 5001)

model_sessions = LRU(MAX_SESSIONS, callback=cleanup_session)

s3_session = boto3.session.Session()
s3 = s3_session.client(
    service_name="s3",
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    endpoint_url=os.environ["S3_ENDPOINT"]
)

def evict_model(uuid: str):
    if uuid and uuid in model_sessions:
        del model_sessions[uuid]
    os.remove(uuid)
    

def load_model(uuid: str, callback=None):
    if not uuid in model_sessions:
        print(uuid)
        meta = s3.head_object(Bucket="models", Key=uuid)
        total_length = int(meta.get('ContentLength', 0))
        print("total length: ", total_length)
        downloaded = 0
        percent = 0
        def cb(chunk):
            nonlocal downloaded
            nonlocal percent
            downloaded += chunk
            percent = chunk / total_length
        s3.download_file("models", uuid, uuid)
        session = onnx.InferenceSession(uuid)
        model_sessions[uuid] = session
        return

def generate_model_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    new_dict = {}
    for input_name in inputs:
        # TODO: add image conversion
        if isinstance(inputs[input_name], list):
            new_dict[input_name] = np.array(inputs[input_name], dtype=np.float32)
        else:
            new_dict[input_name] = inputs[input_name]
    return new_dict

app = Flask(__name__)
    
@app.get("/evict")
def handle_evict():
    evict_model(request.args.get("model"))
    return "evicted"

@app.post("/inference")
def handle_inference():
    print ("we are in here")
    data = request.json
    if not data:
        return "bad request"
    model = data.get("model")
    print("got down here")
    load_model(model)
    inputs = data.get("inputs", {})
    if not inputs:
        return "bad request"
    print ("this is good")
    res = model_sessions[model].run(None, generate_model_inputs(inputs))
    outputs = {}
    print ("stopping here ")
    for i, output in enumerate(model_sessions[model].get_outputs()):
        outputs[str(output.name)] = res[i].tolist()
    print("got to here")
    return {"result": outputs}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=PORT)