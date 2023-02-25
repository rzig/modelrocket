from multiprocessing import Process
from flask import Flask, request
import onnxruntime as onnx
import os
from lru import LRU
from typing import Any, Dict
import numpy as np

def cleanup_session(model, session):
    del session

MAX_SESSIONS = os.environ.get("MAX_SESSIONS", 10)
PORT = os.environ.get("PORT", 5001)

model_sessions = LRU(MAX_SESSIONS, callback=cleanup_session)

def evict_model(uuid: str):
    if uuid and uuid in model_sessions:
        del model_sessions[uuid]

def load_model(uuid: str):
    if not uuid in model_sessions:
        session = onnx.InferenceSession(uuid)
        model_sessions[uuid] = session

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
    data = request.json
    if not data:
        return "bad request"
    model = data.get("model")
    load_model(model)
    inputs = data.get("inputs", {})
    if not inputs:
        return "bad request"
    res = model_sessions[model].run(None, generate_model_inputs(inputs))
    outputs = {}
    for i, output in enumerate(model_sessions[model].get_outputs()):
        outputs[str(output.name)] = res[i].tolist()
    return {"result": outputs}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=PORT)