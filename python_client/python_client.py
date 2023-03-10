import numpy as np
import torch.onnx
import torch
from io import BytesIO
import requests
# from torchvision import datasets, transforms
import test
import json as JSON
def upload(model, model_name, input_type, input_shape):

    if model == None or model_name == None or model_name == "" or input_type == None or input_shape == None:
        print("Invalid input")
        return
    # set the model to inference mode 
    model.eval() 
    bytes_io_model = BytesIO()
    # Let's create a dummy input tensor  
    dummy_input = torch.randn(input_shape, requires_grad=True)  

    # Export the model   
    torch.onnx.export(
        model,                                                  # model being run 
        dummy_input,                                            # model input (or a tuple for multiple inputs) 
        bytes_io_model,                                         # where to save the model  
        export_params=True,                                     # store the trained parameter weights inside the model file 
        opset_version=10,                                       # the ONNX version to export the model to 
        do_constant_folding=True,                               # whether to execute constant folding for optimization 
        input_names = ['modelInput'],                           # the model's input names 
        output_names = ['modelOutput'],                         # the model's output names 
        dynamic_axes = {'modelInput'  : {0 : 'batch_size'},     # variable length axes 
                        'modelOutput' : {0 : 'batch_size'}
        }
    ) 
    # print(bytes_io_model.getvalue()) 
    print('Model has been converted to ONNX') 
    requests_url = 'http://127.0.0.1:3000/load_model'
    myobj =  {'name': model_name, 'input': {'type': str(input_type), 'shape': str(input_shape)}}
    response = requests.post(requests_url, json=myobj) #,files={ 'model':bytes_io_model })
    try:
        response_json = response.json()
    except Exception:
        print("Empty response")
        return
    key = response_json['key']
    response2 = requests.post(f"http://127.0.0.1:3000/upload_model_file/{key}", files={ "model": bytes_io_model.getvalue() })
    try:
        rj = response2.json()
    except Exception:
        print("Bad reponse")
        return
    print("key:",key)
    print("token:",response_json["token"])
    print(rj)

if __name__ == "__main__":
    model = test.Network() 
    model.load_state_dict(torch.load("./mymodel.pth"))
    input_shape = (1, 4) # input for single input
    input_type = np.float64
    model_name = 'vj model'

    upload(model, model_name, input_type, input_shape)
