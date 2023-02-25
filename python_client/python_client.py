import numpy as np
import torch.onnx
import torch
from io import BytesIO
import requests
from torchvision import datasets, transforms
import test
def upload(model, input_type, input_shape):

    # set the model to inference mode 
    model.eval() 
    bytes_io_model = BytesIO()
    # Let's create a dummy input tensor  
    dummy_input = torch.randn(input_shape, requires_grad=True)  

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "test.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}) 
    print('Model has been converted to ONNX') 

if __name__ == "__main__":
    model = test.Network() 
    model.load_state_dict(torch.load("./mymodel.pth"))
    input_shape = (1, 4) # input for single input
    input_type = np.float64

    upload(model, input_type, input_shape)
