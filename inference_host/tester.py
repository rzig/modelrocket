import requests
import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import onnxruntime as ort

transform=transforms.Compose([
        transforms.ToTensor()
        ])
dataset1 = datasets.MNIST('../data', train=True, download=True,
                    transform=transform)
dataset2 = datasets.MNIST('../data', train=False,
                    transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1, 128)
test_loader = torch.utils.data.DataLoader(dataset2, 1000)




for data, target in test_loader:
    plt.imshow(data[0][0], cmap=plt.get_cmap('gray'))
    plt.show()
    test = (data[0][0]).reshape((1, 1, 28, 28))
    print(test)
    ort_sess = ort.InferenceSession('../python_client/digit_recog.onnx')
    outputs = ort_sess.run(None, {'modelInput': test.tolist()})
    print(outputs)
    break

#test = np.random.rand(1,1,28,28)
test_list = test.tolist()
res = requests.post("http://127.0.0.1:5000/inference", json={"token": "54a55a23-339c-437b-9569-0275334bb4c1", "model": "a7181b35-8670-46e6-b3d6-d57e57d596db", "inputs": {"modelInput": test_list}})
print(res.content)