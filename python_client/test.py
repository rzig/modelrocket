from sklearn.datasets import make_classification
import torch.nn as nn
from sklearn.model_selection import train_test_split

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
# number of features (len of X cols)
input_dim = 4
# number of hidden layers
hidden_layers = 25
# number of classes (unique of y)
output_dim = 3
X, Y = make_classification(
  n_samples=100, n_features=4, n_redundant=0,
  n_informative=3,  n_clusters_per_class=2, n_classes=3
)

X_train, X_test, Y_train, Y_test = train_test_split(
  X, Y, test_size=0.33, random_state=42)


class Data(Dataset):
  def __init__(self, X_train, y_train):
    # need to convert float64 to float32 else 
    # will get the following error
    # RuntimeError: expected scalar type Double but found Float
    self.X = torch.from_numpy(X_train.astype(np.float32))
    # need to convert float64 to Long else 
    # will get the following error
    # RuntimeError: expected scalar type Long but found Float
    self.y = torch.from_numpy(y_train).type(torch.LongTensor)
    self.len = self.X.shape[0]
  
  def __getitem__(self, index):
    return self.X[index], self.y[index]
  def __len__(self):
    return self.len

traindata = Data(X_train, Y_train)

batch_size = 4
trainloader = DataLoader(traindata, batch_size=batch_size, 
                         shuffle=True, num_workers=2)

class Network(nn.Module):
  def __init__(self):
    super(Network, self).__init__()
    self.linear1 = nn.Linear(input_dim, hidden_layers)
    self.linear2 = nn.Linear(hidden_layers, output_dim)
  def forward(self, x):
    x = torch.sigmoid(self.linear1(x))
    x = self.linear2(x)
    return x



if __name__ == "__main__":
    clf = Network()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(clf.parameters(), lr=0.1)
    epochs = 2
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            # set optimizer to zero grad to remove previous epoch gradients
            optimizer.zero_grad()
            # forward propagation
            outputs = clf(inputs)
            loss = criterion(outputs, labels)
            # backward propagation
            loss.backward()
            # optimize
            optimizer.step()
            running_loss += loss.item()
        # display statistics
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')
    PATH = './mymodel.pth'
    torch.save(clf.state_dict(), PATH)