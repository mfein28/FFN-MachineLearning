import pandas as pandas
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn import model_selection





class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, input):
        out = self.fc1(input)
        out = self.relu(out)
        out = self.fc2(out)
        return out



names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data = pandas.read_csv("iris.csv", names=names)
array = data.values
X = array[:, 0:4]

Y = array[:,4]
validation_size = .30

inputSize = data.size
hiddenSize = 500
numClasses = 3
numEpochs = 5
batchSize = 125
learningRate = 0.01

print(inputSize)
net = Net(inputSize, hiddenSize, numClasses)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learningRate)

for epoch in numEpochs:
    for i, (data, labels) in