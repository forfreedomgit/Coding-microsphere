import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns

torch.set_default_tensor_type(torch.DoubleTensor)

# 85*2 inputs vectors
# corebeadsFL_FS


dataPath = 'E:/working/corebeads/corebeads_vector.csv'
data = pd.read_csv(dataPath)

FSFL_vector = list(data['FSFL_vector'])
AFL_vector = list(data['AFL_vector'])
corebeadsFLFS = list(data['corebeadsFL_FS'])

# torch.set_default_tensor_type(torch.DoubleTensor)

FSFL_AFL_vector = []
corebeadsFL_FS = []

for i in range(len(FSFL_vector)):
    FSFL_AFL_vector.append([FSFL_vector[i], AFL_vector[i]])
    corebeadsFL_FS.append([corebeadsFLFS[i]])

FSFL_AFL_vector = torch.tensor(FSFL_AFL_vector)
corebeadsFL_FS = torch.tensor(corebeadsFL_FS)

# print(FSFL_AFL_vector, corebeadsFL_FS)
print(FSFL_AFL_vector.shape, corebeadsFL_FS.shape)

'''



'''


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.bn1 = nn.BatchNorm1d(2)
        self.fc1 = nn.Linear(2, 10)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(10, 1)
        self.bn2 = nn.BatchNorm1d(1)
        self.bn3 = nn.BatchNorm1d(1)

    #         self.Adam = torch.optim.Adam(self)

    def forward(self, x, y):
        out = self.bn1(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)

        y = self.bn3(y)

        return out, y


net = Net()

# 85*2 inputs vectors
# corebeadsFL_A


dataPath = 'E:/working/corebeads/corebeads_vector.csv'
data = pd.read_csv(dataPath)

FSFL_vector = list(data['FSFL_vector'])
AFL_vector = list(data['AFL_vector'])
corebeadsFLvector = list(data['corebeadsFL_vector'])

# corebeadsFL_FS = list(data['corebeadsFL_FS'])
# corebeadsFL_A = list(data['corebeadsFL_A'])

torch.set_default_tensor_type(torch.DoubleTensor)

FSFL_AFL_vector = []
corebeadsFL_vector = []
for i in range(len(FSFL_vector)):
    FSFL_AFL_vector.append([FSFL_vector[i], AFL_vector[i]])
    corebeadsFL_vector.append([corebeadsFLvector[i]])

FSFL_AFL_vector = torch.tensor(FSFL_AFL_vector)
corebeadsFL_vector = torch.tensor(corebeadsFL_vector)

print(FSFL_AFL_vector.shape, corebeadsFL_vector.shape)

running_loss = []

for i in tqdm(range(50000)):

    #     running_loss = 0.0

    out, y = net(FSFL_AFL_vector, corebeadsFL_vector)

    criterion = nn.MSELoss()

    loss = criterion(out, y)

    #     type(loss)

    #     running_loss.append(int(loss))

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if i % 5000 == 0:
        print(loss)