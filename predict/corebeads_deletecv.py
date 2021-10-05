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


# [FSFL_FS, FSFL_A, AFL_FS, AFL_A]  --->  [corebeadsFL_FS, corebeadsFL_A]
# leaky_ReLu
# SGD


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.bn1 = nn.BatchNorm1d(4)
        self.fc1 = nn.Linear(4, 20)
        #         self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(inplace=True)
        self.fc2 = nn.Linear(20, 2)
        self.bn2 = nn.BatchNorm1d(2)
        self.bn3 = nn.BatchNorm1d(2)

    def forward(self, x, y):
        out_FS_A_bn = self.bn1(x)
        out = self.fc1(out_FS_A_bn)
        out = self.relu(out)
        out = self.fc2(out)
        out_pre = self.bn2(out)

        y_gt = self.bn3(y)

        return out, y_gt, out_FS_A_bn, out_pre

        # out_FS_A_bn 85*4
        # out_pre     85*2


net = Net()

dataPath = 'E:/working/predict/corebeads_deletecv.csv'
data = pd.read_csv(dataPath)

FSFL_FS_vector = list(data['FSFL_FS'])
FSFL_A_vector = list(data['FSFL_A'])

AFL_FS_vector = list(data['AFL_FS'])
AFL_A_vector = list(data['AFL_A'])

corebeadsFS = list(data['corebeadsFL_FS'])
corebeadsA = list(data['corebeadsFL_A'])

# torch.set_default_tensor_type(torch.DoubleTensor)

FSFL_AFL_vector = []
corebeadsFS_A_s = []
for i in range(len(FSFL_FS_vector)):
    FSFL_AFL_vector.append([FSFL_FS_vector[i], FSFL_A_vector[i], AFL_FS_vector[i], AFL_A_vector[i]])
    corebeadsFS_A_s.append([corebeadsFS[i], corebeadsA[i]])

FSFL_AFL_vector = torch.tensor(FSFL_AFL_vector)
corebeadsFS_A_s = torch.tensor(corebeadsFS_A_s)
# corebeadsFS_A_s.clone().detach().requires_grad_(True)


print(FSFL_AFL_vector.shape, corebeadsFS_A_s.shape)

running_loss = []

FS_A_pre_points = []  # 85*2*epochs

epochs = 50000

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.Adam(net.parameters())


for i in tqdm(range(epochs)):

    net.train()

    out, y_gt, out_FS_A_bn, out_pre = net(FSFL_AFL_vector, corebeadsFS_A_s)

    out_FS_A_bn = out_FS_A_bn.detach().numpy()

    FS_A_pre_points.append(out_pre.detach().numpy())

    criterion = nn.MSELoss()

    loss = criterion(out, y_gt)

    running_loss.append(loss.detach().numpy())

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if i % 5000 == 0:
        net.eval()

        print(loss)

plt.plot(running_loss)
plt.show()

print('last loss:{}'.format(loss))

FS_A_pre_points_last = FS_A_pre_points[-1]  # predicted points

FS_pre_points_last = []  # x
A_pre_points_last = []  # y

for FS_, A_ in FS_A_pre_points_last:
    FS_pre_points_last.append(FS_)
    A_pre_points_last.append(A_)

FS_A_FL_vector = []  # FS_A points   #  z

for FS_FS, FS_A, A_FS, A_A in out_FS_A_bn:
    t = ((FS_FS ** 2 + FS_A ** 2) + (A_FS ** 2 + A_A ** 2)) ** 0.5
    FS_A_FL_vector.append(t)

len(FS_A_FL_vector)

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(30, 30))
ax = fig.add_subplot(111, projection='3d')

X_ = FS_pre_points_last
Y_ = A_pre_points_last

Z = FS_A_FL_vector

rm_and_lstp_price = ax.plot_trisurf(X_, Y_, Z, color='green')

ax.set_xlabel('FS_pre_points_last')
ax.set_ylabel('A_pre_points_last')
ax.set_zlabel('FS_A_FL_vector')
plt.show()


