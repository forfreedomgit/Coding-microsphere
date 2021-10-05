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

# leaky_ReLu

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.bn1 = nn.BatchNorm1d(2)
        self.fc1 = nn.Linear(2, 10)
        self.relu = nn.LeakyReLU(inplace=True)
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
epochs = 50000

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.Adam(net.parameters())


for i in tqdm(range(epochs)):

#     running_loss = 0.0

    out, y = net(FSFL_AFL_vector, corebeadsFL_vector)

    criterion = nn.MSELoss()


    loss = criterion(out, y)

    running_loss.append(loss.detach().numpy())

    optimizer.zero_grad()

    loss.backward()


    optimizer.step()

    if i % 5000 == 0:
        print(loss)

plt.plot(running_loss)
plt.show()


# for i in range(2000):
#     Net.forward(FSFL_A, AFL_FS)


# criterion = nn.MSELoss()
# loss = criterion(x, y)
#         # optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
#         # optimizer.zero_grad()
#         # loss.backward()
#         # # 更新参数
#         # optimizer.step()
#         print(x.size())  # 结果：[1, 6, 30, 30]
#         x = F.max_pool2d(x, (2, 2))  # 我们使用池化层，计算结果是15
#         x = F.relu(x)
#         print(x.size())  # 结果：[1, 6, 15, 15]
#         # reshape，‘-1’表示自适应
#         # 这里做的就是压扁的操作 就是把后面的[1, 6, 15, 15]压扁，变为 [1, 1350]
#         x = x.view(x.size()[0], -1)
#         print(x.size())  # 这里就是fc1层的的输入1350
#         x = self.fc1(x)













#
# FSFL_FS = list(data['FSFL_FS'])
# AFL_FS = list(data['AFL_FS'])
# FSFL_A = list(data['FSFL_A'])
# AFL_A = list(data['AFL_A'])
#
# corebeadsFL_FS = list(data['corebeadsFL_FS'])
# corebeadsFL_A = list(data['corebeadsFL_A'])
#
# FSFL_FS = torch.tensor(FSFL_FS)
# AFL_FS = torch.tensor(AFL_FS)
# FSFL_A = torch.tensor(FSFL_A)
# AFL_A = torch.tensor(AFL_A)
#
# corebeadsFL_FS = torch.tensor(corebeadsFL_FS)
# corebeadsFL_A = torch.tensor(corebeadsFL_A)
#
#
# '''
# corebeadsFL_FS = FSFL_FS_w * FSFL_FS + AFL_FS_w * AFL_FS
# corebeadsFL_A = FSFL_A_w * FSFL_A + AFL_A_w * AFL_A
# '''
#
#
# # print(FSFL_FS.shape)
#
# FSFL_FS_w = torch.randn((1, 85), requires_grad=True)
# AFL_FS_w = torch.randn((1, 85), requires_grad=True)
# FSFL_A_w = torch.randn((1, 85), requires_grad=True)
# AFL_A_w = torch.randn((1, 85), requires_grad=True)
#
# lr = 0.0005
#
# loss1 = []
# loss2 = []
# # loss3 = []
# # loss4 = []
#
#
# for i in tqdm(range(5000)):
#     corebeadsFL_FS_pred = (FSFL_FS_w * FSFL_FS).sum(1) + (AFL_FS_w * AFL_FS).sum(1)
#     corebeadsFL_FS_pred = torch.sigmoid(corebeadsFL_FS_pred)
#
#     corebeadsFL_A_pred = (FSFL_A_w * FSFL_A).sum(1) + (AFL_A_w * AFL_A).sum(1)
#     corebeadsFL_A_pred = torch.sigmoid(corebeadsFL_A_pred)
#
#     corebeadsFL_FS_loss = (corebeadsFL_FS_pred - corebeadsFL_FS).pow(2).sum()
#     loss1.append(corebeadsFL_FS_loss.detach().numpy())
#
#     corebeadsFL_A_loss = (corebeadsFL_A_pred - corebeadsFL_A).pow(2).sum()
#     loss2.append(corebeadsFL_A_loss.detach().numpy())
#
#     corebeadsFL_FS_loss.backward()
#     corebeadsFL_A_loss.backward()
#
#     with torch.no_grad():
#         FSFL_FS_w -= lr*FSFL_FS_w.grad
#         FSFL_FS_w.grad = None
#         AFL_FS_w -= lr*AFL_FS_w.grad
#         AFL_FS_w.grad = None
#         FSFL_A_w -= lr*FSFL_A_w.grad
#         FSFL_A_w.grad = None
#         AFL_A_w -= lr*AFL_A_w.grad
#         AFL_A_w.grad = None
#
#
# plt.plot(loss1)
# plt.show()

