# 准备数据集
import torchvision
from torch.utils.data import DataLoader

from model import *
from nn_demo.nn_demo import optim

train_data = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 格式化输出形式
print('训练数据集的长度为：{}'.format(train_data_size))
print('测试数据集的长度为：{}'.format(test_data_size))

# 利用dataloader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# 创建网络模型
net1 = Net()

# 损失函数
loss_func = nn.CrossEntropyLoss()

# 优化器
# 随机梯度下降优化算法
learning_rate = 0.01
# learning_rate=1e-2=1×10^(-2)=0.01
optimizer = optim.SGD(net1.parameters(), learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epochs = 10

for i in range(epochs):
    print('第{}轮训练开始'.format(i + 1))
