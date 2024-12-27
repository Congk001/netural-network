# 准备数据集
import torchvision
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *


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
# 添加tensorboard
writer=SummaryWriter(log_dir='./logs_train')
for i in range(epochs):
    print('-------------第{}轮训练开始-----------------'.format(i + 1))
    # 训练步骤
    for data in train_dataloader:
        imgs,targets = data
        outputs=net1(imgs)
        loss=loss_func(outputs,targets)
        #优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            #.item()
            print('训练次数为{}，损失为{}'.format(total_train_step, loss.item()))
            writer.add_scalar('train_loss', loss.item(), total_train_step)
    # 测试步骤
    total_test_loss=0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets = data
            outputs=net1(imgs)
            loss=loss_func(outputs,targets)
            total_test_loss =total_test_loss+loss.item()
    print('整体测试数据集上的损失为{}'.format(total_test_loss))
    writer.add_scalar('total_test_loss', total_test_loss, total_test_step)
    total_test_step=total_test_step+1

    torch.save(net1, 'net1_{}.pth'.format(i))
    print('模型已保存')
writer.close()



