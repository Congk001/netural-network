# # 读取图片
# # \d相当于[0,9]，匹配所有数字字符
# # \D相当于[^0,9],匹配所有非数字字符
# # \w匹配包括下划线的任何单词字符，等价于[A-Za-z0-9_]
# import re
#
# a = '<img src="https://s-media-cache-ak0.pinimg.com/originals/a8/c4/9e/a8c49ef606e0e1f3ee39a7b219b5c05e.jpg">'
#
# # 使用 re.search
# search = re.search('<img src="(.*)">', a)
# # group(0) 是一个完整的分组
# print(search.group(0))
# print(search.group(1))
#
# # 使用 re.findall
# findall = re.findall('<img src="(.*)">', a)
# print(findall)
# # 多个分组的使用（比如我们需要提取 img 字段和图片地址字段）
# re_search = re.search('<(.*) src="(.*)">', a)
# # 打印 img
# print(re_search.group(1))
# # 打印图片地址
# print(re_search.group(2))
# # 打印 img 和图片地址，以元祖的形式
# print(re_search.group(1, 2))
# # 或者使用 groups
# print(re_search.groups())
# ----------------------------------------------------------------------
# 下载、存储数据集
# import os
# from torch.utils.data import Dataset
# class MyData(Dataset):
#     def __init__(self, root_dir, label_dir):
#         self.root_dir = root_dir
#         self.label_dir = label_dir
#         self.path = os.path.join(self.root_dir, self.label_dir)
#         self.img_path = os.listdir(self.path)
#
#     def __getitem__(self, index):
#         img_name = self.img_path[index]
#         img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
#         img = Image.open(img_item_path)
#         label = self.label_dir
#         return img, label
#
#     def __len__(self):
#         return len(self.img_path)
#
# # 给定根文件路径
# root_dir = "D:\\database\\python_study\\model\\hymenoptera_data\\train"
# ants_label_dir = "ants"
# bees_label_dir = "bees"
# ants_dataset = MyData(root_dir, ants_label_dir)
# bees_dataset = MyData(root_dir, bees_label_dir)
# train_dataset = ants_dataset + bees_dataset
# ------------------------------------------------------------------------
# 下载标准数据集CIFAR10
# import torchvision
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from torchvision import transforms
#
# dataset_transform = transforms.Compose([
#     transforms.ToTensor()
# ])
# # 存储数据集信息
# train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=dataset_transform, download=True)
# test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=dataset_transform, download=True)
# # 打印第一个数据的tensor信息
# print(test_set[0])
# # 打印分类信息
# print(test_set.classes)
# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# # 打印图片，不过tensor数据类型无法使用img.show()函数
# # img.show(img)
# print(test_set[0])
# --------------------------------------------------------------------
# # 使用tensorboard查看数据信息
# # 打开终端 运行‘tensorboard --logdir=地址 --port=自己指定的端口’
# writer = SummaryWriter(log_dir='./DataShow')
# for i in range(10):
#     img, target = test_set[i]
#     writer.add_image('dataset_test', img, global_step=i)
# writer.close()
# # 使用dataloader读取数据
# test_loader = DataLoader(test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
# img, target = test_set[0]
# print(img.shape)
# print(target)
# # 使用tensorboard查看数据信息
# writer = SummaryWriter('./Dataloader')
# step = 0
# for data in test_loader:
#     imgs, targets = data
#     # print(imgs.shape)
#     # print(targets)
#     writer.add_images('test_dataloader', imgs, step)
#     step += 1
# writer.close()
# -----------------------------------------------------------------------
# # transform的使用
# from PIL import Image
# from torch.utils.tensorboard import SummaryWriter
# from torchvision import transforms
#
# img_path = "practice/train/ants_image/6240338_93729615ec.jpg"
# img = Image.open(img_path)
# writer = SummaryWriter('logs')
# # 在python中的使用：
# # 1、创建具体的工具
# # 2、使用自己的工具（输入、；输出）
# tensor_trans = transforms.ToTensor()
# tensor_img = tensor_trans(img)
#
# writer.add_image("tensor_img", tensor_img, global_step=5)
# writer.close()
# --------------------------------------------------------------------
# # 一般transofrom的用法
# from PIL import Image
# from torch.utils.tensorboard import SummaryWriter
# from torchvision import transforms
#
# writer = SummaryWriter('logs')
# img = Image.open('D:\\database\\python_study\\model\\practice\\train\\ants_image\\69639610_95e0de17aa.jpg')
# # ToTensor
# trans_totensor = transforms.ToTensor()
# img_tensor = trans_totensor(img)
# writer.add_image('imgtensor', img_tensor)
#
# # Normalize 归一化
# print(img_tensor[0][0][0])
# trans_norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# img_norm = trans_norm(img_tensor)
# print(img_norm[0][0][0])
# writer.add_image('normalize', img_norm)
#
# # Resize
# print(img.size)
# trans_resize = transforms.Resize((512, 512))
# # img PIL -> resize ->img_resize PIL
# img_resize = trans_resize(img)
# img_resize = trans_totensor(img_resize)
# print(img_resize)
# writer.add_image('resize', img_resize, 0)
#
# # Compose - resize - 2
# trans_resize_2 = transforms.Resize(512)
# trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
# img_resize_2 = trans_compose(img)
# writer.add_image('compose', img_resize_2, 1)
#
# # RandomCrop()
# trans_random = transforms.RandomCrop(100)
# trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
# for i in range(10):
#     img_crop = trans_compose_2(img)
#     writer.add_image('random_crop', img_crop, i)
# writer.close()
# ---------------------------------------------------------------------------
# 正式搭建神经网络模型啦
import torch
import torchvision
from torch import nn
from torch.nn import CrossEntropyLoss, Sequential, MaxPool2d, Conv2d, Flatten, Linear

# 下载、存储数据集
dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=8, shuffle=True, drop_last=True)


# 卷积层
# # convolution layer
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 3, stride=1, padding=1)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x

# 只搭建卷积层的神经网络用tensorboard查看信息
# net = Net()
# writer = SummaryWriter(log_dir='./logs')
# step = 0
# for data in dataloader:
#     imgs, targets = data
#     output = net(imgs)
#     writer.add_images('input', imgs, global_step=step)
#     output = torch.reshape(output, (-1, 3, 32, 32))
#     writer.add_images('output', output, global_step=step)
#     step = step + 1
# writer.close()
# input=torch.tensor([[1,2,0,3,1],
#                     [0,1,2,3,1],
#                     [1,2,1,0,0],
#                     [5,2,3,1,1],
#                     [2,1,0,1,1]])
# input=torch.reshape(input,(-1,1,5,5))
# 池化层
# Pooling layer
# class Pool(nn.Module):
#     def __init__(self):
#         super(Pool,self).__init__()
#         self.maxpool1=MaxPool2d(3,ceil_mode=False)
#
#     def forward(self,x):
#         output=self.maxpool1(x)
#         return output
#
# pool=Pool()
# output=pool(input)
# print(output)
# writer = SummaryWriter(log_dir='./pool_logs')
# step = 0
# for data in dataloader:
#     imgs, targets = data
#     writer.add_images('pool_input', imgs, global_step=step)
#     output=pool(imgs)
#     writer.add_images('pool_output', output, global_step=step)
#     step = step + 1
# writer.close()
# kernel=torch.tensor([[1,2,1],
#                      [0,1,0],
#                      [2,1,0]])
# ---------------------------------------------------------------
# 了解一些函数
# torch.nn.function
# input=torch.reshape(input,(1,1,5,5))
# kernel=torch.reshape(kernel,(1,1,3,3))
# print(input.shape)
# print(kernel.shape)
# output=F.conv2d(input,kernel,stride=1)
# print(output)
#
# output2=F.conv2d(input,kernel,stride=2)
# print(output2)
#
# output3=F.conv2d(input,kernel,stride=1,padding=1)
# print(output3)
# -------------------------------------------------------
# structure of natural network
# class Demo_nn(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, input):
#         output = input + 1
#         return output
#
#
# demo_nn = Demo_nn()
# x = torch.tensor(1.0)
# output = demo_nn(x)
# print(output)
# -----------------------------------------------------
# #Nonlinear Activations
# input=torch.tensor([[1,-0.5],
#                     [-1,3]])
# # ReLU
# class Relu(nn.Module):
#     def __init__(self):
#         super(Relu, self).__init__()
#         self.relu1=ReLU(False)
#     def forward(self, input):
#         output=self.relu1(input)
#         return output
# relu=Relu()
# output=relu(input)
# print(output)
# sigmoid
# class Sigo(nn.Module):
#     def __init__(self):
#         super(Sigo, self).__init__()
#         self.sigo1=Sigmoid()
#     def forward(self, input):
#         output=self.sigo1(input)
#         return output
# sigo=Sigo()
# -----------------------------------------------------------------
# # linear
# class Lin(nn.Module):
#     def __init__(self):
#         super(Lin, self).__init__()
#         self.lin1 = Linear(196608, 10)
#
#     def forward(self, input):
#         output = self.lin1(input)
#         return output
#
#
# lin = Lin()
# # writer = SummaryWriter(log_dir='./nonlinear_logs')
# step = 0
# for data in dataloader:
#     imgs, targets = data
#     print(imgs.shape)
#     output = torch.flatten(imgs)
#     print(output.shape)
#     output = lin(output)
#     print(output.shape)
#     # writer.add_images('input', imgs, global_step=step)
#     # writer.add_images('output', output, global_step=step)
#     step = step + 1
# # writer.close()
# ------------------------------------------------------------------
# Sequentical 用法
# class Seq(nn.Module):
#     def __init__(self):
#         super(Seq, self).__init__()
#         self.model1 = Sequential(
#             Conv2d(3, 32, 5, padding=2),
#             MaxPool2d(2),
#             Conv2d(32, 32, 5, padding=2),
#             MaxPool2d(2),
#             Conv2d(32, 64, 5, padding=2),
#             MaxPool2d(2),
#             Flatten(),
#             Linear(1024, 64),
#             Linear(64, 10)
#         )
#
#     def forward(self, x):
#         x = self.model1(x)
#         return x
#
# seq = Seq()
# optim = torch.optim.SGD(seq.parameters(), lr=0.01)
# loss = CrossEntropyLoss()
# for epoch in range(20):
#     running_loss = 0.0
#     for data in dataloader:
#         imgs, targets = data
#         outputs = seq(imgs)
#         result_loss = loss(outputs, targets)
#         optim.zero_grad()
#         result_loss.backward()
#         optim.step()
#         running_loss = running_loss + result_loss
#     print(running_loss)
# ------------------------------------------------------
# # 反向传播
# inputs = torch.tensor([1, 2, 3], dtype=torch.float)
# targets = torch.tensor([1, 2, 5], dtype=torch.float)
# loss = L1Loss(reduction='sum')
# result = loss(inputs, targets)
# loss_mse = MSELoss()
# result1 = loss_mse(inputs, targets)
# print(result1)
# x = torch.tensor([0.1, 0.2, 0.3])
# y = torch.tensor([1])
# x = torch.reshape(x, (1, 3))
# loss_nn = CrossEntropyLoss()
# result_loss = loss_nn(x, y)
# print(result_loss)
# print(-0.2 + log(exp(0.1) + exp(0.2) + exp(0.3)))
