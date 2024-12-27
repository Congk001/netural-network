import torch
import torchvision.datasets
from torch import nn

vgg16_false=torchvision.models.vgg16(pretrained=False)
vgg16_true=torchvision.models.vgg16(pretrained=True)

train_data=torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True)
# 新加一层
vgg16_true.classifier.add_module('add_linear',nn.Linear(1000, 10))
print(vgg16_true)

print(vgg16_false)

# 修改现有层
vgg16_false.classifier[6]=nn.Linear(4096, 10)
print(vgg16_false)

# 模型保存方式1: 模型的结构+参数
torch.save(vgg16_false, 'vgg16_false1.pth')

# 模型保存方式2：模型的参数（官方推荐）
torch.save(vgg16_false.state_dict(), 'vgg16_false2.pth')