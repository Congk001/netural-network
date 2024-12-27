# 把model_pretrained内的定义和类引入
from model_pretrained import *

# 加载方式1>>对应保存方式1
model1 = torch.load('vgg16_false1.pth')
print(model1)
# 加载方式2>>对应保存方式2
# model2=torch.load('vgg16_false2.pth')
# print(model2)
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load('vgg16_false2.pth'))
print(vgg16)
