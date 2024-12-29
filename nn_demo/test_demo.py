import torch
import torchvision
from PIL import Image

image_path = '../practice_dataset/train/ants_image/0013035.jpg'
image = Image.open(image_path)
print(image)
image = image.convert('RGB')
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape)
# 如果加载的是GPU训练的模型，在只有CPU的机器上运行，需要设置map_location
model = torch.load('net1_9.pth', map_location=torch.device('cpu'))
print(model)
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)

print(output.argmax(dim=1))
