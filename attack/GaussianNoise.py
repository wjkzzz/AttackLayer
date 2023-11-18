import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, transforms
from torchvision.utils import save_image
import math

class GaussianNoise(nn.Module):

	def __init__(self, var=0.3, mean=0):
		super(GaussianNoise, self).__init__()
		self.var = var
		self.mean = mean

	def gaussian_noise(self, image, mean, var):
		noise = torch.Tensor(np.random.normal(mean, var ** 0.5, image.shape)).to(image.device)
		out = image + noise
		return out

	def forward(self, image_and_cover):
		image= image_and_cover
		print("image.shape")
		print(image.shape)
		return self.gaussian_noise(image, self.mean, self.var)

img = Image.open("../img/00000.png")
img_np = np.asarray(img)

# 将图片转换为tensor
img_tensor = transforms.ToTensor()(img)
#添加一个维度
img_tensor = img_tensor.unsqueeze(0)
print(img_tensor.size())
noised_image = GaussianNoise()(img_tensor)
print(noised_image.size())

save_image(noised_image, "../img_attacked/gaussian_noise_attacked/gaussian_noise_00000.png")