import torch.nn as nn
import torch
from kornia.filters import GaussianBlur2d
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torchvision.utils import save_image

class GaussianFilter(nn.Module):

	def __init__(self, sigma=1, kernel=7):
		super(GaussianFilter, self).__init__()
		self.gaussian_filter = GaussianBlur2d((kernel, kernel), (sigma, sigma))

	def forward(self, image_and_cover):
		image = image_and_cover
		return self.gaussian_filter(image)

img = Image.open("../img/00000.png")
img_np = np.asarray(img)
# print("img RGB:")
# print(img_np)
# print("img pre")
# print(img)
# 将图片转换为tensor
img_tensor = transforms.ToTensor()(img)
#添加一个维度
img_tensor = img_tensor.unsqueeze(0)
print(img_tensor.size())
noised_image = GaussianFilter()(img_tensor)
# noised_image = img_tensor
print(noised_image.size())

# noised_image = img_tensor.squeeze()
# noised_image = noised_image.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
# im = Image.fromarray(noised_image)
# im.save("0.5ResizeAttack_fall.jpg")
save_image(noised_image, "../img_attacked/gaussian_filter_attacked/gaussian_filter_00000.png")