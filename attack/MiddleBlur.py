import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torchvision.utils import save_image
from kornia.filters import MedianBlur
class MiddleBlur(nn.Module):

	def __init__(self, kernel=9, opt=None):
		super(MiddleBlur, self).__init__()
		self.middle_filters = {
			3: MedianBlur((3, 3)),
		    5: MedianBlur((5, 5)),
		    7: MedianBlur((7, 7)),
		    9: MedianBlur((9, 9))
		}

	def forward(self, tensor, kernel=9):
		blur_result = self.middle_filters[kernel](tensor)
		return blur_result, kernel

img = Image.open("../img/00000.png")
img_np = np.asarray(img)
# 将图片转换为tensor
img_tensor = transforms.ToTensor()(img)
#添加一个维度
img_tensor = img_tensor.unsqueeze(0)
print(img_tensor.size())
noised_image, kernel = MiddleBlur()(img_tensor)
# noised_image = img_tensor
print(noised_image.size())

save_image(noised_image, "../img_attacked/middle_blur_attacked/middle_blur_00000.png")