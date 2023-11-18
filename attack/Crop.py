import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms
from torchvision.utils import save_image


class Crop(nn.Module):
	"""
	Crop randomly sized images and rescale to original size
	"""
	def __init__(self):
		super(Crop, self).__init__()

	def get_random_rectangle_inside(self, image_shape, height_ratio, width_ratio):
		'''返回随机的长方形区域(返回顶点坐标)'''
		#图片大小和待裁剪图片大小
		image_height = image_shape[2]
		image_width = image_shape[3]

		remaining_height = int(height_ratio * image_height)
		remaining_width = int(width_ratio * image_width)

		#裁剪的起始坐标点
		if remaining_height == image_height:
			height_start = 0
		else:
			height_start = np.random.randint(0, image_height - remaining_height)

		if remaining_width == image_width:
			width_start = 0
		else:
			width_start = np.random.randint(0, image_width - remaining_width)

		return height_start, height_start + remaining_height, width_start, width_start + remaining_width

	def forward(self, image, apex=None, min_rate=0.5, max_rate=1.0):
		if min_rate:
			#获取介于min_rate和max_rate之间的随机比例
			self.height_ratio = min_rate + (max_rate-min_rate) * np.random.rand()
			self.width_ratio = min_rate + (max_rate-min_rate) * np.random.rand()
		else:
			self.height_ratio = 0.3 + 0.7 * np.random.rand()
			self.width_ratio = 0.3 + 0.7 * np.random.rand()

		#减小高和宽比例上的差距
		self.height_ratio = min(self.height_ratio,self.width_ratio+0.2)
		self.width_ratio = min(self.width_ratio,self.height_ratio+0.2)

		#直接获取 or 随机获取 裁剪区域(顶点)
		if apex is not None:
			h_start, h_end, w_start, w_end = apex
		else:
			h_start, h_end, w_start, w_end = self.get_random_rectangle_inside(image.shape, self.height_ratio,
																	 self.width_ratio)
		new_images = image[:, :, h_start: h_end, w_start: w_end]

		#重新放缩到原图大小
		scaled_images = F.interpolate(
			new_images,
			size=[image.shape[2], image.shape[3]],
			mode='bilinear')

		return scaled_images, (h_start, h_end, w_start, w_end)

img = Image.open("../img/00000.png")
img_np = np.asarray(img)
# 将图片转换为tensor
img_tensor = transforms.ToTensor()(img)
#添加一个维度
img_tensor = img_tensor.unsqueeze(0)
print(img_tensor.size())
noised_image, pos= Crop()(img_tensor)
# noised_image = img_tensor
print(noised_image.size())

save_image(noised_image, "../img_attacked/crop_attacked/crop_00000.png")