import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms
from torchvision.utils import save_image

class SaltPepperNoise(nn.Module):

	def __init__(self, prob = 0.03):
		super(SaltPepperNoise, self).__init__()
		self.prob = prob

	def sp_noise(self, image, prob):
		prob_zero = prob / 2
		prob_one = 1 - prob_zero
		rdn = torch.rand(image.shape).to(image.device)

		output = torch.where(rdn > prob_one, torch.zeros_like(image).to(image.device), image)
		output = torch.where(rdn < prob_zero, torch.ones_like(output).to(output.device), output)

		return output

	def forward(self, image):
		# image, cover_image = image_and_cover
		return self.sp_noise(image, self.prob)


img = Image.open("../img/00000.png")
img_np = np.asarray(img)
# 将图片转换为tensor
img_tensor = transforms.ToTensor()(img)
#添加一个维度
img_tensor = img_tensor.unsqueeze(0)
print(img_tensor.size())
noised_image = SaltPepperNoise()(img_tensor)
# noised_image = img_tensor
print(noised_image.size())

save_image(noised_image, "../img_attacked/salt_pepper_attacked/salt_pepper_noise_00000.png")