import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torchvision.utils import save_image

class Gaussian(nn.Module):
    '''Adds random noise to a tensor.'''
    def __init__(self):
        super(Gaussian, self).__init__()

    def forward(self, tensor, cover_image=None, mean=0, stddev=0.15):

        self.name="Gaussian"
        noise = torch.nn.init.normal_(torch.Tensor(tensor.size()), mean, stddev)
        mixed = tensor + noise
        mixed = torch.clamp(mixed, 0, 1)

        return mixed

img = Image.open("../img/00000.png")
img_np = np.asarray(img)
# 将图片转换为tensor
img_tensor = transforms.ToTensor()(img)
#添加一个维度
img_tensor = img_tensor.unsqueeze(0)
print(img_tensor.size())
noised_image = Gaussian()(img_tensor)
# noised_image = img_tensor
print(noised_image.size())

save_image(noised_image, "../img_attacked/gaussian_attacked/gaussian_00000.png")