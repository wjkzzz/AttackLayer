import numpy as np
import torch
import torch.nn as nn
import cv2
import skimage as sk

from PIL import Image
from torchvision import datasets, transforms
from torchvision.utils import save_image


class Saturate(nn.Module):
    def __init__(self, coef = (2,0)):
        super(Saturate, self).__init__()
        self.coef = coef

    def saturate(self, image):
        image = sk.color.rgb2hsv(image.numpy().transpose(1,2,0))
        image[:, :, 1] = np.clip(image[:, :, 1] * self.coef[0] + self.coef[1], 0, 1)
        res = sk.color.hsv2rgb(image)
        return transforms.ToTensor()(res)

    def forward(self, images):
        batch_size = images.shape[0]
        attacked_images = torch.empty_like(images)
        for idx in range(batch_size):
            image = images[idx]
            attacked_images[idx:idx + 1] = self.saturate(image)
        return attacked_images

img = Image.open("../img/00000.png")
img_np = np.asarray(img)

# 将图片转换为tensor
img_tensor = transforms.ToTensor()(img)
#添加一个维度
img_tensor = img_tensor.unsqueeze(0)
print(img_tensor.size())
noised_image = Saturate()(img_tensor)
print(noised_image.size())

save_image(noised_image, "../img_attacked/saturate_attacked/saturate_00000.png")