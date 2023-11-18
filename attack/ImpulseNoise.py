import numpy as np
import torch
import torch.nn as nn
import skimage as sk

from PIL import Image
from torchvision import datasets, transforms
from torchvision.utils import save_image
import math

class ImpulseNoise(nn.Module):
    def __init__(self, coef = 0.09):
        super(ImpulseNoise, self).__init__()
        self.coef = coef

    def impulse_noise(self, image):
        res = torch.tensor(sk.util.random_noise(image, mode='s&p', amount=self.coef)).to(image.device)
        return res

    def forward(self, images):
        batch_size = images.shape[0]
        attacked_images = torch.empty_like(images)
        for idx in range(batch_size):
            image = images[idx]
            attacked_images[idx:idx + 1] = self.impulse_noise(image)
        return attacked_images

img = Image.open("../img/00000.png")
img_np = np.asarray(img)

# 将图片转换为tensor
img_tensor = transforms.ToTensor()(img)
#添加一个维度
img_tensor = img_tensor.unsqueeze(0)
print(img_tensor.size())
noised_image = ImpulseNoise()(img_tensor)
print(noised_image.size())

save_image(noised_image, "../img_attacked/impulse_noise_attacked/impulse_noise_00000.png")