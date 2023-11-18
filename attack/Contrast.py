import numpy as np
import torch
import torch.nn as nn
import cv2
import skimage as sk

from PIL import Image
from torchvision import datasets, transforms
from torchvision.utils import save_image


class Contrast(nn.Module):
    def __init__(self, coef = 0.2):
        super(Contrast, self).__init__()
        self.coef = coef

    def contrast(self, image):
        # print("image.shape")
        # print(image.shape)
        # print("np shape")
        # print(image.numpy().shape)
        image = image.numpy().transpose(1,2,0)
        means = np.mean(image, axis = (0, 1), keepdims=True)
        print("means.shape")
        print(means.shape)
        res =(image - means) * self.coef + means
        res = res.transpose(0,1,2)
        return transforms.ToTensor()(res)

    def forward(self, images):
        batch_size = images.shape[0]
        attacked_images = torch.empty_like(images)
        for idx in range(batch_size):
            image = images[idx]
            attacked_images[idx:idx + 1] = self.contrast(image)
        return attacked_images

img = Image.open("../img/00000.png")
img_np = np.asarray(img)

# 将图片转换为tensor
img_tensor = transforms.ToTensor()(img)
#添加一个维度
img_tensor = img_tensor.unsqueeze(0)
print(img_tensor.size())
noised_image = Contrast()(img_tensor)
print(noised_image.size())

save_image(noised_image, "../img_attacked/contrast_attacked/contrast_00000.png")