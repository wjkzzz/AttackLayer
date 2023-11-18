import numpy as np
import torch
import torch.nn as nn
import cv2

from PIL import Image
from torchvision import datasets, transforms
from torchvision.utils import save_image

class Pixelate(nn.Module):
    def __init__(self, coef = 0.25):
        super(Pixelate, self).__init__()
        self.coef = coef

    def pixelate(self, image):
        image = image.resize((int(1024 * self.coef), int(1024 * self.coef)), Image.Resampling.BOX)
        res = image.resize((1024, 1024), Image.Resampling.BOX)
        return transforms.ToTensor()(res)

    def forward(self, image_and_cover):
        image= image_and_cover
        return self.pixelate(image)

    # def forward(self, images):
    #     batch_size = images.shape[0]
    #     attacked_images = torch.empty_like(images)
    #     print(attacked_images.shape)
    #     for idx in range(batch_size):
    #         image = images[idx]
    #         attacked_images[idx:idx + 1] = self.pixelate(image)
    #     return attacked_images

img = Image.open("../img/00000.png")
# img_np = np.asarray(img)

# print(img_np)
# 将图片转换为tensor
# img_tensor = transforms.ToTensor()(img)
# #添加一个维度
# img_tensor = img_tensor.unsqueeze(0)
# print(img_tensor.size())
noised_image = Pixelate()(img)
# print(noised_image.size())
#
save_image(noised_image, "../img_attacked/pixelate_attacked/pixelate_00000.png")