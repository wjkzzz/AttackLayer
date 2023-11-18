import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
from torchvision import datasets, transforms
from torchvision.utils import save_image
import math

class DefocusBlur(nn.Module):
    def __init__(self, radius = 8, alias_blur = 0.5):
        super(DefocusBlur, self).__init__()
        self.radius = radius
        self.alias_blur = alias_blur

    def disk(self, radius, alias_blur, dtype=np.float32):
        if radius <= 8:
            L = np.arange(-8, 8 + 1)
            ksize = (3, 3)
        else:
            L = np.arange(-radius, radius + 1)
            ksize = (5, 5)
        X, Y = np.meshgrid(L, L)
        aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
        aliased_disk /= np.sum(aliased_disk)

        # supersample disk to antialias
        return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)

    def defocus_blur(self, image):
        kernel = self.disk(radius=self.radius, alias_blur=self.alias_blur)
        channels = []
        print(image.numpy().shape)
        for d in range(3):
            channels.append(cv2.filter2D(image.numpy()[d, :, :], -1, kernel))
        res = torch.tensor(np.array(channels)).to(image.device)
        # res = ''
        return  res

    def forward(self, images):
        batch_size = images.shape[0]
        attacked_images = torch.empty_like(images)
        for idx in range(batch_size):
            image = images[idx]
            attacked_images[idx:idx + 1] = self.defocus_blur(image)
        return attacked_images

img = Image.open("../img/00000.png")
img_np = np.asarray(img)

# 将图片转换为tensor
img_tensor = transforms.ToTensor()(img)
#添加一个维度
img_tensor = img_tensor.unsqueeze(0)
print(img_tensor.size())
noised_image = DefocusBlur()(img_tensor)
print(noised_image.size())

save_image(noised_image, "../img_attacked/defocus_blur_attacked/defocus_blur_00000.png")