import numpy as np
import torch
import torch.nn as nn
import cv2

from skimage.filters import gaussian
from PIL import Image
from torchvision import datasets, transforms
from torchvision.utils import save_image

class Spatter(nn.Module):
    def __init__(self, coef = (0.65, 0.3, 3, 0.68, 0.6, 0)):
        super(Spatter, self).__init__()
        self.coef = coef

    def spatter(self, image):
        image = image.numpy().transpose(1,2,0)
        x = np.array(image, dtype=np.float32)
        liquid_layer = np.random.normal(size=x.shape[:2], loc=self.coef[0], scale=self.coef[1])
        liquid_layer = gaussian(liquid_layer, sigma=self.coef[2])
        liquid_layer[liquid_layer < self.coef[3]] = 0
        if self.coef[5] == 0:
            liquid_layer = (liquid_layer * 255).astype(np.uint8)
            dist = 255 - cv2.Canny(liquid_layer, 50, 150)
            dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
            _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
            dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
            dist = cv2.equalizeHist(dist)
            ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
            dist = cv2.filter2D(dist, cv2.CV_8U, ker)
            dist = cv2.blur(dist, (3, 3)).astype(np.float32)

            m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
            m /= np.max(m, axis=(0, 1))
            m *= self.coef[4]

            # water is pale turqouise
            color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                    238 / 255. * np.ones_like(m[..., :1]),
                                    238 / 255. * np.ones_like(m[..., :1])), axis=2)

            color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

            res = cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR)
        else:
            m = np.where(liquid_layer > self.coef[3], 1, 0)
            m = gaussian(m.astype(np.float32), sigma=self.coef[4])
            m[m < 0.8] = 0

            # mud brown
            color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
                                    42 / 255. * np.ones_like(x[...,:1]),
                                   20 / 255. * np.ones_like(x[..., :1])), axis = 2)

            color *= m[..., np.newaxis]
            x *= (1 - m[..., np.newaxis])

            res = np.clip(x + color, 0, 1)

        return transforms.ToTensor()(res)

    def forward(self, images):
        batch_size = images.shape[0]
        attacked_images = torch.empty_like(images)
        print(attacked_images.shape)
        for idx in range(batch_size):
            image = images[idx]
            attacked_images[idx:idx + 1] = self.spatter(image)
        return attacked_images


img = Image.open("../img/00000.png")
# img_np = np.asarray(img)
# print(img_np)
# 将图片转换为tensor
img_tensor = transforms.ToTensor()(img)
#添加一个维度
img_tensor = img_tensor.unsqueeze(0)
# print(img_tensor.size())
noised_image = Spatter()(img_tensor)
# print(noised_image.size())

save_image(noised_image, "../img_attacked/spatter_attacked/spatter_00000.png")