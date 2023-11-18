import numpy as np
import torch
import torch.nn as nn
import cv2

from PIL import Image
from torchvision import datasets, transforms
from torchvision.utils import save_image


class Fog(nn.Module):
    def __init__(self, coef = [2.5,1.7]):
        super(Fog, self).__init__()
        self.coef = coef

    def diamondsquare(self, mapsize, wibbledecay):
        assert (mapsize & (mapsize - 1) == 0)
        maparray = np.empty((mapsize, mapsize), dtype=np.float_)
        maparray[0, 0] = 0
        stepsize = mapsize
        wibble = 100

        def wibbledmean(array):
            return array / 4 + np.random.uniform(-wibble, wibble, array.shape)

        def fillsquares():
            """For each square of points stepsize apart,
               calculate middle value as mean of points + wibble"""
            cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
            squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
            squareaccum += np.roll(squareaccum, shift=-1, axis=1)
            maparray[stepsize // 2:mapsize:stepsize,
            stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

        def filldiamonds():
            """For each diamond of points stepsize apart,
               calculate middle value as mean of points + wibble"""
            mapsize = maparray.shape[0]
            drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
            ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
            ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
            lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
            ltsum = ldrsum + lulsum
            maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
            tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
            tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
            ttsum = tdrsum + tulsum
            maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

        while stepsize >= 2:
            fillsquares()
            filldiamonds()
            stepsize //= 2
            wibble /= wibbledecay

        maparray -= maparray.min()
        return maparray / maparray.max()

    def fog(self, image):
        max_val = image.max()
        image += self.coef[0] * self.diamondsquare(mapsize = 1024, wibbledecay=self.coef[1])[:1024, :1024][..., np.newaxis].transpose(2,1,0)
        res = image * max_val / (max_val + self.coef[0])
        return res

    def forward(self, images):
        batch_size = images.shape[0]
        attacked_images = torch.empty_like(images)
        for idx in range(batch_size):
            image = images[idx]
            attacked_images[idx:idx + 1] = self.fog(image)
        return attacked_images

img = Image.open("../img/00000.png")
img_np = np.asarray(img)

# 将图片转换为tensor
img_tensor = transforms.ToTensor()(img)
#添加一个维度
img_tensor = img_tensor.unsqueeze(0)
print(img_tensor.size())
noised_image = Fog()(img_tensor)
print(noised_image.size())

save_image(noised_image, "../img_attacked/fog_attacked/fog_00000.png")