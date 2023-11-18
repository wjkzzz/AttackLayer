import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from attack import *
from attack.Resize import *
from attack.Crop import *
from attack.Dropout import *
from attack.Gaussian import *
from attack.GaussianBlur import *
from attack.GaussianFilter import *
from attack.GaussianNoise import *
from attack.MiddleBlur import *
from attack.SaltPepperNoise import *
from attack.JpegCompression import *

from PIL import Image

from torchvision import datasets, transforms
from torchvision.utils import save_image, flow_to_image

img = Image.open("../img/00000.png")
img_np = np.asarray(img)
# 将图片转换为tensor
img_tensor = transforms.ToTensor()(img)
#添加一个维度
img_tensor = img_tensor.unsqueeze(0)
#______________________________________________________________________________________________________
commands = [c for c in input("input commands list: ").split()]
print(commands)
noised_image = img_tensor
device = torch.device("cpu")
#_____________________________________________________________________________
for command in commands:
    print(command)
    if command == "Jpeg":
        noised_image = JpegCompression(device)(noised_image)
        continue
    if command == "Resize":
        noised_image = Resize(interpolation_method='nearest')(noised_image)
        continue
    if command == "Crop":
        noised_image, pos = Crop()(noised_image)
        continue
    if command == "Dropout":
        noised_image = Dropout()(noised_image,img_tensor)
        continue
    if command == "Gaussian":
        noised_image = Gaussian()(noised_image)
        continue
    if command == 'GaussianBlur':
        noised_image = GaussianBlur()(noised_image)
        continue
    if command == 'GaussianFilter':
        noised_image = GaussianFilter()(noised_image)
        continue
    if command == 'GaussianNoise':
        noised_image = GaussianNoise()(noised_image)
        continue
    if command == 'MiddleBlur':
        noised_image, kernel = MiddleBlur()(noised_image)
        continue
    if command == 'SaltPepperNoise':
        noised_image = SaltPepperNoise()(noised_image)
        continue
#________________________________________________________
save_image(noised_image,"../img_attacked/test/test1.png")
#______________________________________________________________________________________________________
# print(img_tensor.size())
# noised_image = Gaussian()(img_tensor)
# noised_image = Dropout()(noised_image,img_tensor)jpeg
# # noised_image = img_tensor
# print(noised_image.size())
#
# save_image(noised_image, "../img_attacked/dropout_attacked/gaussian_dropout_00000.png")
# device = torch.device("cpu")
#
# noised_image = JpegCompression(device)(noised_image)
#
# save_image(noised_image, "../img_attacked/combined_attacked/gaussian_dropout_jpegcompression_00000.png")
