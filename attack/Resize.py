import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torchvision.utils import save_image

class Resize(nn.Module):
    """
    Resize the image. The target size is original size * resize_ratio
    """
    def __init__(self, interpolation_method='nearest'):
        super(Resize, self).__init__()
        self.interpolation_method = interpolation_method

    def forward(self, img):
        '''缩小到0.5倍再放大到2倍'''
        noised_image = img
        noised_image = F.interpolate(
                                    noised_image,
                                    scale_factor=(0.5,0.5),
                                    mode=self.interpolation_method)
        noised_image = F.interpolate(
                                    noised_image,
                                    scale_factor=(2,2),
                                    mode=self.interpolation_method)
        return noised_image

img = Image.open("../img/00000.png")
img_np = np.asarray(img)
# print("img RGB:")
# print(img_np)
# print("img pre")
# print(img)
# 将图片转换为tensor
img_tensor = transforms.ToTensor()(img)
#添加一个维度
img_tensor = img_tensor.unsqueeze(0)
print(img_tensor.size())
noised_image = Resize(interpolation_method='nearest')(img_tensor)
# noised_image = img_tensor
print(noised_image.size())

# noised_image = img_tensor.squeeze()
# noised_image = noised_image.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
# im = Image.fromarray(noised_image)
# im.save("0.5ResizeAttack_fall.jpg")
save_image(noised_image, "../img_attacked/resize_attacked/resize_00000.png")