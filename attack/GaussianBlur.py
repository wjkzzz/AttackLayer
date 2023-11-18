import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torchvision.utils import save_image
import math
class GaussianBlur(nn.Module):
    '''
    Adds random noise to a tensor.'''

    def __init__(self, kernel_size=5, opt=None):
        super(GaussianBlur, self).__init__()
        # self.device = config.device
        # self.kernel_size = kernel_size
        self.name = "G_Blur"

    def get_gaussian_kernel(self, kernel_size=5, sigma=3, channels=3):
        # kernel_size = self.kernel_size
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(
                              -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                              (2 * variance)
                          )

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        padding = int((kernel_size-1)/2)
        self.gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, padding=padding, groups=channels, bias=False)

        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False

        return self.gaussian_filter

    def forward(self, tensor, kernel_size=5):
        self.name = "GaussianBlur"
        # blur_result = tensor
        # for kernel in [3, 5, 7, 9]:
        gaussian_layer = self.get_gaussian_kernel(kernel_size)
        blur_result = gaussian_layer(tensor)
            # psnr = self.psnr(self.postprocess(blur_result), self.postprocess(tensor)).item()
            # if psnr>=self.psnr_thresh or kernel==7:
            #     return blur_result, kernel
        ## if none of the above satisfy psnr>30, we abandon the attack
        # print("abandoned gaussian blur, we cannot find a suitable kernel that satisfy PSNR>=25")
        return blur_result

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
noised_image = GaussianBlur()(img_tensor)
# noised_image = img_tensor
print(noised_image.size())

# noised_image = img_tensor.squeeze()
# noised_image = noised_image.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
# im = Image.fromarray(noised_image)
# im.save("0.5ResizeAttack_fall.jpg")
save_image(noised_image, "../img_attacked/gaussian_blur_attacked/gaussian_blur_00000.png")
