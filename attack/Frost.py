import numpy as np
import torch
import torch.nn as nn
import cv2

from PIL import Image
from torchvision import datasets, transforms
from torchvision.utils import save_image


class Frost(nn.Module):
    def __init__(self, coef = [0.8,0.6]):
        super(Frost, self).__init__()
        self.coef = coef

    def frost(self, image):
        idx = np.random.randint(5)
        filename = ['../frost/frost1.png',
                    '../frost/frost2.png',
                    '../frost/frost3.png',
                    '../frost/frost4.jpg',
                    '../frost/frost5.jpg',
                    '../frost/frost6.jpg'][idx]
        frost = transforms.ToTensor()(Image.open(filename).resize( (2048, 2048) ) )[0:3,:,:]
        print("frost shape")
        print(frost.shape)
        # randomly crop and convert to rgb
        x_start, y_start = np.random.randint(0, frost.shape[1] - 1024), np.random.randint(0, frost.shape[2] - 1024)
        frost = frost[:, x_start:x_start + 1024, y_start:y_start + 1024].numpy()
        res = torch.tensor(self.coef[0] * image.numpy() + self.coef[1] * frost).to(image.device)
        return res

    def forward(self, images):
        batch_size = images.shape[0]
        attacked_images = torch.empty_like(images)
        for idx in range(batch_size):
            image = images[idx]
            attacked_images[idx:idx + 1] = self.frost(image)
        return attacked_images

img = Image.open("../img/00000.png")
img_np = np.asarray(img)

# 将图片转换为tensor
img_tensor = transforms.ToTensor()(img)
#添加一个维度
img_tensor = img_tensor.unsqueeze(0)
print(img_tensor.size())
noised_image = Frost()(img_tensor)
print(noised_image.size())

save_image(noised_image, "../img_attacked/frost_attacked/frost_00000.png")