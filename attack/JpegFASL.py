import numpy as np
import torch
import torch.nn as nn
import cv2
from utils.JPEG import DiffJPEG
from PIL import Image
from torchvision import datasets, transforms
from torchvision.utils import save_image

class JpegFASL(nn.Module):
    """
	Jpeg use Forward ASL method
    """
    def __init__(self):
        super(JpegFASL, self).__init__()

    def real_world_attacking_on_ndarray(self,grid, qf_after_blur):
        realworld_attack = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).contiguous().to('cpu', torch.uint8).numpy()

        _, realworld_attack = cv2.imencode('.jpeg', realworld_attack,
                                           (int(cv2.IMWRITE_JPEG_QUALITY), qf_after_blur))
        realworld_attack = cv2.imdecode(realworld_attack, cv2.IMREAD_UNCHANGED)

        realworld_attack = realworld_attack.astype(np.float32) / 255.
        realworld_attack = torch.from_numpy(
            np.ascontiguousarray(np.transpose(realworld_attack, (2, 0, 1)))).contiguous().float()
        realworld_attack = realworld_attack.unsqueeze(0)
        return realworld_attack

    def clamp_with_grad(self,tensor):
        tensor_clamp = torch.clamp(tensor, 0, 1)
        return tensor + (tensor_clamp - tensor).clone().detach()

    def forward(self, attacked_forward, quality):
        batch_size, height_width = attacked_forward.shape[0], attacked_forward.shape[2]
        attacked_real_jpeg = torch.empty_like(attacked_forward)

        #用DiffJPEG作为模拟Jpeg攻击
        jpeg_layer_after_blurring = DiffJPEG(quality, height=attacked_forward.shape[2], width=attacked_forward.shape[2])
        jpeg_result = jpeg_layer_after_blurring(attacked_forward)

        attacked_real_jpeg_simulate = self.clamp_with_grad(jpeg_result)

        for idx_atkimg in range(batch_size):
            grid = attacked_forward[idx_atkimg]
            realworld_attack = self.real_world_attacking_on_ndarray(grid=grid, qf_after_blur=quality)#真实的Jpeg攻击
            attacked_real_jpeg[idx_atkimg:idx_atkimg + 1] = realworld_attack

        attacked_real_jpeg = attacked_real_jpeg.clone().detach()
        attacked_image = attacked_real_jpeg_simulate + (attacked_real_jpeg - attacked_real_jpeg_simulate).clone().detach()

        return attacked_image, attacked_real_jpeg_simulate, quality

img = Image.open("../img/00000.png")
img_np = np.asarray(img)

# 将图片转换为tensor
img_tensor = transforms.ToTensor()(img)
#添加一个维度
img_tensor = img_tensor.unsqueeze(0)
noised_image, attacked_real_jpeg_simulate, quality= JpegFASL()(img_tensor,50)

save_image(noised_image, "../img_attacked/jpegFASL_attacked/jpegFASL_00000.png")