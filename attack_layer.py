import torch
import torch.nn as nn
import numpy as np
from attack.JpegFASL import JpegFASL
from attack.JpegCompression import JpegCompression
from attack.JpegSS import JpegSS
from attack.Crop import Crop
from attack.Resize import Resize
from attack.Saturate import Saturate
from attack.Contrast import Contrast
from attack.Brightness import Brightness
from attack.GaussianBlur import GaussianBlur
from attack.MiddleBlur import MiddleBlur
from attack.DefocusBlur import DefocusBlur
from attack.GaussianNoise import GaussianNoise
from attack.ShotNoise import ShotNoise
from attack.SpeckleNoise import SpeckleNoise
from attack.SaltPepperNoise import SaltPepperNoise
from attack.Fog import Fog
from attack.Frost import Frost
from attack.Spatter import Spatter

class AttackLayer(nn.Module):
    def __init__(self, opts):
        super(AttackLayer, self).__init__()
        self.jpeg_layer = JpegFASL()
        self.crop_layer = Crop()
        self.resize_layer = Resize()

        self.jpeg_min = opts['JPEG'][0]
        self.jpeg_max = opts['JPEG'][1]

        self.crop_min = opts['CROP'][0]
        self.crop_max = opts['CROP'][1]

        self.resize_min = opts['RESIZE'][0]
        self.resize_max = opts['RESIZE'][1]

    
    def forward(self, image_tensor):
        prob = np.random.rand()
        if prob < 0.25:
            attacked = image_tensor  # No Attack
        elif prob < 0.5:
            jpeg_quality = np.random.randint(self.jpeg_min, self.jpeg_max)  # JPEG compression
            attacked, _, _ = self.jpeg_layer(image_tensor, jpeg_quality)
        elif prob < 0.75:
            scale_factor = np.random.rand() * (self.resize_max - self.resize_min) + self.resize_min  # Rescaling
            attacked = self.resize_layer(image_tensor, scale_factor)
        else:
            # 获取介于min_rate和max_rate之间的随机比例
            height_ratio = np.random.rand() * (self.crop_max - self.crop_min) + self.crop_min  # Cropping
            width_ratio = np.random.rand() * (self.crop_max - self.crop_min) + self.crop_min
            attacked, _ = self.crop_layer(image_tensor, height_ratio=height_ratio, width_ratio=width_ratio)
        return attacked


