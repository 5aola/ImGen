import torch.nn as nn

# Mask conv network
def MaskNet(dim, mask_size):
    output_dim = 1
    layers, cur_size = [], 1
    while cur_size < mask_size:
        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        layers.append(nn.BatchNorm2d(dim))
        layers.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        cur_size *= 2
    if cur_size != mask_size:
        raise ValueError('Mask size must be a power of 2')
    layers.append(nn.Conv2d(dim, output_dim, kernel_size=1))
    return nn.Sequential(*layers)