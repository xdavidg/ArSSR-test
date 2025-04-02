import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

# Define the updated RC layer with clamping and absolute value
class RandomConvolutionWithNonLinearity(nn.Module):
    def __init__(self, kernel_size=1, in_channels=1, out_channels=1, min_val=0,
                  max_val=2, clamp_min=0, clamp_max=256,bias=True,min_val_bias=0,
                  max_val_bias=0.01):
        super(RandomConvolutionWithNonLinearity, self).__init__()
        self.max_val=max_val
        self.min_val=min_val
        self.min_val_bias=min_val_bias
        self.max_val_bias=max_val_bias

        self.bias=bias
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, bias=self.bias)
        self.reset_parameters()
        self.clamp_min = clamp_min  # for clamping
        self.clamp_max = clamp_max  # for clamping

    def reset_parameters(self):
        # Sample weights from a uniform distribution and reshift to be zero-centered
        with torch.no_grad():
            self.conv.weight.uniform_(self.min_val, self.max_val)
            self.conv.weight -= (self.max_val - self.min_val) / 2  # zero-centering

            if self.bias:
                self.conv.bias.uniform_(self.min_val_bias, self.max_val_bias)
                self.conv.bias -= (self.max_val_bias - self.min_val_bias) / 2  # zero-centering

    def forward(self, x):
        # Apply convolution
        self.reset_parameters()
        x = self.conv(x)

        return x

# Define the contrast augmentation module with the new non-linear mappings
class RCContrastAugmentationWithNonLinearity(nn.Module):
    def __init__(self, num_layers=4, kernel_size=1,
                  negative_slope=0.2):

        super(RCContrastAugmentationWithNonLinearity, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(RandomConvolutionWithNonLinearity(kernel_size=kernel_size))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope))  # LeakyReLU activation
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    

rc_augmentation = RCContrastAugmentationWithNonLinearity(num_layers=4,
                                                                 kernel_size=1,
                                                                 negative_slope=0.2
                                                                 ).cuda()