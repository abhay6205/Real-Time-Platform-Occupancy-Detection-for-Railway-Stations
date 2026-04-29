"""
CSRNet: Dilated Convolutional Neural Networks for Understanding
the Highly Congested Scenes (CVPR 2018).

Why CSRNet instead of YOLO?
Traditional Object Detectors (like YOLO or SSD) rely on drawing bounding boxes around targets. 
In extremely dense crowds (e.g., 500+ people on a railway platform), people heavily occlude 
(block) each other. Bounding boxes overlap to the point of failure, and the model severely undercounts.

CSRNet solves this by ignoring bounding boxes entirely. Instead, it predicts a "Density Map".
A density map is a 2D heatmap where the pixel intensity represents the likelihood of a human head.
By integrating (summing up) the values across this heatmap, the model yields a highly accurate 
overall crowd count, even when it can't distinguish individual bodies.

Architecture:
1. Frontend: The first 10 convolutional layers of VGG-16 (a famous image classification network). 
   This acts as a powerful generic feature extractor.
2. Backend: A series of "Dilated Convolutions". Normal convolutions lose resolution (spatial data) 
   as they pool deeper. Dilated convolutions expand the receptive field (the area the filter looks at) 
   without losing resolution, which is critical for mapping out exact crowd locations on a high-res heatmap.
"""
import torch
import torch.nn as nn
from torchvision import models


class CSRNet(nn.Module):
    """CSRNet crowd density estimation model."""

    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        
        # VGG-16 architecture configuration: 
        # Numbers are output channels for Conv layers. 'M' stands for Max Pooling (downscaling).
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        
        # Dilated backend configuration:
        # Designed to expand the receptive field to capture crowd context without downscaling further.
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        
        # Construct the sequential PyTorch modules based on the configs
        self.frontend = _make_layers(self.frontend_feat)
        self.backend = _make_layers(self.backend_feat, in_channels=512, dilation=True)
        
        # Final output layer collapses the 64 feature channels down to 1 channel (The 2D Heatmap)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if not load_weights:
            # If we aren't loading pre-trained ShanghaiTech weights from disk, we must
            # download the generic ImageNet VGG-16 weights from PyTorch hub to initialize the frontend.
            mod = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            self._initialize_weights()
            
            # Manually copy the weights from the downloaded VGG model into our frontend
            frontend_state = list(self.frontend.state_dict().items())
            vgg_state = list(mod.state_dict().items())
            for i in range(len(frontend_state)):
                frontend_state[i][1].data[:] = vgg_state[i][1].data[:]

    def forward(self, x):
        """
        The forward pass of the neural network.
        Data flows: Input Image -> VGG Frontend -> Dilated Backend -> 1x1 Conv Output Heatmap.
        """
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        """
        Initializes the custom backend layers using standard Normal distribution 
        before training. Prevents exploding/vanishing gradients.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def _make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    """
    A helper function to dynamically build PyTorch Sequential layers from an array config.
    
    :param cfg: List defining the architecture (e.g., [64, 'M', 128]).
    :param in_channels: Starting input channels (3 for RGB images, 512 for the backend start).
    :param batch_norm: Whether to apply batch normalization (CSRNet standard is False).
    :param dilation: If True, applies a dilation rate of 2 to expand the receptive field.
    """
    d_rate = 2 if dilation else 1
    layers = []
    for v in cfg:
        if v == 'M':
            # Max Pooling halves the resolution to find higher-level abstract features
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # Standard or Dilated Convolution
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3,
                               padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
            
    return nn.Sequential(*layers)
