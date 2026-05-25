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
import torch  # PURPOSE: PyTorch deep learning framework - defines neural network architecture and layers
import torch.nn as nn  # PURPOSE: PyTorch neural network modules - Conv2d, MaxPool2d, ReLU activation functions
from torchvision import models  # PURPOSE: Pre-trained model zoo - downloads ImageNet-pretrained VGG-16 weights


class CSRNet(nn.Module):
    """CSRNet crowd density estimation model using dilated convolutions."""

    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        
        # VGG-16 architecture configuration: 
        # Numbers are output channels for Conv layers. 'M' stands for Max Pooling (downscaling).
        # This is the frontend that extracts initial features from input images
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        
        # Dilated backend configuration:
        # Designed to expand the receptive field to capture crowd context without downscaling further.
        # Uses dilation=2 to see larger areas without losing spatial resolution
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        
        # torch.nn: Construct the sequential PyTorch modules based on the configs
        # _make_layers(): Dynamically builds Conv and Pooling layers from configuration
        self.frontend = _make_layers(self.frontend_feat)
        self.backend = _make_layers(self.backend_feat, in_channels=512, dilation=True)
        
        # torch.nn.Conv2d: Final output layer collapses the 64 feature channels down to 1 channel (The 2D Heatmap)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if not load_weights:
            # If we aren't loading pre-trained ShanghaiTech weights from disk, we must
            # download the generic ImageNet VGG-16 weights from PyTorch hub to initialize the frontend.
            # torchvision.models: Provides pre-trained models (VGG-16, ResNet, etc.)
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
        
        :param x: Input tensor of shape (batch_size, 3, height, width)
        :return: Output density heatmap of shape (batch_size, 1, height_scaled, width_scaled)
        """
        x = self.frontend(x)  # Extract generic image features using VGG-16 layers
        x = self.backend(x)   # Apply dilated convolutions to create high-res density map
        x = self.output_layer(x)  # Collapse to single-channel heatmap
        return x

    def _initialize_weights(self):
        """
        Initializes the custom backend layers using standard Normal distribution 
        before training. Prevents exploding/vanishing gradients.
        
        torch.nn.init: PyTorch initialization utilities for random weight initialization
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # torch.nn.init.normal_: Initialize Conv2d weights with normal distribution (std=0.01)
                # Small standard deviation prevents extreme activation values
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    # torch.nn.init.constant_: Initialize biases to 0 (neutral starting point)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # BatchNorm weights: initialize scale to 1, bias to 0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def _make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    """
    A helper function to dynamically build PyTorch Sequential layers from an array config.
    
    :param cfg: List defining the architecture (e.g., [64, 'M', 128]).
    :param in_channels: Starting input channels (3 for RGB images, 512 for the backend start).
    :param batch_norm: Whether to apply batch normalization (CSRNet standard is False).
    :param dilation: If True, applies a dilation rate of 2 to expand the receptive field.
    
    :return: torch.nn.Sequential containing all layers
    """
    d_rate = 2 if dilation else 1
    layers = []
    for v in cfg:
        if v == 'M':
            # torch.nn.MaxPool2d: Max Pooling halves the resolution to find higher-level abstract features
            # Reduces spatial dimensions but loses fine-grained location information
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # torch.nn.Conv2d: Standard or Dilated Convolution layer
            # dilation parameter: spacing between kernel weights (1=normal, 2=spaced out)
            # padding parameter: adjusted based on dilation to maintain output size
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3,
                               padding=d_rate,      # Padding = dilation to maintain size
                               dilation=d_rate)      # Dilation for receptive field expansion
            if batch_norm:
                # torch.nn.BatchNorm2d: Normalizes activations for faster, more stable training
                # torch.nn.ReLU: Rectified Linear Unit - non-linear activation (max(0, x))
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v  # Update channel count for next layer
            
    # torch.nn.Sequential: Chains all layers together for forward pass
    return nn.Sequential(*layers)
