# generator.py

import torch
import torch.nn as nn
from torchvision import models

# ==========================
# Convolutional Building Blocks
# ==========================

class ConvBlock(nn.Module):
    """
    A convolutional block consisting of two convolutional layers,
    each followed by batch normalization and a ReLU activation.
    """

    def __init__(self, num_filters):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters // 2,
            kernel_size=3,
            padding=1
        )
        self.batchnorm1 = nn.BatchNorm2d(num_filters // 2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=num_filters // 2,
            out_channels=num_filters // 4,
            kernel_size=3,
            padding=1
        )
        self.batchnorm2 = nn.BatchNorm2d(num_filters // 4)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)

        return x

# ==========================
# Decoder Block
# ==========================

class Decoder(nn.Module):
    """
    Decoder block that performs upsampling using transposed convolution,
    concatenates with skip connections, and applies a convolutional block.
    """

    def __init__(self, skip_channels, num_filters):
        super(Decoder, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=2,
            stride=2
        )
        self.conv_block = ConvBlock(num_filters + skip_channels)

    def forward(self, inputs, skip_layer):
        x = self.conv_transpose(inputs)
        x = torch.cat([x, skip_layer], dim=1)  # Concatenate along channel dimension
        x = self.conv_block(x)
        return x

# ==========================
# Convolution Utility Function
# ==========================

def conv3x3(in_planes, out_planes, stride=1):
    """
    Utility function to create a 3x3 convolution with padding.
    """
    return nn.Conv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )

# ==========================
# Attention Mechanisms
# ==========================

class ChannelAttention(nn.Module):
    """
    Channel Attention Module as described in CBAM.
    Applies both average and max pooling, followed by shared MLP and sigmoid activation.
    """

    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.sharedMLP(self.avg_pool(x))
        max_out = self.sharedMLP(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module as described in CBAM.
    Applies average and max pooling along the channel axis,
    concatenates them, and applies a convolution followed by sigmoid activation.
    """

    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compute average and max along the channel axis
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate along the channel dimension
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Sequentially applies channel and spatial attention.
    """

    def __init__(self, inplanes, planes, stride=1):
        super(CBAMBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        # Convolutional layers
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply Channel Attention
        out = self.ca(out) * out
        # Apply Spatial Attention
        out = self.sa(out) * out

        return out

# ==========================
# Generator Model
# ==========================

class Generator(nn.Module):
    """
    Generator model based on a U-Net architecture with CBAM attention blocks.
    Utilizes a pre-trained VGG16 as the encoder.
    """

    def __init__(self, input_channels=3, num_classes=1):
        super(Generator, self).__init__()
        # Load pre-trained VGG16 and extract feature layers
        vgg16_features = models.vgg16(pretrained=True).features

        # Encoder layers
        self.encoder1 = vgg16_features[:4]    
        self.encoder2 = vgg16_features[4:9]   
        self.encoder3 = vgg16_features[9:16]  
        self.encoder4 = vgg16_features[16:23] 
        self.encoder5 = vgg16_features[23:30] 

        # CBAM Attention blocks
        self.att1 = CBAMBlock(inplanes=512, planes=512)
        self.att2 = CBAMBlock(inplanes=256, planes=256)
        self.att3 = CBAMBlock(inplanes=128, planes=128)
        self.att4 = CBAMBlock(inplanes=64, planes=64)

        # Decoder layers with skip connections
        self.decoder1 = Decoder(skip_channels=512, num_filters=512)
        self.decoder2 = Decoder(skip_channels=256, num_filters=256)
        self.decoder3 = Decoder(skip_channels=128, num_filters=128)
        self.decoder4 = Decoder(skip_channels=64, num_filters=64)

        # Final output convolution
        self.output_conv = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder forward pass
        s1 = self.encoder1(x)  # Output size: (64, H, W)
        s2 = self.encoder2(s1) # Output size: (128, H/2, W/2)
        s3 = self.encoder3(s2) # Output size: (256, H/4, W/4)
        s4 = self.encoder4(s3) # Output size: (512, H/8, W/8)
        b1 = self.encoder5(s4) # Bottleneck: (512, H/16, W/16)

        # Apply attention to bottleneck
        s4 = self.att1(s4)
        d1 = self.decoder1(b1, s4)

        # Apply attention and decode
        s3 = self.att2(s3)
        d2 = self.decoder2(d1, s3)

        s2 = self.att3(s2)
        d3 = self.decoder3(d2, s2)

        s1 = self.att4(s1)
        d4 = self.decoder4(d3, s1)

        # Final output
        output = self.output_conv(d4)

        return output

# ==========================
# Testing the Generator
# ==========================

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    generator = Generator().to(device)
    generator.eval()  

    input_image = torch.rand((1, 3, 224, 224)).to(device)  # Example input image

    with torch.no_grad():
        generated_output = generator(input_image)
    
    print(f"Generator output shape: {generated_output.shape}")
