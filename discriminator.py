# discriminator.py

import torch
import torch.nn as nn

# ==========================
# CNN Block for Discriminator
# ==========================

class CNNBlock(nn.Module):
    """
    Basic CNN block used in the discriminator.
    Consists of a convolutional layer followed by batch normalization and ReLU activation.
    """

    def __init__(self, in_channels, out_channels, stride=2):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding_mode='reflect',
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# ==========================
# Discriminator Model
# ==========================

class Discriminator(nn.Module):
    """
    Discriminator model for GAN.
    Takes in both real/fake images and corresponding masks as input.
    """

    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        # Initial layer takes concatenated image and mask
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels + 1,  # +1 for the mask channel
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode='reflect'
            ),
            nn.ReLU(inplace=True)
        )

        layers = []
        in_channels_current = features[0]
        for idx, feature in enumerate(features[1:], start=1):
            stride = 1 if feature == features[-1] else 2
            layers.append(CNNBlock(in_channels_current, feature, stride))
            in_channels_current = feature

        # Final output layer
        layers.append(
            nn.Conv2d(
                in_channels_current,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode='reflect'
            )
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, mask):
        """
        Forward pass for the discriminator.
        :param x: Input image tensor.
        :param mask: Corresponding mask tensor.
        :return: Discriminator output.
        """
        # Concatenate image and mask along the channel dimension
        x = torch.cat([x, mask], dim=1)
        x = self.initial(x)
        x = self.model(x)
        return x

# ==========================
# Testing the Discriminator
# ==========================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    discriminator = Discriminator().to(device)
    discriminator.eval()  


    input_image = torch.rand((1, 3, 224, 224)).to(device)  # Example input image
    dummy_mask = torch.rand((1, 1, 224, 224)).to(device)   # Example mask


    with torch.no_grad():
        discriminator_output = discriminator(input_image, dummy_mask)
    
    print(f"Discriminator output shape: {discriminator_output.shape}")
