import torch
import torch.nn as nn

# reference
# https://medium.com/analytics-vidhya/super-resolution-gan-srgan-5e10438aec0c

class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4, bias=False)
        # I don't understand about the padding -> for same padding
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=1, bias=False)
        
        self.conv3_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=1, bias=False)
        self.conv3_2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=1, bias=False)
        
        self.conv4 = nn.Conv2d(in_channels=9, out_channels=64, kernel_size=9, padding=4, bias=False)
        
        self.bn = nn.BatchNorm2d(64)
        self.ps = nn.PixelShuffle(2)
        # I don't understand about this thing nn.PixelShuffle -> ? just for  upscaling?
        self.prelu = nn.PReLU()

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False)

        self.conv2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        
        self.conv3_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=1, bias=False)
        self.conv3_2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=1, bias=False)
        
        self.conv4 = nn.Conv2d(in_channels=9, out_channels=64, kernel_size=9, padding=4, bias=False)
        
        self.bn = nn.BatchNorm2d(64)
        self.ps = nn.PixelShuffle(2)
        # I don't understand about this thing nn.PixelShuffle -> ? just for  upscaling?
        self.prelu = nn.PReLU()
        
                                