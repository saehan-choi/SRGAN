import torch
import torch.nn as nn

# reference
# https://medium.com/analytics-vidhya/super-resolution-gan-srgan-5e10438aec0c

class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4, bias=False)
        # I don't understand about the padding -> for same padding
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)
        
        self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.conv3_2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1, bias=False)
        
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=3, kernel_size=9, padding=4, bias=False)
        
        self.bn = nn.BatchNorm2d(64)
        self.ps = nn.PixelShuffle(2)
        # I don't understand about this thing nn.PixelShuffle -> ? just for  upscaling?
        self.prelu = nn.PReLU()

    def __forward__(self, input):
        x = self.prelu(self.conv1(input))
        b = self.prelu(self.bn(self.conv2(x)))
        
    def B_residual_block(self, input):
        output = self.conv2(self.prelu(self.bn(self.conv2(input))))
        
    
class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False)

        self.conv2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=1, bias=False)
        
        nn.Flatten()
        self.fc = nn.Linear
        
        
        self.bn = nn.BatchNorm2d(64)
        self.ps = nn.PixelShuffle(2)
        # I don't understand about this thing nn.PixelShuffle -> ? just for  upscaling?
        self.prelu = nn.PReLU()
        
                                