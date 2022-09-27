import torch
import torch.nn as nn

# reference
# https://medium.com/analytics-vidhya/super-resolution-gan-srgan-5e10438aec0c

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, padding=4, bias=False)
        
        self.bn = nn.BatchNorm2d(64)
        self.ps = nn.PixelShuffle(2)
        # I don't understand about this thing nn.PixelShuffle -> ? just for  upscaling?
        self.prelu = nn.PReLU()

    def forward(self, x):
        block1 = self.first_block(x)
        
        block2 = self.residual_block(block1)
        block2 = self.residual_block(block2)
        block2 = self.residual_block(block2)
        block2 = self.residual_block(block2)
        block3 = self.third_block(block2, block1)
        block4 = self.fourth_block(block3)
        
        return block4
        
    def first_block(self, x):
        return self.prelu(self.conv1(x))
        
    def residual_block(self, x):
        return torch.add(self.bn(self.conv2(self.prelu(self.bn(self.conv2(x))))), x)
    
    def third_block(self, x, skip_data):
        # 그냥 + 해도 값은 똑같음
        return torch.add(self.bn(self.conv2(x)), skip_data)
    
    def fourth_block(self, x):
        # print(x.size())
        # print(self.conv3(x).size())
        # print(self.ps(self.conv3(x)).size())
        x = self.prelu(self.ps(self.conv3(x)))
        x = self.prelu(self.ps(self.conv3(x)))
        
        # ps가 두번들어가면 4배
        # Rearranges elements in a tensor of shape (B, C*r^2,H,W) -> (B, C, H*r, W*r)
        x = self.conv4(x)
        return x
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = self.CNNblock(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = self.CNNblock(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3 = self.CNNblock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = self.CNNblock(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv5 = self.CNNblock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = self.CNNblock(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv7 = self.CNNblock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv8 = self.CNNblock(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False)
        
        self.leakyrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x).flatten()
        result = self.MLPblock(x, in_features=x.size(dim=0))
        return result
    
    def CNNblock(self, in_channels, out_channels, kernel_size, padding, stride, bias):
        
        return nn.Sequential(
                            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias),
                            nn.BatchNorm2d(out_channels),
                            nn.LeakyReLU()
                            )

    def MLPblock(self, x, in_features, out_features=1024):
        model = nn.Linear(in_features=in_features, out_features=out_features)
        x = self.leakyrelu(model(x))
        model = nn.Linear(in_features=out_features, out_features=1)
        x = self.sigmoid(model(x))
        
        return x
        


input = torch.randn(1,3,256,256)
Model = Generator()
output = Model(input)

print(f'input size: {input.size()}')
print(f'output size: {output.size()}')








