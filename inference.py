import os
import cv2
import random

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg19_bn

from albumentations.pytorch.transforms import ToTensorV2
from tqdm import tqdm

# reference
# https://medium.com/analytics-vidhya/super-resolution-gan-srgan-5e10438aec0c
# https://www.kaggle.com/code/balraj98/single-image-super-resolution-gan-srgan-pytorch
# ->여기도 보기

class CFG:
    
    dataPath = './data/train/hr/'
    device = 'cuda'
    
    # srgan recommand lr 0.001
    lr = 1e-4
    batch_size = 64
        
    mseLoss = nn.MSELoss()
    bceLoss = nn.BCELoss()
    epochs = 100
    
    HR_patch_size = 64
    LR_patch_size = 16

    weights_path = './weights/'

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
        x = self.prelu(self.ps(self.conv3(x)))
        x = self.prelu(self.ps(self.conv3(x)))
        
        # ps가 두번들어가면 4배
        # Rearranges elements in a tensor of shape (B, C*r^2,H,W) -> (B, C, H*r, W*r)
        x = self.conv4(x)
        return x


def make_patch(LRpath):

    lr_img = cv2.imread(LRpath)
    h, w = lr_img.shape[:2]
    lr_patch = []
    for i in range(0, h, CFG.LR_patch_size):
        for j in range(0, w, CFG.LR_patch_size):
            patch = lr_img[i:i+CFG.LR_patch_size, j:j+CFG.LR_patch_size, :]
            # cv2.imwrite(f'./test/lr/{i+CFG.LR_patch_size}_{j+CFG.LR_patch_size}.png', patch)
            lr_patch.append(patch)
    
    return lr_patch

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(random_seed)
    # np.random.seed(random_seed)


if __name__ == "__main__":
    set_seed(42)

    G_Model = Generator().to(CFG.device)
    G_Model.load_state_dict(torch.load(CFG.weights_path+'Generator_epochs_0.pt'))

    G_Model.eval()
    dataPath = './data/'
    lid = os.listdir(dataPath)
    patchs = []
    for l in lid:
        path = dataPath+l
        kk = make_patch(path)
        print(kk)
        
    # 1개만 먼저 테스트해보기
    
    with torch.no_grad():
        randn = torch.randn((1,3,16,16)).to(CFG.device, dtype=torch.float)
        
        
        result = G_Model(randn)
        print(result.size())