import os
import cv2
import random

import torch
import torch.nn as nn

class CFG:
    valPath = './testdata/'    
    resultPath = './results/'
    device = 'cuda'

    lr = 1e-4
    
    mseLoss = nn.MSELoss()
    bceLoss = nn.BCELoss()
    epochs = 100
    
    HR_patch_size = 160
    LR_patch_size = 40
    
    HR_img_size = 160
    LR_img_size = 40

    weights_path = './weights/epoch99_generator.pt'

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, padding=4, bias=False)

        self.bn = nn.BatchNorm2d(64)
        self.ps = nn.PixelShuffle(2)
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
        return torch.add(self.bn(self.conv2(x)), skip_data)
    
    def fourth_block(self, x):
        x = self.prelu(self.ps(self.conv3(x)))
        x = self.prelu(self.ps(self.conv3(x)))
        x = self.conv4(x)
        return x


def make_patch(LRpath):

    lr_img = cv2.imread(LRpath)
    h, w = lr_img.shape[:2]
    lr_patch = []
    for i in range(0, h, CFG.LR_patch_size):
        for j in range(0, w, CFG.LR_patch_size):
            patch = lr_img[i:i+CFG.LR_patch_size, j:j+CFG.LR_patch_size, :]
            lr_patch.append(patch)

    return lr_patch

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(random_seed)

def resize_img(path):
    HRimg = cv2.imread(path)
    HRimg = cv2.resize(HRimg, (CFG.HR_img_size, CFG.HR_img_size))
    LRimg = cv2.resize(HRimg, (CFG.LR_img_size, CFG.LR_img_size))
    return HRimg, LRimg
