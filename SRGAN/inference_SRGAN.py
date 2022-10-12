import os
import cv2
import random

import torch
import torch.nn as nn

import time

class CFG:
    
    valPath = './data/val/'
    resultPath = './results/'
    device = 'cuda'
    
    # srgan recommand lr 0.001
    lr = 1e-4
    batch_size = 64
        
    mseLoss = nn.MSELoss()
    bceLoss = nn.BCELoss()
    epochs = 100
    
    HR_patch_size = 64
    LR_patch_size = 16
    
    HR_img_size = 160
    LR_img_size = 40

    weights_path = './weights/epoch99_generator.pt.pt'

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

def resize_img(path):
    HRimg = cv2.imread(path)
    HRimg = cv2.resize(HRimg, (CFG.HR_img_size, CFG.HR_img_size))
    LRimg = cv2.resize(HRimg, (CFG.LR_img_size, CFG.LR_img_size))
    return HRimg, LRimg

if __name__ == "__main__":
    set_seed(42)

    G_Model = Generator().to(CFG.device)
    G_Model.load_state_dict(torch.load(CFG.weights_path))

    G_Model.eval()
    with torch.no_grad():
        lid = os.listdir(CFG.valPath)
        patchs = []
        for l in lid:
            st = time.time()
            
            path = CFG.valPath+l
            HRimg, LRimg = resize_img(path)
            # print(HRimg.shape)
            cv2.imwrite(CFG.resultPath+l.replace('.', '_LR.'), LRimg)
            cv2.imwrite(CFG.resultPath+l.replace('.', '_HR.'), HRimg)
            
            LRimg = torch.from_numpy(LRimg).to(CFG.device, dtype=torch.float)
            LRimg = LRimg.permute(2,1,0).unsqueeze(0)
            result = G_Model(LRimg).squeeze(0).permute(2,1,0).cpu().detach().numpy()

            # print(result.shape)

            cv2.imwrite(CFG.resultPath+l.replace('.', '_pred.'), result)

            ed = time.time()
            print(f"{round(ed-st,5)}s passed")

# reference
# https://arxiv.org/abs/1609.04802
# https://medium.com/analytics-vidhya/super-resolution-gan-srgan-5e10438aec0c
# https://www.kaggle.com/code/balraj98/single-image-super-resolution-gan-srgan-pytorch