import os
import cv2
import time
import random

import torch
import torch.nn as nn

class CFG:
    
    valPath = './data/val/'
    resultPath = './results/'
    device = 'cuda'
    
    HR_patch_size = 64
    LR_patch_size = 16

    weights_path = './weights/ESRGAN_epoch_14.pt'
    beta = 0.2
    
    # eta와 lambda는 loss에 들어가는거라서 ㄱㅊ
class DenseBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias)
        
        self.conv2 = nn.Conv2d(in_channels+out_channels, out_channels, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(in_channels+2*out_channels, out_channels, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(in_channels+3*out_channels, out_channels, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(in_channels+4*out_channels, out_channels, 3, 1, 1, bias=bias)
        
        self.relu = nn.ReLU()

    def forward(self, x1):
        x2 = torch.cat((x1, self.relu(self.conv1(x1))), dim=1)
        x3 = torch.cat((x1, self.relu(self.conv1(x1)), self.relu(self.conv2(x2))), dim=1)
        x4 = torch.cat((x1, self.relu(self.conv1(x1)), self.relu(self.conv2(x2)), self.relu(self.conv3(x3))), dim=1)
        x5 = torch.cat((x1, self.relu(self.conv1(x1)), self.relu(self.conv2(x2)), self.relu(self.conv3(x3)), self.relu(self.conv4(x4))), dim=1)
        return self.conv5(x5)


class RRDB(nn.Module):
    def __init__(self, beta=CFG.beta):
        super(RRDB, self).__init__()
        
        self.DenseBlock = DenseBlock()
        self.beta = beta
        
    def forward(self, x1):
        x2 = self.beta * torch.add(x1, self.DenseBlock(x1))
        x3 = self.beta * torch.add(x2, self.DenseBlock(x2))
        x4 = self.beta * torch.add(x3, self.DenseBlock(x3))
        x5 = torch.add(self.beta * x4, x1)

        return x5


class Generator(nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        self.repeat = 3
        
        self.RRDB = RRDB()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.ps = nn.PixelShuffle(4)
        
    def forward(self, x1):
        x1 = self.conv1(x1)
        x2 = self.repeatRRDB(x1)
        x3 = self.ps(torch.add(x1, self.conv2(x2)))
        x3 = self.conv3(x3)
        x4 = self.conv4(x3)
        
        return x4
        
    def repeatRRDB(self, input):
        for i in range(self.repeat):
            if i == 0:
                output = self.RRDB(input)
            else:
                output = self.RRDB(output)       
                     
        return output

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
    HRimg = cv2.resize(HRimg, (100,100))
    LRimg = cv2.resize(HRimg, (25,25))
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

