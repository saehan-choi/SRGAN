import os

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

import cv2

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch.optim as optim

from tqdm import tqdm

# reference
# https://medium.com/analytics-vidhya/super-resolution-gan-srgan-5e10438aec0c

class CFG:
    
    dataPath = './data/'
    device = 'cuda'
    
    lr = 3e-4
    batch_size = 1
    
    discLoss = nn.MSELoss()
    genLoss = nn.MSELoss()
    # mse로 한번 해봅시다

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

class ImageDataset(Dataset):
    def __init__(self, dataPath):
        
        self.dataPath = dataPath
        self.dataList = []
        
        for path in os.listdir(self.dataPath):
            self.dataList.append(CFG.dataPath+path)

    def __len__(self):
        return len(self.dataList)

    def __getitem__(self, index):
        HRimage = cv2.imread(self.dataList[index], cv2.COLOR_BGR2RGB)
        h, w = HRimage.shape[:2]
        LRimage = cv2.resize(HRimage, dsize=(round(w/4), round(h/4)))
        # 이거 사이즈가 맞아야 할듯? 흠... 일단 돌려봅시다.

        HRimage = ToTensorV2()(image=HRimage)
        LRimage = ToTensorV2()(image=LRimage)

        return HRimage, LRimage

# input = torch.randn(1,3,256,256)
# Model = Generator()
# output = Model(input)

# print(f'input size: {input.size()}')
# print(f'output size: {output.size()}')


def train_one_epoch(G_model, D_model, optimizer, dataloader, epoch):
    G_model.train()
    D_model.train()
    
    dataset_size = 0
    running_loss = 0
    
    discLoss = CFG.discLoss
    genLoss = CFG.genLoss
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))

    for step, data in bar:
        # images = data[0].to(device, dtype=torch.float)
        # labels = data[1].to(device, dtype=torch.long)
        HRimages = data[0].to(CFG.device)
        LRimages = data[1].to(CFG.device)

        batch_size = HRimages.size(0)

        # 나중에 dtype 빼보고도 해보기!!! 되는지
        # Discriminator Loss part
        G_outputs = G_model(LRimages).to(CFG.device, dtype=torch.float)
        
        fakeLabel = D_model(G_outputs).to(CFG.device, dtype=torch.float)
        realLabel = D_model(HRimages).to(CFG.device, dtype=torch.float)

        d1Loss = torch.mean(discLoss(fakeLabel, torch.zeros_like(fakeLabel, dtype=torch.float)))
        d2Loss = torch.mean(discLoss(realLabel, torch.ones_like(realLabel, dtype=torch.float)))
        dLoss = d1Loss+d2Loss
        d2Loss.backward()
        d1Loss.backward()
        
        # Generator Loss part
        gLoss = genLoss(fakeLabel, torch.ones_like(fakeLabel))
        

        loss = nn.CrossEntropyLoss()(outputs, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()*batch_size
        dataset_size += batch_size
        epoch_loss = running_loss/dataset_size

        bar.set_postfix(EPOCH=epoch, TRAIN_LOSS=epoch_loss)





if __name__ == "__main__":
    # os.environ['KMP_DUPLICATE_LIB_OK']= True

    G_Model = Generator().to(CFG.device)
    D_Model = Discriminator().to(CFG.device)
    
    G_optimizer = optim.Adam(G_Model.parameters(), lr=CFG.lr)
    D_optimizer = optim.Adam(D_Model.parameters(), lr=CFG.lr)
    # optimizer Adam인지 논문확인
    
    train_dataset = ImageDataset(CFG.dataPath)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=CFG.batch_size)
    
    bar = tqdm(enumerate(train_loader), total=len(train_loader))
    
    for step, data in bar:
        print(data)