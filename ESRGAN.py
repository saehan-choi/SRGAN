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

class CFG:
    
    traindata = './data/train/'
    valdata = './data/val/'
    device = 'cuda'
    
    lr = 2e-4
    batch_size = 1
    beta = 0.2
    
    eta = 1e-2
    Lambda = 5e-3
    
    epochs = 100

    HR_patch_size = 160
    LR_patch_size = 40

    weights_path = './weights/'

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
        model = nn.Linear(in_features=in_features, out_features=out_features).to(CFG.device)
        x = self.leakyrelu(model(x))
        model = nn.Linear(in_features=out_features, out_features=1).to(CFG.device)
        x = model(x)

        return x

    

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19_bn(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features[:6])).to(CFG.device, dtype=torch.float)

    def forward(self, img):
        return self.feature_extractor(img)



class ImageDataset(Dataset):
    def __init__(self, dataPath):

        self.dataPath = dataPath
        self.dataList = []
        
        self.cnt = 0
        for path in os.listdir(self.dataPath):
            self.dataList.append(self.dataPath+path)

        for data in self.dataList:
            HRimage = cv2.imread(data, cv2.COLOR_BGR2RGB)
            h, w = HRimage.shape[:2]
            
            if h < CFG.HR_patch_size and w < CFG.HR_patch_size:
                pass
            else:
                self.cnt+=1
            
    def __len__(self):
        return self.cnt

    def __getitem__(self, index):
        
        HRimage = cv2.imread(self.dataList[index], cv2.COLOR_BGR2RGB)
        HRimage = cv2.resize(HRimage, dsize=(CFG.HR_patch_size, CFG.HR_patch_size))
        h, w = HRimage.shape[:2]
        LRimage = cv2.resize(HRimage, dsize=(round(w/4), round(h/4)))
        
        HRimage = ToTensorV2()(image=HRimage)['image'].to(CFG.device, dtype=torch.float)
        LRimage = ToTensorV2()(image=LRimage)['image'].to(CFG.device, dtype=torch.float)


        return HRimage, LRimage

def PSNR(HRimage, LRimage):
    r = 255
    mse = nn.MSELoss()
    mseloss = mse(HRimage, LRimage)
    psnr = 20*torch.log10(r/torch.sqrt(mseloss))
    return psnr

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(random_seed)
    # np.random.seed(random_seed)

def DiscriminatorLoss(fakeLabel, realLabel):
    bceLoss = nn.BCEWithLogitsLoss()
    sigmoid = nn.Sigmoid()

    d1Loss = bceLoss(realLabel-fakeLabel, torch.ones_like(realLabel, dtype=torch.float))
    d2Loss = bceLoss(fakeLabel-realLabel, torch.zeros_like(fakeLabel, dtype=torch.float))
    
    dLoss = (d1Loss+d2Loss)/2
    
    return dLoss
    
def GeneratorLoss(fakeLabel, G_outputs, HRimages):
    mseLoss = nn.MSELoss()
    bceLoss = nn.BCEWithLogitsLoss()
    
    g1Loss = mseLoss(feature_extractor(G_outputs).detach(), feature_extractor(HRimages).detach())
    
    g2Loss = CFG.Lambda * bceLoss(fakeLabel.detach(), torch.ones_like(fakeLabel))

    g3Loss = CFG.eta * mseLoss(G_outputs, HRimages)
    
    gLoss = (g1Loss+g2Loss+g3Loss)/3

    return gLoss
    
def train_one_epoch(G_model, D_model, G_optimizer, D_optimizer, dataloader, epoch):
    G_model.train()
    D_model.train()
    
    dataset_size = 0
    running_loss = 0

    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))

    for step, data in bar:
        HRimages = data[0]
        LRimages = data[1]

        # Discriminator Loss part
        G_outputs = G_model(LRimages)
        fakeLabel = D_model(G_outputs.detach())
        realLabel = D_model(HRimages)

        dLoss = DiscriminatorLoss(fakeLabel, realLabel)
        
        D_optimizer.zero_grad()
        dLoss.backward(retain_graph=True)
        
        D_optimizer.step()

        G_optimizer.zero_grad()

        # Generator Loss part
        gLoss = GeneratorLoss(fakeLabel, G_outputs, HRimages)
        
        gLoss.backward()
        G_optimizer.step()
        
        running_loss += ((gLoss.item()+dLoss.item())/2)*CFG.batch_size
        dataset_size += CFG.batch_size
        epoch_loss = running_loss/dataset_size

        bar.set_postfix(EPOCH=epoch, TRAIN_LOSS=epoch_loss, PSNR=PSNR(HRimages, G_outputs).item())

def val_one_epoch(G_model, D_model, dataloader, epoch):
    
    G_model.eval()
    D_model.eval()
    with torch.no_grad():
        dataset_size = 0
        running_loss = 0
        
        bar = tqdm(enumerate(dataloader), total=len(dataloader))

        for step, data in bar:
            HRimages = data[0]
            LRimages = data[1]

            # Discriminator Loss part
            G_outputs = G_model(LRimages)
            fakeLabel = D_model(G_outputs.detach())
            realLabel = D_model(HRimages)

            dLoss = DiscriminatorLoss(fakeLabel, realLabel)

            # Generator Loss part
            gLoss = GeneratorLoss(fakeLabel, G_outputs, HRimages)
                        
            running_loss += ((gLoss.item()+dLoss.item())/2)*CFG.batch_size
            dataset_size += CFG.batch_size
            epoch_loss = running_loss/dataset_size

            bar.set_postfix(EPOCH=epoch, VAL_LOSS=epoch_loss, PSNR=PSNR(HRimages, G_outputs).item())

if __name__ == "__main__":
    set_seed(42)

    G_Model = Generator().to(CFG.device)
    D_Model = Discriminator().to(CFG.device)
    
    G_optimizer = optim.Adam(G_Model.parameters(), lr=CFG.lr)
    D_optimizer = optim.Adam(D_Model.parameters(), lr=CFG.lr)

    train_dataset = ImageDataset(CFG.traindata)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=CFG.batch_size)

    val_dataset = ImageDataset(CFG.valdata)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=CFG.batch_size)
    
    feature_extractor = FeatureExtractor()
    feature_extractor.eval()

    for epoch in range(CFG.epochs):
        train_one_epoch(G_Model, D_Model, G_optimizer, D_optimizer, train_loader, epoch)
        print('\n')
        val_one_epoch(G_Model, D_Model, val_loader, epoch)
        print('\n')
        torch.save(G_Model.state_dict(), CFG.weights_path+f'ESRGAN_epoch_{epoch}.pt')

