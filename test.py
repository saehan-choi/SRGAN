# import torch
# import torch.nn as nn


# def PSNR(HRimage, LRimage):
#     # peak signal to noise ratio r -> maximam value
#     r = 255
#     mse = nn.MSELoss()
#     mseloss = mse(HRimage, LRimage)
#     psnr = 20*torch.log10(r/torch.sqrt(mseloss))
#     print(psnr)
#     pass



# HRimage = torch.randn(1,3,256,256)
# LRimage = torch.randn(1,3,256,256)

# PSNR(HRimage, LRimage)

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

path = './data/train/'
result_path = './results/'
lid = os.listdir(path)

for i in lid:
    
    img = cv2.imread(path+i)
    if img.shape[0] < 120 or img.shape[1] < 120:
        pass
    else:
        img_lr = cv2.resize(img, (30, 30))
        img_hr = cv2.resize(img, (120, 120))
        print(i)
        cv2.imwrite(result_path+i.replace('.', 'lr.'), img_lr)
        cv2.imwrite(result_path+i.replace('.', 'hr.'), img_hr)
        
        # cv2.imshow('', img)
        # cv2.waitKey(0)
        
