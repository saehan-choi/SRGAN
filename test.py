# # # import torch
# # # import torch.nn as nn


# # # def PSNR(HRimage, LRimage):
# # #     # peak signal to noise ratio r -> maximam value
# # #     r = 255
# # #     mse = nn.MSELoss()
# # #     mseloss = mse(HRimage, LRimage)
# # #     psnr = 20*torch.log10(r/torch.sqrt(mseloss))
# # #     print(psnr)
# # #     pass



# # # HRimage = torch.randn(1,3,256,256)
# # # LRimage = torch.randn(1,3,256,256)

# # # PSNR(HRimage, LRimage)

# # import os
# # import cv2
# # import random

# # import torch
# # import torch.nn as nn
# # import torch.optim as optim

# # from torch.utils.data import Dataset, DataLoader
# # from torchvision.models import vgg19_bn

# # from albumentations.pytorch.transforms import ToTensorV2
# # from tqdm import tqdm

# # # reference
# # # https://medium.com/analytics-vidhya/super-resolution-gan-srgan-5e10438aec0c
# # # https://www.kaggle.com/code/balraj98/single-image-super-resolution-gan-srgan-pytorch
# # # ->여기도 보기

# # path = './data/train/'
# # result_path = './results/'
# # lid = os.listdir(path)

# # for i in lid:
    
# #     img = cv2.imread(path+i)
# #     if img.shape[0] < 120 or img.shape[1] < 120:
# #         pass
# #     else:
# #         img_lr = cv2.resize(img, (30, 30))
# #         img_hr = cv2.resize(img, (120, 120))
# #         print(i)
# #         cv2.imwrite(result_path+i.replace('.', 'lr.'), img_lr)
# #         cv2.imwrite(result_path+i.replace('.', 'hr.'), img_hr)
        
# #         # cv2.imshow('', img)
# #         # cv2.waitKey(0)
        

# import os
# import cv2
# import random

# import torch
# import torch.nn as nn
# import torch.optim as optim

# from torch.utils.data import Dataset, DataLoader
# from torchvision.models import vgg19_bn

# from albumentations.pytorch.transforms import ToTensorV2
# from tqdm import tqdm

# # reference
# # https://medium.com/analytics-vidhya/super-resolution-gan-srgan-5e10438aec0c
# # https://www.kaggle.com/code/balraj98/single-image-super-resolution-gan-srgan-pytorch
# # ->여기도 보기

# import time

# class CFG:
    
#     valPath = './testdata/'
#     resultPath = './results/'
#     device = 'cuda'
    
#     # srgan recommand lr 0.001
#     lr = 1e-4
#     batch_size = 64
        
#     mseLoss = nn.MSELoss()
#     bceLoss = nn.BCELoss()
#     epochs = 100
    
#     HR_patch_size = 64
#     LR_patch_size = 16

#     weights_path = './weights/Generator_epochs_6.ptGenerator_epochs_99.pt'

# class Generator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4, bias=False)
#         self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1, bias=False)
#         self.conv4 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, padding=4, bias=False)

#         self.bn = nn.BatchNorm2d(64)
#         self.ps = nn.PixelShuffle(2)
#         # I don't understand about this thing nn.PixelShuffle -> ? just for  upscaling?
#         self.prelu = nn.PReLU()

#     def forward(self, x):
#         block1 = self.first_block(x)
        
#         block2 = self.residual_block(block1)
#         block2 = self.residual_block(block2)
#         block2 = self.residual_block(block2)
#         block2 = self.residual_block(block2)
#         block3 = self.third_block(block2, block1)
#         block4 = self.fourth_block(block3)
        
#         return block4

#     def first_block(self, x):
#         return self.prelu(self.conv1(x))
        
#     def residual_block(self, x):
#         return torch.add(self.bn(self.conv2(self.prelu(self.bn(self.conv2(x))))), x)
    
#     def third_block(self, x, skip_data):
#         # 그냥 + 해도 값은 똑같음
#         return torch.add(self.bn(self.conv2(x)), skip_data)
    
#     def fourth_block(self, x):
#         x = self.prelu(self.ps(self.conv3(x)))
#         x = self.prelu(self.ps(self.conv3(x)))
        
#         # ps가 두번들어가면 4배
#         # Rearranges elements in a tensor of shape (B, C*r^2,H,W) -> (B, C, H*r, W*r)
#         x = self.conv4(x)
#         return x


# def resize_img(path):
#     HRimg = cv2.imread(path)
#     # HRimg = cv2.resize(HRimg, (100,100))
#     LRimg = cv2.resize(HRimg, (40,40))
#     return LRimg

# if __name__ == "__main__":

#     G_Model = Generator().to(CFG.device)
#     G_Model.load_state_dict(torch.load(CFG.weights_path))

#     G_Model.eval()
#     with torch.no_grad():
#         lid = os.listdir(CFG.valPath)
#         patchs = []
#         for l in lid:
#             st = time.time()
            
#             path = CFG.valPath+l
#             LRimg = resize_img(path)

#             cv2.imwrite(CFG.resultPath+l.replace('.', '_LR.'), LRimg)
            
#             LRimg = torch.from_numpy(LRimg).to(CFG.device, dtype=torch.float)
#             LRimg = LRimg.permute(2,1,0).unsqueeze(0)
#             result = G_Model(LRimg).squeeze(0).permute(2,1,0).cpu().detach().numpy()

#             # print(result.shape)

#             cv2.imwrite(CFG.resultPath+l.replace('.', '_pred.'), result)

#             ed = time.time()
#             print(f"{round(ed-st,5)}s passed")


info = ["java backend junior pizza 150","python frontend senior chicken 210","python frontend senior chicken 150","cpp backend senior pizza 260","java backend junior chicken 80","python backend senior chicken 50"]
query = ["java and backend and junior and pizza 100","python and frontend and senior and chicken 200","cpp and - and senior and pizza 250","- and backend and senior and - 150","- and - and - and chicken 100","- and - and - and - 150"]


arr = []

for cnt in range(len(query)):
    count = 0
    target = query[cnt].split(" ")
    
    for i in info:
        print(i.split(" ")[cnt])
        
        if i.split(" ")[cnt] == target[cnt]:
            count+=1
        
    arr.append(count)            


print(arr)

